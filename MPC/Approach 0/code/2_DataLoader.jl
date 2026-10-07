# #############################################################################
# DataLoader.jl  -  module DataLoader
# -----------------------------------------------------------------------------
# Reads the input CSVs from a directory and assembles them into the single named tuple `d` that every other file in the pipeline (MCSModel, OneShot, Output) consumes. 
# Input-only module.
# Three groups:
#
#   1. CSV LOADING HELPERS
#      _require_file, _read_csv
#      -- resolve a file inside the input directory and error immediately if
#      it's missing or missing a required column, so a bad input directory
#      fails fast with a clear message instead of a cryptic downstream error.
#
#   2. LIVE ACTIVITY POWER LOADER
#      _LIVE_ACTIVITY_NAMES, load_live_powers
#      -- reads recorded real-world activity power measurements from
#      live_powers.csv, split by activity (digging / loading+swinging /
#      traveling / idling), for use with Common's draw_activity_power_pool_live.
#
#   3. INPUT DATA ASSEMBLY
#      _clock_hours, _psd, _psd_opt, load_input_data, load_data
#      -- parses parameters.csv (required and optional scalar parameters),
#      ev_data.csv, mcs_data.csv, place.csv, time_data.csv, travel_time.csv,
#      and work_flexible.csv, and combines them into the node/set definitions
#      (N, N_g, N_c, M, E, A), SOE/charging bounds, travel times, work-hour
#      targets, electricity price/carbon series, and Bayesian prior parameters
#      that the MILP model and simulation loop both need. `load_data` is a
#      thin wrapper around `load_input_data` with a default input directory.
# #############################################################################

module DataLoader

# external packages used across this file
using CSV
using DataFrames

# everything below that other files are allowed to use
export load_data, load_input_data, load_live_powers

# Builds the full path to a required input file inside dir, and errors immediately if it doesn't exist.
# Used as a guard before every CSV read, so a missing input file fails fast with a clear message instead of a cryptic CSV.read error
_require_file(dir, name) = (p = joinpath(dir, name);
    isfile(p) ? p : error("DataLoader input mode: required file missing -> $p"))

# Reads a CSV file from dir (via _require_file's existence check) and verifies it has every column listed in required_cols, erroring with a clear message naming the missing column if not  
function _read_csv(dir, name; required_cols = String[])
    df = CSV.read(_require_file(dir, name), DataFrame)
    for c in required_cols
        Symbol(c) in propertynames(df) ||
            error("DataLoader input mode: '$name' is missing required column '$c'")
    end
    return df
end

# Fixed activity ordering (dig, load+swing, travel, idle) used to map live_powers.csv rows to activity indices 1-4.
# Same ordering assumption flagged earlier in Common.jl's activity_power_model/refit! and log_realized_row!.
const _LIVE_ACTIVITY_NAMES = ["p_digging", "p_loading_swinging", "p_traveling", "p_idling"]

# Reads live_powers.csv (activity, power_kW columns) and groups the recorded power_kW values by activity, matched case-insensitively against _LIVE_ACTIVITY_NAMES.
# Errors if any of the 4 activities has zero matching rows, since draw_activity_power_pool_live (in Common.jl) can't compute mu/sd from an empty set.
# Returns a Dict{Int, Vector{Float64}} mapping activity index (1-4) to its list of recorded values, in the exact shape draw_activity_power_pool_live expects.
function load_live_powers(input_dir::AbstractString; filename::AbstractString = "live_powers.csv")
    df = _read_csv(input_dir, filename; required_cols = ["activity", "power_kW"])
    names_lc = strip.(lowercase.(string.(df.activity)))
    live = Dict{Int, Vector{Float64}}()
    for (a, nm) in enumerate(_LIVE_ACTIVITY_NAMES)
        rows = names_lc .== lowercase(nm)
        vals = Float64.(df.power_kW[rows])
        isempty(vals) &&
            error("load_live_powers: no rows found for activity '$nm' in $filename")
        live[a] = vals
    end
    return live
end

# Parses a clock string like "HH:MM" (or bare "HH") into a decimal-hours value, e.g. "13:30" -> 13.5.
# Splits on ":", takes the hour part as-is, and adds the minute part (if present) divided by 60; assumes 0 minutes if no ":" is found.
_clock_hours(s) = (parts = split(strip(string(s)), ":");
    parse(Int, parts[1]) + (length(parts) >= 2 ? parse(Int, parts[2]) : 0) / 60)

# Looks up a required scalar parameter by name in the parameters.csv-derived DataFrame (par), matching against its Parameter column and returning the matching Value as a Float64.
# Errors if the key isn't found at all, since this is for parameters the model cannot run without.    
function _psd(par, key)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    idx === nothing && error("DataLoader input mode: parameter '$key' missing in parameters.csv")
    return Float64(par.Value[idx])
end

# Same lookup as _psd, but for optional parameters: returns default instead of erroring if the key isn't found in parameters.csv.
function _psd_opt(par, key, default)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    return idx === nothing ? default : Float64(par.Value[idx])
end

# Reads and assembles all input CSVs into the single named tuple `d` used by every other file in the pipeline.
function load_input_data(input_dir::AbstractString)
    isdir(input_dir) || error("DataLoader input mode: input directory not found -> $input_dir")

    # Reads all seven required input files up front, each with its required columns checked immediately.
    par = _read_csv(input_dir, "parameters.csv"; required_cols = ["Parameter", "Value"])
    evd = _read_csv(input_dir, "ev_data.csv";   required_cols = ["SOE_min","SOE_max","SOE_ini","ch_rate","eta_ch_dch_cev"])
    mcd = _read_csv(input_dir, "mcs_data.csv";  required_cols =
            ["SOE_min","SOE_max","SOE_ini","CH_MCS","DCH_MCS","C_MCS_plug","DCH_MCS_plug","eta_ch_dch_mcs"])
    plc = _read_csv(input_dir, "place.csv";     required_cols = ["site","hours_digging","hours_loading_swinging"])
    tdd = _read_csv(input_dir, "time_data.csv"; required_cols = ["lambda_buy","intensity_tons_emissions"])
    ttm = _read_csv(input_dir, "travel_time.csv")
    wkf = _read_csv(input_dir, "work_flexible.csv"; required_cols = ["Location","EV"])

    # Pulls scalar model parameters out of parameters.csv, required ones via _psd and optional ones (with defaults) via _psd_opt.
    # delta_T is the interval granularity used throughout Section III; rho_miss and rho_labor are the ρ^miss and ρ^travel cost conversion factors in objective function (4); lambda_demand_NC and lambda_demand_OP are the λ^NC and λ^OP demand charge rates in objective function (4); carbon_price_per_ton is the λ^em carbon-to-dollar conversion factor in objective function (4); p_idling is the idle activity power p_a in constraint 8d; scale is the κ^seq precedence ratio in constraint 14c; t_limit_rest is t_limit in constraint 14d.
    # prior_sigma_frac, obs_noise_std, and co2_unit_scale have no constraint counterpart -- they're calibration/unit-conversion settings only.
    delta_T = _psd(par, "delta_T")
    rho_miss         = _psd(par, "rho_miss");          rho_labor = _psd(par, "rho_labor")
    lambda_demand_NC = _psd(par, "lambda_demand_NC");  lambda_demand_OP = _psd(par, "lambda_demand_OP")
    carbon_price_per_ton = _psd_opt(par, "carbon_price_per_ton", 0.0)
    p_idling     = _psd_opt(par, "p_idling", 0.0)
    scale        = Int(round(_psd_opt(par, "scale", 2.0)))
    t_limit_rest = _psd_opt(par, "t_limit_rest", 1.0)
    prior_sigma_frac = _psd_opt(par, "prior_sigma_frac", 0.2)
    obs_noise_std    = _psd_opt(par, "obs_noise_std", 0.05)
    co2_unit_scale   = _psd_opt(par, "co2_unit_scale", 1.0)
    kappa_wt         = Int(round(_psd_opt(par, "kappa_wt", 4.0)))

    # Derives the interval count and horizon start time from time_data.csv, and pulls its per-interval electricity price and carbon intensity series.
    # lambda_whl_elec is λ^elec_t and lambda_CO2 is λ^CO2_t, both from objective function (4).
    n_int   = nrow(tdd)
    t_start = _clock_hours(tdd[1, 1]) - delta_T
    lambda_whl_elec = Float64.(tdd.lambda_buy)
    lambda_CO2      = Float64.(tdd.intensity_tons_emissions) .* co2_unit_scale

    # Builds the ID lists and index lookups for EVs, MCSs, and nodes, and defines the N/E/M index sets from their lengths.
    ev_ids   = strip.(string.(evd[!, 1]))
    mcs_ids  = strip.(string.(mcd[!, 1]))
    node_ids = strip.(string.(plc.site))
    node_idx = Dict(lowercase(id) => i for (i, id) in enumerate(node_ids))
    ev_idx   = Dict(lowercase(id) => e for (e, id) in enumerate(ev_ids))
    N = 1:length(node_ids);  E = 1:length(ev_ids);  M = 1:length(mcs_ids)

    # Builds the CEV-to-node assignment matrix A from place.csv's per-EV columns, then splits nodes into construction sites (N_c, have an assigned EV) and grid nodes (N_g, the rest).
    # Constraint 11a of the paper's MILP formulation (the assignment matrix A_i,e that ρ_m,i,e,t is checked against).
    A = zeros(Int, length(N), length(E))
    for (e, eid) in enumerate(ev_ids)
        Symbol(eid) in propertynames(plc) ||
            error("DataLoader: place.csv missing assignment column '$eid'")
        col = plc[!, Symbol(eid)]
        for r in 1:nrow(plc)
            Int(round(Float64(col[r]))) == 1 && (A[node_idx[lowercase(node_ids[r])], e] = 1)
        end
    end
    N_c = [i for i in N if any(A[i, e] == 1 for e in E)]
    N_g = [i for i in N if !(i in N_c)]
    isempty(N_g) && error("DataLoader: no grid node (a node with no EV assigned) found")
    isempty(N_c) && error("DataLoader: no site node (a node with an EV assigned) found")

    # Pulls the SOE bounds, charge/discharge rates, plug limits, and efficiencies for MCSs and CEVs straight from their respective CSV columns.
    # SOE_MCS_min/max and SOE_CEV_min/max are the bounds in constraints 10c/10d; CH_MCS/DCH_MCS are the capacities CH^MCS_m/DCH^MCS_m in constraints 6a/6b; DCH_MCS_plug and C_MCS_plug are DCH^plug_m and C^plug_m in constraints 7a/7d; CH_CEV is CH^CEV_e in constraint 7b.
    SOE_MCS_ini = Float64.(mcd.SOE_ini); SOE_MCS_max = Float64.(mcd.SOE_max)
    SOE_MCS_min = Float64.(mcd.SOE_min); CH_MCS = Float64.(mcd.CH_MCS); DCH_MCS = Float64.(mcd.DCH_MCS)
    DCH_MCS_plug = Float64.(mcd.DCH_MCS_plug); C_MCS_plug = Int.(mcd.C_MCS_plug); eta_ch_dch_mcs = Float64.(mcd.eta_ch_dch_mcs)
    SOE_CEV_ini = Float64.(evd.SOE_ini); SOE_CEV_max = Float64.(evd.SOE_max)
    SOE_CEV_min = Float64.(evd.SOE_min); CH_CEV = Float64.(evd.ch_rate)
    eta_ch_dch_cev = Float64.(evd.eta_ch_dch_cev)

    # Builds the Bayesian prior mean/std for the 4 activity powers (dig, load, travel, idle) from parameters.csv, falling back to a fraction of the mean (prior_sigma_frac) when no explicit sigma is given.
    # p_digging/p_loading_swinging/p_traveling are unpacked for convenience.
    # These feed p_a, the per-activity power constant in constraint 8d.
    prior_mu    = [_psd(par, "p_digging"), _psd(par, "p_loading_swinging"), _psd(par, "p_traveling"), p_idling]
    sig_dig  = _psd_opt(par, "sigma_digging",          NaN)
    sig_load = _psd_opt(par, "sigma_loading_swinging", NaN)
    sig_trv  = _psd_opt(par, "sigma_traveling",        NaN)
    _sigma_or_frac(explicit, mu) =
        isnan(explicit) ? (mu > 0 ? max(prior_sigma_frac * mu, 0.05) : 0.0) : max(explicit, 0.0)
    prior_sigma = [_sigma_or_frac(sig_dig,  prior_mu[1]),
                   _sigma_or_frac(sig_load, prior_mu[2]),
                   _sigma_or_frac(sig_trv,  prior_mu[3]),
                   0.0]
    p_digging, p_loading_swinging, p_traveling = prior_mu[1], prior_mu[2], prior_mu[3]

    # Pulls each construction node's required digging and loading+swinging work hours from place.csv into per-node arrays.
    # These are H_i,a, the required productive work duration in constraint 14b.
    hours_digging = zeros(length(N)); hours_loading_swinging = zeros(length(N))
    for r in 1:nrow(plc)
        i = node_idx[lowercase(node_ids[r])]
        hours_digging[i]          = Float64(plc.hours_digging[r])
        hours_loading_swinging[i] = Float64(plc.hours_loading_swinging[r])
    end

    # Builds the node-to-node travel time matrix tau_trv from travel_time.csv, matching row/column labels against node_idx and skipping any that don't match a known node.
    # This is τ^trv_i,j, used in constraints 12b/13b (same as Common.jl's normalize_travel_steps, which converts this into integer interval counts).
    tau_trv = zeros(length(N), length(N))
    tt_rows = lowercase.(strip.(string.(ttm[!, 1])))
    tt_cols = lowercase.(strip.(string.(names(ttm)[2:end])))
    for (ri, rn) in enumerate(tt_rows), (ci, cn) in enumerate(tt_cols)
        (haskey(node_idx, rn) && haskey(node_idx, cn)) || continue
        tau_trv[node_idx[rn], node_idx[cn]] = Float64(ttm[ri, ci + 1])
    end

    # Builds the is_working[node, ev, interval] boolean array from work_flexible.csv, marking which (node, ev, interval) combinations have a nonzero flexible-work value.
    # Also defines the day's interval sets K (1..n_day) and T (1..n_day+1, interval boundaries).
    # is_working corresponds to R_e,t / T^work in constraint 8b (whether the CEV's work capacity is active that interval).
    wf_time_cols = names(wkf)[3:end]
    n_full = min(n_int, length(wf_time_cols))
    R_full = zeros(length(N), length(E), n_full)
    for r in 1:nrow(wkf)
        loc = lowercase(strip(string(wkf.Location[r]))); ev = lowercase(strip(string(wkf.EV[r])))
        (haskey(node_idx, loc) && haskey(ev_idx, ev)) || continue
        i = node_idx[loc]; e = ev_idx[ev]
        for k in 1:n_full
            R_full[i, e, k] = Float64(wkf[r, 2 + k])
        end
    end
    n_day = n_int
    K = 1:n_day;  T = 1:(n_day + 1)
    is_working = falses(length(N), length(E), n_day)
    nfill = min(n_full, n_day)
    is_working[:, :, 1:nfill] = R_full[:, :, 1:nfill] .> 0

    # Fixes the activity index set B = [dig, load, travel, idle] and packages every derived quantity into the single named tuple returned as `d`.
    # B is the activity set A referenced throughout constraints 8c/8d/14a-14f.
    B = [1, 2, 3, 4]
    return (; delta_T, K, T, t_start, n_int, n_day, t_limit_rest,
              N, N_g, N_c, M, E, A,
              SOE_MCS_ini, SOE_MCS_max, SOE_MCS_min, CH_MCS, DCH_MCS,
              DCH_MCS_plug, C_MCS_plug, eta_ch_dch_mcs,
              SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, CH_CEV, eta_ch_dch_cev,
              p_digging, p_loading_swinging, p_traveling, p_idling,
              prior_mu, prior_sigma, obs_noise_std,
              hours_digging, hours_loading_swinging, tau_trv,
              lambda_whl_elec, lambda_CO2, is_working,
              rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP,
              carbon_price_per_ton, scale, kappa_wt, B)
end

# Thin wrapper around load_input_data with a default input directory (the data/input_data folder sitting next to this module's parent directory).
# Kept separate from load_input_data so callers have a zero-argument entry point, and so the default path is defined in exactly one place.
function load_data(input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"))
        return load_input_data(input_dir)
end

end