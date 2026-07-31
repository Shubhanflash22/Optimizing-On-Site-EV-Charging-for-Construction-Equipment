# #############################################################################
# DataLoader.jl  —  module DataLoader
# -----------------------------------------------------------------------------
# Loads the full scenario definition into a single NamedTuple `d` that every
# downstream module reads (fleet sizes, battery limits, prices, work to be done,
# activity-power priors, etc.). Two sources are supported, exposed through the
# same field layout so nothing downstream can tell where the numbers came from:
#
#   * :synthetic  -> a small, self-contained example built in code.
#   * :input      -> the 7-CSV real dataset in data/input_data/.
#
# NOMENCLATURE mirrors DataLoader_v4_real.jl (SOE_*, CH_*, DCH_*, R_work,
# lambda_whl_elec, lambda_CO2, hours_digging, ...).
# #############################################################################
module DataLoader

using CSV
using DataFrames

export load_data, build_default_data, load_input_data
# RECEDING HORIZON (multi-day): each 24 h day is split into a PHASE-1 daytime
# window of `n_day` intervals (work + opportunistic charging, solved by the
# cross-day MILP) and an overnight remainder handled deterministically by the
# separate phase. `n_day` is INFERRED from work availability rather than fixed
# to the full day: n_day = (last work-available interval) + a return buffer so
# the MCS can drive home and top the CEVs up before the overnight recharge.
# `n_days` reported days are simulated with one extra buffer day dropped by the
# driver. Powers/curves are fixed (fit-once prior, no online learning).

# =============================================================================
# SYNTHETIC MODE  (the built-in example)
# =============================================================================
function build_default_data()
    # ---- time discretization ----
    delta_T = 0.25                 # hours per step (0.25 h = 15 min)
    n_int   = 96                   # steps in a full 24 h day
    t_start = 8                    # clock hour the horizon begins (08:00)
    work_start_hour = 8; work_end_hour = 17        # productive shift 08:00-17:00 ...
    lunch_start_hour = 12; lunch_end_hour = 14     # ... minus a 12:00-14:00 lunch
    t_limit_rest = 1.0             # rest rule: <=1 h work per rolling (1 h + step) window

    # ---- nodes: 1 = grid, 2.. = construction sites ----
    N   = 1:3
    N_g = [1]
    N_c = collect(2:length(N))

    # ---- fleet sizes: one MCS, two CEVs ----
    M = 1:1
    E = 1:2

    # Assignment A[node, ev] = 1 if that excavator works at that site.
    A = zeros(Int, length(N), length(E))
    A[2, 1] = 1                    # excavator 1 at site 2
    A[3, 2] = 1                    # excavator 2 at site 3

    # ---- MCS (charging-truck) parameters ----
    SOE_MCS_ini = [250.0]; SOE_MCS_max = [250.0]; SOE_MCS_min = [50.0]
    CH_MCS  = [150.0]; DCH_MCS = [150.0]; DCH_MCS_plug = [60.0]
    C_MCS_plug = [2]; eta_ch_dch = [0.95]

    # ---- CEV (excavator) parameters ----
    SOE_CEV_max = [90.0, 60.0]
    SOE_CEV_ini = [72.0, 48.0]     # 80% of max = the end-of-day target
    SOE_CEV_min = [18.0, 12.0]
    CH_CEV      = [45.0, 30.0]

    # ---- activity power draws, kW (B = [dig, load, travel, idle]) ----
    # Idle is pinned to 0 kW with 0 std: no power is lost while idling.
    prior_mu      = [4.6, 3.3, 4.5, 0.0]
    prior_sigma   = [1.0, 1.0, 1.5, 0.0]
    true_powers   = [5.2, 2.8, 5.0, 0.0]
    true_sigma    = [0.3, 0.2, 0.3, 0.0]    # per-interval wobble of the stochastic plant
    obs_noise_std = 0.05
    p_digging          = prior_mu[1]
    p_loading_swinging = prior_mu[2]
    p_traveling        = prior_mu[3]
    p_idling           = prior_mu[4]

    # ---- required work hours per node, PER DAY (only site rows 2,3 are used) ----
    # Work is a PER-DAY schedule (not a single lumpsum): each kept day carries its
    # own digging/loading quota per site. `dig_by_day[dy]` / `load_by_day[dy]` are
    # node-length vectors for reported day dy (1..n_days). `hours_digging` /
    # `hours_loading_swinging` remain the day-1 vectors as a legacy reference.
    n_days = 2                     # synthetic default (reported days KEPT)
    dig_by_day  = [[0.0, 2.5, 1.5], [0.0, 2.0, 1.0]]
    load_by_day = [[0.0, 1.5, 1.0], [0.0, 1.0, 0.5]]
    @assert length(dig_by_day) == n_days == length(load_by_day)
    hours_digging          = copy(dig_by_day[1])
    hours_loading_swinging = copy(load_by_day[1])

    # ---- travel model: tau_trv[i,j] in INTERVALS, k_trv = kWh per arc ----
    tau_trv = [0.0 2.0 3.0;
               2.0 0.0 2.0;
               3.0 2.0 0.0]
    k_trv = 2.0

    # ---- exogenous 24 h series (per interval): price + CO2 intensity ----
    lambda_whl_elec = Float64[]
    lambda_CO2      = Float64[]
    for k in 1:n_int
        hour = mod(t_start + (k - 1) * delta_T, 24)
        price = (16 <= hour < 21) ? 0.45 : (7 <= hour < 16 ? 0.18 : 0.10)
        co2   = 0.30 + 0.15 * sin((hour - 6) / 24 * 2pi)
        push!(lambda_whl_elec, price)
        push!(lambda_CO2, max(co2, 0.05))
    end

    # ---- FULL 24 h horizon (one optimisation over the whole day) ----
    # Build the full-day productive mask (drives R_work below).
    available_full = Bool[
        (work_start_hour <= mod(t_start + (k - 1) * delta_T, 24) < work_end_hour) &&
        !(lunch_start_hour <= mod(t_start + (k - 1) * delta_T, 24) < lunch_end_hour)
        for k in 1:n_int]
    n_day = n_int                                     # FULL 24 h horizon (one optimisation)
    K = 1:n_day
    T = 1:(n_day + 1)

    # ---- per-CEV work-availability cap R_work[node, ev, interval] over the daytime window ----
    R_work = zeros(length(N), length(E), n_day)
    for i in N_c, e in E, k in 1:n_day
        R_work[i, e, k] = (available_full[k] && A[i, e] == 1) ? 1000.0 : 0.0
    end

    # ---- costs / penalties in the objective ----
    rho_miss             = 50.0
    rho_labor            = 30.0
    lambda_demand_NC     = 10.0
    lambda_demand_OP     = 25.0
    carbon_price_per_ton = 50.0

    scale = 2
    B = [1, 2, 3, 4]

    return (; delta_T, K, T, t_start, n_int, n_day, t_limit_rest,
              n_days,
              N, N_g, N_c, M, E, A,
              SOE_MCS_ini, SOE_MCS_max, SOE_MCS_min, CH_MCS, DCH_MCS,
              DCH_MCS_plug, C_MCS_plug, eta_ch_dch,
              SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, CH_CEV,
              p_digging, p_loading_swinging, p_traveling, p_idling,
              prior_mu, prior_sigma, true_powers, true_sigma, obs_noise_std,
              hours_digging, hours_loading_swinging, dig_by_day, load_by_day, tau_trv, k_trv,
              lambda_whl_elec, lambda_CO2, R_work,
              rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP,
              carbon_price_per_ton, scale, B)
end

# Read an OPTIONAL per-day work schedule `work_by_day.csv` (columns:
# site, day, hours_digging, hours_loading_swinging). Returns
# (dig_by_day, load_by_day) as n_days node-length vectors, or `nothing` if the
# file is absent (caller then repeats the single place.csv quota each day).
function _read_work_by_day(input_dir, node_ids, node_idx, n_days)
    path = joinpath(input_dir, "work_by_day.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    for c in ("site", "day", "hours_digging", "hours_loading_swinging")
        Symbol(c) in propertynames(df) ||
            error("DataLoader: work_by_day.csv missing required column '$c'")
    end
    nN = length(node_ids)
    dig_by_day  = [zeros(nN) for _ in 1:n_days]
    load_by_day = [zeros(nN) for _ in 1:n_days]
    for r in 1:nrow(df)
        dy = Int(round(Float64(df.day[r])))
        (1 <= dy <= n_days) || continue
        loc = lowercase(strip(string(df.site[r])))
        haskey(node_idx, loc) || continue
        i = node_idx[loc]
        dig_by_day[dy][i]  = Float64(df.hours_digging[r])
        load_by_day[dy][i] = Float64(df.hours_loading_swinging[r])
    end
    return (dig_by_day, load_by_day)
end

# =============================================================================
# INPUT MODE  (7-CSV real dataset)
# =============================================================================
_require_file(dir, name) = (p = joinpath(dir, name);
    isfile(p) ? p : error("DataLoader input mode: required file missing -> $p"))

function _read_csv(dir, name; required_cols = String[])
    df = CSV.read(_require_file(dir, name), DataFrame)
    for c in required_cols
        Symbol(c) in propertynames(df) ||
            error("DataLoader input mode: '$name' is missing required column '$c'")
    end
    return df
end

# Clock string ("8:15:00") -> decimal hours (8.25).
_clock_hours(s) = (parts = split(strip(string(s)), ":");
    parse(Int, parts[1]) + (length(parts) >= 2 ? parse(Int, parts[2]) : 0) / 60)

# Required / optional scalar lookups in parameters.csv (by Parameter name).
function _psd(par, key)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    idx === nothing && error("DataLoader input mode: parameter '$key' missing in parameters.csv")
    return Float64(par.Value[idx])
end
function _psd_opt(par, key, default)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    return idx === nothing ? default : Float64(par.Value[idx])
end

function load_input_data(input_dir::AbstractString)
    isdir(input_dir) || error("DataLoader input mode: input directory not found -> $input_dir")

    par = _read_csv(input_dir, "parameters.csv"; required_cols = ["Parameter", "Value"])
    evd = _read_csv(input_dir, "ev_data.csv";   required_cols = ["SOE_min","SOE_max","SOE_ini","ch_rate"])
    mcd = _read_csv(input_dir, "mcs_data.csv";  required_cols =
            ["SOE_min","SOE_max","SOE_ini","CH_MCS","DCH_MCS","C_MCS_plug","DCH_MCS_plug","eta_ch_dch"])
    plc = _read_csv(input_dir, "place.csv";     required_cols = ["site","hours_digging","hours_loading_swinging"])
    tdd = _read_csv(input_dir, "time_data.csv"; required_cols = ["lambda_buy","intensity_tons_emissions"])
    ttm = _read_csv(input_dir, "travel_time.csv")
    wkf = _read_csv(input_dir, "work_flexible.csv"; required_cols = ["Location","EV"])

    # ---- scalar settings ----
    delta_T = _psd(par, "delta_T");  k_trv = _psd(par, "k_trv")
    rho_miss         = _psd(par, "rho_miss");          rho_labor = _psd(par, "rho_labor")
    lambda_demand_NC = _psd(par, "lambda_demand_NC");  lambda_demand_OP = _psd(par, "lambda_demand_OP")
    carbon_price_per_ton = _psd_opt(par, "carbon_price_per_ton", 0.0)
    p_idling     = _psd_opt(par, "p_idling", 0.0)
    scale        = Int(round(_psd_opt(par, "scale", 2.0)))
    t_limit_rest = _psd_opt(par, "t_limit_rest", 1.0)
    prior_sigma_frac = _psd_opt(par, "prior_sigma_frac", 0.2)
    obs_noise_std    = _psd_opt(par, "obs_noise_std", 0.05)
    co2_unit_scale   = _psd_opt(par, "co2_unit_scale", 1.0)

    # ---- time series + full-day horizon from time_data.csv ----
    n_int   = nrow(tdd)
    t_start = _clock_hours(tdd[1, 1]) - delta_T
    lambda_whl_elec = Float64.(tdd.lambda_buy)
    lambda_CO2      = Float64.(tdd.intensity_tons_emissions) .* co2_unit_scale

    # ---- id -> index maps ----
    # String IDs from each CSV's first column (whitespace-trimmed).
    ev_ids   = strip.(string.(evd[!, 1]))
    mcs_ids  = strip.(string.(mcd[!, 1]))
    node_ids = strip.(string.(plc.site))
     # Lowercased name -> integer index, so lookups are case-insensitive and consistent across files.
    node_idx = Dict(lowercase(id) => i for (i, id) in enumerate(node_ids))
    ev_idx   = Dict(lowercase(id) => e for (e, id) in enumerate(ev_ids))
        # Integer index sets used everywhere downstream (nodes, excavators, MCS units).
    N = 1:length(node_ids);  E = 1:length(ev_ids);  M = 1:length(mcs_ids)

    # ---- assignment matrix A from place.csv (one column per excavator id) ----
    A = zeros(Int, length(N), length(E))
    for (e, eid) in enumerate(ev_ids)
        Symbol(eid) in propertynames(plc) ||
            error("DataLoader: place.csv missing assignment column '$eid'")
        col = plc[!, Symbol(eid)]
        for r in 1:nrow(plc)
            Int(round(Float64(col[r]))) == 1 && (A[node_idx[lowercase(node_ids[r])], e] = 1)
        end
    end
    # Site nodes = any node with a CEV assigned; grid nodes = all the rest.
    N_c = [i for i in N if any(A[i, e] == 1 for e in E)]
    N_g = [i for i in N if !(i in N_c)]
    isempty(N_g) && error("DataLoader: no grid node (a node with no EV assigned) found")
    isempty(N_c) && error("DataLoader: no site node (a node with an EV assigned) found")

    # ---- MCS / CEV battery parameters ----
    SOE_MCS_ini = Float64.(mcd.SOE_ini); SOE_MCS_max = Float64.(mcd.SOE_max)
    SOE_MCS_min = Float64.(mcd.SOE_min); CH_MCS = Float64.(mcd.CH_MCS); DCH_MCS = Float64.(mcd.DCH_MCS)
    DCH_MCS_plug = Float64.(mcd.DCH_MCS_plug); C_MCS_plug = Int.(mcd.C_MCS_plug); eta_ch_dch = Float64.(mcd.eta_ch_dch)
    SOE_CEV_ini = Float64.(evd.SOE_ini); SOE_CEV_max = Float64.(evd.SOE_max)
    SOE_CEV_min = Float64.(evd.SOE_min); CH_CEV = Float64.(evd.ch_rate)

    # ---- activity powers: known constants seed the learner's prior ----
    prior_mu    = [_psd(par, "p_digging"), _psd(par, "p_loading_swinging"), _psd(par, "p_traveling"), p_idling]
    # PER-ACTIVITY std: prefer explicit sigma_* rows (e.g. written by the step-0
    # Bayesian regression = the posterior SD of each activity power). If a row is
    # missing (NaN), fall back to the old single prior_sigma_frac * mu behaviour so
    # older parameter files still load. Idle (prior_mu[4] == 0) is pinned to 0 std:
    # no power is lost while idling.
    sig_dig  = _psd_opt(par, "sigma_digging",          NaN)
    sig_load = _psd_opt(par, "sigma_loading_swinging", NaN)
    sig_trv  = _psd_opt(par, "sigma_traveling",        NaN)
    _sigma_or_frac(explicit, mu) =
        isnan(explicit) ? (mu > 0 ? max(prior_sigma_frac * mu, 0.05) : 0.0) : max(explicit, 0.0)
    prior_sigma = [_sigma_or_frac(sig_dig,  prior_mu[1]),
                   _sigma_or_frac(sig_load, prior_mu[2]),
                   _sigma_or_frac(sig_trv,  prior_mu[3]),
                   0.0]
    true_powers = copy(prior_mu)
    true_sigma  = copy(prior_sigma)          # per-interval wobble of the stochastic plant
    p_digging, p_loading_swinging, p_traveling = prior_mu[1], prior_mu[2], prior_mu[3]

    # ---- required work hours per node from place.csv ----
    hours_digging = zeros(length(N)); hours_loading_swinging = zeros(length(N))
    for r in 1:nrow(plc)
        i = node_idx[lowercase(node_ids[r])]
        hours_digging[i]          = Float64(plc.hours_digging[r])
        hours_loading_swinging[i] = Float64(plc.hours_loading_swinging[r])
    end

    # ---- travel-time matrix from travel_time.csv (values are in INTERVALS) ----
    tau_trv = zeros(length(N), length(N))
    tt_rows = lowercase.(strip.(string.(ttm[!, 1])))
    tt_cols = lowercase.(strip.(string.(names(ttm)[2:end])))
    for (ri, rn) in enumerate(tt_rows), (ci, cn) in enumerate(tt_cols)
        (haskey(node_idx, rn) && haskey(node_idx, cn)) || continue
        tau_trv[node_idx[rn], node_idx[cn]] = Float64(ttm[ri, ci + 1])
    end

    # ---- work-availability matrix over the FULL 24 h horizon ----
    # Each work_flexible row is (Location, EV) followed by one column per FULL-day
    # interval giving the kW work cap (0 = no work). We read the whole-day caps and
    # keep the full 24 h horizon (n_day = n_int): this is a single-day model, so
    # there is no shift-based horizon inference and no return buffer.
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
    n_day = n_int                            # FULL 24 h horizon (one optimisation)
    K = 1:n_day;  T = 1:(n_day + 1)
    R_work = zeros(length(N), length(E), n_day)   # pad the caps out to the full-day horizon
    nfill = min(n_full, n_day)
    R_work[:, :, 1:nfill] = R_full[:, :, 1:nfill]

    B = [1, 2, 3, 4]
    
    # ---- Receding horizon multi-day schedule ----
    n_days = max(1, Int(round(_psd_opt(par, "n_days", 2.0))))

    wbd = _read_work_by_day(input_dir, node_ids, node_idx, n_days)
    if wbd === nothing
        dig_by_day  = [copy(hours_digging)          for _ in 1:n_days]
        load_by_day = [copy(hours_loading_swinging) for _ in 1:n_days]
    else
        dig_by_day, load_by_day = wbd
        hours_digging          = copy(dig_by_day[1])
        hours_loading_swinging = copy(load_by_day[1])
    end

    return (; delta_T, K, T, t_start, n_int, n_day, t_limit_rest,
              n_days,
              N, N_g, N_c, M, E, A,
              SOE_MCS_ini, SOE_MCS_max, SOE_MCS_min, CH_MCS, DCH_MCS,
              DCH_MCS_plug, C_MCS_plug, eta_ch_dch,
              SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, CH_CEV,
              p_digging, p_loading_swinging, p_traveling, p_idling,
              prior_mu, prior_sigma, true_powers, true_sigma, obs_noise_std,
              hours_digging, hours_loading_swinging, dig_by_day, load_by_day, tau_trv, k_trv,
              lambda_whl_elec, lambda_CO2, R_work,
              rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP,
              carbon_price_per_ton, scale, B)
end

# Dispatcher: pick the data source.
function load_data(mode::Symbol; input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"))
    if mode == :synthetic
        return build_default_data()
    elseif mode == :input
        return load_input_data(input_dir)
    else
        error("DataLoader: unknown data mode :$mode (use :synthetic or :input)")
    end
end

end # module DataLoader
