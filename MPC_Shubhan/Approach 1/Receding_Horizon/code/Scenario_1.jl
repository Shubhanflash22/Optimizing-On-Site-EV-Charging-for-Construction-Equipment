# =============================================================================
# Scenario_1.jl  —  Deterministic Certainty-Equivalent MPC for MCS dispatch
# =============================================================================
#
# Standalone implementation of "Scenario 1 / Approach 1: Deterministic
# Certainty-Equivalent MPC" for dispatching a Mobile Charging Station (MCS) to a
# fleet of construction EVs (excavators). The formal model is in
# docs/math_model.tex; a plain-English map is in docs/constraints_explained.txt,
# and a constraint comparison in docs/constraints_code_vs_model.txt.
#
# Run:  julia Scenario_1.jl        (synthetic mode; see docs/how_to_run.txt)
#
# The MPC runs every 15 min (= one interval). At each step it ALSO ingests the
# realized energy drop observed over the last 15 min and uses it to recursively
# improve the per-activity power estimate -> a feedback loop on BOTH the
# operational state AND the model parameters:
#
#   * Bayesian Uncertainty Engine -> "Certainty Collapse": an offline Bayesian
#     posterior (mean + std) seeds an ONLINE estimator; its current mean is the
#     single median power profile fed to the optimizer.
#   * Single-Instance MPC Solver: a MILP over the cross-day prediction window
#     (rest of today + lookahead_days future daytime blocks).
#   * Operational Execution + Closed-Loop Update: apply ONLY the first interval's
#     decision, advance the realized SOE, observe the 15-min energy drop, UPDATE
#     the power estimate, and re-solve.
#
# Two feedback loops vs. a one-shot 24h open-loop solve:
#   (1) State feedback    : realized SOE re-initializes each window.
#   (2) Parameter feedback : realized energy drop -> Bayesian re-fit of the
#                            activity power estimate (BayesianActivityEstimator below).
# =============================================================================

using JuMP
using HiGHS
using Plots
using DataFrames
using CSV
using Printf
using Dates
using LinearAlgebra
using Random
using Statistics        # mean / std for posterior summaries
using Turing            # exact Bayesian regression (TruncatedNormal priors + NUTS)

gr()
Turing.setprogress!(false)

# =============================================================================
# 1. DATA  (inlined synthetic dataset — replace with your real CSVs if desired)
# =============================================================================
#
# `build_default_data` returns a NamedTuple holding every set and parameter the
# window MILP needs. The activity powers p_digging / p_loading_swinging /
# p_traveling are the certainty-collapsed Bayesian posterior means; replace them
# with your own offline posterior means for a different fleet.

function build_default_data()
    # ---- time discretization ----
    delta_T = 0.25                 # hours per interval (15 min)
    n_int   = 96                   # full 24 h (08:00 -> 08:00); used for overnight prices
    t_start = 8                    # clock hour the horizon starts at (08:00)
    # Phase-1 (daytime MILP) horizon: 08:00 -> day_end_hour (18:00).
    work_start_hour = 8; work_end_hour = 17        # productive shift (with a lunch gap)
    lunch_start_hour = 12; lunch_end_hour = 14     # lunch: no work, but may charge
    day_end_hour = 18                              # CEVs full / MCS home by 18:00
    t_limit_rest = 1.0             # Eq. 12e rest rule: >=1 idle break per rolling (t_limit+dt) window
    kappa_wt = 4                   # travel-pacing: productive intervals per CEV travel
    n_day   = Int(round((day_end_hour - t_start) / delta_T))   # 40 daytime intervals
    K       = 1:n_day              # daytime MILP interval indices
    T       = 1:(n_day + 1)        # daytime boundary indices
    # Receding-horizon multi-day length. n_days = the number of days we KEEP in the
    # results; the simulation actually runs n_days + 1 and DROPS the last (buffer)
    # day, so the last kept day still has a full day of lookahead beyond it.
    n_days  = 2                    # synthetic default (override via run_scenario_1(; n_days=...))

    # ---- nodes: 1 = grid connection, 2..N = construction sites ----
    N   = 1:3
    N_g = [1]
    N_c = collect(2:length(N))     # sites 2,3

    # ---- fleet ----
    M = 1:1                        # one MCS
    E = 1:2                        # two CEVs (excavators)

    # CEV e is assigned to exactly one site (A[i,e] = 1).  Grid row is all zeros.
    A = zeros(Int, length(N), length(E))
    A[2, 1] = 1                    # CEV 1 works at site 2
    A[3, 2] = 1                    # CEV 2 works at site 3

    # ---- MCS parameters ----
    SOE_MCS_ini = [250.0]; SOE_MCS_max = [250.0]; SOE_MCS_min = [50.0]   # start full; 20% floor
    CH_MCS  = [150.0]              # grid->MCS charge rate (kW)
    DCH_MCS = [150.0]             # MCS->site discharge rate (kW)
    DCH_MCS_plug = [60.0]         # per-plug rate to a single CEV (kW)
    C_MCS_plug   = [2]            # plugs per MCS
    eta_ch_dch   = [0.95]         # round-trip-ish efficiency

    # ---- CEV parameters ----
    # Two DIFFERENT excavators. Crucially SOE_ini < SOE_max (start at 80%), so the
    # energy-neutral terminal "end >= SOE_ini" has headroom and never collides with the
    # SOE_max bound -- the batteries are also large vs the daily work, so each CEV can do
    # its whole shift and be topped back up by the shared MCS well before 18:00.
    SOE_CEV_max = [90.0, 60.0]     # CEV 1 is a bigger machine than CEV 2
    SOE_CEV_ini = [72.0, 48.0]     # start at 80% of max (energy-neutral target = start level)
    SOE_CEV_min = [18.0, 12.0]     # 20% floor
    CH_CEV      = [45.0, 30.0]     # CEV accept rate (kW)

    # ---- activity power draws, kW ----
    # B = [digging, loading/swinging, traveling, idling].
    # Idling is a full activity too: a CEV is ALWAYS doing exactly one of these
    # four each interval. Idling draws a small power and is what happens during
    # charging, lunch breaks, and any gap between useful work.
    #
    # prior_mu / prior_sigma : the OFFLINE Bayesian posterior (mean + std). These
    #   seed the ONLINE estimator and are what the MPC uses on step 1 (before any
    #   telematics feedback arrives).
    # true_powers : the (hidden) ground truth the excavators actually draw. The
    #   estimator never sees this; we only use it to SIMULATE the realized SOC
    #   drop each 15 min. With real hardware this comes from telematics instead.
    # obs_noise_std : telematics/measurement noise on the realized energy (kWh).
    prior_mu      = [4.6, 3.3, 4.5, 0.5]
    prior_sigma   = [1.0, 1.0, 1.5, 0.3]
    true_powers   = [5.2, 2.8, 5.0, 0.6]
    obs_noise_std = 0.05

    # back-compat scalars (initial certainty-equivalent values = prior means)
    p_digging          = prior_mu[1]
    p_loading_swinging = prior_mu[2]
    p_traveling        = prior_mu[3]
    p_idling           = prior_mu[4]

    # ---- per-site work-hour requirements (only N_c entries used) ----
    hours_digging          = [0.0, 2.5, 1.5]
    hours_loading_swinging = [0.0, 1.5, 1.0]

    # ---- travel ----
    # tau_trv[i,j] = travel time in INTERVALS; k_trv = kWh consumed per arc traversal.
    tau_trv = [0.0 2.0 3.0;
               2.0 0.0 2.0;
               3.0 2.0 0.0]
    k_trv = 2.0

    # ---- exogenous time series (interval-indexed) ----
    # Time-of-use price with a 16:00-21:00 on-peak bump; smooth CO2 intensity.
    lambda_whl_elec = Float64[]
    lambda_CO2      = Float64[]
    for k in 1:n_int
        hour = mod(t_start + (k - 1) * delta_T, 24)
        price = (16 <= hour < 21) ? 0.45 : (7 <= hour < 16 ? 0.18 : 0.10)
        co2   = 0.30 + 0.15 * sin((hour - 6) / 24 * 2pi)
        push!(lambda_whl_elec, price)
        push!(lambda_CO2, max(co2, 0.05))
    end

    # ---- per-CEV work-availability profile R_work[i,e,k] (kW cap) ----
    # CEV may only work during the day shift here; value caps P_work.
    # Productive work-availability: shift hours EXCLUDING the lunch window.
    R_work = zeros(length(N), length(E), n_day)
    for i in N_c, e in E, k in 1:n_day
        hour = mod(t_start + (k - 1) * delta_T, 24)
        productive = (work_start_hour <= hour < work_end_hour) &&
                     !(lunch_start_hour <= hour < lunch_end_hour)
        R_work[i, e, k] = (productive && A[i, e] == 1) ? 1000.0 : 0.0
    end

    # ---- costs / penalties ----
    rho_miss             = 50.0    # $/h of missed work
    rho_labor            = 30.0    # fixed labour $ per operator (one operator per device: each CEV + each MCS)
    lambda_demand_NC     = 10.0    # $/kW non-coincident demand charge
    lambda_demand_OP     = 25.0    # $/kW on-peak demand charge
    carbon_price_per_ton = 50.0    # $/ton CO2

    scale = 2                      # loading/swinging-vs-digging precedence slack
    B = [1, 2, 3, 4]               # 1=digging, 2=loading/swinging, 3=traveling, 4=idling

    return (; delta_T, K, T, t_start, n_int, n_day, n_days, day_end_hour, t_limit_rest, kappa_wt,
              N, N_g, N_c, M, E, A,
              SOE_MCS_ini, SOE_MCS_max, SOE_MCS_min, CH_MCS, DCH_MCS,
              DCH_MCS_plug, C_MCS_plug, eta_ch_dch,
              SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, CH_CEV,
              p_digging, p_loading_swinging, p_traveling, p_idling,
              prior_mu, prior_sigma, true_powers, obs_noise_std,
              hours_digging, hours_loading_swinging, tau_trv, k_trv,
              lambda_whl_elec, lambda_CO2, R_work,
              rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP,
              carbon_price_per_ton, scale, B)
end

# =============================================================================
# 1a. INPUT-DATA MODE  (loads the 7-CSV dataset from data/input_data/;
#     same file names / column schema as the reference dataset)
# =============================================================================
#
# Required files (exact column names):
#   parameters.csv    Parameter,Value,Unit,Description
#                     core: k_trv, delta_T, rho_miss, rho_labor, lambda_demand_NC,
#                       lambda_demand_OP, carbon_price_per_ton, p_digging,
#                       p_loading_swinging, p_traveling
#                     extras folded into the SAME file: p_idling, scale, kappa_wt,
#                       day_end_hour, t_limit_rest, prior_sigma_frac,
#                       obs_noise_std, co2_unit_scale
#   ev_data.csv       <id>,SOE_min,SOE_max,SOE_ini,ch_rate,work_cap
#   mcs_data.csv      <id>,SOE_min,SOE_max,SOE_ini,CH_MCS,DCH_MCS,C_MCS_plug,
#                       DCH_MCS_plug,eta_ch_dch
#   place.csv         site,<one e<i> column per CEV>,hours_digging,
#                       hours_loading_swinging
#   time_data.csv     <time>,<t-id>,lambda_CO2,lambda_buy,intensity_tons_emissions
#   travel_time.csv   Node,<dest cols...>     (matrix; values in 15-min intervals)
#   work_flexible.csv Location,EV,<one column per interval>  (per-interval kW cap)
#
# Node types: a node with an assigned CEV (place.csv e<i>=1) is a site; the first
# node (no CEV assigned) is the grid. Activity powers are KNOWN constants (no
# activities.csv); the Bayesian estimator is seeded from them (prior_sigma_frac)
# with true_power == those values. Any missing file/column raises a clear error.

_require_file(dir, name) = (p = joinpath(dir, name);
    isfile(p) ? p : error("Scenario_1 input mode: required file missing -> $p"))

function _read_csv(dir, name; required_cols = String[])
    df = CSV.read(_require_file(dir, name), DataFrame)
    for c in required_cols
        Symbol(c) in propertynames(df) ||
            error("Scenario_1 input mode: '$name' is missing required column '$c'")
    end
    return df
end

# "8:15:00" / " 24:00" / "0:15:00" -> decimal hours (8.25 / 24.0 / 0.25). Uses
# `string` (not `String`) so it also accepts a parsed Dates.Time cell, whichever way
# CSV.jl infers the time column.
_clock_hours(s) = (parts = split(strip(string(s)), ":");
    parse(Int, parts[1]) + (length(parts) >= 2 ? parse(Int, parts[2]) : 0) / 60)

# parameters.csv lookup (columns: Parameter,Value,Unit,Description)
function _psd(par, key)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    idx === nothing && error("Scenario_1 input mode: parameter '$key' missing in parameters.csv")
    return Float64(par.Value[idx])
end
function _psd_opt(par, key, default)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    return idx === nothing ? default : Float64(par.Value[idx])
end

function load_input_data(input_dir::AbstractString)
    isdir(input_dir) || error("Scenario_1 input mode: input directory not found -> $input_dir")

    par = _read_csv(input_dir, "parameters.csv"; required_cols = ["Parameter", "Value"])
    evd = _read_csv(input_dir, "ev_data.csv";   required_cols = ["SOE_min","SOE_max","SOE_ini","ch_rate"])
    mcd = _read_csv(input_dir, "mcs_data.csv";  required_cols =
            ["SOE_min","SOE_max","SOE_ini","CH_MCS","DCH_MCS","C_MCS_plug","DCH_MCS_plug","eta_ch_dch"])
    plc = _read_csv(input_dir, "place.csv";     required_cols = ["site","hours_digging","hours_loading_swinging"])
    tdd = _read_csv(input_dir, "time_data.csv"; required_cols = ["lambda_buy","intensity_tons_emissions"])
    ttm = _read_csv(input_dir, "travel_time.csv")
    wkf = _read_csv(input_dir, "work_flexible.csv"; required_cols = ["Location","EV"])

    # ---- scalars (extras default to ASSUMPTIONS; override via parameters.csv) ----
    delta_T = _psd(par, "delta_T");  k_trv = _psd(par, "k_trv")
    rho_miss         = _psd(par, "rho_miss");          rho_labor = _psd(par, "rho_labor")
    lambda_demand_NC = _psd(par, "lambda_demand_NC");  lambda_demand_OP = _psd(par, "lambda_demand_OP")
    carbon_price_per_ton = _psd_opt(par, "carbon_price_per_ton", 0.0)
    p_idling     = _psd_opt(par, "p_idling", 0.0)
    scale        = Int(round(_psd_opt(par, "scale", 2.0)))
    t_limit_rest = _psd_opt(par, "t_limit_rest", 1.0)
    kappa_wt     = Int(round(_psd_opt(par, "kappa_wt", 4.0)))
    day_end_hour = _psd_opt(par, "day_end_hour", 18.0)
    prior_sigma_frac = _psd_opt(par, "prior_sigma_frac", 0.2)
    obs_noise_std    = _psd_opt(par, "obs_noise_std", 0.05)
    co2_unit_scale   = _psd_opt(par, "co2_unit_scale", 1.0)

    # ---- time series + horizon (n_int from time_data; labels are interval END times) ----
    n_int   = nrow(tdd)
    t_start = _clock_hours(tdd[1, 1]) - delta_T
    lambda_whl_elec = Float64.(tdd.lambda_buy)
    lambda_CO2      = Float64.(tdd.intensity_tons_emissions) .* co2_unit_scale
    n_day = Int(round((day_end_hour - t_start) / delta_T))
    K = 1:n_day;  T = 1:(n_day + 1)
    # Receding-horizon multi-day length (kept days); simulate n_days + 1, drop the buffer.
    n_days = max(1, Int(round(_psd_opt(par, "n_days", 1.0))))

    # ---- ids / index maps (case-insensitive) ----
    ev_ids   = strip.(string.(evd[!, 1]))
    mcs_ids  = strip.(string.(mcd[!, 1]))
    node_ids = strip.(string.(plc.site))
    node_idx = Dict(lowercase(id) => i for (i, id) in enumerate(node_ids))
    ev_idx   = Dict(lowercase(id) => e for (e, id) in enumerate(ev_ids))
    N = 1:length(node_ids);  E = 1:length(ev_ids);  M = 1:length(mcs_ids)

    # ---- assignment from place.csv ev-id columns; node type inferred by assignment ----
    A = zeros(Int, length(N), length(E))
    for (e, eid) in enumerate(ev_ids)
        Symbol(eid) in propertynames(plc) ||
            error("simple dataset: place.csv missing assignment column '$eid'")
        col = plc[!, Symbol(eid)]
        for r in 1:nrow(plc)
            Int(round(Float64(col[r]))) == 1 && (A[node_idx[lowercase(node_ids[r])], e] = 1)
        end
    end
    N_c = [i for i in N if any(A[i, e] == 1 for e in E)]   # sites = a CEV is assigned
    N_g = [i for i in N if !(i in N_c)]                    # grid  = everything else
    isempty(N_g) && error("simple dataset: no grid node (a node with no EV assigned) found")
    isempty(N_c) && error("simple dataset: no site node (a node with an EV assigned) found")

    # ---- MCS / CEV parameters ----
    SOE_MCS_ini = Float64.(mcd.SOE_ini); SOE_MCS_max = Float64.(mcd.SOE_max)
    SOE_MCS_min = Float64.(mcd.SOE_min); CH_MCS = Float64.(mcd.CH_MCS); DCH_MCS = Float64.(mcd.DCH_MCS)
    DCH_MCS_plug = Float64.(mcd.DCH_MCS_plug); C_MCS_plug = Int.(mcd.C_MCS_plug); eta_ch_dch = Float64.(mcd.eta_ch_dch)
    SOE_CEV_ini = Float64.(evd.SOE_ini); SOE_CEV_max = Float64.(evd.SOE_max)
    SOE_CEV_min = Float64.(evd.SOE_min); CH_CEV = Float64.(evd.ch_rate)

    # ---- activity powers / Bayesian (folded into parameters.csv; powers are KNOWN) ----
    prior_mu    = [_psd(par, "p_digging"), _psd(par, "p_loading_swinging"), _psd(par, "p_traveling"), p_idling]
    prior_sigma = [max(prior_sigma_frac * prior_mu[j], 0.05) for j in 1:4]
    true_powers = copy(prior_mu)
    p_digging, p_loading_swinging, p_traveling = prior_mu[1], prior_mu[2], prior_mu[3]

    # ---- work demand per node ----
    hours_digging = zeros(length(N)); hours_loading_swinging = zeros(length(N))
    for r in 1:nrow(plc)
        i = node_idx[lowercase(node_ids[r])]
        hours_digging[i]          = Float64(plc.hours_digging[r])
        hours_loading_swinging[i] = Float64(plc.hours_loading_swinging[r])
    end

    # ---- travel-time matrix (values in INTERVALS; node names case-insensitive) ----
    tau_trv = zeros(length(N), length(N))
    tt_rows = lowercase.(strip.(string.(ttm[!, 1])))
    tt_cols = lowercase.(strip.(string.(names(ttm)[2:end])))
    for (ri, rn) in enumerate(tt_rows), (ci, cn) in enumerate(tt_cols)
        (haskey(node_idx, rn) && haskey(node_idx, cn)) || continue
        tau_trv[node_idx[rn], node_idx[cn]] = Float64(ttm[ri, ci + 1])
    end

    # ---- R_work[node, ev, k] from work_flexible (per-interval kW cap; 0 = no work) ----
    R_work = zeros(length(N), length(E), n_day)
    wf_time_cols = names(wkf)[3:end]
    for r in 1:nrow(wkf)
        loc = lowercase(strip(string(wkf.Location[r]))); ev = lowercase(strip(string(wkf.EV[r])))
        (haskey(node_idx, loc) && haskey(ev_idx, ev)) || continue
        i = node_idx[loc]; e = ev_idx[ev]
        for k in 1:min(n_day, length(wf_time_cols))
            R_work[i, e, k] = Float64(wkf[r, 2 + k])
        end
    end

    B = [1, 2, 3, 4]
    return (; delta_T, K, T, t_start, n_int, n_day, n_days, day_end_hour, t_limit_rest, kappa_wt,
              N, N_g, N_c, M, E, A,
              SOE_MCS_ini, SOE_MCS_max, SOE_MCS_min, CH_MCS, DCH_MCS,
              DCH_MCS_plug, C_MCS_plug, eta_ch_dch,
              SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, CH_CEV,
              p_digging, p_loading_swinging, p_traveling, p_idling,
              prior_mu, prior_sigma, true_powers, obs_noise_std,
              hours_digging, hours_loading_swinging, tau_trv, k_trv,
              lambda_whl_elec, lambda_CO2, R_work,
              rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP,
              carbon_price_per_ton, scale, B)
end

# Dispatcher: :synthetic builds artificial data; :input loads the 7-CSV dataset.
function load_data(mode::Symbol; input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"))
    if mode == :synthetic
        return build_default_data()
    elseif mode == :input
        return load_input_data(input_dir)
    else
        error("Unknown data mode :$mode (use :synthetic or :input)")
    end
end

# Integer arc travel-step matrix (>=1 step for any positive off-diagonal time).
function normalize_travel_steps(tau_trv, N)
    n = length(N)
    steps = zeros(Int, n, n)
    for i in N, j in N
        steps[i, j] = i == j ? 0 : max(1, Int(round(tau_trv[i, j])))
    end
    return steps
end

# Is interval k inside the 16:00-21:00 on-peak window?
function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

# =============================================================================
# 1b. ONLINE POWER ESTIMATOR  (the feedback that "improves power estimates")
# =============================================================================
#
# This uses the same Bayesian regression as the offline model:
#
#     x_a    ~ TruncatedNormal(mu_a, sigma_a, lower = 0)     # per-activity power
#     sigma  ~ HalfNormal(std(b))                            # observation noise
#     b_i    ~ Normal( (A x)_i , sigma )                     # energy-balance rows
#
# inferred by NUTS / MCMC (via Turing.jl). It is NOT a Kalman/Gaussian shortcut —
# exact priors, likelihood, and sampler family. Online use = accumulate observation
# rows (A, b) as telematics arrives and re-fit with all data so far (Bayesian
# updating with a fixed prior).
#
# Each observation row is (a, b):
#   a : activity-hours performed in the interval, e.g. [0.083, 0, 0.167]
#   b : realized work energy that interval (kWh), from the SOC drop:
#         b = charging_received*dt - (SOC_next - SOC_now)*battery_cap

# Turing model for the activity-power regression.
# NOTE: we use EXPLICIT scalar parameters x1..x4 (one per activity) instead of an
# array `x[i]`. Newer Turing/FlexiChains versions name array variables in a way
# that `chain[Symbol("x[i]")]` can no longer resolve; scalar names are retrieved
# reliably as `chain[:x1]` across versions. (There are always 4 activities:
# digging, loading/swinging, traveling, idling.)
Turing.@model function activity_power_model(A, b, prior_mu, prior_sigma, sigma_b)
    x1 ~ truncated(Normal(prior_mu[1], prior_sigma[1]); lower = 0.0)        # digging
    x2 ~ truncated(Normal(prior_mu[2], prior_sigma[2]); lower = 0.0)        # loading/swinging
    x3 ~ truncated(Normal(prior_mu[3], prior_sigma[3]); lower = 0.0)        # traveling
    x4 ~ truncated(Normal(prior_mu[4], prior_sigma[4]); lower = 0.0)        # idling
    x = [x1, x2, x3, x4]
    s ~ truncated(Normal(0.0, sigma_b); lower = 0.0)                        # HalfNormal(sigma_b)
    mu = A * x
    for j in eachindex(b)
        b[j] ~ Normal(mu[j], s)
    end
end

mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}      # TruncatedNormal means (the offline Bayesian prior)
    prior_sigma::Vector{Float64}   # TruncatedNormal sigmas
    A_obs::Matrix{Float64}         # accumulated activity-hour rows
    b_obs::Vector{Float64}         # accumulated realized energies (kWh)
    mu::Vector{Float64}            # current posterior mean = the median profile used by MPC
    sd::Vector{Float64}            # current posterior std (uncertainty)
    mcmc_samples::Int
end

function BayesianActivityEstimator(prior_mu, prior_sigma; mcmc_samples = 500)
    k = length(prior_mu)
    return BayesianActivityEstimator(collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     Matrix{Float64}(undef, 0, k), Float64[],
                                     collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     mcmc_samples)
end

# Append one telematics observation (does not re-fit; call refit! to update).
function observe!(est::BayesianActivityEstimator, a::AbstractVector, b::Real)
    est.A_obs = vcat(est.A_obs, reshape(collect(float.(a)), 1, :))
    push!(est.b_obs, float(b))
    return est
end

# Re-run the Bayesian regression on ALL data so far and refresh the estimate.
function refit!(est::BayesianActivityEstimator)
    isempty(est.b_obs) && return est
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0   # HalfNormal scale = std(b)
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, est.prior_sigma, sigma_b)
    chain = sample(model, NUTS(0.9), est.mcmc_samples; progress = false)
    syms = (:x1, :x2, :x3, :x4)                  # scalar parameter names (see model)
    for i in 1:length(est.prior_mu)
        col = vec(chain[syms[i]])
        est.mu[i] = mean(col)
        est.sd[i] = std(col)
    end
    return est
end

# =============================================================================
# 2. WINDOW MILP  (Box 3: Single-Instance MPC Solver, one median scenario)
# =============================================================================
#
# Faithful implementation of the full window model, specialized to a CROSS-DAY
# prediction window K_win (global indices spanning the rest of today plus
# lookahead_days future daytime blocks), started from the realized physical state.
# Nights inside the window are handled by two link rules (MCS reset to full + parked
# at the grid overnight; CEV battery carries over). The terminal energy-neutral SOE
# for the CEVs is applied ONLY when the window reaches the true horizon end; other
# constraints apply exactly, including:
#   * terminal position (MCS parked at a grid node every 18:00),
#   * flow conservation, precedence,
#   * grid-connection exclusivity, plug/presence logic, peak-demand trackers.
#
# Closed-loop carry-in (exact state hand-off between 15-min re-solves):
#   soe_mcs0[m] / soe_cev0[e]          realized state of energy (kWh)
#   mcs_node0[m]                       node the MCS is parked at (0 if in transit)
#   mcs_transit0[m]                    nothing, or (i, j, r): in transit on arc
#                                      (i->j) with r intervals of travel remaining
#   rem_dig[i] / rem_load[i]           remaining work hours per site
#   cum_dig_e[e]/cum_load_e[e]/cum_trv_e[e]  per-CEV hours already done
#   peak_nc0 / peak_op0                realized daily peak grid draw so far (kW)
#
# The only modeling concession vs. a single full-day solve is that work already
# completed before the window enters through rem_* / cum_* (re-planning the
# REMAINING day), which is exactly what a receding-horizon controller should do.

function build_window_model(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, cum_dig_e, cum_load_e, cum_trv_e,
                            peak_nc0, peak_op0, pvec;
                            daily_dig = rem_dig, daily_load = rem_load,
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true,
                            soft_prec::Bool = false,
                            soft_pace::Bool = false,
                            soft_term::Bool = false,
                            enforce_cev_terminal::Bool = true,
                            is_global_terminal::Bool = (last(collect(K_win)) == d.n_day),
                            term_tol::Float64 = 0.0)
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

    # =========================================================================
    # MULTI-DAY WINDOW GEOMETRY (RECEDING HORIZON)
    # =========================================================================
    # K may span SEVERAL days' daytime blocks laid end to end (day 1 daytime, day 2
    # daytime, ...). n_day = daytime intervals per day. For a GLOBAL interval index k:
    #   wd(k)    = its position WITHIN its day (1..n_day) -> indexes the daily price /
    #              work-availability profile, which is the SAME shape every day.
    #   dayof(k) = which day it belongs to (1, 2, ...).
    # The "nights" between day-blocks are not intervals themselves; they are handled by
    # two link rules (below): the MCS battery is recharged to full and the MCS is parked
    # at the grid overnight, while the CEV battery simply CARRIES OVER unchanged.
    n_day = d.n_day
    wd(k)    = mod(k - 1, n_day) + 1
    dayof(k) = div(k - 1, n_day) + 1

    K = collect(K_win)                      # window interval indices (GLOBAL)
    Tb = vcat(K, last(K) + 1)               # window boundary indices (|K|+1)
    K_peak = [k for k in K if in_peak(wd(k), delta_T, d.t_start)]   # on-peak by within-day clock
    blockdays  = sort(unique(dayof.(K)))                  # days this window touches
    firstday   = dayof(first(K))                          # the (possibly partial) current day
    block_ks(dy) = [k for k in K if dayof(k) == dy]       # global intervals of day dy in-window
    # "Evening" intervals = the last daytime interval of each day present in the window
    # (18:00). At each evening the MCS must be parked home; after each evening EXCEPT the
    # final one the MCS battery is recharged (reset to full) for the next morning.
    eve_k      = [k for k in K if wd(k) == n_day]
    night_eve  = [k for k in eve_k if k != last(K)]       # evenings that are followed by another day
    # per-day within-day price/CO2/work-availability lookups (same daily profile each day)
    price_k(k) = d.lambda_whl_elec[wd(k)]
    co2_k(k)   = d.lambda_CO2[wd(k)]
    Rwork(i, e, k) = d.R_work[i, e, wd(k)]
    # productive_k[k] = true iff dig/load/travel are allowed in interval k (shift hours,
    # lunch EXCLUDED). Outside productive hours the CEV still idles on site and may charge.
    productive_k = Dict(k => any(Rwork(i, e, k) > 0 for i in N_c, e in E) for k in K)

    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))   # incl. idling (last)

    # cumulative per-site hours already done (for precedence; sum over assigned CEVs).
    # Only carried into the FIRST (current) day-block; later day-blocks start fresh.
    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    # carried-transit helpers (r intervals of travel still pending at window start)
    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])          # first r window intervals
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    set_time_limit_sec(model, time_limit_sec)
    # Force HiGHS to run serial + deterministic. The multi-threaded MIP path can
    # segfault on Windows for the larger (multi-CEV) models; serial is stable and the
    # per-window problems are small enough that the speed cost is negligible.
    set_attribute(model, "threads", 1)
    set_attribute(model, "parallel", "off")
    # Disable HiGHS's sub-MIP primal heuristics (RENS/RINS). These launch an INTERNAL
    # sub-MIP solver that spins up HiGHS's parallel task deque even when the OUTER model
    # is set serial (threads=1/parallel=off do NOT propagate into the sub-solver), which
    # intermittently segfaults on Windows (EXCEPTION_ACCESS_VIOLATION in HighsSplitDeque).
    # Turning heuristic effort off keeps the solver on the stable serial branch-and-cut
    # path; the per-window MILPs are small, so incumbent quality is unaffected in practice.
    set_attribute(model, "mip_heuristic_effort", 0.0)
    # Same reason: HiGHS's root-node symmetry detection also uses the parallel task deque
    # and is a second source of the Windows segfault. Disable it too.
    set_attribute(model, "mip_detect_symmetry", false)
    # Stop each window as soon as HiGHS has a near-optimal incumbent (1% gap). MPC only
    # applies the FIRST interval of each solve, so proving full optimality every window
    # is wasted effort; this keeps the serial solver fast instead of running to the limit.
    set_attribute(model, "mip_rel_gap", 1.0e-2)

    # ---- power-flow variables (kW) ----
    @variable(model, P_ch_MCS[M, N, K] >= 0)
    @variable(model, P_dch_MCS[M, N, K] >= 0)
    @variable(model, P_MCS_CEV[M, N_c, E, K] >= 0)
    @variable(model, P_work[N_c, E, K] >= 0)
    @variable(model, P_ch_tot[M, K] >= 0)
    @variable(model, P_dch_tot[M, K] >= 0)
    # MULTI-DAY work-quota shortfall slacks, one per (site, day-block in this window).
    # s_miss_dig[i, dy] = hours of that day's CUMULATIVE digging quota still unmet by the
    # end of day-block dy (>= 0). Any shortfall carries into the next day (because the
    # target is cumulative) and is penalised again -> a soft "leftover work rolls over
    # with a penalty" exactly as requested.
    @variable(model, s_miss_dig[N_c, blockdays] >= 0)
    @variable(model, s_miss_load[N_c, blockdays] >= 0)
    # Precedence (12d) + travel-pacing (13) slacks. HARD by default (pinned to 0 below);
    # freed only via the manual soft_prec / soft_pace switches.
    @variable(model, s_prec[N_c, K] >= 0)
    @variable(model, s_pace_hi[E, K] >= 0)   # travel-pacing upper band (13a)
    @variable(model, s_pace_lo[E, K] >= 0)   # travel-pacing lower band (13b)

    # ---- travel energy (kWh) ----
    @variable(model, L_trv[M, N, N, K] >= 0)
    @variable(model, L_trv_tot[M, K] >= 0)

    # ---- state of energy (boundary-indexed over the window) ----
    @variable(model, SOE_MCS[M, Tb] >= 0)
    @variable(model, SOE_CEV[E, Tb] >= 0)

    # ---- binaries ----
    @variable(model, u[E, N, B, K], Bin)          # activity selection
    @variable(model, mu[N, E, K], Bin)            # CEV in charging mode
    @variable(model, rho[M, N, E, K], Bin)        # CEV plugged into MCS
    @variable(model, z[M, N, K], Bin)             # MCS parked at node
    @variable(model, g_ch[M, N_g, K], Bin)        # MCS charge-active at grid node
    @variable(model, x[M, N, N, K], Bin)          # MCS departs i->j
    @variable(model, y_trv[M, N, N, K], Bin)      # MCS in transit on arc
    @variable(model, beta_arr[M, N, K], Bin)
    @variable(model, beta_dep[M, N, K], Bin)
    @variable(model, P_peak_NC >= 0)
    @variable(model, P_peak_OP >= 0)
    # CEV energy-neutral terminal slack (Eq. 8b). In the default HARD mode this is
    # pinned to 0 (so SOE_CEV[end] == SOE_CEV_ini exactly); it is only freed via the
    # manual soft_term switch.
    # NOTE: the MCS has NO Phase-1 terminal-energy constraint -- its energy-neutral
    # cycle (back to SOE_MCS_ini) is restored OVERNIGHT in Phase 2 (smart charge).
    @variable(model, s_term_cev[E] >= 0)

    # ---- objective: Eq. (1) operating cost ----
    # J = grid energy + monetized carbon + NC/OP demand charges + missed-work penalty
    #     (rho_miss * sum s_miss, the model's own soft slack from Eq. 12c) + MCS towing
    #     labour (rho_labor * dt * sum y_trv). This is EXACTLY Eq. (1) -- no extra penalties.
    obj = @expression(model,
        sum(price_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        sum((d.carbon_price_per_ton / 1000.0) * co2_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        d.rho_miss * (sum(s_miss_dig[i, dy] for i in N_c, dy in blockdays) +
                      sum(s_miss_load[i, dy] for i in N_c, dy in blockdays)) +
        d.lambda_demand_NC * P_peak_NC +
        d.lambda_demand_OP * P_peak_OP +
        d.rho_labor * delta_T * sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K))

    # HARD MODE (default): each soft slack is PINNED to zero, so subactivity precedence
    # (12d), travel pacing (13) and the CEV energy-neutral terminal (8b) hold EXACTLY, and
    # the objective is exactly Eq. (1). Setting a soft_* flag frees that one group
    # (penalising its slack instead) -- available per-group for manual relaxation.
    W_prec = 8.0e2; W_pace = 1.0e2; W_term = 1.5e2
    soft_prec || @constraint(model, [i in N_c, k in K], s_prec[i, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_hi[e, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_lo[e, k] == 0)
    soft_term || @constraint(model, [e in E], s_term_cev[e] == 0)
    @objective(model, Min, obj +
        (soft_prec ? W_prec * sum(s_prec[i, k] for i in N_c, k in K) : AffExpr(0.0)) +
        (soft_pace ? W_pace * sum(s_pace_hi[e, k] + s_pace_lo[e, k] for e in E, k in K) : AffExpr(0.0)) +
        (soft_term ? W_term * sum(s_term_cev[e] for e in E) : AffExpr(0.0)))

    # ---- power aggregation & node feasibility ----
    @constraint(model, [m in M, k in K], P_ch_tot[m, k]  == sum(P_ch_MCS[m, i, k]  for i in N_g))
    @constraint(model, [m in M, k in K], P_dch_tot[m, k] == sum(P_dch_MCS[m, i, k] for i in N_c))
    @constraint(model, [m in M, i in N_g, k in K], P_dch_MCS[m, i, k] == 0)
    @constraint(model, [m in M, i in N_c, k in K], P_ch_MCS[m, i, k]  == 0)
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] == sum(P_MCS_CEV[m, i, e, k] for e in E))
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] <= d.DCH_MCS[m] * z[m, i, k])

    # grid-connection exclusivity
    @constraint(model, [m in M, i in N_g, k in K], P_ch_MCS[m, i, k] <= d.CH_MCS[m] * g_ch[m, i, k])
    @constraint(model, [m in M, i in N_g, k in K], g_ch[m, i, k] <= z[m, i, k])
    @constraint(model, [i in N_g, k in K], sum(g_ch[m, i, k] for m in M) <= 1)

    # plug-level and CEV-acceptance limits
    @constraint(model, [m in M, i in N_c, e in E, k in K],
        P_MCS_CEV[m, i, e, k] <= d.DCH_MCS_plug[m] * rho[m, i, e, k])
    @constraint(model, [i in N_c, e in E, k in K],
        sum(P_MCS_CEV[m, i, e, k] for m in M) <= d.CH_CEV[e] * mu[i, e, k])

    # peak demand trackers (seeded with the realized daily peak so far)
    @constraint(model, P_peak_NC >= peak_nc0)
    @constraint(model, P_peak_OP >= peak_op0)
    @constraint(model, [k in K], P_peak_NC >= sum(P_ch_tot[m, k] for m in M))
    @constraint(model, [k in K_peak], P_peak_OP >= sum(P_ch_tot[m, k] for m in M))
    if peak_demand_limit !== nothing
        @constraint(model, [k in K], sum(P_ch_tot[m, k] for m in M) <= peak_demand_limit)
    end

    # ---- travel energy bookkeeping (NOTE: k_trv * delta_T, as in the original) ----
    for m in M, i in N, j in N, k in K
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k] == 1)        # continue a trip begun earlier
        else
            @constraint(model, y_trv[m, i, j, k] == sum(x[m, i, j, tau]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end
    @constraint(model, [m in M, i in N, j in N, k in K],
        L_trv[m, i, j, k] == d.k_trv * delta_T * y_trv[m, i, j, k])
    @constraint(model, [m in M, k in K],
        L_trv_tot[m, k] == sum(L_trv[m, i, j, k] for i in N, j in N))

    # ---- energy dynamics (boundary-indexed within the window) ----
    @constraint(model, [m in M], SOE_MCS[m, first(Tb)] == soe_mcs0[m])
    @constraint(model, [e in E], SOE_CEV[e, first(Tb)] == soe_cev0[e])
    # MCS energy flow WITHIN each day. The night boundaries (k in night_eve) are handled
    # separately below: the MCS is recharged OVERNIGHT (Phase 2) back to SOE_MCS_ini, so
    # the next morning starts full regardless of how far it drained during the day. We
    # therefore SKIP the intraday flow link across a night and pin the next-morning SOE.
    @constraint(model, [m in M, k in K; !(k in night_eve)],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
            d.eta_ch_dch[m] * P_ch_tot[m, k] * delta_T -
            (P_dch_tot[m, k] * delta_T) / d.eta_ch_dch[m] -
            L_trv_tot[m, k])
    # Overnight bridge: each MCS starts the next day recharged to its start-of-day level.
    @constraint(model, [m in M, k in night_eve], SOE_MCS[m, k + 1] == d.SOE_MCS_ini[m])
    # CEV battery CARRIES OVER continuously across nights (no reset) -- the single flow
    # link below applies to every interval, night boundaries included.
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
            sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -
            sum(P_work[i, e, k] for i in N_c) * delta_T)

    # SOE bounds at every boundary
    @constraint(model, [m in M, t in Tb], d.SOE_MCS_min[m] <= SOE_MCS[m, t] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb], d.SOE_CEV_min[e] <= SOE_CEV[e, t] <= d.SOE_CEV_max[e])

    # No MCS recharge gating: daytime grid charging is allowed normally (capped by
    # g_ch / CH_MCS above). Because Phase 1 carries NO MCS terminal-energy reward, the
    # cost-minimiser draws from the grid during the day ONLY when needed to respect the
    # 20% floor; the bulk recharge back to SOE_MCS_ini is deferred to the cheap
    # overnight hours (Phase 2).

    # Day-closing target (only when the window reaches the end of the daytime horizon,
    # i.e. 18:00). Energy-neutral terminal (8b) for the CEV: each excavator MUST end
    # the day back at its START-OF-DAY SOE (SOE_CEV_ini).
    #
    # HARD mode (default): a "return to (essentially) full" as the one-sided HARD
    # inequality SOE_CEV[end] >= SOE_CEV_ini - term_tol. The upper side is already enforced
    # by the SOE_max bound (SOE_CEV_ini <= SOE_CEV_max), so this is the exact equality when
    # term_tol = 0. A SMALL term_tol > 0 gives just enough margin to absorb
    # the online estimator's certainty-equivalent drift (the estimated powers used to PLAN
    # differ slightly from the TRUE powers that move the battery), which otherwise makes a
    # zero-margin equality-at-max infeasible on re-plan.
    #
    # soft_term (manual switch): relax to a penalised two-sided slack instead.
    #
    # MULTI-DAY RECEDING HORIZON: `enforce_cev_terminal` gates this daily wrap-up.
    # On the KEPT real days it is set FALSE, so the CEV battery is NOT forced back to
    # its start level each 18:00 -- it simply flows into the next day (only the SOE
    # floor/ceiling bounds apply). It is set TRUE only on the final BUFFER day (which we
    # then drop), so the artificial "return to initial SOE" end-effect lands on the
    # discarded day and never distorts the reported days.
    if is_global_terminal && enforce_cev_terminal
        if soft_term
            @constraint(model, [e in E],  SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e] <= s_term_cev[e])
            @constraint(model, [e in E], -(SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e]) <= s_term_cev[e])
        else
            @constraint(model, [e in E], SOE_CEV[e, last(Tb)] >= d.SOE_CEV_ini[e] - term_tol)
        end
    end

    # ---- plugging / presence logic ----
    @constraint(model, [m in M, i in N_c, k in K], sum(rho[m, i, e, k] for e in E) <= d.C_MCS_plug[m])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= d.A[i, e])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])
    @constraint(model, [m in M, i in N, k in K], x[m, i, i, k] == 0)

    # presence partition: parked at one node OR in transit on one arc
    @constraint(model, [m in M, k in K],
        sum(z[m, i, k] for i in N) + sum(y_trv[m, i, j, k] for i in N, j in N if i != j) == 1)

    # initial position: parked-at / departing-from the realized node, OR continue
    # an in-progress trip (handled by the carried y_trv constraints above).
    for m in M
        if mcs_transit0[m] === nothing
            p = mcs_node0[m]
            @constraint(model, z[m, p, first(K)] + sum(x[m, p, j, first(K)] for j in N if j != p) == 1)
        end
    end

    @constraint(model, [m in M, i in N, k in K],
        beta_dep[m, i, k] == sum(x[m, i, j, k] for j in N if j != i))
    for m in M, j in N, k in K
        if carried_arrival_k(m) == k && j == mcs_transit0[m][2]
            @constraint(model, beta_arr[m, j, k] == 1)        # arrival of the carried trip
        else
            terms = Any[]
            for i in N
                i == j && continue
                tau = k - travel_steps[i, j]
                tau in K && push!(terms, x[m, i, j, tau])
            end
            @constraint(model, beta_arr[m, j, k] == (isempty(terms) ? 0 : sum(terms)))
        end
    end
    @constraint(model, [m in M, i in N, k in K[2:end]],
        beta_arr[m, i, k] - beta_dep[m, i, k] == z[m, i, k] - z[m, i, k - 1])
    @constraint(model, [m in M, i in N, k in K],
        beta_arr[m, i, k] + beta_dep[m, i, k] <= 1)

    # flow conservation per node, generalized for an MPC window that may start with
    # the MCS parked at ANY node (a site, not just the grid) or mid-trip, and that
    # (when terminal) ends parked at a grid node:
    #
    #     arrivals(i) - departures(i) = present_end(i) - present_start(i)
    #
    # present_start(i) = 1 if the MCS is parked at node i at the window start (0 when
    #   mid-trip; a carried trip's dangling arrival is already on the arrivals side).
    # present_end(i)   = z[m, i, last(K)] (the parked node at the window end).
    #
    # This reduces to the original arr == dep balance when the MCS both starts and
    # ends at the grid. The original (destination +1) form made every travelling
    # window INFEASIBLE, and a plain arr == dep form is infeasible whenever a window
    # starts with the MCS parked at a site (it departs once more than it arrives).
    for m in M, i in N
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k] for k in K) - sum(beta_dep[m, i, k] for k in K) ==
            z[m, i, last(K)] - start_here)
    end

    # Terminal position (Eq. 10e): at the END of EVERY daytime block present in the
    # window (each 18:00 = eve_k), the MCS must be parked at a grid node, ready for that
    # night's overnight Phase-2 recharge. This covers both the interior nights of a
    # multi-day window and the final horizon end. The planner always reserves enough
    # time to drive home, so this is feasible every window.
    @constraint(model, [m in M, k in eve_k], sum(z[m, i, k] for i in N_g) == 1)

    # optional site-visit rules (match original flags)
    if require_site_visit
        @constraint(model, [m in M], sum(beta_arr[m, i, k] for i in N_c, k in K) >= 1)
    end
    if single_visit_per_site
        @constraint(model, [m in M, i in N_c], sum(beta_arr[m, i, k] for k in K) <= 1)
        @constraint(model, [m in M, i in N_c], sum(beta_dep[m, i, k] for k in K) <= 1)
    end

    # ---- activity scheduling ----
    # Throughout the DAYTIME horizon a CEV is always PRESENT at its own site doing
    # exactly one activity (idling included), so idling is the residual that fills any
    # interval not spent on useful work. Useful work (dig/load/travel) is allowed only
    # in PRODUCTIVE intervals (shift hours, lunch excluded); during lunch and the
    # 17:00-18:00 wind-down the CEV idles and may charge.
    @constraint(model, [i in N_c, e in E, k in K],
        sum(u[e, i, a, k] for a in B) == d.A[i, e])
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= d.A[i, e])
    # Eq. (5a): productive work power is capped by the CEV work capacity R_{e,t}
    # (per site/interval; 0 outside work hours) times the assignment A_{i,e}, and forced
    # to ZERO while the CEV is charging via the (1 - mu) factor. Because R_work = 0
    # outside productive hours, this also gates dig/load/travel to work hours (idling,
    # which carries its own power, stays free at all times).
    @constraint(model, [i in N_c, e in E, k in K],
        sum(p_activity[a] * u[e, i, a, k] for a in (B[1], B[2], B[3])) <=
        Rwork(i, e, k) * d.A[i, e] * (1 - mu[i, e, k]))
    # a CEV can only be charged while it is idling (charging => idle activity)
    @constraint(model, [i in N_c, e in E, k in K], mu[i, e, k] <= u[e, i, B[4], k])
    # per-interval CEV power draw (idling draws its small idle power)
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- remaining work demand: DAILY quota as a soft, cumulative TARGET ------------
    # Each day a fresh quota (daily_dig / daily_load) arrives; the window-start remaining
    # rem_* already holds the CURRENT day's outstanding hours plus any carried leftover.
    # For every day-block dy present in the window we require the CUMULATIVE work done
    # through the END of dy to reach the CUMULATIVE target (rem_* + one fresh quota per
    # subsequent morning). Any shortfall s_miss_* >= 0 is penalised (rho_miss) and, because
    # the target is cumulative, automatically ROLLS OVER into the next day -- exactly the
    # "leftover work carries to the next day with a penalty" behaviour. Working AHEAD is
    # allowed (slack simply hits 0), so no upper-bound infeasibility.
    for (p, dy) in enumerate(blockdays)
        Kupto = [k for k in K if dayof(k) <= dy]           # window intervals up to end of dy
        @constraint(model, [i in N_c],
            s_miss_dig[i, dy] >= (max(rem_dig[i], 0.0) + (p - 1) * daily_dig[i]) -
                                 delta_T * sum(u[e, i, B[1], k] for e in E, k in Kupto))
        @constraint(model, [i in N_c],
            s_miss_load[i, dy] >= (max(rem_load[i], 0.0) + (p - 1) * daily_load[i]) -
                                  delta_T * sum(u[e, i, B[2], k] for e in E, k in Kupto))
    end

    # precedence: cumulative loading/swinging <= scale * cumulative digging, evaluated
    # WITHIN each day-block (counters restart each morning; carried realized work is
    # seeded only into the first, current day-block).
    # Soft (+s_prec): realized drift may seed a momentary violation at window start.
    bstart(k) = first(block_ks(dayof(k)))                  # first in-window interval of k's day
    @constraint(model, [i in N_c, k in K],
        (((dayof(k) == firstday) ? cum_load_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[2], tau] for tau in bstart(k):k, e in E)) <=
        d.scale * (((dayof(k) == firstday) ? cum_dig_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[1], tau] for tau in bstart(k):k, e in E)) +
        s_prec[i, k])

    # ---- rest rule (Eq. 12e): operator break ----
    # Over any rolling window of (t_limit_rest + delta_T) hours, a CEV may perform
    # construction-related work (dig/load/travel) for at most t_limit_rest hours --
    # i.e. >= 1 idle break interval per window. The idle interval may be used to charge.
    rest_cap = Int(round(d.t_limit_rest / delta_T))      # max work intervals per window
    rest_win = rest_cap + 1                               # window length in intervals
    if length(K) >= rest_win
        # Only apply to rolling windows that stay WITHIN a single day-block (a night is a
        # full rest, so a break need not be enforced across it).
        rest_starts = [k0 for k0 in first(K):(last(K) - rest_win + 1)
                       if dayof(k0) == dayof(k0 + rest_win - 1)]
        @constraint(model, [i in N_c, e in E, k0 in rest_starts],
            sum(u[e, i, a, k] for a in (B[1], B[2], B[3]), k in k0:(k0 + rest_win - 1)) <= rest_cap)
    end

    # ---- travel pacing (Eq. 13): CEV repositioning cadence ----
    # Tie each CEV's cumulative traveling (repositioning) to its cumulative productive
    # work: about one travel interval per kappa_wt productive intervals (two-sided
    # band). Counts are seeded with realized work so the band holds over the whole day;
    # the soft slacks s_pace_hi/lo absorb closed-loop drift.
    kappa = d.kappa_wt
    for e in E, kk in K
        carry = (dayof(kk) == firstday)                    # seed realized only in current day
        bs = bstart(kk)                                    # first in-window interval of kk's day
        trv_cum  = (carry ? cum_trv_e[e] / delta_T : 0.0) +
                   sum(u[e, i, B[3], tau] for i in N_c, tau in bs:kk)
        work_cum = (carry ? (cum_dig_e[e] + cum_load_e[e]) / delta_T : 0.0) +
                   sum(u[e, i, a, tau] for i in N_c, a in (B[1], B[2]), tau in bs:kk)
        @constraint(model, kappa * trv_cum <= work_cum + s_pace_hi[e, kk])
        @constraint(model, kappa * trv_cum >= work_cum - kappa - s_pace_lo[e, kk])
    end

    # Solve. HiGHS's native MIP path can, rarely and non-deterministically on Windows,
    # throw a memory fault (e.g. ReadOnlyMemoryError) on a particular window. We catch it
    # so a single bad solve does NOT kill a multi-hour closed-loop run: the caller checks
    # has_values(model) and, finding none, treats this interval as infeasible and HOLDS
    # state (identical to the no-fallback behaviour for a genuinely infeasible window).
    try
        optimize!(model)
    catch err
        @warn "Solver threw during optimize!; treating this window as no-solution (hold state)." exception = err
    end
    return model
end

# =============================================================================
# 3. MPC LOOP  (Box 4 + Closed-Loop Update arrow)
# =============================================================================
#
# Multi-day, cross-day receding horizon: at each interval we solve a window that
# spans the rest of today plus lookahead_days future daytime blocks (global index
# g0 = (day-1)*nK + k0) from the realized physical state, apply only g0's decisions,
# advance the realized state (incl. MCS in-transit carry-over and daily-peak
# carry-over), then move to the next interval and re-solve. The CEV energy-neutral
# terminal fires only at the true horizon end (the dropped buffer day); the MCS is
# parked at the grid + recharged overnight every night, and the CEV battery and any
# unfinished work carry across days.
#
# Realized execution can DIFFER from the plan: within a 15-min interval the CEV
# may split work across several sub-activities, and it draws the (unknown) TRUE
# power. Each interval we observe that realized energy drop + activity-time mix
# and re-fit the Bayesian power estimate before the next solve.

# Realized within-interval activity durations (hours) for CEV e in interval k0.
# The MILP plans at 15-min granularity (one activity per interval), but in
# reality the machine SPLITS the interval across sub-activities, e.g. "5 min
# digging, 10 min traveling". Real telemetry reports these exact durations; here
# we synthesize a plausible split so the estimator receives richer, better-
# conditioned multi-activity regression rows (each interval -> one mixed row).
#
# Returns a vector over B = [digging, loading/swinging, traveling, idling] in
# HOURS, summing to delta_T. Idling is the physical residual: whatever fraction
# of the interval is not spent on the planned activity is spent idling.
function realized_activity_durations(rng, model, e, k0, d; multi::Bool = true)
    dt = d.delta_T
    a = zeros(length(d.B))
    idle = length(d.B)                         # idling is the last activity

    # planned activity index for this interval (0 = none found -> all idle)
    planned = 0
    for i in d.N_c, (ai, act) in enumerate(d.B)
        if value(model[:u][e, i, act, k0]) > 0.5
            planned = ai
        end
    end
    if planned == 0
        a[idle] = dt                           # safety: treat as a full idle interval
        return a
    end

    if !multi
        a[planned] = dt                        # single-activity interval (old behavior)
        return a
    end

    # 60-100% of the interval on the planned activity; the remainder is idling
    # (machine momentarily stopped / repositioning pause between tasks).
    frac = 0.6 + 0.4 * rand(rng)
    a[planned] += dt * frac
    a[idle]    += dt * (1.0 - frac)
    return a
end

# Determine the MCS's realized state at the START of the next interval (k0+1)
# from the solved window. Returns (node, transit):
#   node    = parked node index, or 0 if the MCS is in transit at k0+1
#   transit = nothing, or (i, j, r): in transit on arc i->j with r intervals left
function advance_mcs_state(model, m, k0, nK, d)
    z = model[:z]; y = model[:y_trv]
    Kw = axes(z)[3]                       # interval axis of the solved window
    knext = k0 + 1
    if knext > nK || !(knext in Kw)
        node = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)
        return (node === nothing ? first(d.N_g) : node, nothing)
    end
    node = findfirst(i -> value(z[m, i, knext]) > 0.5, d.N)
    node !== nothing && return (node, nothing)
    for i in d.N, j in d.N
        i == j && continue
        if value(y[m, i, j, knext]) > 0.5
            r = 0; k = knext
            while k <= nK && value(y[m, i, j, k]) > 0.5
                r += 1; k += 1
            end
            return (0, (i, j, r))
        end
    end
    node0 = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)
    return (node0 === nothing ? first(d.N_g) : node0, nothing)
end

# ---- worker-facing readouts (the simple front-end CSV) ----------------------
# Map the MILP's chosen activity / charging decisions for the APPLIED interval k0
# into plain words a site worker can act on. Everything else (powers, SOE, prices,
# estimates) stays in the detailed analyst CSV.
const _ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# The single activity the plan wants CEV e to do this interval ("Off (home)" when
# the shift is over and the machine is powered down).
function _planned_activity(model, d, e, k0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    site === nothing && return "Off (home)"
    vals = [value(model[:u][e, site, a, k0]) for a in eachindex(d.B)]
    sum(vals) < 0.5 && return "Off (home)"
    return _ACT_NAME[d.B[argmax(vals)]]
end

# Should CEV e be plugged into the MCS to charge this interval?
function _cev_should_charge(model, d, e, k0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    return (site !== nothing && value(model[:mu][site, e, k0]) > 0.5) ? "Yes" : "No"
end

# Should the MCS be drawing from the grid this interval?
_mcs_should_charge(model, d, k0) =
    (sum(value(model[:P_ch_tot][m, k0]) for m in d.M) > 1e-6) ? "Yes" : "No"

# =============================================================================
# 2b. PHASE 2 — OVERNIGHT SMART-CHARGE  (deterministic; NOT an MPC)
# =============================================================================
# Once Phase 1 ends (18:00) the MCS is parked at a grid node with some SOE. The
# overnight task is trivial and deterministic: restore each MCS back to its
# START-OF-DAY level (SOE_MCS_ini -> energy-neutral over the full 24 h) by buying
# the CHEAPEST available overnight 15-min slots, at rate <= CH_MCS, never exceeding
# soe_max. No routing, no work, no learning -- exactly a "smart charger" handed a
# known deficit and a price curve. Greedy cheapest-first fill is optimal here
# because the MCS only charges (monotone SOE) and the target (= SOE_ini) <= soe_max.
#
# Returns (df, P_ov, ov_k):
#   df   : per-overnight-interval schedule (price, per-MCS charge kW / SOE / flag)
#   P_ov : MCS x overnight-interval charge-power matrix (kW)
#   ov_k : the overnight interval indices (n_day+1 .. n_int)
function phase2_overnight_charge(d, soe_mcs_end)
    dt   = d.delta_T
    ov_k = (d.n_day + 1):d.n_int               # overnight intervals (18:00 -> 08:00)
    nov  = length(ov_k)
    P_ov = zeros(length(d.M), nov)             # per-MCS overnight charge power (kW)
    soe_path = [fill(float(soe_mcs_end[m]), nov + 1) for m in d.M]

    for m in d.M
        eta  = d.eta_ch_dch[m]
        rate = d.CH_MCS[m]
        deficit = d.SOE_MCS_ini[m] - soe_mcs_end[m]    # SOE to restore (energy-neutral)
        if deficit > 1e-9
            # cheapest-first: fill the lowest-price overnight slots at full rate.
            order = sort(collect(1:nov); by = j -> d.lambda_whl_elec[ov_k[j]])
            remaining = deficit
            for j in order
                remaining <= 1e-9 && break
                gain = min(eta * rate * dt, remaining)   # SOE gained this interval
                P_ov[m, j] = gain / (eta * dt)           # grid power needed for that gain
                remaining -= gain
            end
        end
        soe = float(soe_mcs_end[m])
        for j in 1:nov
            soe += eta * P_ov[m, j] * dt
            soe_path[m][j + 1] = soe
        end
    end

    df = DataFrame(k = collect(ov_k),
                   clock = [clock_label(d, k) for k in ov_k],
                   price = [d.lambda_whl_elec[k] for k in ov_k])
    for m in d.M
        df[!, Symbol("MCS$(m)_charge_kW")] = P_ov[m, :]
        df[!, Symbol("MCS$(m)_soe_kWh")]   = soe_path[m][2:end]
        df[!, Symbol("MCS$(m)_charging")]  = [P_ov[m, j] > 1e-6 ? "Yes" : "No" for j in 1:nov]
    end
    return df, P_ov, ov_k
end

function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,  # LEGACY: ignored (the horizon is the cross-day window set by lookahead_days)
                          time_limit_sec::Float64 = 60.0,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          refit_every::Int = 8, mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(@__DIR__), "output", String(mode)),
                          soft_prec::Bool = false, soft_pace::Bool = false,
                          soft_term::Bool = false,
                          term_tol::Float64 = 0.1,
                          n_days::Union{Nothing, Int} = nothing,
                          lookahead_days::Int = 1,
                          seed::Int = 1)
    # Reproducibility: seed the GLOBAL RNG (used by Turing/NUTS) so repeated runs give
    # identical KPIs/trajectories. The local telematics-noise RNG below is seeded too.
    Random.seed!(seed)
    # Path fallback: default input dataset is <root>/data/input_data; if it is absent, try
    # a couple of legacy layouts so :input mode still works if the data lives elsewhere.
    if mode == :input && !isdir(input_dir)
        for alt in (joinpath(@__DIR__, "input_data"),
                    joinpath(dirname(@__DIR__), "input_data"))
            if isdir(alt); input_dir = alt; break; end
        end
    end
    d = load_data(mode; input_dir = input_dir)         # :synthetic or :input
    K_all = collect(d.K)
    nK = length(K_all)

    # =========================================================================
    # RECEDING-HORIZON MULTI-DAY SETUP
    # =========================================================================
    # n_days_keep  = number of days we KEEP in the reported results.
    # D_total      = n_days_keep + 1: we actually SIMULATE one extra "buffer" day and
    #                DROP it from every output/KPI. That extra 24 h gives the last kept
    #                day a full day of lookahead, so its schedule is not distorted by the
    #                artificial end-of-horizon "return to start" wrap-up (which now lands
    #                on the discarded buffer day instead).
    # State FLOWS across days: CEV battery SOE and any unfinished work carry from one day
    # into the next. The daily "CEV back to start SOE by 18:00" rule is enforced ONLY on
    # the buffer day (enforce_cev_terminal below). The MCS is recharged overnight each
    # night (Phase 2) so it starts every day ready to work.
    n_days_keep = n_days === nothing ? d.n_days : max(1, n_days)
    D_total     = n_days_keep + 1

    # ---- realized physical state (CARRIED across days) ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    # Per-day work quota that ARRIVES each morning; whatever is left unfinished stays in
    # rem_* and carries into the next day (a soft, penalised carry via the MILP s_miss).
    daily_dig  = copy(float.(d.hours_digging))
    daily_load = copy(float.(d.hours_loading_swinging))
    rem_dig    = zeros(length(d.hours_digging))
    rem_load   = zeros(length(d.hours_loading_swinging))

    # ---- ONLINE Bayesian estimator seeded with the offline TruncatedNormal prior ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)                       # for simulated telematics noise

    # ---- closed-loop logs (a `day` column tags which simulated day each row is) ----
    log = DataFrame(
        day = Int[], gstep = Int[],
        k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[],
    )

    # ---- simple WORKER-FACING schedule (front-end CSV) ----
    fe_time = String[]
    fe_act  = [String[] for _ in d.E]    # per-CEV planned activity
    fe_chg  = [String[] for _ in d.E]    # per-CEV "plug in to charge?" Yes/No
    fe_mcs  = String[]                   # MCS "charge from grid?" Yes/No

    # ---- per-day overnight schedules + replanning grids (kept days written out) ----
    overnight_by_day = Dict{Int, DataFrame}()
    replan_by_day    = Dict{Int, NamedTuple}()

    G = D_total * nK                  # total daytime intervals across the whole horizon
    println("Running Scenario 1 (RECEDING horizon, closed-loop MPC, 15-min steps, CROSS-DAY lookahead):")
    println("  keeping $n_days_keep day(s); simulating $D_total (last = dropped buffer day); $nK steps/day")
    println("  window spans current + $lookahead_days lookahead day(s) of daytime blocks; nights via MCS overnight recharge + CEV carry-over")
    println("  prior power estimate : ", round.(est.mu, digits = 2), " kW")
    println("  (hidden) true power  : ", d.true_powers, " kW")
    t0 = time()
    n_obs_total  = 0
    n_infeasible = 0                  # windows infeasible under the HARD constraints (state held)
    gstep        = 0                  # global 15-min step counter across all days (for plots)
    missed_kept  = 0.0                # unfinished work (hours) at the end of the last KEPT day

    # =========================================================================
    # OUTER LOOP OVER DAYS. Each interval is solved over a CROSS-DAY window that spans
    # the remainder of the current day PLUS `lookahead_days` full future daytime blocks
    # (capped at the buffer day). Global interval index g0 = (day-1)*nK + k0 runs
    # 1..G; the MILP indexes prices / work / nights by the within-day clock internally.
    # =========================================================================
    for day in 1:D_total
        # ---- start-of-day resets ----
        # New work quota arrives; leftover from prior days is already in rem_* (carried).
        rem_dig  .+= daily_dig
        rem_load .+= daily_load
        # Precedence / travel-pacing counters restart each day (they describe a within-day
        # cadence), as do the daily demand-charge peak trackers.
        cum_dig_e  = zeros(length(d.E))
        cum_load_e = zeros(length(d.E))
        cum_trv_e  = zeros(length(d.E))
        peak_nc = 0.0; peak_op = 0.0
        mcs_node = [first(d.N_g) for _ in d.M]         # MCS starts the day parked at grid
        mcs_transit = Any[nothing for _ in d.M]

        # per-day replanning grids (within-day nK x nK, recording only the current day's
        # slice of each cross-day forward plan)
        plan_grid_kW = fill(NaN, nK, nK)
        plan_mcs_soe = fill(NaN, nK, nK)
        plan_cev_soe = [fill(NaN, nK, nK) for _ in d.E]
        plan_cev_act = [fill("", nK, nK)  for _ in d.E]

        day_off = (day - 1) * nK                        # global offset of this day's block

        for k0 in 1:nK
            gstep += 1
            g0    = day_off + k0                         # GLOBAL interval index of this step
            clk   = clock_day_label(d, day, k0)
            # cross-day window: rest of today + `lookahead_days` future daytime blocks,
            # never past the buffer day. Guarantees the schedule always "sees" tomorrow.
            view_end_day = min(D_total, day + lookahead_days)
            Kend  = view_end_day * nK
            K_win = g0:Kend
            is_glob_term = (Kend == G)                   # window reaches the true horizon end

            # Box 3: solve the window MILP using the CURRENT estimated power profile.
            model = build_window_model(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                       rem_dig, rem_load, cum_dig_e, cum_load_e, cum_trv_e,
                                       peak_nc, peak_op, est.mu;
                                       daily_dig = daily_dig, daily_load = daily_load,
                                       require_site_visit = require_site_visit,
                                       single_visit_per_site = single_visit_per_site,
                                       time_limit_sec = time_limit_sec,
                                       soft_prec = soft_prec, soft_pace = soft_pace,
                                       soft_term = soft_term, term_tol = term_tol,
                                       enforce_cev_terminal = true,
                                       is_global_terminal = is_glob_term)

            cur_node = mcs_node[1]
            if !has_values(model)
                n_infeasible += 1
                @warn "No feasible solution at day=$day k=$k0 under HARD constraints; holding state." status=termination_status(model)
                push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                            0.0, 0.0, 0.0, soe_mcs[1], _cev(soe_cev, 1), _cev(soe_cev, 2), cur_node,
                            est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                            est.sd[1], est.sd[2], est.sd[3], est.sd[4],
                            n_obs_total))
                push!(fe_time, clk)
                for e in d.E
                    push!(fe_act[e], "Idle"); push!(fe_chg[e], "No")
                end
                push!(fe_mcs, "No")
                continue
            end

            # ---- apply ONLY the first interval's decisions (global index g0) ----
            grid_kW = sum(value(model[:P_ch_tot][m, g0]) for m in d.M)
            dch_kW  = sum(value(model[:P_dch_tot][m, g0]) for m in d.M)
            cur_node = let nh = findfirst(i -> value(model[:z][1, i, g0]) > 0.5, d.N)
                nh === nothing ? 0 : nh                  # 0 = MCS in transit during g0
            end

            # ---- record the CURRENT-DAY slice of this window's forward plan ----
            for k in K_win
                div(k - 1, nK) + 1 == day || continue    # keep only today's intervals
                kl = k - day_off                          # within-day column index
                plan_grid_kW[k0, kl] = sum(value(model[:P_ch_tot][m, k]) for m in d.M)
                plan_mcs_soe[k0, kl] = value(model[:SOE_MCS][1, k + 1])
                for e in d.E
                    plan_cev_soe[e][k0, kl] = value(model[:SOE_CEV][e, k + 1])
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    if site !== nothing
                        vals = [value(model[:u][e, site, a, k]) for a in eachindex(d.B)]
                        plan_cev_act[e][k0, kl] = sum(vals) < 0.5 ? "" : _ACT_NAME[d.B[argmax(vals)]]
                    end
                end
            end

            # ---- worker-facing front-end row for the APPLIED interval ----
            push!(fe_time, clk)
            for e in d.E
                push!(fe_act[e], _planned_activity(model, d, e, g0))
                push!(fe_chg[e], _cev_should_charge(model, d, e, g0))
            end
            push!(fe_mcs, _mcs_should_charge(model, d, g0))

            # Realized within-interval activity breakdown per CEV ("to the dot").
            a_real = Dict(e => realized_activity_durations(rng, model, e, g0, d;
                                                           multi = multi_activity) for e in d.E)

            # ---- FEEDBACK STEP: each realized row improves the Bayesian estimate ----
            for e in d.E
                row = a_real[e]
                if sum(row) > 1e-9
                    b_obs = dot(row, d.true_powers) + d.obs_noise_std * randn(rng)
                    observe!(est, row, b_obs)
                    n_obs_total += 1
                end
            end
            if n_obs_total > 0 && gstep % refit_every == 0
                refit!(est)                              # NUTS re-fit on all data so far
            end

            # ---- advance realized MCS energy (flow) + position ----
            # Advance the MCS battery by the APPLIED interval's realized flows (NOT by
            # reading SOE_MCS[g0+1], which the model resets to full at a night boundary).
            for m in d.M
                ch  = value(model[:P_ch_tot][m, g0])
                dch = value(model[:P_dch_tot][m, g0])
                ltr = value(model[:L_trv_tot][m, g0])
                soe_mcs[m] = soe_mcs[m] + d.eta_ch_dch[m] * ch * d.delta_T -
                             (dch * d.delta_T) / d.eta_ch_dch[m] - ltr
                if k0 == nK
                    mcs_node[m] = first(d.N_g); mcs_transit[m] = nothing   # parked at grid overnight
                else
                    mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, g0, Kend, d)
                end
            end

            # ---- realized CEV energy (TRUE powers over the realized mix) ----
            for e in d.E
                charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T
                work_true = dot(a_real[e], d.true_powers)
                soe_cev[e] = clamp(soe_cev[e] + charged - work_true, d.SOE_CEV_min[e], d.SOE_CEV_max[e])
            end

            # ---- work-completion accounting uses REALIZED durations (per CEV) ----
            for e in d.E
                site_e = findfirst(i -> d.A[i, e] == 1, d.N)
                if site_e !== nothing
                    rem_dig[site_e]  = max(rem_dig[site_e]  - a_real[e][1], 0.0)
                    rem_load[site_e] = max(rem_load[site_e] - a_real[e][2], 0.0)
                end
                cum_dig_e[e]  += a_real[e][1]
                cum_load_e[e] += a_real[e][2]
                cum_trv_e[e]  += a_real[e][3]
            end

            # ---- daily-peak carry-over (so demand charges reflect the whole day) ----
            peak_nc = max(peak_nc, grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, grid_kW))

            work_kW = sum(dot(a_real[e], d.true_powers) for e in d.E) / d.delta_T

            push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        grid_kW, dch_kW, work_kW,
                        soe_mcs[1], _cev(soe_cev, 1), _cev(soe_cev, 2), cur_node,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4],
                        n_obs_total))
        end

        # Snapshot unfinished work at the end of the LAST kept day (before the buffer day
        # gets a chance to mop it up) -- this is the "missed work" we report.
        day == n_days_keep && (missed_kept = sum(rem_dig) + sum(rem_load))

        # ---- end-of-day: overnight smart-charge, then MCS starts next day recharged ----
        ov_df, _, _ = phase2_overnight_charge(d, soe_mcs)
        overnight_by_day[day] = ov_df
        soe_mcs = copy(float.(d.SOE_MCS_ini))          # MCS restored overnight (ready for next day)

        replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act)
    end

    n_obs_total > 0 && refit!(est)          # final Bayesian fit on all data
    elapsed = time() - t0
    @printf("MPC loop done in %.1f s (%d telematics observations, %d simulated days)\n",
            elapsed, n_obs_total, D_total)
    n_infeasible > 0 && @printf("  NOTE: %d windows were INFEASIBLE under the HARD constraints (no fallback);\n        the plant HELD state (no work / no charging) for those intervals.\n", n_infeasible)
    println("  final power estimate : ", round.(est.mu, digits = 2), " kW")
    println("  (hidden) true power  : ", d.true_powers, " kW")

    # =========================================================================
    # DROP THE BUFFER DAY: everything reported below covers ONLY days 1..n_days_keep.
    # =========================================================================
    keep_row = log.day .<= n_days_keep
    klog = log[keep_row, :]
    n_kept_steps = count(keep_row)

    total_energy = sum(klog.grid_kW) * d.delta_T
    total_cost   = sum(klog.grid_kW .* klog.price) * d.delta_T
    total_co2    = sum(klog.grid_kW .* klog.co2)  * d.delta_T
    nc_peak      = isempty(klog.grid_kW) ? 0.0 : maximum(klog.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in klog.k]
    op_peak      = any(op_mask) ? maximum(klog.grid_kW[op_mask]) : 0.0
    missed       = missed_kept                    # unfinished work at end of last kept day
    transit_intervals = count(==(0), klog.mcs_node)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals

    # overnight recharge cost/energy over KEPT days only
    overnight_energy = 0.0; overnight_cost = 0.0
    for day in 1:n_days_keep
        ov = overnight_by_day[day]
        for m in d.M
            col = ov[!, Symbol("MCS$(m)_charge_kW")]
            overnight_energy += sum(col) * d.delta_T
            overnight_cost   += sum(col .* ov.price) * d.delta_T
        end
    end

    println("\n==== Scenario 1 RECEDING-horizon KPIs (kept days 1..$n_days_keep; buffer day dropped) ====")
    @printf("Total grid energy   : %.2f kWh\n", total_energy)
    @printf("Total energy cost   : \$%.2f\n", total_cost)
    total_co2 > 1e-9 && @printf("Total CO2 emissions : %.2f kg\n", total_co2)
    @printf("NC peak demand      : %.2f kW\n", nc_peak)
    @printf("On-peak demand      : %.2f kW\n", op_peak)
    @printf("Missed work (hours) : %.2f\n", missed)
    @printf("Labour (towing)     : \$%.2f  (%.2f h in transit @ \$%.2f/h)\n",
            labour_cost, transit_intervals * d.delta_T, d.rho_labor)
    @printf("Overnight recharge  : %.2f kWh grid  ->  \$%.2f (cheapest hours, kept days)\n",
            overnight_energy, overnight_cost)

    # ---- export (KEPT days only) ----
    mkpath(out_dir)
    # (a) detailed ANALYST trajectory (buffer day dropped).
    CSV.write(joinpath(out_dir, "closed_loop_trajectory.csv"), klog)
    # (a2) overnight MCS smart-charge schedule per kept day.
    for day in 1:n_days_keep
        CSV.write(joinpath(out_dir, "overnight_mcs_charge_day$(day).csv"), overnight_by_day[day])
    end
    # (b) simple WORKER schedule (kept days only).
    fe = DataFrame(time = fe_time[1:n_kept_steps])
    for e in d.E
        fe[!, Symbol("CEV$(e)_activity")]       = fe_act[e][1:n_kept_steps]
        fe[!, Symbol("CEV$(e)_plug_in_charge")] = fe_chg[e][1:n_kept_steps]
    end
    fe[!, :MCS_charge_from_grid] = fe_mcs[1:n_kept_steps]
    CSV.write(joinpath(out_dir, "worker_schedule.csv"), fe)
    # (c) REPLANNING GRIDS per kept day (one subfolder each).
    for day in 1:n_days_keep
        g = replan_by_day[day]
        gdir = joinpath(out_dir, "replan_grids", "day$(day)"); mkpath(gdir)
        write_replan_grid(joinpath(gdir, "plan_grid_kW.csv"), g.plan_grid_kW, d, nK)
        write_replan_grid(joinpath(gdir, "plan_mcs_soe.csv"), g.plan_mcs_soe, d, nK)
        for e in d.E
            write_replan_grid(joinpath(gdir, "plan_cev$(e)_soe.csv"),      g.plan_cev_soe[e], d, nK)
            write_replan_grid(joinpath(gdir, "plan_cev$(e)_activity.csv"), g.plan_cev_act[e], d, nK)
        end
    end
    make_plots(d, klog, out_dir)
    println("\nResults written to: $(abspath(out_dir))")
    println("  - worker_schedule.csv         (simple, for site workers; kept days)")
    println("  - closed_loop_trajectory.csv  (detailed; kept days, buffer dropped)")
    println("  - overnight_mcs_charge_day*.csv (Phase 2 overnight per kept day)")
    println("  - replan_grids/day*/*.csv     (per-step forward plans + replanning)")
    return klog
end

# safe CEV accessor for logging (datasets may have a different fleet size)
_cev(v, i) = i <= length(v) ? v[i] : NaN

# ---- replanning-grid cell formatting + writer -------------------------------
_cell(v::AbstractString) = v
_cell(v::Real) = isnan(v) ? "" : round(v, digits = 3)

# Write a replanning grid to CSV. Row label `replan_at` = clock at the re-plan step k0;
# one column per interval (labelled by its clock). For row k0 (the 15-min re-plan at that
# clock):
#   * columns k <  k0 (PAST): the decision already APPLIED at interval k -- i.e. the
#     diagonal value mat[k, k] -- now FIXED and no longer re-planned.
#   * column  k == k0 (diagonal): the decision applied to the plant this step.
#   * columns k >  k0 (FUTURE): the fresh forward plan made at step k0 for interval k.
# So each row reads left-to-right as "already-fixed past  +  newly re-planned future"
# (e.g. at 08:15 the 08:00 slot is fixed and 08:15 onward is re-planned; at 08:30 the
# 08:00 and 08:15 slots are fixed; and so on). Reading DOWN a column shows how that one
# interval's plan was revised each step until it became fixed on the diagonal.
function write_replan_grid(path, mat, d, nK)
    df = DataFrame(replan_at = [clock_label(d, k0) for k0 in 1:nK])
    for k in 1:nK
        df[!, Symbol(clock_label(d, k))] =
            Any[_cell(k < k0 ? mat[k, k] : mat[k0, k]) for k0 in 1:nK]
    end
    CSV.write(path, df)
    write_replan_grid_html(replace(path, r"\.csv$" => ".html"), mat, d, nK)
end

# Colored companion view of a replanning grid (open in any browser). Each cell is shaded
#   GREEN  = complete (a PAST interval whose decision is already fixed, k < k0), and
#   YELLOW = pending (the current/applied step k == k0 and the forward plan k > k0).
# Blank cells (an infeasible step's un-planned future) are left uncolored.
function write_replan_grid_html(path, mat, d, nK)
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif}")
    println(io, "table{border-collapse:collapse;font-size:11px}")
    println(io, "th,td{border:1px solid #ccc;padding:2px 6px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".done{background:#c6efce}")   # green  = complete (fixed)
    println(io, ".pend{background:#ffeb9c}")   # yellow = pending (planned)
    println(io, "</style></head><body>")
    println(io, "<p><b>How to read this grid.</b> Every cell is a <i>PLANNED</i> value (nothing here is ",
                "\"actual\").<br>",
                "&nbsp;&nbsp;\u2022 <b>Each ROW</b> = one 15-min re-plan step, labelled by the clock time the plan was made at.<br>",
                "&nbsp;&nbsp;\u2022 <b>Each COLUMN</b> = the interval that is being planned <i>for</i>, labelled by its clock time.<br>",
                "&nbsp;&nbsp;\u2022 <b>A cell</b> = what the optimiser planned for that column's interval, as decided at that row's re-plan step.<br>",
                "&nbsp;&nbsp;\u2022 The <b>diagonal</b> (row time == column time) is the decision actually applied to the plant that step.</p>")
    println(io, "<p><b>Colour:</b> <span class=\"done\">&nbsp;&nbsp;&nbsp;</span> complete (a past interval, already fixed) &nbsp;&nbsp; ",
                "<span class=\"pend\">&nbsp;&nbsp;&nbsp;</span> pending (the current step + the forward plan)</p>")
    println(io, "<table><tr><th>re-plan made at &darr; &nbsp;\\&nbsp; interval &rarr;</th>")
    for k in 1:nK
        print(io, "<th>", clock_label(d, k), "</th>")
    end
    println(io, "</tr>")
    for k0 in 1:nK
        print(io, "<tr><th>", clock_label(d, k0), "</th>")
        for k in 1:nK
            cell = _cell(k < k0 ? mat[k, k] : mat[k0, k])
            cls  = cell == "" ? "" : (k < k0 ? "done" : "pend")
            print(io, "<td class=\"", cls, "\">", cell, "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# clock label for interval k0 (start boundary)
function clock_label(d, k0)
    m = mod(Int(round(d.t_start * 60 + (k0 - 1) * d.delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

# day-tagged clock label for the multi-day run, e.g. "D2 08:15".
clock_day_label(d, day, k0) = string("D", day, " ", clock_label(d, k0))

# =============================================================================
# 4. PLOTTING
# =============================================================================
function make_plots(d, log, out_dir)
    # Global step index (k repeats each day; gstep is continuous across the kept days).
    x = (:gstep in propertynames(log)) ? log.gstep : log.k
    # grid draw + price overlay
    p1 = plot(x, log.grid_kW, label = "Grid charging (kW)", lw = 2, color = :steelblue,
              xlabel = "Interval", ylabel = "Power (kW)", title = "Scenario 1: closed-loop grid draw")
    plot!(twinx(), x, log.price, label = "Price (\$/kWh)", lw = 2, color = :red, ylabel = "Price (\$/kWh)")
    savefig(p1, joinpath(out_dir, "01_grid_draw_vs_price.png"))

    # SOE trajectories
    p2 = plot(x, log.soe_mcs, label = "MCS SOE", lw = 2,
              xlabel = "Interval", ylabel = "SOE (kWh)", title = "Scenario 1: state of energy")
    plot!(p2, x, log.soe_cev1, label = "CEV 1 SOE", lw = 2)
    plot!(p2, x, log.soe_cev2, label = "CEV 2 SOE", lw = 2)
    savefig(p2, joinpath(out_dir, "02_state_of_energy.png"))

    # work power
    p3 = plot(x, log.work_kW, label = "Total work (kW)", lw = 2, color = :forestgreen,
              xlabel = "Interval", ylabel = "Power (kW)", title = "Scenario 1: CEV work power")
    savefig(p3, joinpath(out_dir, "03_work_power.png"))

    # online power-estimate convergence (the feedback loop in action)
    p4 = plot(xlabel = "Interval (15 min each)", ylabel = "Estimated power (kW)",
              title = "Scenario 1: online power estimate -> truth", legend = :right)
    names_ = ["Digging", "Loading/Swinging", "Traveling", "Idling"]
    ests   = [log.est_dig, log.est_load, log.est_trv, log.est_idle]
    uncs   = [log.unc_dig, log.unc_load, log.unc_trv, log.unc_idle]
    cols   = [:steelblue, :darkorange, :purple, :seagreen]
    for j in 1:4
        plot!(p4, x, ests[j], ribbon = uncs[j], lw = 2, color = cols[j], label = names_[j] * " est.")
        hline!(p4, [d.true_powers[j]], lw = 1.5, ls = :dash, color = cols[j],
               label = names_[j] * " true")
    end
    savefig(p4, joinpath(out_dir, "04_power_estimate_convergence.png"))
end

# =============================================================================
# 5. ENTRY POINT
# =============================================================================
# Run on launch so it "just works" whether started from the command line
# (`julia Scenario_1.jl`) or via an editor's Run button (e.g. the VS Code Julia
# extension, which `include`s the file in the REPL where PROGRAM_FILE is empty).
# A test harness can `include` this file WITHOUT triggering the full run by
# defining `SCENARIO1_NO_AUTORUN = true` before the include.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
