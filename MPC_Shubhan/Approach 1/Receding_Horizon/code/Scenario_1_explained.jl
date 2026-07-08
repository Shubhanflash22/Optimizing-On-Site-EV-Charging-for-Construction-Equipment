# #############################################################################

# =============================================================================
# WHAT THIS PROGRAM DOES (the 30-second version)
# =============================================================================
# We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
# fleet of electric excavators (Construction EVs, "CEVs"). Over a work day the
# MCS must drive around and top the excavators up so none of them ever runs flat,
# while paying the least possible for electricity (time-of-use price + demand
# charges + carbon), and getting all the digging/loading work done.
#
# The catch: we do NOT know exactly how much power each excavator uses when it
# digs vs. loads vs. drives. We only have a noisy guess. So we do two things at
# once, every 15 minutes:
#   (1) OPTIMISE: solve a mixed-integer linear program (MILP) for the best plan
#       across a CROSS-DAY window (the rest of today + one or more future days),
#       given our CURRENT best power guess.
#   (2) LEARN: watch the energy actually consumed in the last 15 min and update
#       (Bayesian regression) our guess of the per-activity power draws.
# Apply only the FIRST 15-min action, then repeat. This is Model Predictive
# Control (MPC) with an online-learned model — "certainty-equivalent" because
# the optimiser plans on the single best (mean) guess, not the full uncertainty.

# --- Packages we rely on ------------------------------------------------------
using JuMP              # modelling language for the optimisation (variables/constraints/objective)
using HiGHS             # the actual MILP solver that JuMP hands the problem to
using Plots             # to draw the result figures (PNG files)
using DataFrames        # tabular data (used to build the CSV outputs)
using CSV               # read/write CSV files
using Printf            # @printf / @sprintf for nicely formatted numbers
using Dates             # time parsing helpers (for the input-CSV clock column)
using LinearAlgebra     # dot(...) for vector dot products
using Random            # RNG for reproducible simulated telemetry noise
using Statistics        # mean / std to summarise the Bayesian posterior
using Turing            # the Bayesian engine: priors + likelihood + NUTS sampler

gr()                        # pick the GR backend for Plots (fast, file-friendly)
Turing.setprogress!(false)  # silence Turing's sampling progress bar

# =============================================================================
# 1. DATA  (the built-in "synthetic" scenario)
# =============================================================================
# `build_default_data` hard-codes a small, self-contained example problem and
# returns it as a NamedTuple (a bag of named fields). Everything the optimiser
# needs — fleet sizes, battery limits, prices, work to be done — lives in here.
# (Section 1a below shows how to load the SAME fields from real CSV files.)

function build_default_data()
    # ---- time discretization ----
    delta_T = 0.25                 # length of one time step, in hours (0.25 h = 15 min)
    n_int   = 96                   # number of steps in a full 24 h day (96 * 15 min = 24 h)
    t_start = 8                    # the clock hour the horizon begins at (08:00)
    # The daytime optimisation (Phase 1) only covers 08:00 -> day_end_hour (18:00).
    work_start_hour = 8; work_end_hour = 17        # productive shift runs 08:00–17:00 ...
    lunch_start_hour = 12; lunch_end_hour = 14     # ... but with a 12:00–14:00 lunch (no work; may charge)
    day_end_hour = 18                              # by 18:00 CEVs must be "full" and MCS parked home
    t_limit_rest = 1.0             # rest rule (Eq. 12e): <=1 h of work per rolling (1 h + 15 min) window
    kappa_wt = 4                   # travel-pacing knob: ~1 travel step allowed per 4 productive steps
    n_day   = Int(round((day_end_hour - t_start) / delta_T))   # number of daytime steps = (18-8)/0.25 = 40
    K       = 1:n_day              # index set of the 40 daytime intervals
    T       = 1:(n_day + 1)        # boundary indices (41 points bracket the 40 intervals)

    # ---- nodes: node 1 = grid connection, nodes 2.. = construction sites ----
    N   = 1:3                      # three physical locations in total
    N_g = [1]                      # grid node(s): where the MCS can plug into the grid
    N_c = collect(2:length(N))     # construction sites: nodes 2 and 3

    # ---- fleet sizes ----
    M = 1:1                        # one MCS (the charging truck)
    E = 1:2                        # two CEVs (excavators)

    # Assignment matrix A[node, ev] = 1 if that excavator works at that site.
    A = zeros(Int, length(N), length(E))   # start all-zeros
    A[2, 1] = 1                    # excavator 1 is stationed at site 2
    A[3, 2] = 1                    # excavator 2 is stationed at site 3
    # (row 1 = grid stays all zeros: no excavator "works" at the grid node)

    # ---- MCS (charging-truck) parameters ----
    SOE_MCS_ini = [250.0]; SOE_MCS_max = [250.0]; SOE_MCS_min = [50.0]   # start full (250), capacity 250, 20% floor (50)
    CH_MCS  = [150.0]              # max rate the MCS can draw FROM the grid (kW)
    DCH_MCS = [150.0]             # max total rate the MCS can push OUT to sites (kW)
    DCH_MCS_plug = [60.0]         # max rate through ONE plug to a single CEV (kW)
    C_MCS_plug   = [2]            # how many CEVs the MCS can charge at once (plugs)
    eta_ch_dch   = [0.95]         # charge/discharge efficiency factor (~5% loss)

    # ---- CEV (excavator) parameters ----
    # Two DIFFERENT machines. Note SOE_ini < SOE_max (they start at 80%), which
    # leaves headroom so "end the day back at the start level" never collides with
    # the max-capacity limit. The batteries are also big vs. the day's work, so
    # each machine can finish its shift and be topped back up before 18:00.
    SOE_CEV_max = [90.0, 60.0]     # capacities: excavator 1 is the bigger machine
    SOE_CEV_ini = [72.0, 48.0]     # start-of-day energy (= 80% of max = the end-of-day target)
    SOE_CEV_min = [18.0, 12.0]     # 20% reserve floor for each machine
    CH_CEV      = [45.0, 30.0]     # max rate each excavator can ACCEPT while charging (kW)

    # ---- activity power draws, kW ----
    # There are exactly FOUR activities, indexed by B = [1,2,3,4]:
    #   1 = digging, 2 = loading/swinging, 3 = traveling, 4 = idling.
    # A machine is ALWAYS doing exactly one of these each interval. "Idling" is
    # the catch-all: it is what happens during charging, lunch, or any gap, and
    # it draws only a little power.
    #
    #   prior_mu / prior_sigma : our OFFLINE best guess of each activity's power
    #     (mean + uncertainty). This SEEDS the online learner and is what step 1
    #     of the MPC uses before any live telemetry has arrived.
    #   true_powers : the HIDDEN real values the excavators actually draw. The
    #     learner never sees these; we only use them to SIMULATE the measured
    #     energy each 15 min. On real hardware this comes from the machines.
    #   obs_noise_std : measurement noise on that simulated energy reading (kWh).
    prior_mu      = [4.6, 3.3, 4.5, 0.5]   # guessed mean power for [dig, load, travel, idle]
    prior_sigma   = [1.0, 1.0, 1.5, 0.3]   # how unsure we are about each guess
    true_powers   = [5.2, 2.8, 5.0, 0.6]   # (pretend) ground truth used only to simulate readings
    obs_noise_std = 0.05                     # noise added to each simulated energy reading

    # Convenience scalars: the day-1 point estimates are just the prior means.
    p_digging          = prior_mu[1]
    p_loading_swinging = prior_mu[2]
    p_traveling        = prior_mu[3]
    p_idling           = prior_mu[4]

    # ---- how much work each site needs (only site rows 2,3 are used) ----
    hours_digging          = [0.0, 2.5, 1.5]   # hours of digging required at [grid, site2, site3]
    hours_loading_swinging = [0.0, 1.5, 1.0]   # hours of loading/swinging required per node

    # ---- travel model ----
    # tau_trv[i,j] = time to drive node i -> j, in INTERVALS. k_trv = kWh burned per drive.
    tau_trv = [0.0 2.0 3.0;        # from grid: 2 steps to site2, 3 steps to site3
               2.0 0.0 2.0;        # from site2
               3.0 2.0 0.0]        # from site3
    k_trv = 2.0                    # energy consumed by the MCS per arc traversal (kWh)

    # ---- exogenous time series over the full 24 h (one value per interval) ----
    # We fill two price/impact curves the optimiser reacts to. This loop walks all
    # 96 intervals, converts each to a clock hour, and pushes a time-of-use
    # electricity price (with a 16:00–21:00 on-peak spike) and a smooth CO2
    # intensity curve into their vectors.
    lambda_whl_elec = Float64[]    # $/kWh electricity price per interval
    lambda_CO2      = Float64[]    # CO2 intensity per interval (kg/kWh-ish)
    for k in 1:n_int
        hour = mod(t_start + (k - 1) * delta_T, 24)                       # clock hour of interval k
        price = (16 <= hour < 21) ? 0.45 : (7 <= hour < 16 ? 0.18 : 0.10) # peak / day / night price
        co2   = 0.30 + 0.15 * sin((hour - 6) / 24 * 2pi)                  # gentle daily CO2 wave
        push!(lambda_whl_elec, price)
        push!(lambda_CO2, max(co2, 0.05))                                 # keep CO2 positive
    end

    # ---- per-CEV work-availability cap R_work[node, ev, interval] ----
    # This 3-D array says "how much work power is allowed for excavator e at site i
    # in interval k". The loop marks an interval as productive only during shift
    # hours EXCLUDING lunch, and only for the (site, excavator) pair that is
    # actually assigned there; a big cap (1000) means "work allowed", 0 means "no".
    R_work = zeros(length(N), length(E), n_day)
    for i in N_c, e in E, k in 1:n_day
        hour = mod(t_start + (k - 1) * delta_T, 24)
        productive = (work_start_hour <= hour < work_end_hour) &&
                     !(lunch_start_hour <= hour < lunch_end_hour)
        R_work[i, e, k] = (productive && A[i, e] == 1) ? 1000.0 : 0.0
    end

    # ---- costs / penalties that appear in the objective ----
    rho_miss             = 50.0    # $ penalty per hour of work left unfinished
    rho_labor            = 30.0    # $ labour per hour the MCS spends driving (towing)
    lambda_demand_NC     = 10.0    # $/kW charge on the day's overall peak grid draw
    lambda_demand_OP     = 25.0    # $/kW extra charge on the on-peak (16–21) peak draw
    carbon_price_per_ton = 50.0    # $ per ton of CO2

    scale = 2                      # precedence factor: loading can't outrun 2x the digging done
    B = [1, 2, 3, 4]               # the activity index set (dig, load, travel, idle)

    # Bundle EVERYTHING into one NamedTuple and return it. Downstream code reads
    # fields as d.delta_T, d.SOE_CEV_ini, etc.
    return (; delta_T, K, T, t_start, n_int, n_day, day_end_hour, t_limit_rest, kappa_wt,
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
# 1a. INPUT-DATA MODE  (load the SAME fields from a folder of 7 CSV files)
# =============================================================================
# Instead of the built-in numbers, we can read a real dataset from disk. The
# folder must contain seven CSVs with fixed names/columns (parameters, ev_data,
# mcs_data, place, time_data, travel_time, work_flexible). Any missing file or
# column raises a clear error rather than silently using a wrong default.

# Return the full path to `name` inside `dir`, or error if it is missing.
_require_file(dir, name) = (p = joinpath(dir, name);
    isfile(p) ? p : error("Scenario_1 input mode: required file missing -> $p"))

# Read one CSV into a DataFrame and check that every column in `required_cols`
# is present (this loop is just the validation check over the required names).
function _read_csv(dir, name; required_cols = String[])
    df = CSV.read(_require_file(dir, name), DataFrame)
    for c in required_cols
        Symbol(c) in propertynames(df) ||
            error("Scenario_1 input mode: '$name' is missing required column '$c'")
    end
    return df
end

# Turn a clock string like "8:15:00" into decimal hours (8.25). Splits on ":"
# and adds hours + minutes/60. Uses `string` so it also accepts a parsed Time.
_clock_hours(s) = (parts = split(strip(string(s)), ":");
    parse(Int, parts[1]) + (length(parts) >= 2 ? parse(Int, parts[2]) : 0) / 60)

# Look up a REQUIRED scalar in parameters.csv by its Parameter name.
function _psd(par, key)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))   # find the row
    idx === nothing && error("Scenario_1 input mode: parameter '$key' missing in parameters.csv")
    return Float64(par.Value[idx])                                     # return its Value
end
# Same, but OPTIONAL: return `default` if the parameter row is absent.
function _psd_opt(par, key, default)
    idx = findfirst(==(String(key)), strip.(string.(par.Parameter)))
    return idx === nothing ? default : Float64(par.Value[idx])
end

# Build the data NamedTuple from the 7 CSVs in `input_dir`. This mirrors
# build_default_data() field-for-field, just sourced from files.
function load_input_data(input_dir::AbstractString)
    isdir(input_dir) || error("Scenario_1 input mode: input directory not found -> $input_dir")

    # Read each file and assert its required columns exist.
    par = _read_csv(input_dir, "parameters.csv"; required_cols = ["Parameter", "Value"])
    evd = _read_csv(input_dir, "ev_data.csv";   required_cols = ["SOE_min","SOE_max","SOE_ini","ch_rate"])
    mcd = _read_csv(input_dir, "mcs_data.csv";  required_cols =
            ["SOE_min","SOE_max","SOE_ini","CH_MCS","DCH_MCS","C_MCS_plug","DCH_MCS_plug","eta_ch_dch"])
    plc = _read_csv(input_dir, "place.csv";     required_cols = ["site","hours_digging","hours_loading_swinging"])
    tdd = _read_csv(input_dir, "time_data.csv"; required_cols = ["lambda_buy","intensity_tons_emissions"])
    ttm = _read_csv(input_dir, "travel_time.csv")
    wkf = _read_csv(input_dir, "work_flexible.csv"; required_cols = ["Location","EV"])

    # ---- scalar settings from parameters.csv (optional ones fall back to defaults) ----
    delta_T = _psd(par, "delta_T");  k_trv = _psd(par, "k_trv")
    rho_miss         = _psd(par, "rho_miss");          rho_labor = _psd(par, "rho_labor")
    lambda_demand_NC = _psd(par, "lambda_demand_NC");  lambda_demand_OP = _psd(par, "lambda_demand_OP")
    carbon_price_per_ton = _psd_opt(par, "carbon_price_per_ton", 0.0)
    p_idling     = _psd_opt(par, "p_idling", 0.0)
    scale        = Int(round(_psd_opt(par, "scale", 2.0)))
    t_limit_rest = _psd_opt(par, "t_limit_rest", 1.0)
    kappa_wt     = Int(round(_psd_opt(par, "kappa_wt", 4.0)))
    day_end_hour = _psd_opt(par, "day_end_hour", 18.0)
    prior_sigma_frac = _psd_opt(par, "prior_sigma_frac", 0.2)   # prior std as a fraction of the mean
    obs_noise_std    = _psd_opt(par, "obs_noise_std", 0.05)
    co2_unit_scale   = _psd_opt(par, "co2_unit_scale", 1.0)

    # ---- time series + horizon derived from time_data.csv ----
    n_int   = nrow(tdd)                                # number of intervals = rows in time_data
    t_start = _clock_hours(tdd[1, 1]) - delta_T        # start clock = first END time minus one step
    lambda_whl_elec = Float64.(tdd.lambda_buy)         # price curve
    lambda_CO2      = Float64.(tdd.intensity_tons_emissions) .* co2_unit_scale  # carbon curve
    n_day = Int(round((day_end_hour - t_start) / delta_T))   # daytime interval count
    K = 1:n_day;  T = 1:(n_day + 1)

    # ---- build id -> index lookup maps (case-insensitive) ----
    ev_ids   = strip.(string.(evd[!, 1]))              # excavator ids (first column)
    mcs_ids  = strip.(string.(mcd[!, 1]))              # MCS ids
    node_ids = strip.(string.(plc.site))               # node/site names
    node_idx = Dict(lowercase(id) => i for (i, id) in enumerate(node_ids))  # name -> node index
    ev_idx   = Dict(lowercase(id) => e for (e, id) in enumerate(ev_ids))    # name -> ev index
    N = 1:length(node_ids);  E = 1:length(ev_ids);  M = 1:length(mcs_ids)

    # ---- assignment matrix A from place.csv (one column per excavator id) ----
    # For each excavator column, this loop marks A[node, ev] = 1 wherever the CSV
    # cell is 1, i.e. which node each excavator is assigned to work at.
    A = zeros(Int, length(N), length(E))
    for (e, eid) in enumerate(ev_ids)
        Symbol(eid) in propertynames(plc) ||
            error("simple dataset: place.csv missing assignment column '$eid'")
        col = plc[!, Symbol(eid)]
        for r in 1:nrow(plc)
            Int(round(Float64(col[r]))) == 1 && (A[node_idx[lowercase(node_ids[r])], e] = 1)
        end
    end
    N_c = [i for i in N if any(A[i, e] == 1 for e in E)]   # sites = any excavator assigned there
    N_g = [i for i in N if !(i in N_c)]                    # grid = every node with no excavator
    isempty(N_g) && error("simple dataset: no grid node (a node with no EV assigned) found")
    isempty(N_c) && error("simple dataset: no site node (a node with an EV assigned) found")

    # ---- MCS / CEV battery parameters (each column becomes a vector) ----
    SOE_MCS_ini = Float64.(mcd.SOE_ini); SOE_MCS_max = Float64.(mcd.SOE_max)
    SOE_MCS_min = Float64.(mcd.SOE_min); CH_MCS = Float64.(mcd.CH_MCS); DCH_MCS = Float64.(mcd.DCH_MCS)
    DCH_MCS_plug = Float64.(mcd.DCH_MCS_plug); C_MCS_plug = Int.(mcd.C_MCS_plug); eta_ch_dch = Float64.(mcd.eta_ch_dch)
    SOE_CEV_ini = Float64.(evd.SOE_ini); SOE_CEV_max = Float64.(evd.SOE_max)
    SOE_CEV_min = Float64.(evd.SOE_min); CH_CEV = Float64.(evd.ch_rate)

    # ---- activity powers: KNOWN constants; the learner's prior is built from them ----
    prior_mu    = [_psd(par, "p_digging"), _psd(par, "p_loading_swinging"), _psd(par, "p_traveling"), p_idling]
    prior_sigma = [max(prior_sigma_frac * prior_mu[j], 0.05) for j in 1:4]  # std = frac*mean (min 0.05)
    true_powers = copy(prior_mu)                        # "truth" == known values in input mode
    p_digging, p_loading_swinging, p_traveling = prior_mu[1], prior_mu[2], prior_mu[3]

    # ---- required work hours per node, read from place.csv ----
    # This loop copies each place-row's digging/loading hour requirement into the
    # per-node vectors, indexed by the node's position.
    hours_digging = zeros(length(N)); hours_loading_swinging = zeros(length(N))
    for r in 1:nrow(plc)
        i = node_idx[lowercase(node_ids[r])]
        hours_digging[i]          = Float64(plc.hours_digging[r])
        hours_loading_swinging[i] = Float64(plc.hours_loading_swinging[r])
    end

    # ---- travel-time matrix from travel_time.csv (values are in INTERVALS) ----
    # This double loop matches each (row-node, column-node) name to its index and
    # fills tau_trv[i,j]; unknown node names are skipped.
    tau_trv = zeros(length(N), length(N))
    tt_rows = lowercase.(strip.(string.(ttm[!, 1])))
    tt_cols = lowercase.(strip.(string.(names(ttm)[2:end])))
    for (ri, rn) in enumerate(tt_rows), (ci, cn) in enumerate(tt_cols)
        (haskey(node_idx, rn) && haskey(node_idx, cn)) || continue
        tau_trv[node_idx[rn], node_idx[cn]] = Float64(ttm[ri, ci + 1])
    end

    # ---- per-interval work cap R_work from work_flexible.csv ----
    # Each row is a (Location, EV) pair followed by one column per interval giving
    # the kW work cap (0 = no work allowed). The nested loop copies those per-
    # interval caps into R_work[node, ev, interval].
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
    # Same NamedTuple shape as build_default_data — the rest of the program can't
    # tell whether the data came from code or from CSVs.
    return (; delta_T, K, T, t_start, n_int, n_day, day_end_hour, t_limit_rest, kappa_wt,
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

# Dispatcher: pick the data source. :synthetic uses the built-in numbers;
# :input reads the 7-CSV folder.
function load_data(mode::Symbol; input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"))
    if mode == :synthetic
        return build_default_data()
    elseif mode == :input
        return load_input_data(input_dir)
    else
        error("Unknown data mode :$mode (use :synthetic or :input)")
    end
end

# Convert travel times (which may be fractional) into WHOLE interval counts.
# Any positive off-diagonal time becomes at least 1 step; the diagonal (i->i) is 0.
function normalize_travel_steps(tau_trv, N)
    n = length(N)
    steps = zeros(Int, n, n)
    for i in N, j in N
        steps[i, j] = i == j ? 0 : max(1, Int(round(tau_trv[i, j])))
    end
    return steps
end

# True if interval k falls inside the 16:00–21:00 on-peak window (used for the
# extra on-peak demand charge). Computes the interval's start/stop clock hours.
function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

# =============================================================================
# 1b. ONLINE POWER ESTIMATOR  (the "learning" half of the loop)
# =============================================================================
# This is the Bayesian regression that turns measured energy into a refined guess
# of each activity's power. The idea:
#
#     energy_used_in_an_interval ≈ (hours_digging)*p_dig + (hours_load)*p_load
#                                  + (hours_travel)*p_trav + (hours_idle)*p_idle
#
# So if we record, each interval, the HOURS spent on each activity (row of A) and
# the MEASURED energy (b), we can regress b on A to recover the powers p. Doing it
# the Bayesian way means we start from a prior belief and get back a full
# posterior (mean + uncertainty), not just a point estimate.
#
# Each observation is a pair (a, b):
#   a : hours spent on each activity this interval, e.g. [0.083, 0, 0.167, 0]
#   b : measured energy that interval (kWh). On real hardware:
#         b = energy_charged_in - (SOC_drop) * battery_capacity

# The Turing probabilistic model. It declares the priors, the noise, and how the
# measured energies `b` are generated from the powers `x`. NUTS then samples the
# posterior of x given the data. We use explicit scalars x1..x4 (one per activity)
# instead of an array because newer Turing versions retrieve named scalars more
# reliably than array elements.
Turing.@model function activity_power_model(A, b, prior_mu, prior_sigma, sigma_b)
    x1 ~ truncated(Normal(prior_mu[1], prior_sigma[1]); lower = 0.0)        # digging power (>=0)
    x2 ~ truncated(Normal(prior_mu[2], prior_sigma[2]); lower = 0.0)        # loading/swinging power
    x3 ~ truncated(Normal(prior_mu[3], prior_sigma[3]); lower = 0.0)        # traveling power
    x4 ~ truncated(Normal(prior_mu[4], prior_sigma[4]); lower = 0.0)        # idling power
    x = [x1, x2, x3, x4]                                                    # stack into a vector
    s ~ truncated(Normal(0.0, sigma_b); lower = 0.0)                        # observation-noise std (HalfNormal)
    mu = A * x                                                              # predicted energy for each row = A·x
    # Likelihood: each measured energy b[j] is Normal around the predicted mu[j].
    # This loop just attaches that likelihood term for every observation row.
    for j in eachindex(b)
        b[j] ~ Normal(mu[j], s)
    end
end

# A small mutable container that holds the estimator's state between updates:
# the fixed prior, all observations gathered so far, and the CURRENT posterior
# summary (mean `mu` = what the optimiser uses; `sd` = its uncertainty).
mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}      # prior means (the offline starting belief)
    prior_sigma::Vector{Float64}   # prior stds
    A_obs::Matrix{Float64}         # every activity-hours row observed so far (stacked)
    b_obs::Vector{Float64}         # every measured energy observed so far
    mu::Vector{Float64}            # current posterior MEAN -> the profile fed to the MILP
    sd::Vector{Float64}            # current posterior STD  -> uncertainty (for plots / Scenario 2)
    mcmc_samples::Int              # how many NUTS samples to draw per re-fit
end

# Constructor: start with NO observations and the posterior initialised to the
# prior (so before any data arrives, "the estimate" is just the prior belief).
function BayesianActivityEstimator(prior_mu, prior_sigma; mcmc_samples = 500)
    k = length(prior_mu)
    return BayesianActivityEstimator(collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     Matrix{Float64}(undef, 0, k), Float64[],
                                     collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     mcmc_samples)
end

# Record ONE new telemetry observation (a row of activity-hours + its energy).
# This only appends the data; it does not re-run inference (that is refit! below).
function observe!(est::BayesianActivityEstimator, a::AbstractVector, b::Real)
    est.A_obs = vcat(est.A_obs, reshape(collect(float.(a)), 1, :))   # add a row to A_obs
    push!(est.b_obs, float(b))                                       # add the matching energy
    return est
end

# Re-run the Bayesian regression on ALL data so far and refresh mu / sd.
function refit!(est::BayesianActivityEstimator)
    isempty(est.b_obs) && return est                                     # nothing to fit yet
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0    # scale for the noise prior
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, est.prior_sigma, sigma_b)
    chain = sample(model, NUTS(0.9), est.mcmc_samples; progress = false) # run the NUTS sampler
    syms = (:x1, :x2, :x3, :x4)                                          # the four power parameters
    # For each activity, pull its posterior samples out of the chain and store the
    # sample mean (the new estimate) and sample std (the new uncertainty).
    for i in 1:length(est.prior_mu)
        col = vec(chain[syms[i]])
        est.mu[i] = mean(col)
        est.sd[i] = std(col)
    end
    return est
end

# =============================================================================
# 2. WINDOW MILP  (the "optimise" half — build + solve one plan)
# =============================================================================
# This is the heart of the controller. Given the CURRENT physical state and the
# CURRENT power estimate, it builds a mixed-integer linear program (MILP) that
# plans everything from "now" (the first interval of K_win) across a CROSS-DAY
# window, and solves it. The MPC loop calls this once every 15 minutes.
#
# K_win holds GLOBAL interval indices that may span several days' daytime blocks. The
# nights in between are handled by two link rules (see the geometry block below): the
# MCS is parked at the grid + recharged to full overnight, while the CEV battery carries
# over unchanged. The "return CEVs to their start level" rule fires only when the window
# reaches the true horizon end (the buffer day's 18:00).
#
# The many arguments after `d` are the CLOSED-LOOP CARRY-IN — the real state
# handed from the previous step:
#   soe_mcs0/soe_cev0 : measured battery levels now (kWh)
#   mcs_node0         : node the MCS is parked at (0 if mid-drive)
#   mcs_transit0      : nothing, or (i,j,r) = mid-drive on arc i->j, r steps left
#   rem_dig/rem_load  : work hours still remaining at each site
#   cum_*_e           : hours each excavator has already done (for precedence/pacing)
#   peak_nc0/peak_op0 : the biggest grid draw seen so far today (for demand charges)
#   pvec              : the current per-activity power estimate (est.mu)

function build_window_model(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, cum_dig_e, cum_load_e, cum_trv_e,
                            peak_nc0, peak_op0, pvec;
                            daily_dig = rem_dig, daily_load = rem_load,  # fresh quota each morning
                            require_site_visit::Bool = false,      # optional: MCS must visit a site
                            single_visit_per_site::Bool = false,   # optional: at most one visit/site
                            peak_demand_limit = nothing,           # optional: hard cap on grid draw
                            time_limit_sec::Float64 = 30.0, silent::Bool = true,
                            soft_prec::Bool = false,               # relax precedence to a penalty?
                            soft_pace::Bool = false,               # relax travel pacing to a penalty?
                            soft_term::Bool = false,               # relax the CEV end-level to a penalty?
                            enforce_cev_terminal::Bool = true,     # apply the CEV end-level rule at all?
                            # does this window reach the TRUE end of the whole horizon (the buffer
                            # day's 18:00)? Only then do we force the CEVs back to their start level.
                            is_global_terminal::Bool = (last(collect(K_win)) == d.n_day),
                            term_tol::Float64 = 0.0)               # slack (kWh) on the hard CEV end-level
    # Pull the frequently-used sets/scalars out of `d` for readability.
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)   # integer drive times

    # =========================================================================
    # MULTI-DAY WINDOW GEOMETRY (CROSS-DAY RECEDING HORIZON)
    # =========================================================================
    # This is the key change from the single-day model. K_win now holds GLOBAL interval
    # indices that can span SEVERAL days' daytime blocks laid end to end (day-1 daytime,
    # then day-2 daytime, ...). `n_day` = daytime intervals in one day (40). For a global
    # interval index k:
    #   wd(k)    = its position WITHIN its own day (1..n_day). The daily price / carbon /
    #              work-availability profiles have the SAME shape every day, so we always
    #              look them up by wd(k) rather than the raw (possibly >40) global index.
    #   dayof(k) = which day it belongs to (1, 2, ...).
    # The "nights" between day-blocks are NOT intervals; they are handled by two link
    # rules further down: the MCS battery is recharged to full + parked at the grid
    # overnight, while the CEV battery simply CARRIES OVER unchanged into the next morning.
    n_day = d.n_day
    wd(k)    = mod(k - 1, n_day) + 1
    dayof(k) = div(k - 1, n_day) + 1

    K = collect(K_win)                      # the (GLOBAL) interval indices this window covers
    Tb = vcat(K, last(K) + 1)               # boundary indices (one more than intervals)
    K_peak = [k for k in K if in_peak(wd(k), delta_T, d.t_start)]   # on-peak (by within-day clock)
    blockdays  = sort(unique(dayof.(K)))                  # which days this window touches
    firstday   = dayof(first(K))                          # the (possibly partial) current day
    block_ks(dy) = [k for k in K if dayof(k) == dy]       # the in-window intervals of day dy
    # "Evening" intervals = the last daytime interval (18:00) of each day present in the
    # window. At every evening the MCS must be parked at a grid node (10e); after every
    # evening EXCEPT the final one the MCS battery is reset to full for the next morning.
    eve_k      = [k for k in K if wd(k) == n_day]
    night_eve  = [k for k in eve_k if k != last(K)]       # evenings followed by another day
    # Per-day within-day lookups (identical daily profile each day, indexed by wd).
    price_k(k) = d.lambda_whl_elec[wd(k)]                 # electricity price this interval
    co2_k(k)   = d.lambda_CO2[wd(k)]                      # grid carbon intensity this interval
    Rwork(i, e, k) = d.R_work[i, e, wd(k)]                # work-availability cap this interval
    # productive_k[k] = true if any excavator is allowed to work in interval k
    # (shift hours, lunch excluded). Outside those hours the machine idles (and may charge).
    productive_k = Dict(k => any(Rwork(i, e, k) > 0 for i in N_c, e in E) for k in K)

    # Map activity index -> its (estimated) power draw, e.g. p_activity[1] = dig power.
    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))

    # Helper functions: total hours already done at a SITE = sum over the excavators
    # assigned to that site (used to seed the precedence rule). These carried counts are
    # only injected into the FIRST (current) day-block; later day-blocks start fresh.
    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    # Helpers for a drive that was ALREADY in progress when this window began:
    #  - is_carried_trv: is (i,j,k) part of that still-pending drive?
    #  - carried_arrival_k: the interval at which that carried drive finally arrives.
    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])          # first r window intervals
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    # ---- create the optimisation model and configure the solver ----
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)                 # suppress solver chatter
    set_time_limit_sec(model, time_limit_sec)   # cap solve time per window
    # Force single-threaded, deterministic solving. HiGHS's parallel MIP path can
    # crash on Windows for the bigger models; serial is stable and fast enough here.
    set_attribute(model, "threads", 1)
    set_attribute(model, "parallel", "off")
    # Disable HiGHS's sub-MIP primal heuristics (RENS/RINS). Those launch an INTERNAL
    # sub-MIP that spins up HiGHS's parallel task deque even though the OUTER model is
    # serial (threads=1/parallel=off do NOT propagate into the sub-solver), which can
    # segfault on Windows (EXCEPTION_ACCESS_VIOLATION in HighsSplitDeque). Turning the
    # heuristic effort off keeps HiGHS on the stable serial branch-and-cut path; these
    # small per-window MILPs solve fine without the heuristics.
    set_attribute(model, "mip_heuristic_effort", 0.0)
    # HiGHS's root-node symmetry detection also uses the parallel task deque, so it is a
    # second source of the same Windows segfault. Disable it too.
    set_attribute(model, "mip_detect_symmetry", false)
    # Accept a solution within 1% of optimal. Since MPC only applies the FIRST
    # interval anyway, proving full optimality every window would be wasted time.
    set_attribute(model, "mip_rel_gap", 1.0e-2)

    # ---- CONTINUOUS decision variables: power flows (kW) ----
    @variable(model, P_ch_MCS[M, N, K] >= 0)      # grid -> MCS charge power, per node
    @variable(model, P_dch_MCS[M, N, K] >= 0)     # MCS -> site discharge power, per node
    @variable(model, P_MCS_CEV[M, N_c, E, K] >= 0)# MCS -> specific excavator power
    @variable(model, P_work[N_c, E, K] >= 0)      # power an excavator spends working
    @variable(model, P_ch_tot[M, K] >= 0)         # total grid draw by the MCS
    @variable(model, P_dch_tot[M, K] >= 0)        # total discharge out of the MCS
    # UNFINISHED work (hours), now ONE slack per (site, day-block). s_miss_dig[i, dy] =
    # how far site i's CUMULATIVE digging quota through the end of day dy is still unmet.
    # Because the target is cumulative, a shortfall automatically ROLLS OVER into the next
    # day (and is penalised again) — a soft "leftover work carries to tomorrow" rule.
    @variable(model, s_miss_dig[N_c, blockdays] >= 0)
    @variable(model, s_miss_load[N_c, blockdays] >= 0)
    # Slacks that let us OPTIONALLY relax precedence (12d) and travel pacing (13).
    # Kept at zero by default (hard); freed only via the soft_* switches.
    @variable(model, s_prec[N_c, K] >= 0)
    @variable(model, s_pace_hi[E, K] >= 0)   # travel-pacing upper band slack
    @variable(model, s_pace_lo[E, K] >= 0)   # travel-pacing lower band slack

    # ---- travel energy (kWh) ----
    @variable(model, L_trv[M, N, N, K] >= 0)      # energy burned driving each arc
    @variable(model, L_trv_tot[M, K] >= 0)        # total driving energy per interval

    # ---- battery state of energy, indexed at interval BOUNDARIES ----
    @variable(model, SOE_MCS[M, Tb] >= 0)         # MCS energy at each boundary
    @variable(model, SOE_CEV[E, Tb] >= 0)         # each excavator's energy at each boundary

    # ---- BINARY (yes/no) decision variables ----
    @variable(model, u[E, N, B, K], Bin)          # which activity each excavator does
    @variable(model, mu[N, E, K], Bin)            # is the excavator charging?
    @variable(model, rho[M, N, E, K], Bin)        # is the excavator plugged into the MCS?
    @variable(model, z[M, N, K], Bin)             # is the MCS parked at this node?
    @variable(model, g_ch[M, N_g, K], Bin)        # is the MCS actively grid-charging here?
    @variable(model, x[M, N, N, K], Bin)          # does the MCS depart i -> j this interval?
    @variable(model, y_trv[M, N, N, K], Bin)      # is the MCS in transit on arc i -> j?
    @variable(model, beta_arr[M, N, K], Bin)      # MCS arrival indicator at a node
    @variable(model, beta_dep[M, N, K], Bin)      # MCS departure indicator at a node
    @variable(model, P_peak_NC >= 0)              # tracked whole-day peak grid draw
    @variable(model, P_peak_OP >= 0)              # tracked on-peak peak grid draw
    # Slack for the CEV "end the day at your start level" rule. Zero (hard) by
    # default; freed only via soft_term. (The MCS has no such Phase-1 rule — it is
    # refilled overnight in Phase 2 instead.)
    @variable(model, s_term_cev[E] >= 0)

    # ---- OBJECTIVE: total operating cost to minimise ----
    # Sum of: grid energy cost + carbon cost + missed-work penalty + two demand
    # charges + MCS towing labour. (Everything is linear in the decision vars.)
    obj = @expression(model,
        sum(price_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +                                          # energy $
        sum((d.carbon_price_per_ton / 1000.0) * co2_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +        # carbon $
        d.rho_miss * (sum(s_miss_dig[i, dy] for i in N_c, dy in blockdays) +
                      sum(s_miss_load[i, dy] for i in N_c, dy in blockdays)) +                                   # missed-work $
        d.lambda_demand_NC * P_peak_NC +                                                                         # whole-day demand $
        d.lambda_demand_OP * P_peak_OP +                                                                         # on-peak demand $
        d.rho_labor * delta_T * sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K))                       # towing labour $

    # HARD MODE (default): pin the optional slacks to zero so precedence, pacing,
    # and the CEV end-level hold EXACTLY. If a soft_* switch is on, we instead add
    # a big penalty on that slack (letting it be violated at a cost). The W_* are
    # those penalty weights.
    W_prec = 8.0e2; W_pace = 1.0e2; W_term = 1.5e2
    soft_prec || @constraint(model, [i in N_c, k in K], s_prec[i, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_hi[e, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_lo[e, k] == 0)
    soft_term || @constraint(model, [e in E], s_term_cev[e] == 0)
    @objective(model, Min, obj +
        (soft_prec ? W_prec * sum(s_prec[i, k] for i in N_c, k in K) : AffExpr(0.0)) +
        (soft_pace ? W_pace * sum(s_pace_hi[e, k] + s_pace_lo[e, k] for e in E, k in K) : AffExpr(0.0)) +
        (soft_term ? W_term * sum(s_term_cev[e] for e in E) : AffExpr(0.0)))

    # ---- power aggregation & where power may flow ----
    @constraint(model, [m in M, k in K], P_ch_tot[m, k]  == sum(P_ch_MCS[m, i, k]  for i in N_g))   # total grid draw = sum over grid nodes
    @constraint(model, [m in M, k in K], P_dch_tot[m, k] == sum(P_dch_MCS[m, i, k] for i in N_c))   # total discharge = sum over sites
    @constraint(model, [m in M, i in N_g, k in K], P_dch_MCS[m, i, k] == 0)                         # no discharging at the grid
    @constraint(model, [m in M, i in N_c, k in K], P_ch_MCS[m, i, k]  == 0)                         # no grid-charging at a site
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] == sum(P_MCS_CEV[m, i, e, k] for e in E))                                # discharge at a site = sum to its excavators
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] <= d.DCH_MCS[m] * z[m, i, k])                                            # can only discharge where parked, capped

    # grid-connection exclusivity: the MCS can only grid-charge where it is parked,
    # up to its charge rate, and at most one MCS uses a given grid plug per interval.
    @constraint(model, [m in M, i in N_g, k in K], P_ch_MCS[m, i, k] <= d.CH_MCS[m] * g_ch[m, i, k])
    @constraint(model, [m in M, i in N_g, k in K], g_ch[m, i, k] <= z[m, i, k])
    @constraint(model, [i in N_g, k in K], sum(g_ch[m, i, k] for m in M) <= 1)

    # plug-level and excavator-acceptance limits: power to an excavator needs a plug
    # (rho) and is capped per plug; each excavator's intake is capped and only while charging (mu).
    @constraint(model, [m in M, i in N_c, e in E, k in K],
        P_MCS_CEV[m, i, e, k] <= d.DCH_MCS_plug[m] * rho[m, i, e, k])
    @constraint(model, [i in N_c, e in E, k in K],
        sum(P_MCS_CEV[m, i, e, k] for m in M) <= d.CH_CEV[e] * mu[i, e, k])

    # peak-demand trackers: P_peak_* must be >= the running grid draw AND >= the
    # peak already seen earlier today (so the demand charge reflects the whole day).
    @constraint(model, P_peak_NC >= peak_nc0)
    @constraint(model, P_peak_OP >= peak_op0)
    @constraint(model, [k in K], P_peak_NC >= sum(P_ch_tot[m, k] for m in M))
    @constraint(model, [k in K_peak], P_peak_OP >= sum(P_ch_tot[m, k] for m in M))
    if peak_demand_limit !== nothing
        @constraint(model, [k in K], sum(P_ch_tot[m, k] for m in M) <= peak_demand_limit)   # optional hard cap
    end

    # ---- travel energy bookkeeping ----
    # This loop links "in transit" (y_trv) to the departure decisions (x): normally
    # an arc is in-transit for the `travel_steps` intervals after a departure; but if
    # a drive was already underway at window start, it is forced to continue (==1).
    for m in M, i in N, j in N, k in K
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k] == 1)        # continue a drive begun before this window
        else
            @constraint(model, y_trv[m, i, j, k] == sum(x[m, i, j, tau]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end
    @constraint(model, [m in M, i in N, j in N, k in K],
        L_trv[m, i, j, k] == d.k_trv * delta_T * y_trv[m, i, j, k])          # energy per in-transit interval
    @constraint(model, [m in M, k in K],
        L_trv_tot[m, k] == sum(L_trv[m, i, j, k] for i in N, j in N))        # total driving energy per interval

    # ---- battery dynamics: next SOE = now + charged - discharged/worked - losses ----
    @constraint(model, [m in M], SOE_MCS[m, first(Tb)] == soe_mcs0[m])       # MCS starts at its measured level
    @constraint(model, [e in E], SOE_CEV[e, first(Tb)] == soe_cev0[e])       # each excavator starts at its measured level
    # MCS flow WITHIN each day. We SKIP this link across a night boundary (k in
    # night_eve): the MCS is recharged overnight (Phase 2) back to full, so how far it
    # drained during the day is irrelevant to the next morning.
    @constraint(model, [m in M, k in K; !(k in night_eve)],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
            d.eta_ch_dch[m] * P_ch_tot[m, k] * delta_T -                     # + energy charged (with efficiency)
            (P_dch_tot[m, k] * delta_T) / d.eta_ch_dch[m] -                  # - energy discharged (with efficiency)
            L_trv_tot[m, k])                                                 # - driving losses
    # Overnight bridge: each MCS starts the next day recharged to its start-of-day level.
    @constraint(model, [m in M, k in night_eve], SOE_MCS[m, k + 1] == d.SOE_MCS_ini[m])
    # The CEV battery, by contrast, CARRIES OVER continuously across nights (no reset):
    # this single link applies to every interval, night boundaries included.
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
            sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -      # + energy received from the MCS
            sum(P_work[i, e, k] for i in N_c) * delta_T)                     # - energy spent working

    # keep every battery within [floor, capacity] at every boundary
    @constraint(model, [m in M, t in Tb], d.SOE_MCS_min[m] <= SOE_MCS[m, t] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb], d.SOE_CEV_min[e] <= SOE_CEV[e, t] <= d.SOE_CEV_max[e])

    # NOTE: there is deliberately NO "MCS must end the day full" rule here. During
    # the day the MCS only grid-charges as needed to stay above its 20% floor; the
    # big refill back to full happens overnight in the cheap Phase-2 charge.

    # ---- CEV energy neutrality (ONLY at the TRUE end of the whole horizon) ----
    # In the multi-day receding horizon the excavators are NOT forced back to their
    # start level every evening — their battery flows across days. The "return to start"
    # rule (8b) is applied ONLY when the window reaches the true horizon end (the buffer
    # day's 18:00 -> is_global_terminal). In HARD mode it is the one-sided inequality
    # SOE_end >= SOE_ini - term_tol (term_tol is a tiny margin that absorbs learning
    # drift; = exact equality at term_tol=0, since the SOE_max bound already caps the
    # top). In soft mode it becomes a two-sided penalised slack instead.
    if enforce_cev_terminal && is_global_terminal
        if soft_term
            @constraint(model, [e in E],  SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e] <= s_term_cev[e])
            @constraint(model, [e in E], -(SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e]) <= s_term_cev[e])
        else
            @constraint(model, [e in E], SOE_CEV[e, last(Tb)] >= d.SOE_CEV_ini[e] - term_tol)
        end
    end

    # ---- plugging / presence logic ----
    @constraint(model, [m in M, i in N_c, k in K], sum(rho[m, i, e, k] for e in E) <= d.C_MCS_plug[m])  # at most #plugs excavators at once
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= d.A[i, e])                  # only plug an excavator at its own site
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])                 # only plug where the MCS is parked
    @constraint(model, [m in M, i in N, k in K], x[m, i, i, k] == 0)                                    # can't "drive" from a node to itself

    # presence partition: at any interval the MCS is EITHER parked at exactly one
    # node OR in transit on exactly one arc — never both, never neither.
    @constraint(model, [m in M, k in K],
        sum(z[m, i, k] for i in N) + sum(y_trv[m, i, j, k] for i in N, j in N if i != j) == 1)

    # initial position: unless it is mid-drive (handled above), the MCS must either
    # be parked at, or departing from, the node where it actually is right now.
    for m in M
        if mcs_transit0[m] === nothing
            p = mcs_node0[m]
            @constraint(model, z[m, p, first(K)] + sum(x[m, p, j, first(K)] for j in N if j != p) == 1)
        end
    end

    # departures follow from the drive decisions x
    @constraint(model, [m in M, i in N, k in K],
        beta_dep[m, i, k] == sum(x[m, i, j, k] for j in N if j != i))
    # arrivals: this loop sets beta_arr. If a carried drive lands at interval k, mark
    # that arrival; otherwise an arrival at node j equals a departure toward j one
    # travel-time earlier.
    for m in M, j in N, k in K
        if carried_arrival_k(m) == k && j == mcs_transit0[m][2]
            @constraint(model, beta_arr[m, j, k] == 1)        # the carried drive arrives here
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
    # parked-status changes only via an arrival or departure, and you can't arrive
    # and depart the same node in the same interval.
    @constraint(model, [m in M, i in N, k in K[2:end]],
        beta_arr[m, i, k] - beta_dep[m, i, k] == z[m, i, k] - z[m, i, k - 1])
    @constraint(model, [m in M, i in N, k in K],
        beta_arr[m, i, k] + beta_dep[m, i, k] <= 1)

    # flow conservation per node, written to also work when the window STARTS with
    # the MCS parked at a site or mid-drive, and (when terminal) ENDS at the grid:
    #     arrivals(i) - departures(i) = present_end(i) - present_start(i)
    # This loop writes that balance for every node.
    for m in M, i in N
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k] for k in K) - sum(beta_dep[m, i, k] for k in K) ==
            z[m, i, last(K)] - start_here)
    end

    # terminal position: at the END of EVERY daytime block in the window (each 18:00 =
    # eve_k), the MCS must be parked at a grid node, ready for that night's overnight
    # refill. This covers the interior nights of a multi-day window and the horizon end.
    @constraint(model, [m in M, k in eve_k], sum(z[m, i, k] for i in N_g) == 1)

    # optional site-visit rules (off by default)
    if require_site_visit
        @constraint(model, [m in M], sum(beta_arr[m, i, k] for i in N_c, k in K) >= 1)
    end
    if single_visit_per_site
        @constraint(model, [m in M, i in N_c], sum(beta_arr[m, i, k] for k in K) <= 1)
        @constraint(model, [m in M, i in N_c], sum(beta_dep[m, i, k] for k in K) <= 1)
    end

    # ---- activity scheduling ----
    # An assigned excavator does EXACTLY one activity per interval (idling counts),
    # and only activities at its own site are possible.
    @constraint(model, [i in N_c, e in E, k in K],
        sum(u[e, i, a, k] for a in B) == d.A[i, e])
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= d.A[i, e])
    # Work power (dig/load/travel, not idle) is capped by the work-availability cap
    # R_work and forced to zero while charging (the (1 - mu) factor). Since R_work is
    # 0 outside shift hours, this also confines real work to the shift.
    @constraint(model, [i in N_c, e in E, k in K],
        sum(p_activity[a] * u[e, i, a, k] for a in (B[1], B[2], B[3])) <=
        Rwork(i, e, k) * d.A[i, e] * (1 - mu[i, e, k]))
    # An excavator may only charge while it is idling (charge => idle activity).
    @constraint(model, [i in N_c, e in E, k in K], mu[i, e, k] <= u[e, i, B[4], k])
    # Define the excavator's per-interval work power from its chosen activity's draw.
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- daily work quota: a soft, CUMULATIVE target that rolls over ----
    # Each morning a fresh quota (daily_dig/daily_load) arrives; the window-start rem_*
    # already holds today's outstanding hours + any carried leftover. For every day-block
    # dy in the window we require the CUMULATIVE work done through the END of dy to reach
    # the cumulative target (rem_* + one fresh quota per subsequent morning). Any shortfall
    # s_miss_* >= 0 is penalised (rho_miss) and, since the target is cumulative, automatically
    # ROLLS OVER into the next day. Working AHEAD is fine (the slack simply hits 0), so there
    # is no upper-bound infeasibility.
    for (p, dy) in enumerate(blockdays)
        Kupto = [k for k in K if dayof(k) <= dy]           # window intervals up to end of day dy
        @constraint(model, [i in N_c],
            s_miss_dig[i, dy] >= (max(rem_dig[i], 0.0) + (p - 1) * daily_dig[i]) -
                                 delta_T * sum(u[e, i, B[1], k] for e in E, k in Kupto))
        @constraint(model, [i in N_c],
            s_miss_load[i, dy] >= (max(rem_load[i], 0.0) + (p - 1) * daily_load[i]) -
                                  delta_T * sum(u[e, i, B[2], k] for e in E, k in Kupto))
    end

    # precedence: cumulative loading can't exceed `scale` x cumulative digging at any
    # point (you can't load faster than you have dug), evaluated WITHIN each day-block
    # (counters restart each morning; carried realized work seeds only the current day).
    bstart(k) = first(block_ks(dayof(k)))                  # first in-window interval of k's day
    @constraint(model, [i in N_c, k in K],
        (((dayof(k) == firstday) ? cum_load_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[2], tau] for tau in bstart(k):k, e in E)) <=
        d.scale * (((dayof(k) == firstday) ? cum_dig_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[1], tau] for tau in bstart(k):k, e in E)) +
        s_prec[i, k])

    # ---- rest rule (operator break) ----
    # Over any rolling window of (t_limit_rest + one step), an excavator may do at
    # most t_limit_rest hours of real work — i.e. at least one idle break per window.
    # This loop imposes that cap on every rolling window (only if the window is long
    # enough to contain one).
    rest_cap = Int(round(d.t_limit_rest / delta_T))      # max work intervals per rolling window
    rest_win = rest_cap + 1                               # rolling window length in intervals
    if length(K) >= rest_win
        # Only enforce on rolling windows that stay WITHIN a single day-block (a night is a
        # full rest, so no break need be enforced across it).
        rest_starts = [k0 for k0 in first(K):(last(K) - rest_win + 1)
                       if dayof(k0) == dayof(k0 + rest_win - 1)]
        @constraint(model, [i in N_c, e in E, k0 in rest_starts],
            sum(u[e, i, a, k] for a in (B[1], B[2], B[3]), k in k0:(k0 + rest_win - 1)) <= rest_cap)
    end

    # ---- travel pacing ----
    # Keep each excavator's cumulative TRAVEL roughly proportional to its cumulative
    # WORK (about one travel step per kappa productive steps), as a two-sided band.
    # This loop adds the upper/lower band inequality at every interval, seeded with
    # work already completed; the s_pace slacks absorb closed-loop drift if made soft.
    kappa = d.kappa_wt
    for e in E, kk in K
        carry = (dayof(kk) == firstday)                    # seed realized work only in current day
        bs = bstart(kk)                                    # first in-window interval of kk's day
        trv_cum  = (carry ? cum_trv_e[e] / delta_T : 0.0) +
                   sum(u[e, i, B[3], tau] for i in N_c, tau in bs:kk)
        work_cum = (carry ? (cum_dig_e[e] + cum_load_e[e]) / delta_T : 0.0) +
                   sum(u[e, i, a, tau] for i in N_c, a in (B[1], B[2]), tau in bs:kk)
        @constraint(model, kappa * trv_cum <= work_cum + s_pace_hi[e, kk])
        @constraint(model, kappa * trv_cum >= work_cum - kappa - s_pace_lo[e, kk])
    end

    # Hand the whole thing to HiGHS and solve. We wrap it in try/catch because HiGHS's
    # native MIP path can, rarely and non-deterministically on Windows, throw a memory
    # fault (e.g. ReadOnlyMemoryError) on one particular window. Catching it means a
    # single bad solve does NOT kill a multi-hour run: the caller checks has_values(model)
    # and, finding none, treats this interval as infeasible and HOLDS state (exactly the
    # no-fallback behaviour used for a genuinely infeasible window).
    try
        optimize!(model)
    catch err
        @warn "Solver threw during optimize!; treating this window as no-solution (hold state)." exception = err
    end
    return model         # return the (possibly unsolved) model; caller reads decisions out of it
end

# =============================================================================
# 3. MPC LOOP HELPERS
# =============================================================================
# The functions below support the main loop: simulating what "really happened"
# in an interval, tracking the MCS position across steps, and turning the plan
# into plain-language worker instructions.

# Simulate the REALIZED within-interval activity split for excavator e at step k0.
# The MILP plans ONE activity per 15-min interval, but a real machine mixes tasks
# (e.g. "9 min digging, 6 min idle"). We read the planned activity out of the
# solved model, then (if multi=true) spend 60–100% of the interval on it and the
# rest idling — giving the learner a richer, mixed regression row. Returns hours
# per activity (summing to delta_T).
function realized_activity_durations(rng, model, e, k0, d; multi::Bool = true)
    dt = d.delta_T
    a = zeros(length(d.B))                      # output: hours per activity
    idle = length(d.B)                          # idling is the last activity index (4)

    # Find which activity the plan chose for this interval (scan the u binaries).
    # This loop leaves `planned` = the chosen activity index, or 0 if none.
    planned = 0
    for i in d.N_c, (ai, act) in enumerate(d.B)
        if value(model[:u][e, i, act, k0]) > 0.5
            planned = ai
        end
    end
    if planned == 0
        a[idle] = dt                            # nothing planned -> a full idle interval
        return a
    end

    if !multi
        a[planned] = dt                         # single-activity mode: whole interval on the plan
        return a
    end

    # multi-activity mode: 60–100% on the planned task, remainder idling.
    frac = 0.6 + 0.4 * rand(rng)
    a[planned] += dt * frac
    a[idle]    += dt * (1.0 - frac)
    return a
end

# Figure out where the MCS will be at the START of the NEXT interval (k0+1), from
# the solved plan. Returns (node, transit):
#   node    = parked node index, or 0 if the MCS is mid-drive at k0+1
#   transit = nothing, or (i, j, r): mid-drive on arc i->j with r intervals left
function advance_mcs_state(model, m, k0, nK, d)
    z = model[:z]; y = model[:y_trv]
    Kw = axes(z)[3]                             # the interval axis of the solved window
    knext = k0 + 1
    if knext > nK || !(knext in Kw)             # no "next" interval -> report the current park node
        node = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)
        return (node === nothing ? first(d.N_g) : node, nothing)
    end
    node = findfirst(i -> value(z[m, i, knext]) > 0.5, d.N)  # parked somewhere next interval?
    node !== nothing && return (node, nothing)
    # Otherwise it is mid-drive: find the arc and count how many more intervals the
    # drive lasts (this loop scans arcs, then walks forward while still in transit).
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
    node0 = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)    # fallback: stay put
    return (node0 === nothing ? first(d.N_g) : node0, nothing)
end

# ---- turn plan decisions into plain words for the worker-facing CSV ----
const _ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# The single activity the plan wants excavator e to do at interval k0 (or "Off
# (home)" if the shift is over / it is powered down).
function _planned_activity(model, d, e, k0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    site === nothing && return "Off (home)"
    vals = [value(model[:u][e, site, a, k0]) for a in eachindex(d.B)]
    sum(vals) < 0.5 && return "Off (home)"
    return _ACT_NAME[d.B[argmax(vals)]]
end

# "Should excavator e be plugged in to charge this interval?" -> Yes/No.
function _cev_should_charge(model, d, e, k0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    return (site !== nothing && value(model[:mu][site, e, k0]) > 0.5) ? "Yes" : "No"
end

# "Should the MCS draw from the grid this interval?" -> Yes/No.
_mcs_should_charge(model, d, k0) =
    (sum(value(model[:P_ch_tot][m, k0]) for m in d.M) > 1e-6) ? "Yes" : "No"

# =============================================================================
# 2b. PHASE 2 — OVERNIGHT SMART-CHARGE  (deterministic; NOT an optimisation)
# =============================================================================
# After 18:00 the MCS is parked at the grid with some energy. The overnight job is
# simple: buy back exactly the energy it is short by, using the CHEAPEST overnight
# 15-min slots first, capped at its charge rate and capacity. Because it only
# charges (energy goes up monotonically) and the target is <= capacity, greedy
# "fill cheapest slots first" is provably optimal — no MILP needed.
# Returns (df, P_ov, ov_k): a schedule table, a power matrix, and the overnight
# interval indices.
function phase2_overnight_charge(d, soe_mcs_end)
    dt   = d.delta_T
    ov_k = (d.n_day + 1):d.n_int               # overnight interval indices (18:00 -> 08:00)
    nov  = length(ov_k)
    P_ov = zeros(length(d.M), nov)             # overnight charge power per MCS per interval
    soe_path = [fill(float(soe_mcs_end[m]), nov + 1) for m in d.M]   # SOE trace per MCS

    # For each MCS, compute how much it must refill and greedily assign that energy
    # to the cheapest overnight slots; then walk the SOE forward for the record.
    for m in d.M
        eta  = d.eta_ch_dch[m]
        rate = d.CH_MCS[m]
        deficit = d.SOE_MCS_ini[m] - soe_mcs_end[m]    # energy to restore for energy-neutrality
        if deficit > 1e-9
            order = sort(collect(1:nov); by = j -> d.lambda_whl_elec[ov_k[j]])   # cheapest first
            remaining = deficit
            for j in order
                remaining <= 1e-9 && break
                gain = min(eta * rate * dt, remaining)   # SOE we can add this slot
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

    # Assemble the per-interval overnight schedule into a DataFrame (this loop adds
    # a charge-power / SOE / on-off column per MCS).
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

# =============================================================================
# 3. MAIN MPC LOOP  (ties optimise + learn + apply together)
# =============================================================================
# This is the driver. It runs a MULTI-DAY, CROSS-DAY receding horizon: it simulates
# `n_days` reported days plus one dropped BUFFER day, and for every 15-min interval
# of each day repeats the classic MPC cycle:
#     (1) solve the MILP over the cross-day window (rest of today + lookahead_days
#         future daytime blocks) from the current state + estimate,
#     (2) APPLY only the first interval's decisions to the "plant",
#     (3) simulate what really happened and feed it to the Bayesian learner,
#     (4) advance the real state, and move on.
# Each night it runs the deterministic overnight charge (Phase 2) and resets the MCS
# to full; the CEV battery and any unfinished work carry into the next day. At the end
# it drops the buffer day, prints KPIs, and writes all the CSVs / plots.
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(@__DIR__), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,        # LEGACY: ignored (horizon = cross-day window set by lookahead_days)
                          time_limit_sec::Float64 = 60.0,             # solver time cap per window
                          multi_activity::Bool = false,               # simulate mixed within-interval work?
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          refit_every::Int = 8, mcmc_samples::Int = 500,  # re-fit cadence + NUTS samples
                          out_dir::String = joinpath(dirname(@__DIR__), "output", String(mode)),
                          soft_prec::Bool = false, soft_pace::Bool = false,
                          soft_term::Bool = false,
                          term_tol::Float64 = 0.1,                    # tiny margin on the CEV end-level
                          n_days::Union{Nothing, Int} = nothing,      # days to KEEP in the results
                          lookahead_days::Int = 1,                    # cross-day window depth
                          seed::Int = 1)
    # Seed the RNG so runs are reproducible (NUTS + the simulated telemetry noise).
    Random.seed!(seed)
    # If the default input folder is missing, try a couple of legacy locations so
    # :input mode still finds the data.
    if mode == :input && !isdir(input_dir)
        for alt in (joinpath(@__DIR__, "input_data"),
                    joinpath(dirname(@__DIR__), "input_data"))
            if isdir(alt); input_dir = alt; break; end
        end
    end
    d = load_data(mode; input_dir = input_dir)         # load the chosen dataset
    K_all = collect(d.K)
    nK = length(K_all)                                  # number of daytime steps per day (40)

    # =========================================================================
    # MULTI-DAY (CROSS-DAY RECEDING HORIZON) SETUP
    # =========================================================================
    # n_days_keep = the days we KEEP in the reported results. We SIMULATE one extra
    # "buffer" day (D_total = n_days_keep + 1) and DROP it from every output/KPI. That
    # extra day gives the last kept day a full day of lookahead, so the "return CEVs to
    # start" wrap-up lands on the discarded buffer day instead of a real reported day.
    # State FLOWS across days: CEV battery SOE and any unfinished work carry into the
    # next day. The MCS is recharged overnight each night so it starts each day full.
    n_days_keep = n_days === nothing ? d.n_days : max(1, n_days)
    D_total     = n_days_keep + 1
    G           = D_total * nK                          # total daytime intervals in the horizon

    # ---- the REAL physical state we CARRY ACROSS DAYS (the "plant") ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))              # MCS energy now
    soe_cev  = copy(float.(d.SOE_CEV_ini))              # excavator energies now
    # Per-day work quota that ARRIVES each morning; whatever is left unfinished stays in
    # rem_* and carries into the next day (a soft, penalised carry via the MILP s_miss).
    daily_dig  = copy(float.(d.hours_digging))
    daily_load = copy(float.(d.hours_loading_swinging))
    rem_dig    = zeros(length(d.hours_digging))         # outstanding digging (starts empty; quota added below)
    rem_load   = zeros(length(d.hours_loading_swinging))# outstanding loading

    # ---- the online learner, seeded with the offline prior ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)                       # separate RNG for the simulated telemetry noise

    # ---- detailed ANALYST log (one row per applied interval; `day` tags the sim day) ----
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

    # ---- simple WORKER-FACING schedule columns (what to do, in plain words) ----
    fe_time = String[]                   # clock label per applied interval
    fe_act  = [String[] for _ in d.E]    # per-excavator planned activity
    fe_chg  = [String[] for _ in d.E]    # per-excavator "plug in?" Yes/No
    fe_mcs  = String[]                   # MCS "charge from grid?" Yes/No

    # ---- per-day overnight schedules + replanning grids (kept days are written out) ----
    # Each kept day gets its own overnight Phase-2 schedule and its own nK x nK replan
    # grids, stored keyed by day number.
    overnight_by_day = Dict{Int, DataFrame}()
    replan_by_day    = Dict{Int, NamedTuple}()

    println("Running Scenario 1 (RECEDING horizon, closed-loop MPC, 15-min steps, CROSS-DAY lookahead):")
    println("  keeping $n_days_keep day(s); simulating $D_total (last = dropped buffer day); $nK steps/day")
    println("  window spans current + $lookahead_days lookahead day(s); nights via MCS overnight recharge + CEV carry-over")
    println("  prior power estimate : ", round.(est.mu, digits = 2), " kW")
    println("  (hidden) true power  : ", d.true_powers, " kW")
    t0 = time()
    n_obs_total = 0                   # count of telemetry observations gathered
    n_infeasible = 0                  # count of windows that could not be solved (state held)
    gstep       = 0                   # global 15-min step counter across all days
    missed_kept = 0.0                 # unfinished work (hours) at the end of the last KEPT day

    # ===================== THE CLOSED LOOP: OUTER OVER DAYS, INNER PER 15-MIN =====================
    # For each simulated day we walk the 40 daytime intervals. At each interval we solve a
    # CROSS-DAY window (the rest of today + `lookahead_days` future daytime blocks), apply
    # only the first interval, learn from the realized result, and advance the real state.
    # Global interval index g0 = (day-1)*nK + k0 runs 1..G; the MILP indexes prices / work /
    # nights by the within-day clock internally.
    for day in 1:D_total
        # ---- start-of-day resets ----
        # A fresh work quota arrives; leftover from prior days is already in rem_* (carried).
        rem_dig  .+= daily_dig
        rem_load .+= daily_load
        # Precedence / pacing counters and the demand-charge peak trackers restart each day.
        cum_dig_e  = zeros(length(d.E))   # hours each excavator has dug today
        cum_load_e = zeros(length(d.E))   # ... loaded today
        cum_trv_e  = zeros(length(d.E))   # ... travelled today
        peak_nc = 0.0; peak_op = 0.0
        mcs_node = [first(d.N_g) for _ in d.M]   # MCS starts each day parked at the grid
        mcs_transit = Any[nothing for _ in d.M]  # ...and not mid-drive

        # per-day replanning grids (nK x nK; we record only the CURRENT day's slice of
        # each cross-day forward plan)
        plan_grid_kW = fill(NaN, nK, nK)
        plan_mcs_soe = fill(NaN, nK, nK)
        plan_cev_soe = [fill(NaN, nK, nK) for _ in d.E]
        plan_cev_act = [fill("", nK, nK)  for _ in d.E]

        day_off = (day - 1) * nK          # global offset of this day's daytime block

        for k0 in 1:nK
            gstep += 1
            g0    = day_off + k0                          # GLOBAL interval index of this step
            clk   = clock_day_label(d, day, k0)           # e.g. "D2 08:15"
            # cross-day window: rest of today + `lookahead_days` future daytime blocks, never
            # past the buffer day. This is what makes the plan always "see" tomorrow.
            view_end_day = min(D_total, day + lookahead_days)
            Kend  = view_end_day * nK
            K_win = g0:Kend
            is_glob_term = (Kend == G)                    # window reaches the true horizon end

            # (1) OPTIMISE: solve the MILP from the current real state + current estimate.
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

            # NO FALLBACK: if the hard model is infeasible from this drifted state, we do
            # NOT relax anything — we log it, hold the plant still for this interval, and
            # move on. (The block below records that "held" interval and skips the rest.)
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
                continue                                   # skip to the next interval
            end

            # (2) APPLY: read out just interval g0's decisions (grid draw, discharge, node).
            grid_kW = sum(value(model[:P_ch_tot][m, g0]) for m in d.M)
            dch_kW  = sum(value(model[:P_dch_tot][m, g0]) for m in d.M)
            cur_node = let nh = findfirst(i -> value(model[:z][1, i, g0]) > 0.5, d.N)
                nh === nothing ? 0 : nh                  # 0 = MCS in transit during g0
            end

            # Save the CURRENT-DAY slice of this window's forward plan into the replan grids.
            for k in K_win
                div(k - 1, nK) + 1 == day || continue     # keep only today's intervals
                kl = k - day_off                           # within-day column index
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

            # Worker-facing row for the applied interval (plain-words instructions).
            push!(fe_time, clk)
            for e in d.E
                push!(fe_act[e], _planned_activity(model, d, e, g0))
                push!(fe_chg[e], _cev_should_charge(model, d, e, g0))
            end
            push!(fe_mcs, _mcs_should_charge(model, d, g0))

            # (3) SIMULATE what really happened this interval (the mixed activity split).
            a_real = Dict(e => realized_activity_durations(rng, model, e, g0, d;
                                                           multi = multi_activity) for e in d.E)

            # (3b) LEARN: each realized row is a new regression observation.
            for e in d.E
                row = a_real[e]
                if sum(row) > 1e-9
                    b_obs = dot(row, d.true_powers) + d.obs_noise_std * randn(rng)
                    observe!(est, row, b_obs)
                    n_obs_total += 1
                end
            end
            if n_obs_total > 0 && gstep % refit_every == 0
                refit!(est)                              # re-run NUTS every refit_every steps
            end

            # (4) ADVANCE the real MCS energy + position. We advance the battery by the
            # APPLIED interval's realized FLOWS (not by reading SOE_MCS[g0+1], which the
            # model resets to full at a night boundary). Position resets to the grid overnight.
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

            # Advance each excavator's REAL energy using the TRUE powers over the real mix.
            for e in d.E
                charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T
                work_true = dot(a_real[e], d.true_powers)
                soe_cev[e] = clamp(soe_cev[e] + charged - work_true, d.SOE_CEV_min[e], d.SOE_CEV_max[e])
            end

            # Update remaining/cumulative work using the REALIZED durations (per excavator).
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

            # Track the day's peak grid draw (so demand charges reflect the whole day).
            peak_nc = max(peak_nc, grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, grid_kW))

            # realized average work power this interval (for logging/plots)
            work_kW = sum(dot(a_real[e], d.true_powers) for e in d.E) / d.delta_T

            # Append the detailed analyst row for this interval.
            push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        grid_kW, dch_kW, work_kW,
                        soe_mcs[1], _cev(soe_cev, 1), _cev(soe_cev, 2), cur_node,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4],
                        n_obs_total))
        end

        # Snapshot unfinished work at the end of the LAST kept day (before the buffer day
        # can mop it up) — this is the "missed work" we report.
        day == n_days_keep && (missed_kept = sum(rem_dig) + sum(rem_load))

        # ---- end-of-day: overnight smart-charge, then MCS starts next day recharged ----
        ov_df, _, _ = phase2_overnight_charge(d, soe_mcs)
        overnight_by_day[day] = ov_df
        soe_mcs = copy(float.(d.SOE_MCS_ini))          # MCS restored overnight (ready for next day)

        replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act)
    end
    # =========================== END OF THE CLOSED LOOP ===========================

    n_obs_total > 0 && refit!(est)          # one final learner update on all data
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

    # ---- KPIs computed from the KEPT-day trajectory ----
    total_energy = sum(klog.grid_kW) * d.delta_T                       # kWh bought during the day(s)
    total_cost   = sum(klog.grid_kW .* klog.price) * d.delta_T         # $ for that energy
    total_co2    = sum(klog.grid_kW .* klog.co2)  * d.delta_T          # carbon of that energy
    nc_peak      = isempty(klog.grid_kW) ? 0.0 : maximum(klog.grid_kW) # whole-horizon peak draw
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in klog.k]  # which rows are on-peak (within-day clock)
    op_peak      = any(op_mask) ? maximum(klog.grid_kW[op_mask]) : 0.0 # on-peak peak draw
    missed       = missed_kept                                         # unfinished work at end of last kept day
    transit_intervals = count(==(0), klog.mcs_node)                    # MCS towing time
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals

    # Overnight recharge cost/energy summed over the KEPT days only (each night's Phase-2).
    overnight_energy = 0.0; overnight_cost = 0.0
    for day in 1:n_days_keep
        ov = overnight_by_day[day]
        for m in d.M
            col = ov[!, Symbol("MCS$(m)_charge_kW")]
            overnight_energy += sum(col) * d.delta_T
            overnight_cost   += sum(col .* ov.price) * d.delta_T
        end
    end

    # ---- print a human-readable KPI summary ----
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

    # ---- write all outputs (KEPT days only) ----
    mkpath(out_dir)
    # (a) detailed ANALYST trajectory (buffer day dropped).
    CSV.write(joinpath(out_dir, "closed_loop_trajectory.csv"), klog)
    # (a2) overnight MCS smart-charge schedule, one file per kept day.
    for day in 1:n_days_keep
        CSV.write(joinpath(out_dir, "overnight_mcs_charge_day$(day).csv"), overnight_by_day[day])
    end
    # (b) the simple worker schedule (kept days only).
    fe = DataFrame(time = fe_time[1:n_kept_steps])
    for e in d.E
        fe[!, Symbol("CEV$(e)_activity")]       = fe_act[e][1:n_kept_steps]
        fe[!, Symbol("CEV$(e)_plug_in_charge")] = fe_chg[e][1:n_kept_steps]
    end
    fe[!, :MCS_charge_from_grid] = fe_mcs[1:n_kept_steps]
    CSV.write(joinpath(out_dir, "worker_schedule.csv"), fe)
    # (c) the replanning grids, one subfolder per kept day (CSV + coloured HTML each).
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
    make_plots(d, klog, out_dir)                                       # (d) the PNG figures (kept days)
    println("\nResults written to: $(abspath(out_dir))")
    println("  - worker_schedule.csv         (simple, for site workers; kept days)")
    println("  - closed_loop_trajectory.csv  (detailed; kept days, buffer dropped)")
    println("  - overnight_mcs_charge_day*.csv (Phase 2 overnight per kept day)")
    println("  - replan_grids/day*/*.csv     (per-step forward plans + replanning)")
    return klog
end

# Safe accessor for logging: return v[i] if it exists, else NaN (so datasets with
# a different number of excavators don't crash the fixed-width log).
_cev(v, i) = i <= length(v) ? v[i] : NaN

# ---- replanning-grid cell formatting + writers ------------------------------
_cell(v::AbstractString) = v                              # strings pass through
_cell(v::Real) = isnan(v) ? "" : round(v, digits = 3)     # numbers: blank if NaN, else rounded

# Write one replanning grid to CSV. Rows are re-plan steps (labelled by the clock
# at k0); columns are intervals (labelled by their clock). For each cell we show:
#   * k <  k0 (PAST):   the value already APPLIED at interval k (the diagonal mat[k,k]) — now fixed.
#   * k == k0 (diagonal): the decision applied to the plant this step.
#   * k >  k0 (FUTURE):  the fresh forward plan made at step k0 for interval k.
# So each row reads "already-fixed past + newly re-planned future". The inner loop
# builds one column at a time across all re-plan rows.
function write_replan_grid(path, mat, d, nK)
    df = DataFrame(replan_at = [clock_label(d, k0) for k0 in 1:nK])
    for k in 1:nK
        df[!, Symbol(clock_label(d, k))] =
            Any[_cell(k < k0 ? mat[k, k] : mat[k0, k]) for k0 in 1:nK]
    end
    CSV.write(path, df)
    write_replan_grid_html(replace(path, r"\.csv$" => ".html"), mat, d, nK)   # also write the coloured view
end

# Write the same grid as a coloured HTML table (nicer to look at in a browser):
#   GREEN  = a past interval already fixed (k < k0)
#   YELLOW = the current/applied step and the forward plan (k >= k0)
# Blank cells (an infeasible step's un-planned future) stay uncoloured. The two
# nested loops emit the header row of interval labels, then one table row per
# re-plan step with each cell shaded by its class.
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
        print(io, "<th>", clock_label(d, k), "</th>")           # column header per interval
    end
    println(io, "</tr>")
    for k0 in 1:nK
        print(io, "<tr><th>", clock_label(d, k0), "</th>")      # row header per re-plan step
        for k in 1:nK
            cell = _cell(k < k0 ? mat[k, k] : mat[k0, k])       # past->fixed value, else planned value
            cls  = cell == "" ? "" : (k < k0 ? "done" : "pend") # colour class
            print(io, "<td class=\"", cls, "\">", cell, "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# Convert an interval index k0 into an "HH:MM" clock label at its start boundary.
function clock_label(d, k0)
    m = mod(Int(round(d.t_start * 60 + (k0 - 1) * d.delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

# Day-tagged clock label for the multi-day run, e.g. "D2 08:15".
clock_day_label(d, day, k0) = string("D", day, " ", clock_label(d, k0))

# =============================================================================
# 4. PLOTTING  (four PNGs summarising the run)
# =============================================================================
function make_plots(d, log, out_dir)
    # x-axis = the continuous global step across the kept days (falls back to k if absent).
    x = (:gstep in propertynames(log)) ? log.gstep : log.k

    # Figure 1: grid draw with the electricity price overlaid on a second axis.
    p1 = plot(x, log.grid_kW, label = "Grid charging (kW)", lw = 2, color = :steelblue,
              xlabel = "Interval", ylabel = "Power (kW)", title = "Scenario 1: closed-loop grid draw")
    plot!(twinx(), x, log.price, label = "Price (\$/kWh)", lw = 2, color = :red, ylabel = "Price (\$/kWh)")
    savefig(p1, joinpath(out_dir, "01_grid_draw_vs_price.png"))

    # Figure 2: state-of-energy trajectories for the MCS and both excavators.
    p2 = plot(x, log.soe_mcs, label = "MCS SOE", lw = 2,
              xlabel = "Interval", ylabel = "SOE (kWh)", title = "Scenario 1: state of energy")
    plot!(p2, x, log.soe_cev1, label = "CEV 1 SOE", lw = 2)
    plot!(p2, x, log.soe_cev2, label = "CEV 2 SOE", lw = 2)
    savefig(p2, joinpath(out_dir, "02_state_of_energy.png"))

    # Figure 3: total realized work power over the day.
    p3 = plot(x, log.work_kW, label = "Total work (kW)", lw = 2, color = :forestgreen,
              xlabel = "Interval", ylabel = "Power (kW)", title = "Scenario 1: CEV work power")
    savefig(p3, joinpath(out_dir, "03_work_power.png"))

    # Figure 4: the LEARNING in action — each activity's estimate (with an
    # uncertainty ribbon) converging toward the hidden true power (dashed line).
    # The loop draws one estimate-vs-truth pair per activity.
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
# Run automatically when the file is launched (command line or an editor's Run
# button). A test harness can `include` this file WITHOUT auto-running by defining
# `SCENARIO1_NO_AUTORUN = true` beforehand.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
