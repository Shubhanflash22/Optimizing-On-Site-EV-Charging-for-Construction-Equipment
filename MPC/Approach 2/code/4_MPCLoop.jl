# #############################################################################
# 4_MPCLoop.jl  -  module MPCLoop
# -----------------------------------------------------------------------------
# The closed-loop shrinking-horizon MPC driver and the simulated plant it controls.
# At every 15-minute interval it draws a fresh set of power scenarios from 2b_ScenarioSampler.jl, solves the stochastic window MILP of 3_MCSModel.jl from the real state over the rest of the day, applies only the first interval to the plant, and advances the real state.
# The plant draws the realized CEV activity power from the shared sample pool, or pins it to the planning mean when the plant is deterministic.
# The planning power constants are the fitted means and standard deviations from parameters.csv, and they are never refitted during the run.
# The scenarios are drawn around these means, and the first interval of the plan is the same in every scenario.
# Seven groups:
#
#   1. ACTIVITY AND STATUS LABELS
#      ACT_NAME, applied_act_index, activity_label, serving_mcs, mcs_status_label, mcs_node_label
#      -- the activity names, the index of the activity the MILP scheduled for a CEV, the MCS serving a CEV, and the text labels used in the plan grids and the detailed logs.
#   2. REALIZED ACTIVITY SPLIT
#      realized_activity_durations
#      -- split one interval of a CEV into hours of digging, loading+swinging, traveling and idling.
#
#   3. MCS STATE ADVANCE
#      advance_mcs_state
#      -- find where an MCS will be at the start of the next interval, either parked at a node or part-way along a trip.
#
#   4. PLANT STEP
#      apply_and_simulate!
#      -- apply the first interval of the plan, draw the realized power, update the CEV and MCS energy, the MCS position, the remaining work and the activity history.
#
#   5. SCENARIO VIEW
#      _ScenarioOneView, _ScenarioOneVar
#      -- a read-only view of the solved stochastic model that shows one scenario's variables with the scenario index hidden, so the plant step and the label helpers can read the model as if it had a single scenario.
#
#   6. CLOSED LOOP
#      run_mpc
#      -- run the day by day, interval by interval loop, log the plans and the realized trajectory, and compute the KPIs returned to 5_Output.jl.
#      The plan grids and the planned power in the logs are read from scenario min(3, n_scenarios), which is the near-average scenario when there are 5.
#
#   7. TERMINAL SOE SHORTFALL
#      _terminal_soe_shortfall
#      -- compute the end-of-run CEV energy shortfall against its initial SOE and the penalty for it.
#      This penalty is added to the total cost outside objective function (4) of the paper's MILP formulation, and it is zero when the plant is deterministic.
# #############################################################################
module MPCLoop

# external packages used across this file
using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, build_time_labels, build_time_labels_days, safe_get,
                BayesianActivityEstimator,
                ActivityPowerPool, new_cursor, next_power!,
                DetailedPlanLog, RealizedTupleLog, log_plan_row!, log_realized_row!,
                MCSPlanLog, MCSRealizedLog, log_mcs_plan_row!, log_mcs_realized_row!,
                to_dataframe
using ..MCSModel: build_window_model_stochastic
using ..ScenarioSampler: sample_scenarios, equal_weights, DEFAULT_N_SCENARIOS

# everything below that other files are allowed to use
export run_mpc

# Fixed activity index-to-name mapping (dig, load+swing, travel, idle), matching the same 4-activity ordering assumed elsewhere.
const ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# Returns the time in hours a CEV spends on [digging, loading+swinging, traveling, idling] during the current interval k0, summing to delta_T.
# delta_T is 0.25 h, which is 15 minutes, so 0.25 means the whole interval.
# k0 is the first interval of the current window, which is the interval the plant is applying.
# With no activity scheduled the whole interval is idle, with multi off the whole interval goes to the scheduled activity, and with multi on a random 60 to 100% goes to it and the rest is idle.
function realized_activity_durations(rng, model, e, k0, d; multi::Bool = true)
    dt = d.delta_T
    a = zeros(length(d.B))
    idle = length(d.B)
    planned = 0
    for i in d.N_c, (ai, act) in enumerate(d.B)
        if value(model[:u][e, i, act, k0]) > 0.5
            planned = ai
        end
    end
    planned == 0 && (a[idle] = dt; return a)
    !multi && (a[planned] = dt; return a)
    frac = 0.6 + 0.4 * rand(rng)
    a[planned] += dt * frac
    a[idle]    += dt * (1.0 - frac)
    return a
end

# Returns where MCS m will be at the start of interval k0+1 as (node, transit), using only what has been decided up to and including interval k0.
# An MCS that is parked in interval k0 is reported as parked at that node even if the plan has it leaving in interval k0+1, because that departure is for the next re-solve to decide.
# node is the parked node index, or 0 if the MCS is mid-trip, and transit is nothing or (i, j, r) for a trip from node i to node j with r intervals still to go.
# If there is no next interval in the window, it keeps the node at k0, using the first grid node if no parked node is found.
function advance_mcs_state(model, m, k0, nK, d)
    z = model[:z]; y = model[:y_trv]
    Kw = axes(z)[3]
    knext = k0 + 1
    if knext > nK || !(knext in Kw)
        node = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)
        return (node === nothing ? first(d.N_g) : node, nothing)
    end
    node_now = findfirst(i -> value(z[m, i, k0]) > 0.5, d.N)
    node_now !== nothing && return (node_now, nothing)
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

# Returns the index in d.B of the activity the plan scheduled for CEV e at interval k0, which is 1 for digging, 2 for loading+swinging, 3 for traveling and 4 for idle.
# Returns 4 when nothing is scheduled, which only happens off-shift, so off-shift time is recorded in the idle slot of the history and the hour buckets.
# The label for that case is "Off", which run_mpc sets separately.
function applied_act_index(model, d, e, k0)
    for i in d.N_c, (ai, act) in enumerate(d.B)
        value(model[:u][e, i, act, k0]) > 0.5 && return ai
    end
    return length(d.B)   
end

# Returns the text label of CEV e at its site in interval k of the model, used for the plan grids and the detailed plan log.
# The label is "Charging" if any MCS delivers power to the CEV, otherwise "Off" if no activity is scheduled, otherwise the name of the scheduled activity.
function activity_label(model, d, e, site, k)
    vals   = [value(model[:u][e, site, a, k]) for a in eachindex(d.B)]
    p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
    return p_into > 1e-6 ? "Charging" : (sum(vals) < 0.5 ? "Off" : ACT_NAME[d.B[argmax(vals)]])
end

# Returns the index of the MCS delivering power to CEV e at the given site during interval k, or nothing if no MCS is connected to it.
# At most one MCS can serve a CEV in an interval (constraint 7c), so the first MCS with nonzero delivered power is the serving one.
function serving_mcs(model, d, site, e, k)
    for m in d.M
        value(model[:P_MCS_CEV][m, site, e, k]) > 1e-6 && return m
    end
    return nothing
end

# Returns the text status of MCS m in interval k, checked in this order: "Charging (grid)" if it draws grid power, "Serving CEV" if it discharges, "Traveling" if it is not parked at any node, otherwise "Idle".
function mcs_status_label(model, d, m, k)
    pch    = value(model[:P_ch_tot][m, k])
    pdch   = value(model[:P_dch_tot][m, k])
    parked = any(value(model[:z][m, i, k]) > 0.5 for i in d.N)
    return pch  > 1e-6 ? "Charging (grid)" :
           pdch > 1e-6 ? "Serving CEV"     :
           !parked     ? "Traveling"       : "Idle"
end

# Returns the node index where MCS m is parked in interval k as a string, or "Transit" if it is not parked at any node.
function mcs_node_label(model, d, m, k)
    node = findfirst(i -> value(model[:z][m, i, k]) > 0.5, d.N)
    return node === nothing ? "Transit" : string(node)
end

# Estimates a hidden end-of-run cost for CEVs that finish below their initial/target SOE (soe_cev_end < d.SOE_CEV_ini), converting the shortfall from kWh into an equivalent number of missed work-hours, so it can be priced using the same rho_miss ($/hour) rate as genuinely missed digging/loading work.
# Only counts a CEV that ended LOWER than its target (max(...,0.0)); ending higher is never penalized, consistent with the MILP's terminal SOE_CEV constraint being a >= inequality (10b).
# That relaxes the paper's exact equality, and the two are equivalent whenever SOE_CEV_ini equals SOE_CEV_max, as in the paper's Table IX.
# avg_work_power_kW is the power-weighted average of digging vs loading+swinging (which activity draws more of the "required" work hours matters more), used purely as a conversion factor from kWh to hours -- it isn't itself a modeled quantity from the paper, just this simulation's own way to translate an unmet energy promise into the paper's missed-work cost units.
function _terminal_soe_shortfall(d, soe_cev_end, rem_dig, rem_load, n_day_run::Int = 1)
    shortfall_kWh = sum(max(d.SOE_CEV_ini[e] - soe_cev_end[e], 0.0) for e in d.E; init = 0.0)


    required_dig_h  = n_day_run * sum(d.hours_digging)
    required_load_h = n_day_run * sum(d.hours_loading_swinging)
    required_h      = required_dig_h + required_load_h
    required_kWh    = required_dig_h * d.p_digging + required_load_h * d.p_loading_swinging
    avg_work_power_kW = required_h > 1e-9 ? required_kWh / required_h :
                                             (d.p_digging + d.p_loading_swinging) / 2

    shortfall_hours = shortfall_kWh / avg_work_power_kW
    shortfall_penalty_cost = d.rho_miss * shortfall_hours
    return (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost)
end

# Executes the first interval k0 of the freshly solved plan, read from a one-scenario view of the stochastic model, against the plant, and updates all the running simulation state in place: soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist, cursor, and the real_* logging arrays.
# The plant draws the realized CEV power from the shared sample pool (plant_mode :sampled) or pins it to the planning mean (plant_mode :mean).
# Returns a NamedTuple summarizing what happened this interval, for run_mpc to log and accumulate.
function apply_and_simulate!(model, k0, nK, d, pool::ActivityPowerPool, cursor, rng, multi_activity,
                             soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                             real_P_ch, real_P_dch, real_loc, real_P_work;
                             plant_mode::Symbol = :sampled, gidx::Int = k0)
    plant_mode in (:sampled, :mean) ||
        error("apply_and_simulate!: plant_mode must be :sampled or :mean, got :$plant_mode")
    use_mean = plant_mode === :mean

    # Reads the plan's total grid charge/discharge power for this interval (summed over all MCSs), and records each MCS's individual charge/discharge power and physical node into the per-MCS, per-interval logging arrays.
    grid_kW = sum(value(model[:P_ch_tot][m, k0]) for m in d.M)
    dch_kW  = sum(value(model[:P_dch_tot][m, k0]) for m in d.M)
    for m in d.M
        real_P_ch[m, gidx]  = value(model[:P_ch_tot][m, k0])
        real_P_dch[m, gidx] = value(model[:P_dch_tot][m, k0])
        real_loc[m, gidx]   = let nh = findfirst(i -> value(model[:z][m, i, k0]) > 0.5, d.N)
            nh === nothing ? 0 : nh
        end
    end

    # Draws each CEV's realized activity durations from the fixed plan (see realized_activity_durations), then draws the REAL power for each active activity from the shared pool -- use_mean pins it to the planning mean from parameters.csv (deterministic reference mode, where realized equals planned), otherwise it takes the next sample from the pool.
    # Off-shift intervals are forced to zero power regardless of duration, since a CEV that isn't scheduled to work shouldn't draw work power even if realized_activity_durations somehow returned a nonzero row for it.
    # n_obs_added counts how many CEVs actually produced a new real observation this interval, for the Bayesian estimator's calibration bookkeeping (the estimator is never refitted in this loop, so it is only accumulated as a count, but it is kept for consistency with the shared pool machinery in Common.jl).
    a_real = Dict(e => realized_activity_durations(rng, model, e, k0, d;
                                                   multi = multi_activity && !use_mean) for e in d.E)
    p_true = Dict{Int, Vector{Float64}}()
    n_obs_added = 0
    for e in d.E
        row = a_real[e]
        pt  = zeros(length(row))
        site = findfirst(i -> d.A[i, e] == 1, d.N)
        working = site !== nothing && d.is_working[site, e, k0]
        for a in eachindex(row)
            row[a] > 1e-9 || continue
            if !working
                pt[a] = 0.0   
                continue
            end
            pt[a] = use_mean ? d.prior_mu[a] : next_power!(pool, cursor, e, a)
        end
        p_true[e] = pt
        sum(row) > 1e-9 && working && (n_obs_added += 1)
    end

    # Advances each MCS's SOE by the plan's change over this interval (plan SOE at k0+1 minus plan SOE at k0), added to its current realized SOE and clamped to its SOE bounds.
    # The window's first SOE is pinned to the realized SOE, so this equals the plan's SOE at k0+1, and the clamp is only a safety net.
    # The CEV overflow refund in the next block is added on top of this value.
    # Also updates the MCS node and transit state for the start of the next interval via advance_mcs_state.
    for m in d.M
        soe_mcs[m] = clamp(soe_mcs[m] + value(model[:SOE_MCS][m, k0 + 1]) - value(model[:SOE_MCS][m, k0]),
                           d.SOE_MCS_min[m], d.SOE_MCS_max[m])
        mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, k0, nK, d)
    end

    # For each CEV: computes how much energy it actually received this interval (raw_by_mcs/total_raw/charged), and how much energy its realized activities would actually cost (work_true).
    # If the realized work would drain the CEV below SOE_CEV_min (work_true > headroom), scales down the realized durations proportionally and dumps the freed time into idle -- this is the SOE-floor capping (capped[e]), reflecting that a CEV physically cannot keep working once its battery is empty, regardless of what the plan assumed.
    # Updates soe_cev with the (possibly capped) realized work.
    # If that pushes the CEV over SOE_CEV_max, the surplus energy is refunded to the MCS(s) that supplied it, in proportion to their share of the delivery.
    # This happens when the CEV drew less work energy than planned in earlier intervals, so its SOE has run ahead of the plan and the planned charge no longer fits.
    # The refund is converted to MCS battery energy by dividing by the MCS discharge efficiency (matching the discharge term of constraint 9a), and is capped at SOE_MCS_max.
    # Finally clamps soe_cev defensively into bounds.
    n_capped = 0
    capped = Dict{Int,Bool}()
    for e in d.E
        capped[e] = false
        raw_by_mcs = Dict(m => sum(value(model[:P_MCS_CEV][m, i, e, k0]) for i in d.N_c) * d.delta_T for m in d.M)
        total_raw  = sum(values(raw_by_mcs))
        charged    = d.eta_ch_dch_cev[e] * total_raw
        headroom  = soe_cev[e] + charged - d.SOE_CEV_min[e]      
        work_true = dot(a_real[e], p_true[e])                    

        # If the realized work this CEV would do (work_true) exceeds what its remaining headroom to SOE_CEV_min can cover, scales digging/loading/traveling down proportionally so exactly the affordable amount gets done, and pushes the rest of the interval into idle.
        # Recomputes work_true from only the scaled dig/load/travel durations (indices 1:3), deliberately excluding idle -- so the leftover time is labeled idle for logging/history purposes, but costs zero energy regardless of p_idling, since the CEV has already run out of battery and physically cannot keep drawing power.
        if work_true > headroom && work_true > 1e-9
            scale = max(headroom, 0.0) / work_true                
            a_real[e][1] *= scale                                 
            a_real[e][2] *= scale                                 
            a_real[e][3] *= scale                                
            a_real[e][4] = d.delta_T - sum(a_real[e][1:3])       
            work_true = dot(a_real[e][1:3], p_true[e][1:3])          
            n_capped += 1
            capped[e] = true
        end

        soe_cev[e] = soe_cev[e] + charged - work_true
        overflow = soe_cev[e] - d.SOE_CEV_max[e]
        if overflow > 1e-9 && total_raw > 1e-9
            overflow_raw = overflow / d.eta_ch_dch_cev[e]
            for m in d.M
                share = raw_by_mcs[m] / total_raw
                soe_mcs[m] = min(soe_mcs[m] + overflow_raw * share / d.eta_ch_dch_mcs[m], d.SOE_MCS_max[m])
            end
        end
        soe_cev[e] = clamp(soe_cev[e], d.SOE_CEV_min[e], d.SOE_CEV_max[e])  
    end

    # Deducts the realized digging/loading hours from each site's remaining work backlog (rem_dig/rem_load), records this interval's applied activity + realized durations into hist[e] (consumed by the next window's build_window_model_stochastic call for the cumulative work and rest history, and reset at the start of each day), and records the realized work power into real_P_work for the CEV's assigned site.
    for e in d.E
        site_e = findfirst(i -> d.A[i, e] == 1, d.N)
        if site_e !== nothing
            rem_dig[site_e]  = max(rem_dig[site_e]  - a_real[e][1], 0.0)
            rem_load[site_e] = max(rem_load[site_e] - a_real[e][2], 0.0)
        end
        push!(hist[e], (applied_act_index(model, d, e, k0), copy(a_real[e])))
        site_e = findfirst(i -> d.A[i, e] == 1, d.N)
        if site_e !== nothing
            real_P_work[site_e, e, gidx] =
                (a_real[e][1]*p_true[e][1] + a_real[e][2]*p_true[e][2] +
                 a_real[e][3]*p_true[e][3] + a_real[e][4]*p_true[e][4]) / d.delta_T
        end
    end

    # Total realized work power across all CEVs this interval, for the plain log's work_kW column.
    work_kW = sum(dot(a_real[e], p_true[e]) for e in d.E) / d.delta_T

    return (; grid_kW, dch_kW, a_real, p_true, n_obs_added, work_kW, n_capped, capped)
end

# A read-only view of a solved stochastic model that exposes one scenario, s_ref, as if the model had no scenario index.
# The plant step and the label helpers were written for a single-scenario model, so they read the stochastic model through this view without any change.
struct _ScenarioOneView
    model::Model
    s_ref::Int
end

# The slice of one variable family of the model at scenario s_ref, which hides the last (scenario) index of the variable.
struct _ScenarioOneVar
    var::Any
    s_ref::Int
end

# Returns the named variable of the wrapped model as a one-scenario slice, so model[:z] on the view behaves like model[:z] on a single-scenario model.
Base.getindex(v::_ScenarioOneView, sym::Symbol) = _ScenarioOneVar(v.model[sym], v.s_ref)

# Reads one element of the slice by appending the scenario index, so z[m, i, k] on the view is z[m, i, k, s_ref] on the model.
Base.getindex(v::_ScenarioOneVar, idx...) = v.var[idx..., v.s_ref]

# Returns the axes of the slice without the scenario axis, so code that asks for the interval axis, such as advance_mcs_state, gets the same answer as on a single-scenario model.
Base.axes(v::_ScenarioOneVar) = axes(v.var)[1:end-1]

# Returns one axis of the slice by position, which is only meaningful for the non-scenario positions.
Base.axes(v::_ScenarioOneVar, d::Int) = axes(v.var)[d]

# Runs the closed-loop shrinking-horizon MPC for n_day_run consecutive days at 15-minute steps, and returns the logs, the realized trajectories and the KPIs used by 5_Output.jl.
# At every interval it draws a fresh set of power scenarios, solves the stochastic window MILP from the realized state to the end of the day, applies only the first interval to the plant through apply_and_simulate!, logs what happened, and moves on to the next interval.
# plant is :sampled to draw the realized CEV power from pool, or :mean to pin it to the planning mean.
# time_limit_sec limits each window solve, and multi_activity splits each interval between the scheduled activity and idle.
# n_scenarios is the number of power scenarios drawn at every re-solve, and the plan grids and the planned power in the logs are read from scenario min(3, n_scenarios).
# n_day_run repeats the same input day and carries over the remaining work and the demand peaks.
# detailed_output also builds the four detailed plan and realized logs, and mcmc_samples is not used for anything here.
# The end-of-run CEV energy shortfall penalty is returned separately from total_cost.
function run_mpc(d, pool::ActivityPowerPool;
                    time_limit_sec::Float64 = Inf,
                    multi_activity::Bool = false,
                    mcmc_samples::Int = 500,
                    plant::Symbol = :sampled,
                    n_scenarios::Int = DEFAULT_N_SCENARIOS,
                    n_day_run::Int = 1,
                    seed::Int = 1,
                    detailed_output::Bool = false)
    plant in (:sampled, :mean) ||
        error("run_mpc: plant must be :sampled or :mean, got :$plant")
    n_scenarios >= 1 || error("run_mpc: n_scenarios must be >= 1, got $n_scenarios")
    n_day_run >= 1 || error("run_mpc: n_day_run must be >= 1, got $n_day_run")
    Random.seed!(seed)
    K_all = collect(d.K)
    nKd = length(K_all)                    
    n_kept = n_day_run * nKd               
    time_labels = n_day_run == 1 ? build_time_labels(d.t_start, d.delta_T, nKd) :
                                    build_time_labels_days(d.t_start, d.delta_T, n_day_run, nKd)
    cursor = new_cursor(pool)
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    mcs_node = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]
    nN_work  = length(d.hours_digging)
    rem_dig  = zeros(nN_work)
    rem_load = zeros(nN_work)
    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    peak_nc = 0.0; peak_op = 0.0

    # Creates the estimator, which only holds the planning power means and standard deviations, and the random generator that drives the activity split.
    # rng_scenarios is a second generator that drives the scenario sampling, so the scenarios never share a random stream with the plant.
    # s_plan is the scenario that the plan grids and the logs read from, which is scenario 3, or the last scenario when there are fewer than 3.
    # log is the plain per-interval summary of the realized run, with one SOE column per MCS and per CEV, one node column per MCS (0 means in transit), and the planning power means and standard deviations.
    # solve_log has one row per window solve with its status, objective, gap in percent and solve time.
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)
    rng_scenarios = MersenneTwister(seed + 1_000_000)
    s_plan = min(3, n_scenarios)

    log = DataFrame(day = Int[], k = Int[], clock = String[], price = Float64[], co2 = Float64[],
                    grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[])
    for m in d.M; log[!, Symbol("soe_mcs$m")]  = Float64[]; end
    for e in d.E; log[!, Symbol("soe_cev$e")]  = Float64[]; end
    for m in d.M; log[!, Symbol("mcs_node$m")] = Int[];     end
    log[!, :est_dig]  = Float64[]; log[!, :est_load] = Float64[]
    log[!, :est_trv]  = Float64[]; log[!, :est_idle] = Float64[]
    log[!, :unc_dig]  = Float64[]; log[!, :unc_load] = Float64[]
    log[!, :unc_trv]  = Float64[]; log[!, :unc_idle] = Float64[]
    log[!, :n_obs]    = Int[]
    solve_log = DataFrame(day = Int[], step = Int[], clock = String[], status = String[],
                          objective = Float64[], gap_percent = Float64[], solve_time_s = Float64[])

    # Pre-allocates the whole-run result arrays, with one column per interval across all days, indexed per MCS, per CEV, or per node as appropriate.
    # The two SOE arrays have one extra column to hold the end-of-run state.
    # real_mcs_act holds one status label vector per MCS, and replan_by_day will hold each day's re-plan grids.
    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, n_kept)
    real_P_dch = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)

    replan_by_day = Dict{Int, NamedTuple}()

    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = [fill("", n_kept) for _ in d.M]

    # The four detailed plan and realized logs exist only when detailed_output is true.
    plan_log         = detailed_output ? DetailedPlanLog()     : nothing
    realized_log     = detailed_output ? RealizedTupleLog()    : nothing
    mcs_plan_log     = detailed_output ? MCSPlanLog()          : nothing
    mcs_realized_log = detailed_output ? MCSRealizedLog()      : nothing

    # Prints a run header with the number of scenarios, the plant mode, the planning power, the plant sampling spread (sampled mode only) and the solver time limit.
    # Then starts the timer and the counters for plant realizations, infeasible windows and intervals capped by the SOE floor.
    println("Running Approach 2 (stochastic scenario-based MPC, 15-min steps, Shrinking horizon): $n_kept steps ($n_day_run day(s))")
    println("  scenarios / re-solve : ", n_scenarios, " (equal weights, resampled fresh every window, plan read from scenario ", s_plan, ")")
    println("  plant                : ", plant === :mean ?
            ":mean (DETERMINISTIC -- realized power pinned to mu)" :
            ":sampled (stochastic -- realized power drawn from the shared pool)")
    println("  planning power (mu)  : ", round.(est.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd    : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit    : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s / window" : "none (solve each window to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_infeasible = 0
    n_capped_total = 0
    day_snapshot_rows = NamedTuple[]

    for day in 1:n_day_run
        # At the start of each day, resets the CEV activity history, adds the day's required digging and loading hours onto any backlog carried over from earlier days, and allocates the day's re-plan grids.
        # The re-plan grids hold, for each re-solve interval k0 and each interval k of its window, the planned grid power, the planned SOE of every MCS and every CEV, and the planned status of every MCS and activity of every CEV.
        hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
        rem_dig  .+= float.(d.hours_digging)
        rem_load .+= float.(d.hours_loading_swinging)

        plan_grid_kW = fill(NaN, nKd, nKd)
        plan_mcs_soe = [fill(NaN, nKd, nKd) for _ in d.M]
        plan_mcs_act = [fill("", nKd, nKd)  for _ in d.M]
        plan_cev_soe = [fill(NaN, nKd, nKd) for _ in d.E]
        plan_cev_act = [fill("", nKd, nKd)  for _ in d.E]

        for k0 in 1:nKd  
            # At every interval k0, records the realized SOE of every battery at the start of the interval, draws a fresh set of power scenarios, and solves the stochastic window MILP from k0 to the end of the day starting from the realized state.
            # gidx converts the day-local interval k0 into the index across all days used by the whole-run arrays.                   
            gidx = (day - 1) * nKd + k0       

            for m in d.M; real_SOE_MCS[m, gidx] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, gidx] = soe_cev[e]; end

            K_win = k0:nKd

            scenarios = sample_scenarios(est.mu, est.sd, n_scenarios; rng = rng_scenarios)
            weights   = equal_weights(n_scenarios)

            model = build_window_model_stochastic(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                       rem_dig, rem_load, hist,
                                       peak_nc, peak_op, scenarios, weights;
                                       time_limit_sec = time_limit_sec)
            stat = string(termination_status(model))

            # If the window has no solution, the plant holds for this interval: nothing is charged or discharged, no work is done, every MCS stays where it is, and every CEV is recorded as idle for the rest rule.
            # The interval is counted in n_infeasible, logged, and skipped.
            if !has_values(model)
                n_infeasible += 1
                @warn "No feasible solution at day=$day, step k=$k0 under HARD constraints; holding state (no fallback)." status=stat
                # Records the status and solve time of this window, with no objective or gap.
                push!(solve_log, (day, k0, clock_label(d.t_start, d.delta_T, k0), stat, NaN, NaN,
                                  try solve_time(model) catch; NaN end))
                # Holds the realized state for this interval: every MCS stays at its node, and the plain log gets a row with no grid power, no discharge and no work at the unchanged SOE.
                for m in d.M; real_loc[m, gidx] = mcs_node[m]; end
                    push!(log, (day, k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                                0.0, 0.0, 0.0,
                                (soe_mcs[m] for m in d.M)...,
                                (soe_cev[e] for e in d.E)...,
                                (real_loc[m, gidx] for m in d.M)...,
                                est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                                est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
                # Labels every CEV and every MCS as "Idle" for this interval.
                # Records a full idle interval in each CEV's history, which counts as a rest for the rest rule.
                for e in d.E; real_cev_act[e][gidx] = "Idle"; end
                for m in d.M; real_mcs_act[m][gidx] = "Idle"; end
                for e in d.E; push!(hist[e], (length(d.B), [0.0, 0.0, 0.0, d.delta_T])); end
                # When detailed_output is on, logs this interval's realized rows with zero power: the "No plan (infeasible)" label and the infeasible flag for every CEV, and the idle status for every MCS.
                # The MCS SOE in the CEV rows is NaN, because no MCS serves any CEV in a held interval.
                if detailed_output
                    for e in d.E
                        log_realized_row!(realized_log, d; day, step = k0, cev = e,
                                          p_tuple = [0.0, 0.0, 0.0, 0.0], activity_executed = "Idle",
                                          activity_planned = "No plan (infeasible)", planned_power_kW = 0.0,
                                          soe_cev_kWh = safe_get(soe_cev, e), soe_mcs_kWh = NaN,
                                          infeasible_flag = true)
                    end
                    for m in d.M
                        log_mcs_realized_row!(mcs_realized_log, d; day, step = k0, mcs = m,
                                              mcs_status_realized = "Idle",
                                              mcs_node_realized = mcs_node[m] == 0 ? "Transit" : string(mcs_node[m]),
                                              grid_charge_kW_realized = 0.0,
                                              grid_discharge_kW_realized = 0.0,
                                              soe_mcs_kWh = soe_mcs[m])
                    end
                end
                continue
            end

            # Records the solver status, objective, gap in percent and solve time of this window.
            push!(solve_log, (day, k0, clock_label(d.t_start, d.delta_T, k0), stat, objective_value(model),
                              100 * (try relative_gap(model) catch; NaN end),
                              try solve_time(model) catch; NaN end))

            # Makes the one-scenario view of the solved model at scenario s_plan, through which the plan grids, the logs and the plant step read it.
            # The raw model is only used for the solver status, the objective, the gap and the solve time.
            vmodel = _ScenarioOneView(model, s_plan)

            # Stores the window's plan from scenario s_plan for the re-plan grids, for every interval k from k0 to the end of the day: total grid power, and the SOE and status or activity of every MCS and every CEV.
            for k in K_win
                plan_grid_kW[k0, k] = sum(value(vmodel[:P_ch_tot][m, k]) for m in d.M)
                for m in d.M; plan_mcs_soe[m][k0, k] = value(vmodel[:SOE_MCS][m, k + 1]); end
                for e in d.E
                    plan_cev_soe[e][k0, k] = value(vmodel[:SOE_CEV][e, k + 1])
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    if site !== nothing
                        plan_cev_act[e][k0, k] = activity_label(vmodel, d, e, site, k)
                    end
                end
                for m in d.M; plan_mcs_act[m][k0, k] = mcs_status_label(vmodel, d, m, k); end
            end

            # When detailed_output is on, logs one plan row per CEV and one per MCS for every interval of the window, for every scenario, with scenario_id saying which scenario the row comes from.
            if detailed_output
                for s in 1:n_scenarios
                    svmodel = _ScenarioOneView(model, s)
                    for k in K_win, e in d.E
                        site = findfirst(i -> d.A[i, e] == 1, d.N)
                        site === nothing && continue
                        p_into = sum(value(svmodel[:P_MCS_CEV][m, site, e, k]) for m in d.M)
                        m_srv = serving_mcs(svmodel, d, site, e, k)
                        log_plan_row!(plan_log, d; day, resolve_step = k0, offset_step = k,
                                      activity_planned = activity_label(svmodel, d, e, site, k),
                                      planned_power_kW = value(svmodel[:P_work][site, e, k]),
                                      planned_charging = p_into > 1e-6,
                                      soe_cev_planned_kWh = value(svmodel[:SOE_CEV][e, k + 1]),
                                      soe_mcs_planned_kWh = m_srv === nothing ? NaN : value(svmodel[:SOE_MCS][m_srv, k + 1]),
                                      cev = e, scenario_id = s)
                    end
                    for k in K_win, m in d.M
                        log_mcs_plan_row!(mcs_plan_log, d; day, resolve_step = k0, offset_step = k,
                                          mcs = m, mcs_status_planned = mcs_status_label(svmodel, d, m, k),
                                          mcs_node_planned = mcs_node_label(svmodel, d, m, k),
                                          grid_charge_kW_planned = value(svmodel[:P_ch_tot][m, k]),
                                          grid_discharge_kW_planned = value(svmodel[:P_dch_tot][m, k]),
                                          soe_mcs_planned_kWh = value(svmodel[:SOE_MCS][m, k + 1]),
                                          scenario_id = s)
                    end
                end
            end

            # Records each MCS's status for this interval from the plan, because the plant never changes what an MCS does.
            for m in d.M; real_mcs_act[m][gidx] = plan_mcs_act[m][k0, k0]; end

            # Applies the first interval of the plan to the plant, which updates the realized state in place.
            # Then adds this interval's plant realizations and capped CEVs to the running totals.
            step = apply_and_simulate!(vmodel, k0, nKd, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_loc, real_P_work;
                                       plant_mode = plant, gidx = gidx)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

            # Builds each CEV's realized label for this interval, adding the realized minutes when they are less than the full interval, for example "Digging (10 min)" when the SOE floor capped the work.
            for e in d.E
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                idx = applied_act_index(vmodel, d, e, k0)
                p_into = site !== nothing ? sum(value(vmodel[:P_MCS_CEV][m, site, e, k0]) for m in d.M) : 0.0
                planned_label = p_into > 1e-6 ? "Charging" :
                                (site !== nothing && !d.is_working[site, e, k0]) ? "Off" : ACT_NAME[idx]
                realized_min  = round(Int, step.a_real[e][idx] * 60)
                full_min      = round(Int, d.delta_T * 60)
                real_cev_act[e][gidx] = realized_min < full_min ? "$(planned_label) ($(realized_min) min)" : planned_label
            end

            # When detailed_output is on, logs this interval's realized row for every CEV and every MCS.
            # The CEV row carries planned and realized values side by side, the flag for work blocked by the SOE floor, and the SOE of the MCS serving that CEV, or NaN when none is connected.
            if detailed_output
                for e in d.E
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    m_srv = site === nothing ? nothing : serving_mcs(vmodel, d, site, e, k0)
                    log_realized_row!(realized_log, d; day, step = k0, cev = e,
                                    p_tuple = step.p_true[e], activity_executed = real_cev_act[e][gidx],
                                    activity_planned = site === nothing ? "Off" : plan_cev_act[e][k0, k0],
                                    planned_power_kW = site === nothing ? 0.0 : value(vmodel[:P_work][site, e, k0]),
                                    soe_cev_kWh = safe_get(soe_cev, e), soe_mcs_kWh = m_srv === nothing ? NaN : soe_mcs[m_srv],
                                    infeasible_flag = false,
                                    planned_activity_blocked_by_min_soe = step.capped[e])
                end
                for m in d.M
                    log_mcs_realized_row!(mcs_realized_log, d; day, step = k0, mcs = m,
                                          mcs_status_realized = mcs_status_label(vmodel, d, m, k0),
                                          mcs_node_realized = real_loc[m, gidx] == 0 ? "Transit" : string(real_loc[m, gidx]),
                                          grid_charge_kW_realized = real_P_ch[m, gidx],
                                          grid_discharge_kW_realized = real_P_dch[m, gidx],
                                          soe_mcs_kWh = soe_mcs[m])
                end
            end

            # Updates the running non-coincident demand peak, and the on-peak demand peak when this interval is on-peak, from the total grid power over all MCSs.
            # Then appends this interval's row to the plain log, with the SOE of every MCS and every CEV after the interval and the node of every MCS during it.
            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            push!(log, (day, k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        (soe_mcs[m] for m in d.M)...,
                        (soe_cev[e] for e in d.E)...,
                        (real_loc[m, gidx] for m in d.M)...,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
        end
        # At the end of each day, records that day's MCS transit hours and travel labour cost across all MCSs, plus the cumulative missed work and terminal shortfall so far.
        # These rows form the per-day KPI table day_snapshots_df.
        day_cols = (day - 1) * nKd + 1 : day * nKd
        transit_hours_day = count(==(0), real_loc[:, day_cols]) * d.delta_T
        labour_cost_day   = d.rho_labor * transit_hours_day
        (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
            _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, day)
        push!(day_snapshot_rows, (; day,
                                     missed_work_cumulative_h = sum(rem_dig) + sum(rem_load),
                                     shortfall_kWh_cumulative = shortfall_kWh,
                                     shortfall_penalty_cost_cumulative = shortfall_penalty_cost,
                                     transit_hours_day, labour_cost_day))

        # Saves the day's re-plan grids, which 5_Output.jl writes out.
        replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act, plan_mcs_act)
    end

    # Records the end-of-run SOE of every battery in the extra last column of the SOE arrays.
    for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end

    # Prints the completion summary, with notes on any infeasible windows and on intervals capped by the SOE floor, and the planning power and plant sampling spread.
    elapsed = time() - t0
    @printf("Approach 2 (stochastic, %d scenarios, plant = :%s) done in %.1f s (%d plant realizations, %d day(s))\n",
            n_scenarios, plant, elapsed, n_obs_total, n_day_run)
    n_infeasible > 0 && @printf("  NOTE: %d/%d windows were INFEASIBLE under the HARD constraints (no fallback);\n        the plant HELD state for those intervals.\n", n_infeasible, n_kept)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy (task could not fully\n        complete before hitting the SOE floor); the shortfall is reflected honestly in\n        rem_dig/rem_load.\n", n_capped_total)
    println("  fixed power model (mu) : ", round.(est.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")

    # Computes the whole-run KPIs behind the terms of objective function (4) of the paper's MILP formulation.
    # Grid energy, energy cost, CO2 and the non-coincident and on-peak peaks come from the plain log, missed work from the remaining backlog, and MCS transit and its labour cost from real_loc over all MCSs.
    # 5_Output.jl prices carbon, demand charges, missed work, travel and the shortfall into the cost components.
    total_energy = sum(log.grid_kW) * d.delta_T
    total_cost   = sum(log.grid_kW .* log.price) * d.delta_T
    total_co2    = sum(log.grid_kW .* log.co2)  * d.delta_T
    nc_peak      = isempty(log.grid_kW) ? 0.0 : maximum(log.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in log.k]
    op_peak      = any(op_mask) ? maximum(log.grid_kW[op_mask]) : 0.0
    missed       = sum(rem_dig) + sum(rem_load)
    transit_intervals = count(==(0), real_loc)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals
    
    # Computes the end-of-run CEV energy shortfall and its penalty.
    (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
        _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, n_day_run)    

    # Converts the detailed logs into DataFrames, or nothing when detailed_output is off.
    detailed_plan_df     = detailed_output ? to_dataframe(plan_log)     : nothing
    detailed_realized_df = detailed_output ? to_dataframe(realized_log) : nothing
    detailed_mcs_plan_df     = detailed_output ? to_dataframe(mcs_plan_log)     : nothing
    detailed_mcs_realized_df = detailed_output ? to_dataframe(mcs_realized_log) : nothing

    # Returns the logs, the realized trajectories, the re-plan grids and the KPIs as one NamedTuple for 5_Output.jl.
    return (; d, time_labels, log, solve_log, replan_by_day, day_snapshots_df = DataFrame(day_snapshot_rows),
              real_P_ch, real_P_dch, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              est, nK = n_kept, nKd, n_day_run, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              shortfall_kWh, shortfall_hours, shortfall_penalty_cost,
              n_obs_total, n_infeasible, elapsed,
              detailed_plan_df, detailed_realized_df,
              detailed_mcs_plan_df, detailed_mcs_realized_df,
            approach = 2, plant, n_capped = n_capped_total, n_scenarios)
end

end