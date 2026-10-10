# #############################################################################
# OneShot.jl  -  module OneShot
# -----------------------------------------------------------------------------
# Implements "Approach 0": a naive one-shot baseline, not a true MPC. For each
# day, it solves the FULL day's MILP exactly once (one 8:00 whole-day plan via
# build_window_model over the entire day's K), then walks that fixed plan
# interval by interval and simulates what actually happens -- it never
# re-solves during the day, even as realized activity powers (drawn from the
# stochastic pool) deviate from the planned mean. This is what makes it a
# useful baseline: it shows how a plan-once-and-commit strategy holds up
# against reality, in contrast to a real receding-horizon MPC that re-plans
# as new information arrives (that logic lives in the other Approaches, not
# here). Five groups:
#
#   1. LABELING HELPERS (read-only, no state change)
#      activity_label, mcs_status_label, mcs_node_label, applied_act_index, serving_mcs
#      -- turn a solved model's decision variables at one interval into
#      human-readable strings ("Charging", "Digging (10 min)", node numbers)
#      for logging and display, without touching any simulation state.
#
#   2. PLANT SIMULATION STEP
#      realized_activity_durations, advance_mcs_state, apply_and_simulate!
#      -- the core "what actually happens" logic for one interval: splits a
#      CEV's single planned activity into realized fractional durations,
#      draws real (possibly noisy) activity powers from the shared
#      ActivityPowerPool, advances each MCS to its next physical location,
#      caps a CEV's realized work if it would drive SOE below its floor
#      (flagging which CEV was capped), and updates soe_mcs/soe_cev in place.
#      This is the one piece that makes the fixed plan's evaluation honest
#      rather than just replaying the optimizer's own assumptions back.
#
#   3. END-OF-RUN ACCOUNTING
#      _terminal_soe_shortfall
#      -- computes the missed-work-equivalent penalty for CEVs that didn't
#      return to their initial SOE by the end of the run (the energy-neutral
#      terminal condition from the paper isn't enforced by execution, only
#      by the plan, so this measures how far reality fell short of it).
#
#   4. MAIN ENTRY POINT
#      run_one_shot
#      -- for each day: builds the whole-day model once, errors out if it's
#      infeasible (a one-shot plan that can't even be built has nothing to
#      execute), then steps through every interval calling apply_and_simulate!,
#      accumulating the plain log (log), solver stats (solve_log), per-day
#      snapshots (day_snapshot_rows), and -- when detailed_output is true --
#      the full plan-vs-realized logs from Common.jl (DetailedPlanLog,
#      RealizedTupleLog, MCSPlanLog, MCSRealizedLog). Returns one large
#      NamedTuple of results (costs, emissions, peaks, realized arrays, logs)
#      for 5_Output.jl to consume.
#
#   5. NOTE ON REALISM
#      Because the plan is fixed at 8:00 and never revised, apply_and_simulate!
#      is deliberately "honest" about consequences it can't hide: if realized
#      power draw is higher than planned, a CEV's work gets capped once SOE
#      hits its floor (rem_dig/rem_load absorb the shortfall as missed work)
#      rather than silently letting SOE go negative or ignoring the mismatch.
# #############################################################################

module OneShot

# external packages used across this file
using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, build_time_labels, build_time_labels_days, safe_get,
                ActivityPowerPool, new_cursor, next_power!,
                DetailedPlanLog, RealizedTupleLog, log_plan_row!, log_realized_row!,
                MCSPlanLog, MCSRealizedLog, log_mcs_plan_row!, log_mcs_realized_row!,
                to_dataframe
using ..MCSModel: build_window_model

# everything below that other files are allowed to use
export run_one_shot

# Fixed activity index-to-name mapping (dig, load+swing, travel, idle), matching the same 4-activity ordering assumed elsewhere.
const ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# Determines what a CEV actually does during interval k0, given the fixed plan's activity-assignment variable u.
# Finds which (site, activity) pair the plan set to 1 for this CEV -- if none is set, the CEV wasn't scheduled to do anything this interval (off-hours or planned idle), so it returns a full-interval idle duration immediately.
# If multi is false, the CEV is assumed to spend the entire interval on its single planned activity, matching the plan exactly.
# If multi is true, the interval is split: a random fraction between 60% and 100% of the interval goes to the planned activity, and the remainder is idle -- modeling that a CEV doesn't necessarily work a full clean interval even when the plan says it should.
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

# Figures out where MCS m physically ends up right after interval k0 finishes, by reading the SOLVED plan's z (presence) and y_trv (traveling) variables one interval ahead (knext = k0+1).
# If knext falls outside the model's own built window (the very last interval has no "next" to check), falls back to reporting the MCS's CURRENT node at k0 with no transit info.
# If the MCS is parked somewhere at knext, returns (that node, nothing).
# Otherwise it must be traveling (per constraint 12c, an MCS is always parked-or-traveling, never neither); finds which (i,j) path it's on and counts how many more consecutive intervals that same trip's y_trv stays active, returning (0, (i, j, remaining_intervals)) as a sentinel for "in transit, not at a node".
# Falls back to the current node at k0 if neither case matches (defensive, shouldn't normally trigger given the exactly-one-state constraint).
# Since Approach 0 solves one whole-day MILP and never re-solves within a day, this function's real purpose is carrying MCS state ACROSS DAY BOUNDARIES in multi-day runs (n_day_run > 1) -- the next day's model build needs to know where each MCS physically ended up, not just what the plan said.
function advance_mcs_state(model, m, k0, nK, d)
    z = model[:z]; y = model[:y_trv]
    Kw = axes(z)[3]
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

# Returns which activity index (1-4) the solved plan assigned CEV e to at interval k0, checked across every construction node.
# Falls back to length(d.B) (the idle index) if no activity is set to 1 -- covers both "not scheduled this interval" and "plan assigned idle".
function applied_act_index(model, d, e, k0)
    for i in d.N_c, (ai, act) in enumerate(d.B)
        value(model[:u][e, i, act, k0]) > 0.5 && return ai
    end
    return length(d.B)   
end

# Builds a human-readable label for what CEV e is doing at a given site and interval.
# Checks charging power into the CEV first ("Charging" takes priority over any activity label), then falls back to "Off" if no activity variable is set at all, otherwise names the activity with the largest u-value (argmax guards against solver floating-point noise rather than requiring an exact 1.0).
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

# Builds a human-readable status label for MCS m in interval k, based on that MCS's own grid charging power, discharging power, and whether it is parked at a node.
# By constraints 5c, 5d and 12c, an MCS is at one node or on the road, charges only at grid nodes, and discharges only at construction nodes, so at most one of "Charging (grid)", "Serving CEV" and "Traveling" can apply to it in an interval
function mcs_status_label(model, d, m, k)
    pch    = value(model[:P_ch_tot][m, k])
    pdch   = value(model[:P_dch_tot][m, k])
    parked = any(value(model[:z][m, i, k]) > 0.5 for i in d.N)
    return pch  > 1e-6 ? "Charging (grid)" :
           pdch > 1e-6 ? "Serving CEV"     :
           !parked     ? "Traveling"       : "Idle"
end

# Returns the node MCS m is parked at during interval k, as a string, or "Transit" if it isn't parked anywhere (i.e. it's traveling).
# Correctly takes m and looks up that specific MCS's presence variable.
function mcs_node_label(model, d, m, k)
    node = findfirst(i -> value(model[:z][m, i, k]) > 0.5, d.N)
    return node === nothing ? "Transit" : string(node)
end

# Executes interval k0 of the FIXED plan against the (possibly stochastic) plant, and updates all the running simulation state in place: soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist, and the real_* logging arrays.
# Returns a NamedTuple summarizing what happened this interval, for run_one_shot to log and accumulate.
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
    # n_obs_added counts how many CEVs actually produced a new real observation this interval, for the Bayesian estimator's calibration bookkeeping (not used by Approach 0 itself, but kept for consistency with the shared pool machinery in Common.jl).
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

    # Advances each MCS's SOE by the plan's change over this interval (plan SOE at k0+1 minus plan SOE at k0), added to the MCS's current realized SOE and clamped to its SOE bounds.
    # Applying the plan's change rather than its absolute level keeps any energy refunded to the MCS by the CEV overflow step below, instead of overwriting it at the next interval.
    # Also updates the MCS node/transit state for the next interval via advance_mcs_state.
    for m in d.M
        soe_planned_step = soe_mcs[m] + value(model[:SOE_MCS][m, k0 + 1]) - value(model[:SOE_MCS][m, k0])
        soe_mcs[m] = clamp(soe_planned_step, d.SOE_MCS_min[m], d.SOE_MCS_max[m])
        # If the battery was already too full to take the planned charge, the extra energy was never stored, so the grid never delivered it: cut the logged grid power by that amount.
        not_stored_kWh = soe_planned_step - d.SOE_MCS_max[m]
        if not_stored_kWh > 1e-9
            cut_kW = min(not_stored_kWh / (d.eta_ch_dch_mcs[m] * d.delta_T), real_P_ch[m, gidx])
            real_P_ch[m, gidx] -= cut_kW
            grid_kW -= cut_kW
            println("  NOTE: MCS ", m, " already full at k0=", k0, "; grid charge cut by ", round(cut_kW, digits = 2), " kW")
        end
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

    # Deducts the realized digging/loading hours from each site's remaining work backlog (rem_dig/rem_load), records this interval's applied activity + realized durations into hist[e] (consumed by the next day's build_window_model call for cumulative history), and records the realized work power into real_P_work for the CEV's assigned site.
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

# Runs Approach 0 for n_day_run day(s): builds ONE whole-day MILP per day and executes it interval by interval against the (possibly stochastic) plant, with no replanning inside the day.
# Accumulates a plain per-interval log, optional detailed plan/realized logs, and whole-run KPIs, and returns everything as one NamedTuple for 5_Output.jl.
function run_one_shot(d, pool::ActivityPowerPool; time_limit_sec::Float64 = Inf,
                      multi_activity::Bool = false,
                      plant::Symbol = :sampled,
                      n_day_run::Int = 1,
                      seed::Int = 1,
                      detailed_output::Bool = false)
    
    # Validates the plant mode and day count, seeds the RNG, and builds the day's interval list, the global step count (n_kept), and the clock labels (single-day or multi-day format).
    # cursor is this run's own independent walker over the shared ActivityPowerPool.
    plant in (:sampled, :mean) ||
        error("run_one_shot: plant must be :sampled or :mean, got :$plant")
    n_day_run >= 1 || error("run_one_shot: n_day_run must be >= 1, got $n_day_run")
    Random.seed!(seed)
    K_all = collect(d.K)
    nKd = length(K_all)                    
    n_kept = n_day_run * nKd
    time_labels = n_day_run == 1 ? build_time_labels(d.t_start, d.delta_T, nKd) :
                                    build_time_labels_days(d.t_start, d.delta_T, n_day_run, nKd)
    cursor = new_cursor(pool)

    # Initializes the running simulation state: every battery starts at its initial SOE, every MCS starts parked at the first grid node, and the remaining-work backlog and CEV activity history start empty.
    # rem_dig/rem_load accumulate across days by design (missed work carries forward), and peak_nc/peak_op carry the running NC/OP demand peaks into each day's model.
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    mcs_node = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]
    nN_work  = length(d.hours_digging)
    rem_dig  = zeros(nN_work)
    rem_load = zeros(nN_work)
    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    peak_nc = 0.0; peak_op = 0.0
    rng = MersenneTwister(seed)
    
    # Plain per-interval summary table (returned as res.log), used for the whole-run KPI aggregation.
    # Carries one SOE column per MCS (soe_mcs1..M), one per CEV (soe_cev1..N), and one node column per MCS (mcs_node1..M, where 0 means in transit).
    # The est_/unc_ columns hold the planning power mean and sd for the 4 activities and are the same on every row.
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

    # Pre-allocates the whole-run result arrays, one column per global step (n_kept), indexed per MCS, per CEV, or per node as appropriate.
    # The two SOE arrays have one extra column to hold the final end-of-run state.
    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, n_kept)
    real_P_dch = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)
    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = [fill("", n_kept) for _ in d.M]

    # The four detailed plan/realized logs exist only when detailed_output is true.
    # solve_log records one solver summary per day, and day_snapshot_rows one cumulative snapshot per day for the per-day KPI table.
    plan_log         = detailed_output ? DetailedPlanLog()     : nothing
    realized_log     = detailed_output ? RealizedTupleLog()    : nothing
    mcs_plan_log     = detailed_output ? MCSPlanLog()          : nothing
    mcs_realized_log = detailed_output ? MCSRealizedLog()      : nothing

    solve_log = DataFrame(day = Int[], status = String[], objective = Float64[],
                          gap_percent = Float64[], solve_time_s = Float64[])
    day_snapshot_rows = NamedTuple[]

     # Prints a run header summarizing the plant mode, the planning power, the sampling spread (sampled mode only), and the solver time limit, then starts the timer and the running totals.
    pmode_txt = plant === :mean ?
        ":mean (DETERMINISTIC -- realized power pinned to mu; realized == planned)" :
        ":sampled (stochastic -- realized power drawn from the shared pool)"
    println("Running Approach 0 (one-shot 8:00 plan per day, executed open-loop, no replanning): $n_kept steps ($n_day_run day(s))")
    println("  plant                  : ", pmode_txt)
    println("  planning power (mu)    : ", round.(d.prior_mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit      : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s" : "none (solve to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_capped_total = 0

    for day in 1:n_day_run
        # Resets the CEV activity history for the new day, adds this day's required work onto any backlog carried over from earlier days, and solves the whole-day MILP once.
        # The model is the paper's objective (4) with constraints (5)-(14), built over the FULL day's interval set (never a shrinking window).
        # Errors immediately if the model is infeasible, since there is then no plan to execute and no later replan to recover from.
        # Records the solver status, objective, MIP gap, and solve time for this day.
        hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]

        rem_dig  .+= float.(d.hours_digging)
        rem_load .+= float.(d.hours_loading_swinging)

        model = build_window_model(d, K_all, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                   rem_dig, rem_load, hist,
                                   peak_nc, peak_op, d.prior_mu;
                                   time_limit_sec = time_limit_sec)
        stat = string(termination_status(model))
        has_values(model) || error("Approach 0 (one-shot): day $day's 8:00 whole-day MILP was INFEASIBLE ",
                                   "(status=$stat); there is no fixed plan to execute.")
        day_solve_s = try solve_time(model) catch; NaN end
        push!(solve_log, (day, stat, objective_value(model),
                          100 * (try relative_gap(model) catch; NaN end),
                          isnan(day_solve_s) ? 0.0 : day_solve_s))

        # When detailed_output is on, logs the full day's plan for every interval right after the solve.
        # resolve_step is always 1 here, since Approach 0 only ever solves once per day.
        # CEV rows record the planned activity, work power, charging flag, and planned SOE, plus the planned SOE of the MCS serving that CEV in that interval (NaN when no MCS is connected).
        # MCS rows record the planned status, node, grid charge/discharge power, and SOE.
        if detailed_output
            for k in 1:nKd
                for e in d.E
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    site === nothing && continue
                    p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
                    m_srv = serving_mcs(model, d, site, e, k)
                    log_plan_row!(plan_log, d; day, resolve_step = 1, offset_step = k,
                                  activity_planned = activity_label(model, d, e, site, k),
                                  planned_power_kW = value(model[:P_work][site, e, k]),
                                  planned_charging = p_into > 1e-6,
                                  soe_cev_planned_kWh = value(model[:SOE_CEV][e, k + 1]),
                                  soe_mcs_planned_kWh = m_srv === nothing ? NaN : value(model[:SOE_MCS][m_srv, k + 1]),
                                  cev = e)
                end
                for m in d.M
                    log_mcs_plan_row!(mcs_plan_log, d; day, resolve_step = 1, offset_step = k,
                                      mcs = m, mcs_status_planned = mcs_status_label(model, d, m, k),
                                      mcs_node_planned = mcs_node_label(model, d, m, k),
                                      grid_charge_kW_planned = value(model[:P_ch_tot][m, k]),
                                      grid_discharge_kW_planned = value(model[:P_dch_tot][m, k]),
                                      soe_mcs_planned_kWh = value(model[:SOE_MCS][m, k + 1]))
                end
            end
        end

        for k0 in 1:nKd  
            # Walks the fixed plan one interval at a time; gidx converts the day-local interval k0 into the global step index used by the whole-run arrays.
            # Snapshots each SOE as it stands BEFORE this interval executes, then runs apply_and_simulate! to execute the interval against the plant and update the running state.                    
            gidx = (day - 1) * nKd + k0        

            for m in d.M; real_SOE_MCS[m, gidx] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, gidx] = soe_cev[e]; end

            step = apply_and_simulate!(model, k0, nKd, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_loc, real_P_work;
                                       plant_mode = plant, gidx = gidx)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

            # Builds each CEV's realized activity label for this interval, adding the realized minutes whenever they are less than a full interval (for example when the SOE floor capped the work, giving "Digging (10 min)").
            # Also builds each MCS's status label from the plan.
            for e in d.E
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                idx = applied_act_index(model, d, e, k0)
                p_into = site !== nothing ? sum(value(model[:P_MCS_CEV][m, site, e, k0]) for m in d.M) : 0.0
                planned_label = p_into > 1e-6 ? "Charging" :
                                (site !== nothing && !d.is_working[site, e, k0]) ? "Off" : ACT_NAME[idx]
                realized_min  = round(Int, step.a_real[e][idx] * 60)
                full_min      = round(Int, d.delta_T * 60)
                real_cev_act[e][gidx] = realized_min < full_min ? "$(planned_label) ($(realized_min) min)" : planned_label
            end
            for m in d.M
                real_mcs_act[m][gidx] = mcs_status_label(model, d, m, k0)
            end

            # When detailed_output is on, logs this interval's realized row for every CEV and every MCS.
            # The CEV row carries planned and realized values side by side, plus the flag for work blocked by the SOE floor.
            # It also carries the SOE of the MCS serving that CEV in that interval (NaN when no MCS is connected).
            if detailed_output
                for e in d.E
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    m_srv = site !== nothing ? serving_mcs(model, d, site, e, k0) : nothing
                    activity_planned = site !== nothing ? activity_label(model, d, e, site, k0) : "Off"
                    planned_power_kW = site !== nothing ? value(model[:P_work][site, e, k0]) : 0.0
                    log_realized_row!(realized_log, d; day, step = k0, cev = e,
                                    p_tuple = step.p_true[e], activity_executed = real_cev_act[e][gidx],
                                    activity_planned, planned_power_kW,
                                    planned_activity_blocked_by_min_soe = step.capped[e],
                                    soe_cev_kWh = safe_get(soe_cev, e), soe_mcs_kWh = m_srv === nothing ? NaN : soe_mcs[m_srv],
                                    infeasible_flag = false)
                end
                for m in d.M
                    log_mcs_realized_row!(mcs_realized_log, d; day, step = k0, mcs = m,
                                          mcs_status_realized = real_mcs_act[m][gidx],
                                          mcs_node_realized = real_loc[m, gidx] == 0 ? "Transit" : string(real_loc[m, gidx]),
                                          grid_charge_kW_realized = real_P_ch[m, gidx],
                                          grid_discharge_kW_realized = real_P_dch[m, gidx],
                                          soe_mcs_kWh = soe_mcs[m])
                end
            end

            # Updates the running NC demand peak, and the OP demand peak when this interval is on-peak, then appends this interval's summary row to the plain log.
            # step.grid_kW is summed over all MCSs.
            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            # Appends this interval's summary row, with one SOE value per MCS, one per CEV, and one node value per MCS, in the same order the columns were declared.
            # The MCS node comes from real_loc, which is set per MCS in apply_and_simulate!, so it is correct for every MCS.
            push!(log, (day, k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        (soe_mcs[m] for m in d.M)...,
                        (soe_cev[e] for e in d.E)...,
                        (real_loc[m, gidx] for m in d.M)...,
                        d.prior_mu[1], d.prior_mu[2], d.prior_mu[3], d.prior_mu[4],
                        d.prior_sigma[1], d.prior_sigma[2], d.prior_sigma[3], d.prior_sigma[4], n_obs_total))
        end
        
        # At the end of each day, computes that day's transit hours and labour cost across ALL MCSs (via real_loc), plus a cumulative missed-work and terminal-shortfall snapshot.
        # These feed the per-day KPI table in 5_Output.jl.
        # The labour cost mirrors the last term of objective function (4) of the paper's MILP formulation.
        day_mask = log.day .== day
        transit_hours_day = count(==(0), real_loc[:, day_mask]) * d.delta_T
        labour_cost_day   = d.rho_labor * count(==(0), real_loc[:, day_mask]) * d.delta_T
        (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
            _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, day)
        push!(day_snapshot_rows, (; day,
                                     missed_work_cumulative_h = sum(rem_dig) + sum(rem_load),
                                     shortfall_kWh_cumulative = shortfall_kWh,
                                     shortfall_penalty_cost_cumulative = shortfall_penalty_cost,
                                     transit_hours_day, labour_cost_day))
    end

    # Records the final end-of-run SOE, prints the completion summary, and notes how many intervals had work capped by the SOE floor.
    for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end

    elapsed = time() - t0
    @printf("Approach 0 one-shot (plant = :%s) done in %.1f s (%d plant realizations, %d day(s))\n",
            plant, elapsed, n_obs_total, n_day_run)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy (task could not fully\n        complete before hitting the SOE floor); the shortfall is reflected honestly in\n        rem_dig/rem_load.\n", n_capped_total)

    # Computes the whole-run KPIs.
    # Energy, cost, CO2, and the NC/OP peaks come from the plain log, missed work from the remaining backlog, and transit/labour cost from real_loc across all MCSs.
    # These mirror the cost terms of objective function (4) of the paper's MILP formulation (energy, carbon, NC/OP demand charges, missed work, and travel labour).
    total_energy = sum(log.grid_kW) * d.delta_T
    total_cost   = sum(log.grid_kW .* log.price) * d.delta_T
    total_co2    = sum(log.grid_kW .* log.co2)  * d.delta_T
    nc_peak      = isempty(log.grid_kW) ? 0.0 : maximum(log.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in log.k]
    op_peak      = any(op_mask) ? maximum(log.grid_kW[op_mask]) : 0.0
    missed       = sum(rem_dig) + sum(rem_load)
    transit_intervals = count(==(0), real_loc[:, :])
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals
    (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
        _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, n_day_run)   

    # Converts the detailed logs into DataFrames (or nothing when detailed_output is off).
    detailed_plan_df         = detailed_output ? to_dataframe(plan_log)         : nothing
    detailed_realized_df     = detailed_output ? to_dataframe(realized_log)     : nothing
    detailed_mcs_plan_df     = detailed_output ? to_dataframe(mcs_plan_log)     : nothing
    detailed_mcs_realized_df = detailed_output ? to_dataframe(mcs_realized_log) : nothing

    # Returns everything the caller needs as one NamedTuple.
    # n_infeasible is always 0 because any infeasible day errors out earlier, so this point is only reached when every day solved.
    return (; d, time_labels, log, solve_log, day_snapshots_df = DataFrame(day_snapshot_rows),
              real_P_ch, real_P_dch, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              nK = n_kept, nKd, n_day_run, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              shortfall_kWh, shortfall_hours, shortfall_penalty_cost,
              n_obs_total, n_infeasible = 0, elapsed,
              detailed_plan_df, detailed_realized_df,
              detailed_mcs_plan_df, detailed_mcs_realized_df,
              approach = 0, plant, n_capped = n_capped_total)
end


end 
