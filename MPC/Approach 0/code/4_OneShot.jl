# #############################################################################
# 4_OneShot.jl  —  module OneShot
# -----------------------------------------------------------------------------
# APPROACH 0 -- the one-shot, no-replanning executor. This is now its OWN,
# self-contained module: it used to live borrowed inside Approach 1's or
# Approach 2's 4_MPCLoop.jl (the `run_one_shot` function), reached via
# `A1ShrinkingApp.MPCLoop.run_one_shot` / `A2ShrinkingApp.MPCLoop.run_one_shot`
# in the comparison driver, with an `approach0_source` switch deciding which.
# That switch is gone: Approach 0 is no longer "borrowed" from whichever of
# Approach 1/2 happened to be picked.
#
# What Approach 0 actually does: at 08:00 each day, solve ONE whole-day MILP
# (the full 24h window, no shrinking, no replanning), then execute that fixed
# plan open-loop for the rest of the day, whatever the plant actually
# realizes. Contrast with Approach 1/2's `run_mpc`, which re-solves a shrinking
# window every 15 minutes.
#
# `apply_and_simulate!` and the small labelling helpers below (`activity_label`,
# `mcs_status_label`, `mcs_node_label`, `advance_mcs_state`,
# `realized_activity_durations`, `applied_act_index`) are copied byte-for-byte
# from Approach 1/Shrinking_Horizon/code/4_MPCLoop.jl, where `run_mpc` uses the
# exact same functions -- this guarantees Approach 0 and Approach 1/2 apply,
# simulate, and label a plan the SAME way, so a comparison between them
# isolates the value of replanning and nothing else. `run_mpc` itself (the
# replanning loop) is NOT copied here -- Approach 0 has no use for it.
#
# DETAILED OUTPUT (opt-in via `detailed_output = true`, same convention as
# Approach 1/2's `run_mpc`): every 15-minute plan and realized decision is
# logged for BOTH the CEV(s) (`DetailedPlanLog`/`RealizedTupleLog`, the same
# structs Approach 1/2 use) and the MCS itself (`MCSPlanLog`/`MCSRealizedLog`).
# Since Approach 0 only ever resolves ONCE per day (at 08:00), `resolve_step`
# is always 1 and `offset_step` runs 1..nKd -- this is the one and only plan
# Approach 0 ever makes for that day, logged in full.
#
# DAY-TO-DAY RESET (multi-day runs, n_day_run > 1): the applied-activity
# history `hist` -- which feeds the rest rule, the precedence rule, and the
# travel-pacing rule inside `build_window_model` (see 3_MCSModel.jl) -- is
# cleared at the START of every day's loop iteration. Only `hist` resets:
# battery SOE, MCS location, and the rem_dig/rem_load work backlog still carry
# over from day to day exactly as before. Without this reset, those three
# rules silently accumulate a running tally across the WHOLE multi-day run
# instead of each day being independently solvable, which produces a
# demand-charge peak that climbs day after day even though every day requires
# identical work -- see docs/README.md, "Why hist resets every day" for the
# full worked example.
# #############################################################################
module OneShot

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

export run_one_shot

# Plain-language activity names for the worker-facing schedule + plan grids.
const ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

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

# -----------------------------------------------------------------------------
# Where will the MCS be at the START of the NEXT interval (k0+1)? Returns
# (node, transit): node = parked node index (0 if mid-drive); transit = nothing
# or (i,j,r) = mid-drive on arc i->j with r intervals left.
# -----------------------------------------------------------------------------
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

# Applied (scheduled) activity index 1..4 for CEV e at k0, read from the u decision
# actually executed (1=dig, 2=load, 3=travel, 4=idle). Used to append to the shared
# history so the window model knows what was really done.
function applied_act_index(model, d, e, k0)
    for i in d.N_c, (ai, act) in enumerate(d.B)
        value(model[:u][e, i, act, k0]) > 0.5 && return ai
    end
    return length(d.B)   # nothing scheduled -> idle (a break)
end

# -----------------------------------------------------------------------------
# Human-readable label for what CEV e's `model` schedules at `site` during
# interval `k`: "Charging" if real power is delivered, else the argmax activity,
# else "". Factored out so the replanning-grid capture (every column of every
# re-plan step, MPC-only) and a single-interval lookup (used by BOTH approaches
# to fill in real_cev_act) share the exact same rule instead of two copies that
# could quietly drift apart.
# -----------------------------------------------------------------------------
function activity_label(model, d, e, site, k)
    vals   = [value(model[:u][e, site, a, k]) for a in eachindex(d.B)]
    p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
    return p_into > 1e-6 ? "Charging" : (sum(vals) < 0.5 ? "Off" : ACT_NAME[d.B[argmax(vals)]])
end

# Mutually-exclusive MCS status label for interval k (same rule used by the
# replanning grid and by the single-interval lookup for both approaches).
function mcs_status_label(model, d, k)
    pch    = sum(value(model[:P_ch_tot][m, k])  for m in d.M)
    pdch   = sum(value(model[:P_dch_tot][m, k]) for m in d.M)
    parked = any(value(model[:z][m, i, k]) > 0.5 for m in d.M, i in d.N)
    return pch  > 1e-6 ? "Charging (grid)" :
           pdch > 1e-6 ? "Serving CEV"     :
           !parked     ? "Traveling"       : "Idle"
end

# Human-readable node label for MCS unit `m` at interval `k`: the parked
# node's index as a string, or "Transit" if the MCS isn't parked anywhere
# this interval (mid-drive). Shared by the MCS plan/realized detailed logs
# below so the same rule is used whether reading a planned or a realized
# state.
function mcs_node_label(model, d, m, k)
    node = findfirst(i -> value(model[:z][m, i, k]) > 0.5, d.N)
    return node === nothing ? "Transit" : string(node)
end

# =============================================================================
# SHARED PLANT STEP  (the module Avik asked for)
# -----------------------------------------------------------------------------
# Given a `model` that has interval k0's decisions available (whether that
# model was just solved for a window starting at k0 -- Approach 1's closed
# loop -- or was solved ONCE for the whole day and is simply being replayed at
# k0 -- Approach 0's one-shot executor), this function is the single place
# that: (2) reads what the plan says to do this interval, (3) simulates the
# REALIZED within-interval activity split, and draws the REALIZED per-activity
# power from the shared `pool`/`cursor` (instead of a fresh independent
# `randn` draw), and (4) advances the real MCS/CEV physical state. Both
# `run_mpc` and `run_one_shot` call this every interval so they draw power
# from -- and update state exactly like -- the same plant model.
# -----------------------------------------------------------------------------
function apply_and_simulate!(model, k0, nK, d, pool::ActivityPowerPool, cursor, rng, multi_activity,
                             soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                             real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work;
                             plant_mode::Symbol = :sampled, gidx::Int = k0)
    # k0    -- DAY-LOCAL index (1..nK): used for every read from `model`, since the
    #          model itself is always a day-local shrinking window (see run_mpc).
    # gidx  -- GLOBAL index (1..n_day_run*nK): used for every WRITE into the real_*
    #          output arrays below, which span the WHOLE multi-day run (CHANGE 5).
    #          Defaults to k0, so single-day callers (run_one_shot, or run_mpc at
    #          n_day_run = 1) are an EXACT passthrough -- gidx == k0 in that case.
    # PLANT MODE (see run_one_shot / run_mpc):
    #   :sampled -> the stochastic plant. Realized per-activity power is the next
    #               unused draw from the shared pool; the within-interval activity
    #               split may be randomized (multi_activity).
    #   :mean    -> the DETERMINISTIC plant. Realized power is pinned to the same
    #               mean mu the MILP planned on, and the interval realizes the single
    #               planned activity for its whole length (multi_activity is forced
    #               off, since a random split would reintroduce randomness). No pool
    #               sample is consumed, so a :mean run leaves the cursor untouched and
    #               cannot perturb a :sampled run sharing the same pool. Realized ==
    #               planned by construction, so the outcome IS the MILP's own optimum.
    plant_mode in (:sampled, :mean) ||
        error("apply_and_simulate!: plant_mode must be :sampled or :mean, got :$plant_mode")
    use_mean = plant_mode === :mean
    # (2) APPLY interval k0's decisions.
    grid_kW = sum(value(model[:P_ch_tot][m, k0]) for m in d.M)
    dch_kW  = sum(value(model[:P_dch_tot][m, k0]) for m in d.M)
    cur_node = let nh = findfirst(i -> value(model[:z][1, i, k0]) > 0.5, d.N)
        nh === nothing ? 0 : nh
    end
    for m in d.M
        real_P_ch[m, gidx]  = value(model[:P_ch_tot][m, k0])
        real_P_dch[m, gidx] = value(model[:P_dch_tot][m, k0])
        real_loc[m, gidx]   = let nh = findfirst(i -> value(model[:z][m, i, k0]) > 0.5, d.N)
            nh === nothing ? 0 : nh
        end
    end

    # (3) SIMULATE realized activity split.
    a_real = Dict(e => realized_activity_durations(rng, model, e, k0, d;
                                                   multi = multi_activity && !use_mean) for e in d.E)

    # (2.5)/(3) STOCHASTIC PLANT: draw the realized per-activity power from the
    # SHARED pool, one draw per (entity, activity) OCCURRENCE this interval
    # (skipping activities not actually realized this step, so the 20-sample
    # budget per pair is spent only on real occurrences).
    p_true = Dict{Int, Vector{Float64}}()
    n_obs_added = 0
    for e in d.E
        row = a_real[e]
        pt  = zeros(length(row))
        site = findfirst(i -> d.A[i, e] == 1, d.N)
        # Off-shift (Eq. 8b: is_working false) means NO activity concept applies at
        # all -- the machine is off, unattended, zero draw. Without this gate,
        # applied_act_index's idle fallback would otherwise route every off-shift
        # interval through next_power!, drawing a real nonzero LIVE_DATA_MODE idle
        # sample for hours the CEV was never actually on and idling.
        working = site !== nothing && d.is_working[site, e, k0]
        for a in eachindex(row)
            row[a] > 1e-9 || continue
            if !working
                pt[a] = 0.0   # off-shift: no pool draw, cursor untouched, power forced to 0
                continue
            end
            # :mean pins the realized power to the planning mean and does NOT advance
            # the cursor; :sampled consumes the next pre-drawn sample for this pair.
            pt[a] = use_mean ? pool.mu[a] : next_power!(pool, cursor, e, a)
        end
        p_true[e] = pt
        sum(row) > 1e-9 && working && (n_obs_added += 1)
    end

    # (4) ADVANCE the real MCS energy + position.
    for m in d.M
        soe_mcs[m] = value(model[:SOE_MCS][m, k0 + 1])
        mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, k0, nK, d)
    end
    # CAP each CEV's realized dig/load hours by what its available energy could
    # actually pay for, BEFORE crediting rem_dig/rem_load or logging hist. This
    # replaces the old after-the-fact SOE clamp, which silently created/discarded
    # energy instead of reflecting that the machine ran out of charge mid-task.
    n_capped = 0
    for e in d.E
        raw_by_mcs = Dict(m => sum(value(model[:P_MCS_CEV][m, i, e, k0]) for i in d.N_c) * d.delta_T for m in d.M)
        total_raw  = sum(values(raw_by_mcs))
        charged    = d.eta_ch_dch_cev[e] * total_raw
        headroom  = soe_cev[e] + charged - d.SOE_CEV_min[e]      # energy available before hitting the floor
        work_true = dot(a_real[e], p_true[e])                    # energy the sampled draw would actually cost

        if work_true > headroom && work_true > 1e-9
            scale = max(headroom, 0.0) / work_true                # fraction of the task actually affordable
            a_real[e][1] *= scale                                 # dig hours, capped
            a_real[e][2] *= scale                                 # load hours, capped
            a_real[e][3] *= scale                                 # travel hours, capped (same treatment)
            a_real[e][4] += d.delta_T - sum(a_real[e][1:3])       # remainder of the interval becomes idle
            work_true = dot(a_real[e], p_true[e])                 # recompute cost against the capped hours
            n_capped += 1
        end

        soe_cev[e] = soe_cev[e] + charged - work_true
        overflow = soe_cev[e] - d.SOE_CEV_max[e]
        if overflow > 1e-9 && total_raw > 1e-9
            overflow_raw = overflow / d.eta_ch_dch_cev[e]
            for m in d.M
                share = raw_by_mcs[m] / total_raw
                soe_mcs[m] = min(soe_mcs[m] + overflow_raw * share, d.SOE_MCS_max[m])
            end
        end
        soe_cev[e] = clamp(soe_cev[e], d.SOE_CEV_min[e], d.SOE_CEV_max[e])  # safety net only; should not bite now
    end

    # Update remaining work + append this interval to the SHARED history.
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

    # Logged work power must be the power the plant ACTUALLY used this interval, i.e.
    # the same p_true that drained the CEV batteries above. (It previously read
    # d.true_powers -- a Fork-A hidden-truth curve that the Fork-B pool plant never
    # uses; in :synthetic those two vectors differ, so the log disagreed with the
    # batteries and with real_P_work.)
    work_kW = sum(dot(a_real[e], p_true[e]) for e in d.E) / d.delta_T

    return (; grid_kW, dch_kW, cur_node, a_real, p_true, n_obs_added, work_kW, n_capped)
end

# =============================================================================
# APPROACH 0 -- ONE-SHOT EXECUTOR (this is the whole point of this module)
# =============================================================================
function _terminal_soe_shortfall(d, soe_cev_end, rem_dig, rem_load, n_day_run::Int = 1)
    shortfall_kWh = sum(max(d.SOE_CEV_ini[e] - soe_cev_end[e], 0.0) for e in d.E; init = 0.0)

    # Total REQUIRED work across the whole run is n_day_run copies of one day's
    # requirement (CHANGE 5 -- same work every day), not just one day's worth.
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

function run_one_shot(d, pool::ActivityPowerPool; time_limit_sec::Float64 = Inf,
                      multi_activity::Bool = false,
                      require_site_visit::Bool = false,
                      single_visit_per_site::Bool = false,
                      plant::Symbol = :sampled,
                      n_day_run::Int = 1,
                      seed::Int = 1,
                      # DETAILED OUTPUT (opt-in): same flag/shape as run_mpc's
                      # kwarg of the same name. Since Approach 0 only ever
                      # resolves ONCE per day (at 08:00), `resolve_step` is
                      # always 1 and `offset_step` runs 1..nKd -- this is the
                      # one and only plan A0 ever makes for that day, logged
                      # in full (every step, not just the one applied first).
                      # See 1_Common.jl's DetailedPlanLog/RealizedTupleLog/
                      # MCSPlanLog/MCSRealizedLog.
                      detailed_output::Bool = false)
    plant in (:sampled, :mean) ||
        error("run_one_shot: plant must be :sampled or :mean, got :$plant")
    n_day_run >= 1 || error("run_one_shot: n_day_run must be >= 1, got $n_day_run")
    Random.seed!(seed)
    K_all = collect(d.K)
    nKd = length(K_all)                    # ONE day's interval count (was `nK`)
    n_kept = n_day_run * nKd
    time_labels = n_day_run == 1 ? build_time_labels(d.t_start, d.delta_T, nKd) :
                                    build_time_labels_days(d.t_start, d.delta_T, n_day_run, nKd)
    # This run's OWN walk through the shared pool -- independent of run_mpc's.
    cursor = new_cursor(pool)

    # ---- REAL physical state carried across steps AND across day boundaries ----
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

    log = DataFrame(
        day = Int[], k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[])

    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, n_kept)
    real_P_dch = zeros(nM, n_kept)
    real_L_trv = zeros(nM, n_kept)  # transit does not draw from the battery; kept at zero for the output schema
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)
    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = fill("", n_kept)

    # ---- DETAILED OUTPUT (opt-in, see run_one_shot's docstring on the kwarg) ----
    plan_log         = detailed_output ? DetailedPlanLog()     : nothing
    realized_log     = detailed_output ? RealizedTupleLog()    : nothing
    mcs_plan_log     = detailed_output ? MCSPlanLog()          : nothing
    mcs_realized_log = detailed_output ? MCSRealizedLog()      : nothing

    # ---- PER-DAY DIAGNOSTICS (always collected; cheap, and what 5_Output.jl's
    # daily KPI table needs for multi-day runs) ----
    solve_log = DataFrame(day = Int[], status = String[], objective = Float64[],
                          gap_percent = Float64[], solve_time_s = Float64[])
    day_snapshot_rows = NamedTuple[]

    pmode_txt = plant === :mean ?
        ":mean (DETERMINISTIC -- realized power pinned to mu; realized == planned)" :
        ":sampled (stochastic -- realized power drawn from the shared pool)"
    println("Running Approach 0 (one-shot 8:00 plan per day, executed open-loop, no replanning): $n_kept steps ($n_day_run day(s))")
    println("  plant                  : ", pmode_txt)
    println("  planning power (mu)    : ", round.(pool.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit      : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s" : "none (solve to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_capped_total = 0

    for day in 1:n_day_run
        # RESET PER DAY (Approach 0 only): clear the applied-activity history
        # so the rest rule, the precedence rule, and the travel-pacing rule
        # (all seeded from `hist` inside build_window_model -- see
        # 3_MCSModel.jl) start fresh each day instead of carrying a running
        # tally across the whole n_day_run window. This is the ONLY thing
        # that resets here: physical state (battery SOE, MCS location) and
        # the rem_dig/rem_load work backlog just below are UNCHANGED and
        # still carry over normally day to day, exactly as before.
        hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]

        # CHANGE 5 -- same work requirement every day, ADDED on top of whatever
        # is still outstanding from the previous day (backlog accumulates).
        rem_dig  .+= float.(d.hours_digging)
        rem_load .+= float.(d.hours_loading_swinging)

        # (1) OPTIMISE -- ONCE per day, over that day's own 24h window. No
        # fallback: if a day's own 8:00 whole-day plan is infeasible there is
        # nothing to execute for that day. K_all is DAY-LOCAL (1..nKd), unchanged
        # from the single-day version -- each new day re-solves its OWN fresh
        # 8:00 plan, exactly matching Approach 0's "commit once per day" identity.
        model = build_window_model(d, K_all, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                   rem_dig, rem_load, hist,
                                   peak_nc, peak_op, pool.mu;
                                   require_site_visit = require_site_visit,
                                   single_visit_per_site = single_visit_per_site,
                                   time_limit_sec = time_limit_sec)
        stat = string(termination_status(model))
        has_values(model) || error("Approach 0 (one-shot): day $day's 8:00 whole-day MILP was INFEASIBLE ",
                                   "(status=$stat); there is no fixed plan to execute.")
        day_solve_s = try solve_time(model) catch; NaN end
        push!(solve_log, (day, stat, objective_value(model),
                          100 * (try relative_gap(model) catch; NaN end),
                          isnan(day_solve_s) ? 0.0 : day_solve_s))

        # DETAILED OUTPUT -- FULL-DAY PLAN CAPTURE: A0 only ever resolves once
        # per day (right here), so this is the one and only plan it makes for
        # `day` -- capture the WHOLE day's plan (every offset step, not just
        # the one that gets applied first), for both the CEV(s) and the MCS.
        if detailed_output
            for k in 1:nKd
                for e in d.E
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    site === nothing && continue
                    p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
                    log_plan_row!(plan_log, d; day, resolve_step = 1, offset_step = k,
                                  activity_planned = activity_label(model, d, e, site, k),
                                  planned_power_kW = value(model[:P_work][site, e, k]),
                                  planned_charging = p_into > 1e-6,
                                  soe_cev_planned_kWh = value(model[:SOE_CEV][e, k + 1]),
                                  soe_mcs_planned_kWh = value(model[:SOE_MCS][1, k + 1]),
                                  cev = e)
                end
                for m in d.M
                    log_mcs_plan_row!(mcs_plan_log, d; day, resolve_step = 1, offset_step = k,
                                      mcs = m, mcs_status_planned = mcs_status_label(model, d, k),
                                      mcs_node_planned = mcs_node_label(model, d, m, k),
                                      grid_charge_kW_planned = value(model[:P_ch_tot][m, k]),
                                      grid_discharge_kW_planned = value(model[:P_dch_tot][m, k]),
                                      soe_mcs_planned_kWh = value(model[:SOE_MCS][m, k + 1]))
                end
            end
        end

        for k0 in 1:nKd                       # k0 is DAY-LOCAL (1..nKd)
            gidx = (day - 1) * nKd + k0        # gidx is GLOBAL (1..n_kept)

            for m in d.M; real_SOE_MCS[m, gidx] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, gidx] = soe_cev[e]; end

            # (2)+(3)+(4) APPLY / SIMULATE (shared pool draw) / ADVANCE -- the
            # the byte-identical copy of the plant step run_mpc calls, replayed
            # against this day's SAME model.
            step = apply_and_simulate!(model, k0, nKd, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work;
                                       plant_mode = plant, gidx = gidx)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

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
            real_mcs_act[gidx] = mcs_status_label(model, d, k0)

            # DETAILED OUTPUT -- REALIZED CAPTURE: what actually happened this
            # interval, for both the CEV(s) and the MCS.
            if detailed_output
                for e in d.E
                    log_realized_row!(realized_log, d; day, step = k0, cev = e,
                                      p_tuple = step.p_true[e], activity_executed = real_cev_act[e][gidx],
                                      soe_cev_kWh = safe_get(soe_cev, e), soe_mcs_kWh = soe_mcs[1],
                                      infeasible_flag = false)
                end
                for m in d.M
                    log_mcs_realized_row!(mcs_realized_log, d; day, step = k0, mcs = m,
                                          mcs_status_realized = real_mcs_act[gidx],
                                          mcs_node_realized = real_loc[m, gidx] == 0 ? "Transit" : string(real_loc[m, gidx]),
                                          grid_charge_kW_realized = real_P_ch[m, gidx],
                                          grid_discharge_kW_realized = real_P_dch[m, gidx],
                                          soe_mcs_kWh = soe_mcs[m])
                end
            end

            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            push!(log, (day, k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                        pool.mu[1], pool.mu[2], pool.mu[3], pool.mu[4],
                        pool.sd[1], pool.sd[2], pool.sd[3], pool.sd[4], n_obs_total))
        end

        # ---- END-OF-DAY SNAPSHOT (for 5_Output.jl's per-day KPI table) ----
        # Cumulative-to-date figures -- backlog and terminal shortfall are, by
        # this model's own design (CHANGE 5), carried forward day to day, not
        # reset -- so "as of end of day X" is the only meaningful reading.
        day_mask = log.day .== day
        transit_hours_day = count(==(0), log.mcs_node[day_mask]) * d.delta_T
        labour_cost_day   = d.rho_labor * count(==(0), log.mcs_node[day_mask]) * d.delta_T
        (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
            _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, day)
        push!(day_snapshot_rows, (; day,
                                     missed_work_cumulative_h = sum(rem_dig) + sum(rem_load),
                                     shortfall_kWh_cumulative = shortfall_kWh,
                                     shortfall_penalty_cost_cumulative = shortfall_penalty_cost,
                                     transit_hours_day, labour_cost_day))
    end

    for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end

    elapsed = time() - t0
    @printf("Approach 0 one-shot (plant = :%s) done in %.1f s (%d plant realizations, %d day(s))\n",
            plant, elapsed, n_obs_total, n_day_run)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy (task could not fully\n        complete before hitting the SOE floor); the shortfall is reflected honestly in\n        rem_dig/rem_load.\n", n_capped_total)

    total_energy = sum(log.grid_kW) * d.delta_T
    total_cost   = sum(log.grid_kW .* log.price) * d.delta_T
    total_co2    = sum(log.grid_kW .* log.co2)  * d.delta_T
    nc_peak      = isempty(log.grid_kW) ? 0.0 : maximum(log.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in log.k]
    op_peak      = any(op_mask) ? maximum(log.grid_kW[op_mask]) : 0.0
    missed       = sum(rem_dig) + sum(rem_load)
    transit_intervals = count(==(0), log.mcs_node)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals
    (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
        _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, n_day_run)   # Change 3, n_day_run-aware

    # DETAILED OUTPUT: converted to DataFrames once here -- see the matching
    # note in run_mpc. `nothing` when detailed_output = false, so callers
    # that don't ask for this pay no cost and see no change to the result
    # shape they already use.
    detailed_plan_df         = detailed_output ? to_dataframe(plan_log)         : nothing
    detailed_realized_df     = detailed_output ? to_dataframe(realized_log)     : nothing
    detailed_mcs_plan_df     = detailed_output ? to_dataframe(mcs_plan_log)     : nothing
    detailed_mcs_realized_df = detailed_output ? to_dataframe(mcs_realized_log) : nothing

    return (; d, time_labels, log, solve_log, day_snapshots_df = DataFrame(day_snapshot_rows),
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
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


end # module OneShot
