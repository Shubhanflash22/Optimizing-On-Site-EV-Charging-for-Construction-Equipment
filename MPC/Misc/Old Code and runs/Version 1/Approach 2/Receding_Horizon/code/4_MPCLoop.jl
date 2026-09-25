# #############################################################################
# MPCLoop.jl  —  module MPCLoop
# -----------------------------------------------------------------------------
# The closed-loop driver that ties the OPTIMISE and LEARN halves together. For
# every 15-min interval it: (1) solves the receding-horizon window MILP over the
# full remaining 24 h from the current real state + the FIXED (once-fitted) power
# model, (2) APPLIES only the first interval's decisions to the "plant", (3) draws
# the realized per-activity power from the fixed posterior (Fork B) and advances
# the real state. The MCS overnight recharge is part of the single 24 h MILP
# (there is no separate Phase 2), and the power model is never re-fitted online.
#
# Besides the analyst log it CAPTURES realized per-interval arrays (charging /
# discharging / travel energy / work power / SOE / location) so the reporting
# and plotting modules can render the same figure set as the v4_real reference
# from the trajectory that was actually realized.
#
# -----------------------------------------------------------------------------
# APPROACH 2 ADDITION — `run_mpc` below is now the STOCHASTIC (scenario-based)
# multi-day controller: at every re-solve it draws `n_scenarios` fresh samples
# from the current posterior and solves `build_window_model_stochastic` instead
# of the certainty-equivalent `build_window_model`. `run_one_shot` (the
# Approach 0 baseline, one-shot PER DAY) is UNCHANGED. See the Shrinking-Horizon
# sibling's MPCLoop.jl for the full explanation of `apply_and_simulate_stochastic!`
# / the scenario-1 view wrapper — identical mechanism here, just threaded through
# the day-loop's `out_idx`/`Kend` bookkeeping.
# #############################################################################
module MPCLoop

using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, clock_day_label, build_time_labels_days,
                multiday_xticks, safe_get, BayesianActivityEstimator,
                ActivityPowerPool, new_cursor, next_power!
using ..MCSModel: build_window_model, build_window_model_stochastic
using ..ScenarioSampler: sample_scenarios, equal_weights, DEFAULT_N_SCENARIOS

export run_mpc, run_one_shot

# Plain-language activity names for the worker-facing schedule + plan grids.
const ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# -----------------------------------------------------------------------------
# Simulate the REALIZED within-interval activity split for excavator e at k0.
# The MILP plans ONE activity per interval; a real machine mixes tasks. In
# multi-activity mode we spend 60-100% of the interval on the planned task and
# the rest idling, giving the learner a richer regression row. Returns hours per
# activity (summing to delta_T).
# -----------------------------------------------------------------------------
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
# GLOBAL interval `g`: "Charging" if real power is delivered, else the argmax
# activity, else "". Factored out so the replanning-grid capture (every column
# of every re-plan step, MPC-only) and a single-interval lookup (used by BOTH
# approaches to fill in real_cev_act) share the exact same rule instead of two
# copies that could quietly drift apart.
# -----------------------------------------------------------------------------
function activity_label(model, d, e, site, g)
    vals   = [value(model[:u][e, site, a, g]) for a in eachindex(d.B)]
    p_into = sum(value(model[:P_MCS_CEV][m, site, e, g]) for m in d.M)
    return p_into > 1e-6 ? "Charging" : (sum(vals) < 0.5 ? "" : ACT_NAME[d.B[argmax(vals)]])
end

# Mutually-exclusive MCS status label for GLOBAL interval g (same rule used by
# the replanning grid and by the single-interval lookup for both approaches).
function mcs_status_label(model, d, g)
    pch    = sum(value(model[:P_ch_tot][m, g])  for m in d.M)
    pdch   = sum(value(model[:P_dch_tot][m, g]) for m in d.M)
    parked = any(value(model[:z][m, i, g]) > 0.5 for m in d.M, i in d.N)
    return pch  > 1e-6 ? "Charging (grid)" :
           pdch > 1e-6 ? "Serving CEV"     :
           !parked     ? "Traveling"       : "Idle"
end

# =============================================================================
# SHARED PLANT STEP  (the module Avik asked for)
# -----------------------------------------------------------------------------
# Given a `model` that has GLOBAL interval `g0`'s decisions available (whether
# that model was just solved for the 24h window starting at g0 -- Approach 1's
# closed loop -- or was solved ONCE for the whole day containing g0 and is
# simply being replayed at g0 -- Approach 0's one-shot-per-day executor), this
# function is the single place that: (2) reads what the plan says to do this
# interval, (3) simulates the REALIZED within-interval activity split and draws
# the REALIZED per-activity power from the shared `pool`/`cursor` (instead of a
# fresh independent `randn` draw), and (4) advances the real MCS/CEV physical
# state. Both `run_mpc` and `run_one_shot` call this every interval so they draw
# power from -- and update state exactly like -- the same plant model.
#
# `out_idx` is where to write the realized per-interval arrays: the KEPT-day
# index for a kept interval, or `nothing` to skip those writes entirely (used
# by run_mpc on the dropped buffer day, which has no `real_*` storage).
# `Kend` is the model's own window's last global index (needed by
# advance_mcs_state to tell "end of window" from "mid-window").
# -----------------------------------------------------------------------------
function apply_and_simulate!(model, g0, Kend, d, pool::ActivityPowerPool, cursor, rng, multi_activity,
                             soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                             real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work,
                             out_idx::Union{Int, Nothing};
                             plant_mode::Symbol = :sampled)
    # PLANT MODE (see run_mpc / run_one_shot):
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

    # (2) APPLY interval g0's decisions.
    grid_kW = sum(value(model[:P_ch_tot][m, g0]) for m in d.M)
    dch_kW  = sum(value(model[:P_dch_tot][m, g0]) for m in d.M)
    cur_node = let nh = findfirst(i -> value(model[:z][1, i, g0]) > 0.5, d.N)
        nh === nothing ? 0 : nh
    end
    if out_idx !== nothing
        for m in d.M
            real_P_ch[m, out_idx]  = value(model[:P_ch_tot][m, g0])
            real_P_dch[m, out_idx] = value(model[:P_dch_tot][m, g0])
            real_L_trv[m, out_idx] = value(model[:L_trv_tot][m, g0])
            real_loc[m, out_idx]   = let nh = findfirst(i -> value(model[:z][m, i, g0]) > 0.5, d.N)
                nh === nothing ? 0 : nh
            end
        end
    end

    # (3) SIMULATE realized activity split.
    a_real = Dict(e => realized_activity_durations(rng, model, e, g0, d;
                                                   multi = multi_activity && !use_mean) for e in d.E)

    # (2.5)/(3) STOCHASTIC PLANT: draw the realized per-activity power from the
    # SHARED pool, one draw per (entity, activity) OCCURRENCE this interval
    # (skipping activities not actually realized this step, so the n_samples
    # budget per pair is spent only on real occurrences).
    p_true = Dict{Int, Vector{Float64}}()
    n_obs_added = 0
    for e in d.E
        row = a_real[e]
        pt  = zeros(length(row))
        for a in eachindex(row)
            row[a] > 1e-9 || continue
            # :mean pins the realized power to the planning mean and does NOT advance
            # the cursor; :sampled consumes the next pre-drawn sample for this pair.
            pt[a] = use_mean ? pool.mu[a] : next_power!(pool, cursor, e, a)
        end
        p_true[e] = pt
        sum(row) > 1e-9 && (n_obs_added += 1)
        if out_idx !== nothing
            site = findfirst(i -> d.A[i, e] == 1, d.N)
            if site !== nothing
                real_P_work[site, e, out_idx] =
                    (row[1] * pt[1] + row[2] * pt[2] + row[3] * pt[3]) / d.delta_T
            end
        end
    end

    # (4) ADVANCE the real MCS energy + position. SOE_MCS is read directly from
    # the model's own energy-balance variable (same approach as the
    # Shrinking_Horizon sibling) rather than a hand-recomputed formula, so a
    # future change to the battery equation in 3_MCSModel.jl can't silently
    # drift out of sync with what actually gets applied here.
    for m in d.M
        soe_mcs[m] = value(model[:SOE_MCS][m, g0 + 1])
        mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, g0, Kend, d)
    end
    # # The CEV balance is CLAMPED to [SOE_min, SOE_max]. The clamp is a guard, not
    # # physics: whenever it bites, energy is silently created (low side) or discarded
    # # (high side) and the reported SOE stops matching the integrated power. It can
    # # only bite in :sampled mode (in :mean mode the realized balance equals the MILP's
    # # own, which already respects the bounds), so we COUNT the events and report them
    # # rather than letting a "0 infeasible windows" run hide them.
    # n_clamped = 0
    # for e in d.E
    #     charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T
    #     work_true = dot(a_real[e], p_true[e])
    #     raw       = soe_cev[e] + charged - work_true
    #     soe_cev[e] = clamp(raw, d.SOE_CEV_min[e], d.SOE_CEV_max[e])
    #     abs(raw - soe_cev[e]) > 1e-9 && (n_clamped += 1)
    # end

    # CAP each CEV's realized dig/load/travel hours by what its available energy
    # could actually pay for, BEFORE crediting rem_dig/rem_load or logging hist.
    # This replaces the old after-the-fact SOE clamp, which silently created or
    # discarded energy instead of reflecting that the machine ran out of charge
    # mid-task.
    n_capped = 0
    for e in d.E
        charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T
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
        soe_cev[e] = clamp(soe_cev[e], d.SOE_CEV_min[e], d.SOE_CEV_max[e])  # safety net only; should not bite now
    end
    # Update remaining work + append this interval to the SHARED history.
    for e in d.E
        site_e = findfirst(i -> d.A[i, e] == 1, d.N)
        if site_e !== nothing
            rem_dig[site_e]  = max(rem_dig[site_e]  - a_real[e][1], 0.0)
            rem_load[site_e] = max(rem_load[site_e] - a_real[e][2], 0.0)
        end
        push!(hist[e], (applied_act_index(model, d, e, g0), copy(a_real[e])))
    end

    # Logged work power must be the power the plant ACTUALLY used this interval, i.e.
    # the same p_true that drained the CEV batteries above. (It previously read
    # d.true_powers -- a Fork-A hidden-truth curve that the Fork-B pool plant never
    # uses; in :synthetic those two vectors differ, so the log disagreed with the
    # batteries and with real_P_work.)
    work_kW = sum(dot(a_real[e], p_true[e]) for e in d.E) / d.delta_T

    # return (; grid_kW, dch_kW, cur_node, a_real, p_true, n_obs_added, work_kW, n_clamped)
    return (; grid_kW, dch_kW, cur_node, a_real, p_true, n_obs_added, work_kW, n_capped)
end

# -----------------------------------------------------------------------------
# STOCHASTIC PLANT STEP  (Approach 2, Receding). Same idea as the Shrinking-
# Horizon sibling: `model` was solved by `build_window_model_stochastic`, so
# every variable carries a trailing scenario index; non-anticipativity means
# scenario 1's slice at g0 IS the shared here-and-now decision. This is a thin
# index-adapting wrapper around `apply_and_simulate!` (identical signature here,
# `out_idx`/`Kend` included, so the day-loop bookkeeping is untouched).
# -----------------------------------------------------------------------------
function apply_and_simulate_stochastic!(model, g0, Kend, d, pool::ActivityPowerPool, cursor, rng, multi_activity,
                             soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                             real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work,
                             out_idx::Union{Int, Nothing};
                             plant_mode::Symbol = :sampled, s_ref::Int = 1)
    view_model = _ScenarioOneView(model, s_ref)
    return apply_and_simulate!(view_model, g0, Kend, d, pool, cursor, rng, multi_activity,
                               soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                               real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work,
                               out_idx; plant_mode = plant_mode)
end

# A tiny read-only wrapper around a solved stochastic model — see the
# Shrinking-Horizon MPCLoop.jl for the full explanation. `[:varname]` returns a
# container that forwards further indexing to `[..., s_ref]` on the underlying
# scenario-indexed variable, so every existing single-scenario helper in this
# file works unchanged against a stochastic model.
struct _ScenarioOneView
    model::Model
    s_ref::Int
end
struct _ScenarioOneVar
    var::Any
    s_ref::Int
end
Base.getindex(v::_ScenarioOneView, sym::Symbol) = _ScenarioOneVar(v.model[sym], v.s_ref)
Base.getindex(v::_ScenarioOneVar, idx...) = v.var[idx..., v.s_ref]
# `advance_mcs_state` calls `axes(z)[3]` to find the window's interval axis; drop
# the trailing scenario axis so it sees the same shape as a non-scenario container.
Base.axes(v::_ScenarioOneVar) = axes(v.var)[1:end-1]
Base.axes(v::_ScenarioOneVar, d::Int) = axes(v.var)[d]

# =============================================================================
# CHANGE 3 — TERMINAL SOE_CEV SHORTFALL PENALTY (all five approaches)
# -----------------------------------------------------------------------------
# See the identical docstring in Approach 1/Shrinking_Horizon/code/4_MPCLoop.jl
# for the full rationale. Summary: this is a pure END-OF-DAY snapshot check
# against SOE_CEV_ini (Eq 8b's recovery target) -- unrelated to the physical
# SOE_CEV_min floor, which is already enforced live every interval inside
# apply_and_simulate! (capped work already flows into rem_dig/rem_load/missed,
# identically for every approach; nothing new needed there). Applies uniformly
# to all five approaches.
# =============================================================================
function _terminal_soe_shortfall(d, soe_cev_end, rem_dig, rem_load)
    shortfall_kWh = sum(max(d.SOE_CEV_ini[e] - soe_cev_end[e], 0.0) for e in d.E; init = 0.0)

    realized_dig_h  = sum(d.hours_digging)          - sum(rem_dig)
    realized_load_h = sum(d.hours_loading_swinging) - sum(rem_load)
    realized_h      = realized_dig_h + realized_load_h
    realized_kWh    = realized_dig_h * d.p_digging + realized_load_h * d.p_loading_swinging
    avg_work_power_kW = realized_h > 1e-9 ? realized_kWh / realized_h :
                                             (d.p_digging + d.p_loading_swinging) / 2

    shortfall_hours = shortfall_kWh / avg_work_power_kW
    shortfall_penalty_cost = d.rho_miss * shortfall_hours
    return (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost)
end

# =============================================================================
# MAIN CLOSED LOOP
# =============================================================================
function run_mpc(d, pool::ActivityPowerPool; plant::Symbol = :sampled,
                    time_limit_sec::Float64 = Inf,
                    multi_activity::Bool = false,
                    require_site_visit::Bool = false,
                    single_visit_per_site::Bool = false,
                    mcmc_samples::Int = 500,
                    n_days::Union{Nothing, Int} = nothing,
                    # APPROACH 2: how many scenarios each re-solve samples from the
                    # current posterior and hedges across (see 2b_ScenarioSampler.jl).
                    n_scenarios::Int = DEFAULT_N_SCENARIOS,
                    seed::Int = 1)
    n_scenarios >= 1 || error("run_mpc: n_scenarios must be >= 1, got $n_scenarios")
    Random.seed!(seed)
    # Dedicated, independent RNG stream for scenario sampling — see the
    # Shrinking-Horizon sibling's run_mpc for why this must never share a
    # stream with the plant's own rng/cursor.
    rng_scenarios = MersenneTwister(seed + 1_000_000)
    nKd = length(collect(d.K))
    # Position of global interval k within its day-block (1..nKd), wrapping across day
    # boundaries. Mirrors wd() in build_window_model (3_MCSModel.jl) - that copy is local
    # to the MILP's own K window, this one is needed here for the replan-grid capture.
    wd(k) = mod1(k, nKd)

    n_days_keep = n_days === nothing ? d.n_days : max(1, n_days)
    D_total     = n_days_keep + 1
    n_kept      = n_days_keep * nKd

    # ---- REAL physical state carried across steps (the "plant") ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    nN_work  = length(d.hours_digging)
    rem_dig  = zeros(nN_work)
    rem_load = zeros(nN_work)

    # Days beyond the given work data (the buffer day, and any day further out a
    # window happens to reach) repeat the LAST defined day's quota rather than 0.
    quota_dig(day)  = float.(d.dig_by_day[clamp(day, 1, length(d.dig_by_day))])
    quota_load(day) = float.(d.load_by_day[clamp(day, 1, length(d.load_by_day))])

    # ---- CARRIED STATE (persists across the WHOLE run, day boundaries included) ----
    # The window is now a fixed-length 24h rolling horizon, so nothing about "today"
    # vs "tomorrow" should reset the real plant state or the shared applied-activity
    # history; only the work QUOTA (above) is a once-per-calendar-day event.
    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    mcs_node    = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]

    # ---- online learner (logged only; the SHARED pool below is what the plant draws from) ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)
    # This run's OWN walk through the shared pool: same underlying samples as
    # any other approach run against `pool`, independent consumption order.
    cursor = new_cursor(pool)

    # ---- analyst log (one row per applied interval) ----
    log = DataFrame(
        day = Int[], gstep = Int[],
        k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[])
    solve_log = DataFrame(day = Int[], step = Int[], clock = String[], status = String[],
                          objective = Float64[], gap_percent = Float64[], solve_time_s = Float64[])

    # ---- realized per-interval capture for the reference-style figures ----
    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch    = zeros(nM, n_kept)
    real_P_dch   = zeros(nM, n_kept)
    real_L_trv   = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)

    # ---- replanning grids (row = re-plan step, col = interval planned) ----
    replan_by_day = Dict{Int, NamedTuple}()

    # ---- REALIZED (applied) activity per interval, one label per step ----
    # This is the diagonal of the plan grids: the activity the loop ACTUALLY
    # executed each step (vs the 08:00 forward plan). Used by the plan-vs-actual
    # activity report to highlight where the realised day diverged from the plan.
    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = fill("", n_kept)

    plant in (:sampled, :mean) ||
        error("run_mpc: plant must be :sampled or :mean, got :$plant")
    println("Running Approach 2 (STOCHASTIC scenario-based MPC, 15-min steps, multi-day receding horizon): $n_kept steps ($n_days_keep kept days)")
    println("  scenarios / re-solve : ", n_scenarios, " (equal-weight, resampled fresh every window)")
    println("  plant                : ", plant === :mean ?
            ":mean (DETERMINISTIC -- realized power pinned to mu)" :
            ":sampled (stochastic -- realized power drawn from the shared pool)")
    println("  posterior mean (mu)  : ", round.(est.mu, digits = 2), " kW")
    println("  posterior std  (sd)  : ", round.(est.sd, digits = 2), " kW  (scenario sampling spread)")
    plant === :sampled && println("  plant sampling sd    : ", round.(pool.sd, digits = 2), " kW")
    # SOLVER TIME LIMIT (control point #2 -> forwarded to build_window_model_stochastic).
    println("  solver time limit    : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s / window" : "none (solve each window to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_infeasible = 0
    n_capped_total = 0
    gstep = 0
    missed_kept = 0.0
    # Change 3 fix: pre-declare the kept-day snapshot vars BEFORE the loop, same
    # defensive pattern as missed_kept just above and the same fix applied to the
    # Approach 1 Receding sibling (identical UndefVarError observed there too).
    rem_dig_kept  = copy(rem_dig)
    rem_load_kept = copy(rem_load)
    soe_cev_kept  = copy(soe_cev)

    for day in 1:D_total
        rem_dig  .+= quota_dig(day)
        rem_load .+= quota_load(day)
        # `hist` (shared applied-activity history) and mcs_node/mcs_transit are NOT
        # reset here anymore - they carry over continuously from the previous day
        # (see initialization above the day loop). Only the demand-charge peak
        # trackers reset daily, matching the paper's monthly-peak surrogate.
        peak_nc = 0.0; peak_op = 0.0

        plan_grid_kW = fill(NaN, nKd, nKd)
        plan_mcs_soe = fill(NaN, nKd, nKd)
        plan_cev_soe = [fill(NaN, nKd, nKd) for _ in d.E]
        plan_cev_act = [fill("", nKd, nKd)  for _ in d.E]
        plan_mcs_act = fill("", nKd, nKd)   # MCS status: Idle / Charging (grid) / Serving CEV / Traveling

        day_off = (day - 1) * nKd
        kept    = day <= n_days_keep
        # Last interval before the NEXT 8am (same fixed clock-time deadline the MILP's
        # terminal constraint uses - see k_term in build_window_model). Every window
        # solved today can see past this point into tomorrow, but we only ever report
        # the plan up to this fixed cutoff, never the "free" tail beyond it.
        k_term_today = day * nKd

        for k0 in 1:nKd
            gstep += 1
            g0    = day_off + k0
            gk    = kept ? ((day - 1) * nKd + k0) : 0
            clk   = clock_day_label(d.t_start, d.delta_T, day, k0)

            # record start-of-interval MCS/CEV SOE (boundary k0)
            if kept
                for m in d.M; real_SOE_MCS[m, gk] = soe_mcs[m]; end
                for e in d.E; real_SOE_CEV[e, gk] = soe_cev[e]; end
            end

            # Fixed-length rolling window: always exactly 24h (nKd intervals) ahead of
            # NOW (g0), every single step - no shrinking through the day, no jump at
            # the day boundary.
            Kend  = g0 + nKd - 1
            K_win = g0:Kend

            # (0) SAMPLE — fresh scenarios for THIS window only, from the current
            # (frozen, once-fitted) posterior. Never touches the plant's rng/cursor.
            scenarios = sample_scenarios(est.mu, est.sd, n_scenarios; rng = rng_scenarios)
            weights   = equal_weights(n_scenarios)

            # (1) OPTIMISE — the scenario-based MILP: one shared here-and-now action
            # (non-anticipativity) that must be feasible under EVERY sampled scenario.
            model = build_window_model_stochastic(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                       rem_dig, rem_load, hist,
                                       peak_nc, peak_op, scenarios, weights;
                                       require_site_visit = require_site_visit,
                                       single_visit_per_site = single_visit_per_site,
                                       time_limit_sec = time_limit_sec)
            stat = string(termination_status(model))
            cur_node = mcs_node[1]

            # NO FALLBACK: infeasible under HARD constraints -> hold the plant still.
            if !has_values(model)
                kept && (n_infeasible += 1)
                @warn "No feasible solution at step day=$day k=$k0 under HARD constraints; holding state (no fallback)." status=stat
                kept && push!(solve_log, (day, k0, clk, stat, NaN, NaN, try solve_time(model) catch; NaN end))
                push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                            0.0, 0.0, 0.0, soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), cur_node,
                            est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                            est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
                if kept; for m in d.M; real_loc[m, gk] = cur_node; end; end
                # A held (infeasible) interval is a BREAK: idle for every CEV and the MCS.
                # ... and record it in history so the rest rule counts it correctly next window.
                for e in d.E; push!(hist[e], (length(d.B), [0.0, 0.0, 0.0, d.delta_T])); end
                continue
            end

            # solve diagnostics for this window (independent of the plant draw).
            kept && push!(solve_log, (day, k0, clk, stat, objective_value(model),
                                      100 * (try relative_gap(model) catch; NaN end),
                                      try solve_time(model) catch; NaN end))

            # Read through the scenario-1 view (non-anticipativity guarantees g0's
            # action agrees with every scenario; scenario 1 is the representative
            # forward look for k > g0 in the replanning grids — a diagnostic display
            # only, not used in what gets applied or in any KPI).
            vmodel = _ScenarioOneView(model, 1)

            # Save this window's forward plan into the replanning grids, up to (and
            # including) the fixed next-8am cutoff - this is what actually gets
            # recovered-to, so it's the meaningful part of "the solver's intentions".
            # wd(k) wraps any k (today OR tomorrow) into the same 1..nKd "clock time of
            # day" column the grid/HTML writer already labels (8am today .. 7:45am
            # tomorrow), instead of the old day-block offset which only worked for k's
            # still inside TODAY and silently dropped everything past midnight.
            for k in K_win
                k <= k_term_today || continue
                kl = wd(k)
                plan_grid_kW[k0, kl] = sum(value(vmodel[:P_ch_tot][m, k]) for m in d.M)
                plan_mcs_soe[k0, kl] = value(vmodel[:SOE_MCS][1, k + 1])
                for e in d.E
                    plan_cev_soe[e][k0, kl] = value(vmodel[:SOE_CEV][e, k + 1])
                    # "Charging" is shown only when REAL power is delivered into this CEV
                    # (sum_m P_MCS_CEV > 0), not merely when the plug-in permission bit
                    # mu=1 (mu can be 1 with zero power flow) -- see activity_label.
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    site !== nothing && (plan_cev_act[e][k0, kl] = activity_label(vmodel, d, e, site, k))
                end
                plan_mcs_act[k0, kl] = mcs_status_label(vmodel, d, k)
            end

            # The APPLIED cell (row k0, col k0) is what actually happened this step.
            if kept
                for e in d.E; real_cev_act[e][gk] = plan_cev_act[e][k0, k0]; end
                real_mcs_act[gk] = plan_mcs_act[k0, k0]
            end

            # (2)+(3)+(4) APPLY / SIMULATE (shared pool draw) / ADVANCE -- reads the
            # shared (non-anticipative) g0 decision out of the stochastic model and
            # runs it through the SAME plant physics apply_and_simulate! uses for
            # Approach 0 / the deterministic controller.
            step = apply_and_simulate_stochastic!(model, g0, Kend, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work,
                                       kept ? gk : nothing; plant_mode = plant, s_ref = 1)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
        end

        # final boundary SOE snapshot for kept days
        if day == n_days_keep
            missed_kept = sum(rem_dig) + sum(rem_load)
            # Snapshot rem_dig/rem_load/soe_cev HERE too (Change 3) -- the loop keeps
            # running through the trailing buffer day below, which would otherwise
            # keep mutating rem_dig/rem_load/soe_cev past the point klog reports.
            rem_dig_kept  = copy(rem_dig)
            rem_load_kept = copy(rem_load)
            soe_cev_kept  = copy(soe_cev)
            # DIAGNOSTIC (temporary): confirms this branch actually ran, so the
            # pre-loop defaults above are provably just an unused safety net.
            println("  [Change 3 diag] kept-day snapshot taken at day=$(day) (n_days_keep=$(n_days_keep)), soe_cev_kept=", soe_cev_kept)
            for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end
        end

        kept && (replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act, plan_mcs_act))
    end

    elapsed = time() - t0
    @printf("Approach 2 (stochastic, %d scenarios, plant = :%s) done in %.1f s (%d plant realizations)\n",
            n_scenarios, plant, elapsed, n_obs_total)
    n_infeasible > 0 && @printf("  NOTE: %d/%d windows were INFEASIBLE under the HARD constraints (no fallback);\n        the plant HELD state for those intervals.\n", n_infeasible, n_kept)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy (task could not fully\n        complete before hitting the SOE floor); the shortfall is reflected honestly in\n        rem_dig/rem_load.\n", n_capped_total)
    println("  posterior mean (mu)    : ", round.(est.mu, digits = 2), " kW")
    println("  posterior std  (sd)    : ", round.(est.sd, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")

    keep_row = log.day .<= n_days_keep
    klog = log[keep_row, :]

    # ---- Phase-1 KPIs from the realized trajectory ----
    total_energy = sum(klog.grid_kW) * d.delta_T
    total_cost   = sum(klog.grid_kW .* klog.price) * d.delta_T
    total_co2    = sum(klog.grid_kW .* klog.co2)  * d.delta_T
    nc_peak      = isempty(klog.grid_kW) ? 0.0 : maximum(klog.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in klog.k]
    op_peak      = any(op_mask) ? maximum(klog.grid_kW[op_mask]) : 0.0
    missed       = missed_kept
    transit_intervals = count(==(0), klog.mcs_node)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals
    (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
        _terminal_soe_shortfall(d, soe_cev_kept, rem_dig_kept, rem_load_kept)   # Change 3

    time_labels = build_time_labels_days(d.t_start, d.delta_T, n_days_keep, nKd)
    xticks = multiday_xticks(n_days_keep, nKd, d.t_start, d.delta_T)

    # Per-day physical/additive breakdown for the Approach 0 vs Approach 1 report
    # (§ write_approach_comparison, 5_Output.jl). Demand-charge $ and missed-work
    # penalty $ are whole-run concepts (a single peak, a single end-of-run
    # backlog) and are deliberately NOT split per day here -- see that function.
    day_costs = NamedTuple[]
    for day in 1:n_days_keep
        dlog = klog[klog.day .== day, :]
        push!(day_costs, (; day,
                          grid_energy = sum(dlog.grid_kW) * d.delta_T,
                          energy_cost = sum(dlog.grid_kW .* dlog.price) * d.delta_T,
                          co2 = sum(dlog.grid_kW .* dlog.co2) * d.delta_T,
                          nc_peak = isempty(dlog.grid_kW) ? 0.0 : maximum(dlog.grid_kW),
                          transit_h = count(==(0), dlog.mcs_node) * d.delta_T,
                          travel_cost = d.rho_labor * d.delta_T * count(==(0), dlog.mcs_node)))
    end

    return (; d, time_labels, xticks, log = klog, solve_log,
              n_days_keep, replan_by_day, day_costs,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              est, nK = n_kept, nK_day = nKd, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = real_SOE_MCS[:, n_kept + 1],
              shortfall_kWh, shortfall_hours, shortfall_penalty_cost,
              n_obs_total, n_infeasible, elapsed,
              approach = 2, plant, n_capped = n_capped_total, n_scenarios)
end

# =============================================================================
# APPROACH 0 — ONE-SHOT PER DAY: for each kept day, solve that day's FIXED 24h
# window ONCE (at its own 8:00), then EXECUTE that fixed plan for the whole day
# (no re-optimization within the day). The NEXT day's solve starts from the
# REAL (plant-carried) state left at the end of the previous day's execution --
# not from any forecast -- so day-to-day recovery is still genuinely tested.
# Unlike Approach 1, this needs NO buffer day: each day's own window already
# ends exactly at the next day's 8:00, so there is no "looking past the
# horizon" edge case to absorb. Interval-by-interval it still calls
# apply_and_simulate! -- the SAME shared function run_mpc uses -- against the
# SAME `pool`, so the two approaches' realized power draws are comparable; the
# only difference is re-solving every interval vs re-solving once per day.
# NO FALLBACK: if a day's own 8:00 MILP is infeasible, this errors (same
# philosophy as the single-day Shrinking_Horizon sibling's run_one_shot).
#
# TWO PLANT MODES (kwarg `plant`), both replaying the SAME per-day MILPs:
#
#   plant = :mean     APPROACH 0-MEAN — the DETERMINISTIC baseline. The plant uses
#                     the same mean mu the MILP planned on, and each interval
#                     realizes its single planned activity in full. Nothing is
#                     sampled, so realized == planned EXACTLY within each day: the
#                     reported KPIs are the per-day MILPs' own optima, chained
#                     through the real end-of-day carry-over. No pool sample is
#                     consumed, so it cannot disturb a :sampled run sharing the
#                     same pool, and it is reproducible without a seed.
#
#   plant = :sampled  APPROACH 0-SAMPLED — the stochastic baseline (the original
#                     behaviour, still the default). Each day's plan is fixed at
#                     that day's 8:00 but the plant draws realized powers from the
#                     shared pool, so the day drifts with no re-planning to correct
#                     it until the next day's solve picks up the real state.
#
# Exactly ONE mode runs per call; which you pick decides what the reported
# A1-vs-A0 gap means (see run_receding's approach0_plant kwarg).
# =============================================================================
function run_one_shot(d, pool::ActivityPowerPool; plant::Symbol = :sampled,
                      time_limit_sec::Float64 = Inf,
                      multi_activity::Bool = false,
                      require_site_visit::Bool = false,
                      single_visit_per_site::Bool = false,
                      n_days::Union{Nothing, Int} = nothing,
                      seed::Int = 1)
    plant in (:sampled, :mean) ||
        error("run_one_shot: plant must be :sampled or :mean, got :$plant")
    Random.seed!(seed)
    nKd = length(collect(d.K))
    n_days_keep = n_days === nothing ? d.n_days : max(1, n_days)
    n_kept = n_days_keep * nKd
    # This run's OWN walk through the shared pool -- independent of run_mpc's.
    cursor = new_cursor(pool)

    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    nN_work  = length(d.hours_digging)
    rem_dig  = zeros(nN_work)
    rem_load = zeros(nN_work)
    quota_dig(day)  = float.(d.dig_by_day[clamp(day, 1, length(d.dig_by_day))])
    quota_load(day) = float.(d.load_by_day[clamp(day, 1, length(d.load_by_day))])

    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    mcs_node    = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]
    rng = MersenneTwister(seed)

    log = DataFrame(
        day = Int[], gstep = Int[], k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[])

    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch    = zeros(nM, n_kept)
    real_P_dch   = zeros(nM, n_kept)
    real_L_trv   = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)
    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = fill("", n_kept)

    println("Running Approach 0 (one-shot PER DAY: solve once at each day's 8:00, replay open-loop): ",
             "$n_days_keep day(s), $n_kept steps total")
    println("  plant                  : ", plant === :mean ?
            ":mean (DETERMINISTIC -- realized power pinned to mu; realized == planned)" :
            ":sampled (stochastic -- realized power drawn from the shared pool)")
    println("  planning power (mu)    : ", round.(pool.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit      : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s / day-solve" : "none (solve each day to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_capped_total = 0
    gstep = 0
    day_costs = NamedTuple[]

    for day in 1:n_days_keep
        rem_dig  .+= quota_dig(day)
        rem_load .+= quota_load(day)
        peak_nc = 0.0; peak_op = 0.0   # resets daily, matching run_mpc
        day_off = (day - 1) * nKd
        g0_day  = day_off + 1
        Kend    = day_off + nKd
        K_win   = g0_day:Kend

        # (1) OPTIMISE -- ONCE, over exactly this day's fixed 24h window, from
        # the REAL state carried over from the previous day's execution.
        model = build_window_model(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                   rem_dig, rem_load, hist,
                                   peak_nc, peak_op, pool.mu;
                                   require_site_visit = require_site_visit,
                                   single_visit_per_site = single_visit_per_site,
                                   time_limit_sec = time_limit_sec)
        stat = string(termination_status(model))
        has_values(model) || error("Approach 0 (one-shot): day $day's 8:00 whole-day MILP was ",
                                   "INFEASIBLE (status=$stat); there is no fixed plan to execute.")

        row0 = nrow(log) + 1
        for k0 in 1:nKd
            gstep += 1
            g0 = day_off + k0
            gk = day_off + k0
            clk = clock_day_label(d.t_start, d.delta_T, day, k0)

            for m in d.M; real_SOE_MCS[m, gk] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, gk] = soe_cev[e]; end

            # (2)+(3)+(4) APPLY / SIMULATE (shared pool draw) / ADVANCE -- the
            # SAME function run_mpc calls, replayed against the SAME day's model.
            step = apply_and_simulate!(model, g0, Kend, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work, gk;
                                       plant_mode = plant)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

            for e in d.E
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                site !== nothing && (real_cev_act[e][gk] = activity_label(model, d, e, site, g0))
            end
            real_mcs_act[gk] = mcs_status_label(model, d, g0)

            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                        pool.mu[1], pool.mu[2], pool.mu[3], pool.mu[4],
                        pool.sd[1], pool.sd[2], pool.sd[3], pool.sd[4], n_obs_total))
        end

        if day == n_days_keep
            for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end
        end

        dlog = log[row0:nrow(log), :]
        push!(day_costs, (; day,
                          grid_energy = sum(dlog.grid_kW) * d.delta_T,
                          energy_cost = sum(dlog.grid_kW .* dlog.price) * d.delta_T,
                          co2 = sum(dlog.grid_kW .* dlog.co2) * d.delta_T,
                          nc_peak = isempty(dlog.grid_kW) ? 0.0 : maximum(dlog.grid_kW),
                          transit_h = count(==(0), dlog.mcs_node) * d.delta_T,
                          travel_cost = d.rho_labor * d.delta_T * count(==(0), dlog.mcs_node)))
    end

    elapsed = time() - t0
    @printf("Approach 0 one-shot-per-day (plant = :%s) done in %.1f s (%d plant realizations, %d day-solves)\n",
            plant, elapsed, n_obs_total, n_days_keep)
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
        _terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load)   # Change 3

    # csv_mcs_cev_soe (5_Output.jl) indexes res.time_labels, and
    # write_approach_comparison calls it on BOTH res0 and res1 -- so Approach 0
    # must publish the same boundary labels run_mpc does, or the comparison
    # writer dies with `NamedTuple has no field time_labels` AFTER a full run.
    time_labels = build_time_labels_days(d.t_start, d.delta_T, n_days_keep, nKd)

    return (; d, time_labels, log, day_costs, n_days_keep,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              nK = n_kept, nK_day = nKd, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              shortfall_kWh, shortfall_hours, shortfall_penalty_cost,
              n_obs_total, n_infeasible = 0, elapsed,
              approach = 0, plant, n_capped = n_capped_total)
end

end # module MPCLoop
