# #############################################################################
# MPCLoop.jl  —  module MPCLoop
# -----------------------------------------------------------------------------
# The closed-loop driver that ties the OPTIMISE and LEARN halves together. For
# every 15-min interval it: (1) solves the shrinking-horizon window MILP over the
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
# #############################################################################
module MPCLoop

using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, build_time_labels, safe_get,
                BayesianActivityEstimator,
                ActivityPowerPool, new_cursor, next_power!
using ..MCSModel: build_window_model

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
# interval `k`: "Charging" if real power is delivered, else the argmax activity,
# else "". Factored out so the replanning-grid capture (every column of every
# re-plan step, MPC-only) and a single-interval lookup (used by BOTH approaches
# to fill in real_cev_act) share the exact same rule instead of two copies that
# could quietly drift apart.
# -----------------------------------------------------------------------------
function activity_label(model, d, e, site, k)
    vals   = [value(model[:u][e, site, a, k]) for a in eachindex(d.B)]
    p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
    return p_into > 1e-6 ? "Charging" : (sum(vals) < 0.5 ? "" : ACT_NAME[d.B[argmax(vals)]])
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
                             plant_mode::Symbol = :sampled)
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
        real_P_ch[m, k0]  = value(model[:P_ch_tot][m, k0])
        real_P_dch[m, k0] = value(model[:P_dch_tot][m, k0])
        real_L_trv[m, k0] = value(model[:L_trv_tot][m, k0])
        real_loc[m, k0]   = let nh = findfirst(i -> value(model[:z][m, i, k0]) > 0.5, d.N)
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
        for a in eachindex(row)
            row[a] > 1e-9 || continue
            # :mean pins the realized power to the planning mean and does NOT advance
            # the cursor; :sampled consumes the next pre-drawn sample for this pair.
            pt[a] = use_mean ? pool.mu[a] : next_power!(pool, cursor, e, a)
        end
        p_true[e] = pt
        sum(row) > 1e-9 && (n_obs_added += 1)
        site = findfirst(i -> d.A[i, e] == 1, d.N)
        if site !== nothing
            real_P_work[site, e, k0] =
                (row[1] * pt[1] + row[2] * pt[2] + row[3] * pt[3]) / d.delta_T
        end
    end

    # (4) ADVANCE the real MCS energy + position.
    for m in d.M
        soe_mcs[m] = value(model[:SOE_MCS][m, k0 + 1])
        mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, k0, nK, d)
    end
    # # The CEV balance is CLAMPED to [SOE_min, SOE_max]. The clamp is a guard, not
    # # physics: whenever it bites, energy is silently created (low side) or discarded
    # # (high side) and the reported SOE stops matching the integrated power. It can
    # # only bite in :sampled mode (in :mean mode the realized balance equals the MILP's
    # # own, which already respects the bounds), so we COUNT the events and report them
    # # rather than letting a "0 infeasible windows" run hide them.
    # n_clamped = 0
    # for e in d.E
    #     charged   = sum(value(model[:P_MCS_CEV][m, i, e, k0]) for m in d.M, i in d.N_c) * d.delta_T
    #     work_true = dot(a_real[e], p_true[e])
    #     raw       = soe_cev[e] + charged - work_true
    #     soe_cev[e] = clamp(raw, d.SOE_CEV_min[e], d.SOE_CEV_max[e])
    #     abs(raw - soe_cev[e]) > 1e-9 && (n_clamped += 1)
    # end
    # CAP each CEV's realized dig/load hours by what its available energy could
    # actually pay for, BEFORE crediting rem_dig/rem_load or logging hist. This
    # replaces the old after-the-fact SOE clamp, which silently created/discarded
    # energy instead of reflecting that the machine ran out of charge mid-task.
    n_capped = 0
    for e in d.E
        charged   = sum(value(model[:P_MCS_CEV][m, i, e, k0]) for m in d.M, i in d.N_c) * d.delta_T
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
        push!(hist[e], (applied_act_index(model, d, e, k0), copy(a_real[e])))
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

# =============================================================================
# MAIN CLOSED LOOP
# =============================================================================
function run_mpc(d, pool::ActivityPowerPool; shrinking::Bool = true, H::Int = 16,
                    time_limit_sec::Float64 = Inf,
                    multi_activity::Bool = false,
                    require_site_visit::Bool = false,
                    single_visit_per_site::Bool = false,
                    mcmc_samples::Int = 500,
                    plant::Symbol = :sampled,
                    seed::Int = 1)
    plant in (:sampled, :mean) ||
        error("run_mpc: plant must be :sampled or :mean, got :$plant")
    Random.seed!(seed)
    K_all = collect(d.K)
    nK = length(K_all)
    time_labels = build_time_labels(d.t_start, d.delta_T, nK)
    # This run's OWN walk through the shared pool: same underlying samples as
    # any other approach run against `pool`, independent consumption order.
    cursor = new_cursor(pool)

    # ---- REAL physical state carried across steps (the "plant") ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    mcs_node = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]
    rem_dig  = copy(float.(d.hours_digging))
    rem_load = copy(float.(d.hours_loading_swinging))
    # SHARED applied-activity history, one growing list per CEV. Each entry is
    # (act, hrs): act = applied activity index; hrs = realized [dig,load,trv,idle] h.
    # Seeds precedence, pacing AND the rest rule inside build_window_model.
    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    peak_nc = 0.0; peak_op = 0.0

    # ---- online learner ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)

    # ---- analyst log (one row per applied interval) ----
    log = DataFrame(
        k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[])

    # ---- realized per-interval capture for the reference-style figures ----
    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, nK)
    real_P_dch = zeros(nM, nK)
    real_L_trv = zeros(nM, nK)
    real_SOE_MCS = zeros(nM, nK + 1)
    real_SOE_CEV = zeros(nE, nK + 1)
    real_P_work  = zeros(nN, nE, nK)
    real_loc     = zeros(Int, nM, nK)

    # ---- replanning grids (row = re-plan step, col = interval planned) ----
    plan_grid_kW = fill(NaN, nK, nK)
    plan_mcs_soe = fill(NaN, nK, nK)
    plan_cev_soe = [fill(NaN, nK, nK) for _ in d.E]
    plan_cev_act = [fill("", nK, nK)  for _ in d.E]
    plan_mcs_act = fill("", nK, nK)   # MCS status: Idle / Charging (grid) / Serving CEV / Traveling

    # ---- REALIZED (applied) activity per interval, one label per step ----
    # This is the diagonal of the plan grids: the activity the loop ACTUALLY
    # executed each step (vs the 08:00 forward plan). Used by the plan-vs-actual
    # activity report to highlight where the realised day diverged from the plan.
    real_cev_act = [fill("", nK) for _ in d.E]
    real_mcs_act = fill("", nK)

    hmode = shrinking ? "shrinking" : "fixed H=$H"
    println("Running Approach 1 (closed-loop MPC, 15-min steps, $hmode horizon): $nK steps")
    println("  plant                : ", plant === :mean ?
            ":mean (DETERMINISTIC -- realized power pinned to mu)" :
            ":sampled (stochastic -- realized power drawn from the shared pool)")
    println("  planning power (mu)  : ", round.(est.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd    : ", round.(pool.sd, digits = 2), " kW")
    # SOLVER TIME LIMIT (control point #2 -> forwarded to build_window_model).
    println("  solver time limit    : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s / window" : "none (solve each window to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_infeasible = 0
    n_capped_total = 0

    for k0 in 1:nK
        # record start-of-interval MCS/CEV SOE (boundary k0)
        for m in d.M; real_SOE_MCS[m, k0] = soe_mcs[m]; end
        for e in d.E; real_SOE_CEV[e, k0] = soe_cev[e]; end

        K_win = shrinking ? (k0:nK) : (k0:min(k0 + H - 1, nK))

        # (1) OPTIMISE
        model = build_window_model(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                   rem_dig, rem_load, hist,
                                   peak_nc, peak_op, est.mu;
                                   require_site_visit = require_site_visit,
                                   single_visit_per_site = single_visit_per_site,
                                   time_limit_sec = time_limit_sec)
        stat = string(termination_status(model))
        cur_node = mcs_node[1]

        # NO FALLBACK: infeasible under HARD constraints -> hold the plant still.
        if !has_values(model)
            n_infeasible += 1
            @warn "No feasible solution at step k=$k0 under HARD constraints; holding state (no fallback)." status=stat
            push!(log, (k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        0.0, 0.0, 0.0, soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), cur_node,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
            for m in d.M; real_loc[m, k0] = cur_node; end
            # A held (infeasible) interval is a BREAK: idle for every CEV and the MCS.
            for e in d.E; real_cev_act[e][k0] = "Idle"; end
            real_mcs_act[k0] = "Idle"
            # ... and record it in history so the rest rule counts it correctly next window.
            for e in d.E; push!(hist[e], (length(d.B), [0.0, 0.0, 0.0, d.delta_T])); end
            continue
        end

        # Save this window's FULL forward plan into the replanning grids.
        for k in K_win
            plan_grid_kW[k0, k] = sum(value(model[:P_ch_tot][m, k]) for m in d.M)
            plan_mcs_soe[k0, k] = value(model[:SOE_MCS][1, k + 1])
            for e in d.E
                plan_cev_soe[e][k0, k] = value(model[:SOE_CEV][e, k + 1])
                # "Charging" is shown only when REAL power is delivered into this CEV
                # (sum_m P_MCS_CEV > 0), not merely when the plug-in permission bit
                # mu=1 (mu can be 1 with zero power flow) -- see activity_label.
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                site !== nothing && (plan_cev_act[e][k0, k] = activity_label(model, d, e, site, k))
            end
            plan_mcs_act[k0, k] = mcs_status_label(model, d, k)
        end

        # The APPLIED cell (row k0, col k0) is what actually happened this step.
        for e in d.E; real_cev_act[e][k0] = plan_cev_act[e][k0, k0]; end
        real_mcs_act[k0] = plan_mcs_act[k0, k0]

        # (2)+(3)+(4) APPLY / SIMULATE (shared pool draw) / ADVANCE -- see
        # apply_and_simulate! above; identical logic to what run_one_shot calls.
        step = apply_and_simulate!(model, k0, nK, d, pool, cursor, rng, multi_activity,
                                   soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                   real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work;
                                   plant_mode = plant)
        n_obs_total += step.n_obs_added
        n_capped_total += step.n_capped

        peak_nc = max(peak_nc, step.grid_kW)
        in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

        push!(log, (k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                    step.grid_kW, step.dch_kW, step.work_kW,
                    soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                    est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                    est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
    end

    # final boundary SOE
    for m in d.M; real_SOE_MCS[m, nK + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, nK + 1] = soe_cev[e]; end

    elapsed = time() - t0
    @printf("Approach 1 (plant = :%s) done in %.1f s (%d plant realizations)\n",
            plant, elapsed, n_obs_total)
    n_infeasible > 0 && @printf("  NOTE: %d/%d windows were INFEASIBLE under the HARD constraints (no fallback);\n        the plant HELD state for those intervals.\n", n_infeasible, nK)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy (task could not fully\n        complete before hitting the SOE floor); the shortfall is reflected honestly in\n        rem_dig/rem_load.\n", n_capped_total)
    println("  fixed power model (mu) : ", round.(est.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")

    # ---- Phase-1 KPIs from the realized trajectory ----
    total_energy = sum(log.grid_kW) * d.delta_T
    total_cost   = sum(log.grid_kW .* log.price) * d.delta_T
    total_co2    = sum(log.grid_kW .* log.co2)  * d.delta_T
    nc_peak      = isempty(log.grid_kW) ? 0.0 : maximum(log.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in log.k]
    op_peak      = any(op_mask) ? maximum(log.grid_kW[op_mask]) : 0.0
    missed       = sum(rem_dig) + sum(rem_load)
    transit_intervals = count(==(0), log.mcs_node)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals

    return (; d, time_labels, log,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act, plan_mcs_act,
              est, nK, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              n_obs_total, n_infeasible, elapsed,
              approach = 1, plant, n_capped = n_capped_total)
end

# =============================================================================
# APPROACH 0 — ONE-SHOT: solve the FULL 24h MILP ONCE at 8:00, then EXECUTE
# that fixed plan for the whole day (no re-optimization). Interval-by-interval
# it still calls apply_and_simulate! -- the SAME shared function run_mpc uses --
# so the plant physics are identical; only the control strategy differs (replan
# every step vs commit once at 8:00).
#
# TWO PLANT MODES (kwarg `plant`), both replaying the SAME single 08:00 MILP:
#
#   plant = :mean     APPROACH 0-MEAN — the DETERMINISTIC baseline. The plant uses
#                     the same mean mu the MILP planned on, and each interval
#                     realizes its single planned activity in full. Nothing is
#                     sampled, so realized == planned EXACTLY: the reported KPIs
#                     are the whole-day MILP's own optimum, evaluated over the full
#                     day. This is the clean reproduction of Avik's single-shot
#                     deterministic model and the reference cost that any stochastic
#                     run should be measured against. No pool sample is consumed, so
#                     it cannot disturb a :sampled run sharing the same pool.
#
#   plant = :sampled  APPROACH 0-SAMPLED — the stochastic baseline (the original
#                     behaviour, still the default). The plan is fixed at 08:00 but
#                     the plant draws realized per-(CEV, activity) powers from the
#                     shared pool, so the day drifts away from the plan with no
#                     re-planning to correct it.
#
# Running BOTH and comparing decomposes the headline gap:
#   (0-mean  ->  0-sampled)  = the cost of plan drift with NO feedback
#   (0-sampled ->  1)        = the value of re-planning against that drift
#   (0-mean  ->  1)          = the net of the two, i.e. the headline number alone
# =============================================================================
function run_one_shot(d, pool::ActivityPowerPool; time_limit_sec::Float64 = Inf,
                      multi_activity::Bool = false,
                      require_site_visit::Bool = false,
                      single_visit_per_site::Bool = false,
                      plant::Symbol = :sampled,
                      seed::Int = 1)
    plant in (:sampled, :mean) ||
        error("run_one_shot: plant must be :sampled or :mean, got :$plant")
    Random.seed!(seed)
    K_all = collect(d.K)
    nK = length(K_all)
    time_labels = build_time_labels(d.t_start, d.delta_T, nK)
    # This run's OWN walk through the shared pool -- independent of run_mpc's.
    cursor = new_cursor(pool)

    # ---- REAL physical state carried across steps (the "plant") ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    mcs_node = [first(d.N_g) for _ in d.M]
    mcs_transit = Any[nothing for _ in d.M]
    rem_dig  = copy(float.(d.hours_digging))
    rem_load = copy(float.(d.hours_loading_swinging))
    hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
    peak_nc = 0.0; peak_op = 0.0
    rng = MersenneTwister(seed)

    log = DataFrame(
        k = Int[], clock = String[], price = Float64[], co2 = Float64[],
        grid_kW = Float64[], dch_kW = Float64[], work_kW = Float64[],
        soe_mcs = Float64[], soe_cev1 = Float64[], soe_cev2 = Float64[],
        mcs_node = Int[],
        est_dig = Float64[], est_load = Float64[], est_trv = Float64[], est_idle = Float64[],
        unc_dig = Float64[], unc_load = Float64[], unc_trv = Float64[], unc_idle = Float64[],
        n_obs = Int[])

    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, nK)
    real_P_dch = zeros(nM, nK)
    real_L_trv = zeros(nM, nK)
    real_SOE_MCS = zeros(nM, nK + 1)
    real_SOE_CEV = zeros(nE, nK + 1)
    real_P_work  = zeros(nN, nE, nK)
    real_loc     = zeros(Int, nM, nK)
    real_cev_act = [fill("", nK) for _ in d.E]
    real_mcs_act = fill("", nK)

    pmode_txt = plant === :mean ?
        ":mean (DETERMINISTIC -- realized power pinned to mu; realized == planned)" :
        ":sampled (stochastic -- realized power drawn from the shared pool)"
    println("Running Approach 0 (one-shot 8:00 plan, executed open-loop, no replanning): $nK steps")
    println("  plant                  : ", pmode_txt)
    println("  planning power (mu)    : ", round.(pool.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit      : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s" : "none (solve to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_capped_total = 0

    # (1) OPTIMISE -- ONCE, over the whole day. No fallback: if the 8:00
    # whole-day plan itself is infeasible there is nothing to execute.
    model = build_window_model(d, K_all, soe_mcs, soe_cev, mcs_node, mcs_transit,
                               rem_dig, rem_load, hist,
                               peak_nc, peak_op, pool.mu;
                               require_site_visit = require_site_visit,
                               single_visit_per_site = single_visit_per_site,
                               time_limit_sec = time_limit_sec)
    stat = string(termination_status(model))
    has_values(model) || error("Approach 0 (one-shot): the 8:00 whole-day MILP was INFEASIBLE ",
                               "(status=$stat); there is no fixed plan to execute.")

    for k0 in 1:nK
        for m in d.M; real_SOE_MCS[m, k0] = soe_mcs[m]; end
        for e in d.E; real_SOE_CEV[e, k0] = soe_cev[e]; end

        # (2)+(3)+(4) APPLY / SIMULATE (shared pool draw) / ADVANCE -- the
        # SAME function run_mpc calls, replayed against the SAME single model.
        step = apply_and_simulate!(model, k0, nK, d, pool, cursor, rng, multi_activity,
                                   soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                   real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work;
                                   plant_mode = plant)
        n_obs_total += step.n_obs_added
        n_capped_total += step.n_capped

        for e in d.E
            site = findfirst(i -> d.A[i, e] == 1, d.N)
            site !== nothing && (real_cev_act[e][k0] = activity_label(model, d, e, site, k0))
        end
        real_mcs_act[k0] = mcs_status_label(model, d, k0)

        peak_nc = max(peak_nc, step.grid_kW)
        in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

        push!(log, (k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                    step.grid_kW, step.dch_kW, step.work_kW,
                    soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                    pool.mu[1], pool.mu[2], pool.mu[3], pool.mu[4],
                    pool.sd[1], pool.sd[2], pool.sd[3], pool.sd[4], n_obs_total))
    end

    for m in d.M; real_SOE_MCS[m, nK + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, nK + 1] = soe_cev[e]; end

    elapsed = time() - t0
    @printf("Approach 0 one-shot (plant = :%s) done in %.1f s (%d plant realizations)\n",
            plant, elapsed, n_obs_total)
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

    return (; d, time_labels, log,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              nK, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              n_obs_total, n_infeasible = 0, elapsed,
              approach = 0, plant, n_capped = n_capped_total)
end

end # module MPCLoop
