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
# #############################################################################
module MPCLoop

using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, clock_day_label, build_time_labels_days,
                multiday_xticks, safe_get, BayesianActivityEstimator
using ..MCSModel: build_window_model

export run_mpc

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

# =============================================================================
# MAIN CLOSED LOOP
# =============================================================================
function run_mpc(d; time_limit_sec::Float64 = 60.0,
                    multi_activity::Bool = false,
                    require_site_visit::Bool = false,
                    single_visit_per_site::Bool = false,
                    mcmc_samples::Int = 500,
                    n_days::Union{Nothing, Int} = nothing,
                    seed::Int = 1)
    Random.seed!(seed)
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

    # ---- online learner ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)

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

    println("Running Scenario 1 (closed-loop MPC, 15-min steps, multi-day receding horizon): $n_kept steps ($n_days_keep kept days)")
    println("  prior power estimate : ", round.(est.mu, digits = 2), " kW")
    println("  (hidden) true power  : ", d.true_powers, " kW")
    # SOLVER TIME LIMIT (control point #2 -> forwarded to build_window_model).
    println("  solver time limit    : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s / window" : "none (solve each window to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_infeasible = 0
    gstep = 0
    missed_kept = 0.0

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

            # (2) APPLY interval k0's decisions.
            grid_kW = sum(value(model[:P_ch_tot][m, g0]) for m in d.M)   # total grid draw this step (all MCS)
            dch_kW  = sum(value(model[:P_dch_tot][m, g0]) for m in d.M)  # total discharge to CEVs this step
            # Node MCS 1 is parked at now (0 = in transit, i.e. no z bit set).
            cur_node = let nh = findfirst(i -> value(model[:z][1, i, g0]) > 0.5, d.N)
                nh === nothing ? 0 : nh
            end
            kept && push!(solve_log, (day, k0, clk, stat, objective_value(model),
                                      100 * (try relative_gap(model) catch; NaN end),
                                      try solve_time(model) catch; NaN end))

            # realized per-MCS charging / discharging / travel + location
            if kept
                for m in d.M
                    real_P_ch[m, gk]  = value(model[:P_ch_tot][m, g0])
                    real_P_dch[m, gk] = value(model[:P_dch_tot][m, g0])
                    real_L_trv[m, gk] = value(model[:L_trv_tot][m, g0])
                    real_loc[m, gk]   = let nh = findfirst(i -> value(model[:z][m, i, g0]) > 0.5, d.N)
                        nh === nothing ? 0 : nh
                    end
                end
            end

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
                plan_grid_kW[k0, kl] = sum(value(model[:P_ch_tot][m, k]) for m in d.M)
                plan_mcs_soe[k0, kl] = value(model[:SOE_MCS][1, k + 1])
                for e in d.E
                    plan_cev_soe[e][k0, kl] = value(model[:SOE_CEV][e, k + 1])
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    if site !== nothing
                        # Combined activity label. "Charging" is shown only when REAL power
                        # is delivered into this CEV (sum_m P_MCS_CEV > 0), not merely when
                        # the plug-in permission bit mu=1 (mu can be 1 with zero power flow).
                        # Using delivered power keeps this grid consistent with the MCS grid
                        # (P_dch_tot>0 <=> some CEV is being served).
                        vals   = [value(model[:u][e, site, a, k]) for a in eachindex(d.B)]
                        p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
                        plan_cev_act[e][k0, kl] =
                            p_into > 1e-6 ? "Charging" :
                            (sum(vals) < 0.5 ? "" : ACT_NAME[d.B[argmax(vals)]])
                    end
                end
                # MCS status this interval (mutually exclusive by node/state).
                pch  = sum(value(model[:P_ch_tot][m, k])  for m in d.M)
                pdch = sum(value(model[:P_dch_tot][m, k]) for m in d.M)
                parked = any(value(model[:z][m, i, k]) > 0.5 for m in d.M, i in d.N)
                plan_mcs_act[k0, kl] = pch  > 1e-6 ? "Charging (grid)" :
                                       pdch > 1e-6 ? "Serving CEV"     :
                                       !parked     ? "Traveling"       : "Idle"
            end

            # The APPLIED cell (row k0, col k0) is what actually happened this step.
            if kept
                for e in d.E; real_cev_act[e][gk] = plan_cev_act[e][k0, k0]; end
                real_mcs_act[gk] = plan_mcs_act[k0, k0]
            end

            # (3) SIMULATE realized activity split.
            a_real = Dict(e => realized_activity_durations(rng, model, e, g0, d;
                                                           multi = multi_activity) for e in d.E)

            # (2.5) STOCHASTIC PLANT (FORK B): the Bayesian model is fitted ONCE and its
            # posterior (est.mu, est.sd) is held FIXED for the whole day. Each interval the
            # real machine's per-activity power is a fresh PER-EXCAVATOR draw from that fixed
            # curve, Normal(est.mu, est.sd) truncated at 0. Idle has est.sd = 0, so its draw
            # collapses to 0 (no power lost while idle). The SAME p_true[e] drives the battery
            # drain below, so the realized consumption matches what was actually sampled. The
            # controller re-plans on the fixed mean; only the plant realization is random.
            p_true = Dict(e => [max(est.mu[j] + est.sd[j] * randn(rng), 0.0)
                                for j in eachindex(est.mu)] for e in d.E)

            for e in d.E
                row = a_real[e]
                sum(row) > 1e-9 && (n_obs_total += 1)
                # realized work power (dig+load+travel, excluding idle) at the CEV's site
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                if kept && site !== nothing
                    real_P_work[site, e, gk] =
                        (a_real[e][1] * p_true[e][1] + a_real[e][2] * p_true[e][2] +
                         a_real[e][3] * p_true[e][3]) / d.delta_T
                end
            end

            # (4) ADVANCE the real MCS energy + position.
            for m in d.M
                ch  = value(model[:P_ch_tot][m, g0])
                dch = value(model[:P_dch_tot][m, g0])
                ltr = value(model[:L_trv_tot][m, g0])
                soe_mcs[m] = soe_mcs[m] + d.eta_ch_dch[m] * ch * d.delta_T -
                             (dch * d.delta_T) / d.eta_ch_dch[m] - ltr
                # Where the MCS will be at the start of the next step (node, or mid-drive transit).
                mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, g0, Kend, d)
            end
            for e in d.E
                charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T  # kWh actually delivered
                work_true = dot(a_real[e], p_true[e])   # kWh actually consumed = realized hours . sampled powers
                # CEV SOE advances on REALIZED (stochastic) consumption, clamped to its physical range.
                soe_cev[e] = clamp(soe_cev[e] + charged - work_true, d.SOE_CEV_min[e], d.SOE_CEV_max[e])
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

            peak_nc = max(peak_nc, grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, grid_kW))
            work_kW = sum(dot(a_real[e], d.true_powers) for e in d.E) / d.delta_T

            push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        grid_kW, dch_kW, work_kW,
                        soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), cur_node,
                        est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                        est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
        end

        # final boundary SOE snapshot for kept days
        if day == n_days_keep
            missed_kept = sum(rem_dig) + sum(rem_load)
            for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end
        end

        kept && (replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act, plan_mcs_act))
    end

    elapsed = time() - t0
    @printf("MPC loop done in %.1f s (%d stochastic-plant realizations)\n", elapsed, n_obs_total)
    n_infeasible > 0 && @printf("  NOTE: %d/%d windows were INFEASIBLE under the HARD constraints (no fallback);\n        the plant HELD state for those intervals.\n", n_infeasible, n_kept)
    println("  fixed power model (mu) : ", round.(est.mu, digits = 2), " kW")
    println("  plant sampling sd      : ", round.(est.sd, digits = 2), " kW")

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

    time_labels = build_time_labels_days(d.t_start, d.delta_T, n_days_keep, nKd)
    xticks = multiday_xticks(n_days_keep, nKd, d.t_start, d.delta_T)

    return (; d, time_labels, xticks, log = klog, solve_log,
              n_days_keep, replan_by_day,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              est, nK = n_kept, nK_day = nKd, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = real_SOE_MCS[:, n_kept + 1],
              n_obs_total, n_infeasible, elapsed)
end

end # module MPCLoop
