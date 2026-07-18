# #############################################################################
# MPCLoop.jl  —  module MPCLoop  (RECEDING, multi-day / cross-day)
# -----------------------------------------------------------------------------
# The closed-loop driver for the MULTI-DAY receding horizon. It simulates
# `n_days` reported days plus one dropped BUFFER day (D_total = n_days + 1), and
# for every 15-min interval of each day: (1) solves the CROSS-DAY window MILP
# (rest of today + `lookahead_days` future daytime blocks) from the current real
# state + the FIXED (once-calibrated) power model, (2) APPLIES only the first
# interval's decisions, (3) draws the realized per-activity power from the fixed
# posterior (FORK B: Normal(mu, sd) truncated at 0; the model is NEVER re-fitted
# online), and (4) advances the real state. Each night the deterministic overnight
# charge (Phase 2) runs and the MCS is reset to full; the CEV battery and any
# unfinished work carry into the next day. The buffer day is DROPPED from every
# reported output.
#
# Realized per-interval arrays are captured over the KEPT days (concatenated end
# to end) so the plotting/reporting modules can render the v4_real-style figure
# set from the trajectory that was actually realized.
# #############################################################################
module MPCLoop

using JuMP
using DataFrames
using Printf
using LinearAlgebra
using Random

using ..Common: in_peak, clock_label, clock_day_label, build_time_labels_days,
                multiday_xticks, safe_get, BayesianActivityEstimator
using ..MCSModel: build_window_model, phase2_overnight_charge

export run_mpc

# Plain-language activity names for the worker-facing schedule + plan grids.
const ACT_NAME = Dict(1 => "Digging", 2 => "Loading/Swinging", 3 => "Traveling", 4 => "Idle")

# -----------------------------------------------------------------------------
# Simulate the REALIZED within-interval activity split for excavator e at global
# index g0. The MILP plans ONE activity per interval; a real machine mixes tasks.
# In multi-activity mode we spend 60-100% of the interval on the planned task and
# the rest idling. Returns hours per activity (summing to delta_T).
# -----------------------------------------------------------------------------
function realized_activity_durations(rng, model, e, g0, d; multi::Bool = true)
    dt = d.delta_T
    a = zeros(length(d.B))
    idle = length(d.B)
    planned = 0
    for i in d.N_c, (ai, act) in enumerate(d.B)
        if value(model[:u][e, i, act, g0]) > 0.5
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
# Where will the MCS be at the START of the NEXT interval (g0+1)? Returns
# (node, transit): node = parked node index (0 if mid-drive); transit = nothing
# or (i,j,r) = mid-drive on arc i->j with r intervals left.
# -----------------------------------------------------------------------------
function advance_mcs_state(model, m, g0, gend, d)
    z = model[:z]; y = model[:y_trv]
    Kw = axes(z)[3]
    knext = g0 + 1
    if knext > gend || !(knext in Kw)
        node = findfirst(i -> value(z[m, i, g0]) > 0.5, d.N)
        return (node === nothing ? first(d.N_g) : node, nothing)
    end
    node = findfirst(i -> value(z[m, i, knext]) > 0.5, d.N)
    node !== nothing && return (node, nothing)
    for i in d.N, j in d.N
        i == j && continue
        if value(y[m, i, j, knext]) > 0.5
            r = 0; k = knext
            while k <= gend && value(y[m, i, j, k]) > 0.5
                r += 1; k += 1
            end
            return (0, (i, j, r))
        end
    end
    node0 = findfirst(i -> value(z[m, i, g0]) > 0.5, d.N)
    return (node0 === nothing ? first(d.N_g) : node0, nothing)
end

# Applied (scheduled) activity index 1..4 for CEV e at g0, read from the executed
# u decision (1=dig, 2=load, 3=travel, 4=idle). Appended to the shared history so
# the window model knows what was really done.
function applied_act_index(model, d, e, g0)
    for i in d.N_c, (ai, act) in enumerate(d.B)
        value(model[:u][e, i, act, g0]) > 0.5 && return ai
    end
    return length(d.B)   # nothing scheduled -> idle (a break)
end

# Worker-facing planned activity name for CEV e at g0.
function planned_activity(model, d, e, g0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    site === nothing && return "Off (home)"
    vals = [value(model[:u][e, site, a, g0]) for a in eachindex(d.B)]
    sum(vals) < 0.5 && return "Off (home)"
    return ACT_NAME[d.B[argmax(vals)]]
end

# "Charging" is shown only when REAL power is delivered into the CEV
# (sum_m P_MCS_CEV > 0), not merely when the plug-in permission bit mu=1 (mu can
# be 1 with zero power flow). Keeps the worker schedule consistent with the MCS.
function cev_should_charge(model, d, e, g0)
    site = findfirst(i -> d.A[i, e] == 1, d.N)
    site === nothing && return "No"
    p_into = sum(value(model[:P_MCS_CEV][m, site, e, g0]) for m in d.M)
    return p_into > 1e-6 ? "Yes" : "No"
end
mcs_should_charge(model, d, g0) =
    (sum(value(model[:P_ch_tot][m, g0]) for m in d.M) > 1e-6) ? "Yes" : "No"

# =============================================================================
# MAIN MULTI-DAY CLOSED LOOP
# =============================================================================
function run_mpc(d; time_limit_sec::Float64 = 60.0,
                    multi_activity::Bool = false,
                    require_site_visit::Bool = false,
                    single_visit_per_site::Bool = false,
                    mcmc_samples::Int = 500,
                    soft_prec::Bool = false, soft_pace::Bool = false,
                    soft_term::Bool = false, term_tol::Float64 = 0.1,
                    n_days::Union{Nothing, Int} = nothing,
                    lookahead_days::Int = 1,
                    seed::Int = 1)
    Random.seed!(seed)
    nKd = length(collect(d.K))                         # daytime steps per day (e.g. 40)

    # ---- multi-day setup: keep n_days_keep, simulate one extra buffer day ----
    n_days_keep = n_days === nothing ? d.n_days : max(1, n_days)
    D_total     = n_days_keep + 1
    G           = D_total * nKd                        # total daytime intervals in the horizon
    n_kept      = n_days_keep * nKd                    # kept concatenated steps

    # ---- REAL physical state carried ACROSS DAYS (the "plant") ----
    soe_mcs  = copy(float.(d.SOE_MCS_ini))
    soe_cev  = copy(float.(d.SOE_CEV_ini))
    nN_work  = length(d.hours_digging)
    rem_dig  = zeros(nN_work)
    rem_load = zeros(nN_work)
    # PER-DAY work quota: each reported day issues its own per-site dig/load quota
    # (d.dig_by_day / d.load_by_day). The dropped buffer day (and any out-of-range
    # day) gets NO fresh work. Unfinished work rolls over via rem_* below.
    quota_dig(day)  = (1 <= day <= min(n_days_keep, length(d.dig_by_day)))  ? float.(d.dig_by_day[day])  : zeros(nN_work)
    quota_load(day) = (1 <= day <= min(n_days_keep, length(d.load_by_day))) ? float.(d.load_by_day[day]) : zeros(nN_work)

    # ---- fixed (once-calibrated) power model ----
    est = BayesianActivityEstimator(d.prior_mu, d.prior_sigma; mcmc_samples = mcmc_samples)
    rng = MersenneTwister(seed)

    # ---- analyst log (one row per applied interval; `day` tags the sim day) ----
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

    # ---- worker-facing schedule columns (all sim steps; sliced to kept later) ----
    fe_time = String[]
    fe_act  = [String[] for _ in d.E]
    fe_chg  = [String[] for _ in d.E]
    fe_mcs  = String[]

    # ---- realized per-interval capture over the KEPT concatenated horizon ----
    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, n_kept)
    real_P_dch = zeros(nM, n_kept)
    real_L_trv = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_mu      = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)

    # ---- per-day overnight schedules + replan grids (kept days written out) ----
    overnight_by_day = Dict{Int, DataFrame}()
    replan_by_day    = Dict{Int, NamedTuple}()

    println("Running Scenario 1 (RECEDING horizon, closed-loop MPC, 15-min steps, CROSS-DAY lookahead):")
    println("  keeping $n_days_keep day(s); simulating $D_total (last = dropped buffer day); $nKd steps/day")
    println("  window spans current + $lookahead_days lookahead day(s); nights via MCS overnight recharge + CEV carry-over")
    println("  fixed power model (mu) : ", round.(est.mu, digits = 2), " kW")
    println("  plant sampling sd      : ", round.(est.sd, digits = 2), " kW")
    println("  (hidden) true power    : ", d.true_powers, " kW")
    t0 = time()
    n_obs_total = 0; n_infeasible = 0; n_softened = 0; gstep = 0
    missed_kept = 0.0

    for day in 1:D_total
        rem_dig  .+= quota_dig(day)
        rem_load .+= quota_load(day)
        # SHARED applied-activity history for the CURRENT day (reset every morning;
        # a night is a long break, so precedence/pacing counters and the rest-rule
        # seam all restart). Each entry is (act, hrs): act = applied activity index;
        # hrs = realized [dig,load,trv,idle] h. Feeds cum_*/work_hist inside the model.
        hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
        peak_nc = 0.0; peak_op = 0.0
        mcs_node = [first(d.N_g) for _ in d.M]
        mcs_transit = Any[nothing for _ in d.M]

        plan_grid_kW = fill(NaN, nKd, nKd)
        plan_mcs_soe = fill(NaN, nKd, nKd)
        plan_cev_soe = [fill(NaN, nKd, nKd) for _ in d.E]
        plan_cev_act = [fill("", nKd, nKd)  for _ in d.E]
        plan_mcs_act = fill("", nKd, nKd)

        day_off = (day - 1) * nKd
        kept    = day <= n_days_keep

        for k0 in 1:nKd
            gstep += 1
            g0    = day_off + k0                          # GLOBAL interval index
            gk    = kept ? ((day - 1) * nKd + k0) : 0      # kept concatenated index (0 if buffer)
            clk   = clock_day_label(d.t_start, d.delta_T, day, k0)

            # capture start-of-interval SOE for the kept horizon
            if kept
                for m in d.M; real_SOE_MCS[m, gk] = soe_mcs[m]; end
                for e in d.E; real_SOE_CEV[e, gk] = soe_cev[e]; end
            end

            view_end_day = min(D_total, day + lookahead_days)
            Kend  = view_end_day * nKd
            K_win = g0:Kend

            # (1) OPTIMISE the cross-day window.
            model = build_window_model(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                       rem_dig, rem_load, hist,
                                       peak_nc, peak_op, est.mu;
                                       dig_by_day = d.dig_by_day, load_by_day = d.load_by_day,
                                       require_site_visit = require_site_visit,
                                       single_visit_per_site = single_visit_per_site,
                                       time_limit_sec = time_limit_sec,
                                       soft_prec = soft_prec, soft_pace = soft_pace,
                                       soft_term = soft_term, term_tol = term_tol,
                                       enforce_cev_terminal = true)
            stat = string(termination_status(model))
            cur_node = mcs_node[1]

            # GRACEFUL FALLBACK: a hard-infeasible tail means the daily terminal + keep-up
            # reserve are not jointly reachable from the drifted state. Do NOT freeze the
            # plant (holding charges NOTHING, the worst move for a low battery). Re-solve
            # once with the terminal made SOFT and the hard reserve dropped, so the MCS
            # still charges as much as possible toward target; any residual shortfall then
            # surfaces as a penalty rather than a dead interval.
            if !has_values(model)
                n_softened += 1
                @warn "Hard window infeasible at day=$day k=$k0; re-solving with soft terminal + relaxed reserve." status=stat
                model = build_window_model(d, K_win, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                           rem_dig, rem_load, hist,
                                           peak_nc, peak_op, est.mu;
                                           dig_by_day = d.dig_by_day, load_by_day = d.load_by_day,
                                           require_site_visit = require_site_visit,
                                           single_visit_per_site = single_visit_per_site,
                                           time_limit_sec = time_limit_sec,
                                           soft_prec = soft_prec, soft_pace = soft_pace,
                                           soft_term = true, soft_reserve = true, term_tol = term_tol,
                                           enforce_cev_terminal = true)
                stat = string(termination_status(model))
            end

            if !has_values(model)
                n_infeasible += 1
                @warn "No feasible solution even under SOFT constraints at day=$day k=$k0; holding state." status=stat
                kept && push!(solve_log, (day, k0, clk, stat, NaN, NaN, try solve_time(model) catch; NaN end))
                push!(log, (day, gstep, k0, clk, d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                            0.0, 0.0, 0.0, soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), cur_node,
                            est.mu[1], est.mu[2], est.mu[3], est.mu[4],
                            est.sd[1], est.sd[2], est.sd[3], est.sd[4], n_obs_total))
                push!(fe_time, clk)
                for e in d.E
                    push!(fe_act[e], "Idle"); push!(fe_chg[e], "No")
                    # A held (infeasible) interval is a BREAK for every CEV -> record it in
                    # history so the rest rule counts it correctly on the next window.
                    push!(hist[e], (length(d.B), [0.0, 0.0, 0.0, d.delta_T]))
                end
                push!(fe_mcs, "No")
                if kept; for m in d.M; real_loc[m, gk] = cur_node; end; end
                continue
            end

            # (2) APPLY interval g0's decisions.
            grid_kW = sum(value(model[:P_ch_tot][m, g0]) for m in d.M)
            dch_kW  = sum(value(model[:P_dch_tot][m, g0]) for m in d.M)
            cur_node = let nh = findfirst(i -> value(model[:z][1, i, g0]) > 0.5, d.N)
                nh === nothing ? 0 : nh
            end
            kept && push!(solve_log, (day, k0, clk, stat, objective_value(model),
                                      100 * (try relative_gap(model) catch; NaN end),
                                      try solve_time(model) catch; NaN end))

            # realized per-MCS charging / discharging / travel + location (kept days)
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

            # Save this window's current-day slice into the replanning grids.
            for k in K_win
                div(k - 1, nKd) + 1 == day || continue
                kl = k - day_off
                plan_grid_kW[k0, kl] = sum(value(model[:P_ch_tot][m, k]) for m in d.M)
                plan_mcs_soe[k0, kl] = value(model[:SOE_MCS][1, k + 1])
                for e in d.E
                    plan_cev_soe[e][k0, kl] = value(model[:SOE_CEV][e, k + 1])
                    site = findfirst(i -> d.A[i, e] == 1, d.N)
                    if site !== nothing
                        # Combined activity label; "Charging" only when REAL power is
                        # delivered into this CEV (sum_m P_MCS_CEV > 0).
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

            # worker-facing schedule row
            push!(fe_time, clk)
            for e in d.E
                push!(fe_act[e], planned_activity(model, d, e, g0))
                push!(fe_chg[e], cev_should_charge(model, d, e, g0))
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                (kept && site !== nothing) && (real_mu[site, e, gk] = value(model[:mu][site, e, g0]))
            end
            push!(fe_mcs, mcs_should_charge(model, d, g0))

            # (3) SIMULATE realized activity split.
            a_real = Dict(e => realized_activity_durations(rng, model, e, g0, d;
                                                           multi = multi_activity) for e in d.E)

            # (3.5) STOCHASTIC PLANT (FORK B): the posterior (est.mu, est.sd) is held
            # FIXED. Each interval the real machine's per-activity power is a fresh
            # per-excavator draw from that fixed curve, Normal(est.mu, est.sd) truncated
            # at 0. Idle has est.sd = 0, so its draw collapses to 0 (no power lost while
            # idle). The SAME p_true[e] drives the battery drain below.
            p_true = Dict(e => [max(est.mu[j] + est.sd[j] * randn(rng), 0.0)
                                for j in eachindex(est.mu)] for e in d.E)

            for e in d.E
                sum(a_real[e]) > 1e-9 && (n_obs_total += 1)
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                if kept && site !== nothing
                    real_P_work[site, e, gk] =
                        (a_real[e][1] * p_true[e][1] + a_real[e][2] * p_true[e][2] +
                         a_real[e][3] * p_true[e][3]) / d.delta_T
                end
            end

            # (4) ADVANCE the real MCS energy (by applied flows) + position.
            for m in d.M
                ch  = value(model[:P_ch_tot][m, g0])
                dch = value(model[:P_dch_tot][m, g0])
                ltr = value(model[:L_trv_tot][m, g0])
                soe_mcs[m] = soe_mcs[m] + d.eta_ch_dch[m] * ch * d.delta_T -
                             (dch * d.delta_T) / d.eta_ch_dch[m] - ltr
                if k0 == nKd
                    mcs_node[m] = first(d.N_g); mcs_transit[m] = nothing
                else
                    mcs_node[m], mcs_transit[m] = advance_mcs_state(model, m, g0, Kend, d)
                end
            end
            for e in d.E
                charged   = sum(value(model[:P_MCS_CEV][m, i, e, g0]) for m in d.M, i in d.N_c) * d.delta_T
                work_true = dot(a_real[e], p_true[e])
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

        # snapshot unfinished work + final kept-horizon SOE at end of last kept day
        if day == n_days_keep
            missed_kept = sum(rem_dig) + sum(rem_load)
            for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end
        end

        # end-of-day overnight smart-charge, then MCS starts next day recharged to full
        ov_df, _, _ = phase2_overnight_charge(d, soe_mcs)
        overnight_by_day[day] = ov_df
        soe_mcs = copy(float.(d.SOE_MCS_ini))
        replan_by_day[day] = (; plan_grid_kW, plan_mcs_soe, plan_cev_soe, plan_cev_act, plan_mcs_act)
    end

    elapsed = time() - t0
    @printf("MPC loop done in %.1f s (%d stochastic-plant realizations, %d simulated days)\n",
            elapsed, n_obs_total, D_total)
    n_softened > 0 && @printf("  NOTE: %d windows were hard-infeasible and recovered via the SOFT-terminal fallback\n        (charged toward target; any residual shortfall shows as a penalty).\n", n_softened)
    n_infeasible > 0 && @printf("  NOTE: %d windows were infeasible even under SOFT constraints; the plant HELD state.\n", n_infeasible)
    println("  fixed power model (mu) : ", round.(est.mu, digits = 2), " kW")
    println("  plant sampling sd      : ", round.(est.sd, digits = 2), " kW")

    # ---- DROP THE BUFFER DAY: report only days 1..n_days_keep ----
    keep_row = log.day .<= n_days_keep
    klog = log[keep_row, :]
    fe_time_k = fe_time[1:n_kept]
    fe_act_k  = [fe_act[e][1:n_kept] for e in d.E]
    fe_chg_k  = [fe_chg[e][1:n_kept] for e in d.E]
    fe_mcs_k  = fe_mcs[1:n_kept]

    # ---- KPIs from the KEPT-day trajectory ----
    total_energy = sum(klog.grid_kW) * d.delta_T
    total_cost   = sum(klog.grid_kW .* klog.price) * d.delta_T
    total_co2    = sum(klog.grid_kW .* klog.co2)  * d.delta_T
    nc_peak      = isempty(klog.grid_kW) ? 0.0 : maximum(klog.grid_kW)
    op_mask      = [in_peak(k, d.delta_T, d.t_start) for k in klog.k]
    op_peak      = any(op_mask) ? maximum(klog.grid_kW[op_mask]) : 0.0
    missed       = missed_kept
    transit_intervals = count(==(0), klog.mcs_node)
    labour_cost  = d.rho_labor * d.delta_T * transit_intervals

    overnight_energy = 0.0; overnight_cost = 0.0
    for day in 1:n_days_keep
        ov = overnight_by_day[day]
        for m in d.M
            col = ov[!, Symbol("MCS$(m)_charge_kW")]
            overnight_energy += sum(col) * d.delta_T
            overnight_cost   += sum(col .* ov.price) * d.delta_T
        end
    end

    # concatenated kept-horizon labels + multi-day x-ticks (for plotting)
    time_labels = build_time_labels_days(d.t_start, d.delta_T, n_days_keep, nKd)
    xticks = multiday_xticks(n_days_keep, nKd, d.t_start, d.delta_T)

    return (; d, time_labels, xticks, log = klog, solve_log,
              overnight_by_day, replan_by_day,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_mu, real_loc,
              fe_time = fe_time_k, fe_act = fe_act_k, fe_chg = fe_chg_k, fe_mcs = fe_mcs_k,
              est, nK = n_kept, nK_day = nKd, n_days_keep, D_total, ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals, overnight_energy, overnight_cost,
              soe_cev_end = copy(soe_cev), soe_mcs_end = real_SOE_MCS[:, n_kept + 1],
              n_obs_total, n_infeasible, n_softened, elapsed)
end

end # module MPCLoop
