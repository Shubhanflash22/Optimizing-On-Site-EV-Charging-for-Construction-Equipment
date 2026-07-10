# #############################################################################
# MCSModel.jl  —  module MCSModel  (RECEDING, multi-day / cross-day)
# -----------------------------------------------------------------------------
# The "optimise" half of the controller for the MULTI-DAY receding horizon.
# `build_window_model` builds and solves a MILP over a CROSS-DAY window: the rest
# of today plus `lookahead_days` future daytime blocks laid end to end. The MPC
# loop calls it once every 15 minutes; the window slides forward and always
# "sees" tomorrow, so the CEV energy-neutral wrap-up lands on the (dropped)
# buffer day rather than on a reported day.
#
# NOMENCLATURE is identical to MCS_OPTIMAL_v4_real.jl:
#   P_ch_MCS, P_dch_MCS, P_MCS_CEV, P_work, P_ch_tot, P_dch_tot   (power flows)
#   L_trv, L_trv_tot                                              (travel energy)
#   SOE_MCS, SOE_CEV                                              (state of energy)
#   u, mu, rho, z, g_ch, x, y_trv, beta_arr, beta_dep            (binaries)
#   P_peak_NC, P_peak_OP                                          (demand peaks)
#
# NIGHTS between day-blocks are NOT intervals; they are handled by two link
# rules: the MCS battery is reset to full + parked at the grid overnight, while
# the CEV battery carries over unchanged. Missed-work slack is per (site, day)
# and CUMULATIVE, so leftover work rolls over into the next day.
# #############################################################################
module MCSModel

using JuMP
using HiGHS
using DataFrames

using ..Common: normalize_travel_steps, in_peak, clock_label

export build_window_model, phase2_overnight_charge

# =============================================================================
# WINDOW MILP  (CROSS-DAY)
# =============================================================================
function build_window_model(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, cum_dig_e, cum_load_e, cum_trv_e,
                            peak_nc0, peak_op0, pvec;
                            # PER-DAY work schedule (absolute-day indexed vectors of
                            # node-length quotas). `nothing` => repeat d.hours_* each day.
                            dig_by_day = nothing,
                            load_by_day = nothing,
                            # SHARED applied Work(1)/Break(0) flags for the CURRENT day,
                            # one growing list per CEV (seeds the rest-rule seam). `nothing`
                            # => no seam (legacy within-window-only behaviour).
                            work_hist = nothing,
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true,
                            soft_prec::Bool = false,
                            soft_pace::Bool = false,
                            soft_term::Bool = false,
                            enforce_cev_terminal::Bool = true,
                            # does this window reach the TRUE end of the whole horizon
                            # (the buffer day's final interval)? Only then do we force the
                            # CEVs back to their start level.
                            is_global_terminal::Bool = (last(collect(K_win)) == d.n_day),
                            term_tol::Float64 = 0.0)
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

    # ---- multi-day window geometry ----
    # K_win holds GLOBAL interval indices spanning several days' daytime blocks laid
    # end to end. For a global index k: wd(k) = its position WITHIN its day (1..n_day);
    # dayof(k) = which day it belongs to. Daily profiles (price / carbon / work-avail)
    # have the same shape each day, so we index them by wd(k).
    n_day = d.n_day
    wd(k)    = mod(k - 1, n_day) + 1
    dayof(k) = div(k - 1, n_day) + 1

    K = collect(K_win)
    Tb = vcat(K, last(K) + 1)
    K_peak = [k for k in K if in_peak(wd(k), delta_T, d.t_start)]
    blockdays  = sort(unique(dayof.(K)))
    firstday   = dayof(first(K))
    block_ks(dy) = [k for k in K if dayof(k) == dy]
    # "Evening" intervals = the last daytime interval of each day in the window. At every
    # evening the MCS must be parked at a grid node; after every evening EXCEPT the final
    # one the MCS battery is reset to full for the next morning.
    eve_k      = [k for k in K if wd(k) == n_day]
    night_eve  = [k for k in eve_k if k != last(K)]
    price_k(k) = d.lambda_whl_elec[wd(k)]
    co2_k(k)   = d.lambda_CO2[wd(k)]
    Rwork(i, e, k) = d.R_work[i, e, wd(k)]

    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))
    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    # ---- model + solver configuration ----
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    set_time_limit_sec(model, time_limit_sec)
    set_attribute(model, "threads", 1)
    set_attribute(model, "parallel", "off")
    # Disable HiGHS's sub-MIP primal heuristics (RENS/RINS) and root-node symmetry
    # detection: both launch internal sub-solvers that spin up HiGHS's parallel task
    # deque even when the OUTER model is serial, which intermittently segfaults on
    # Windows (EXCEPTION_ACCESS_VIOLATION in HighsSplitDeque).
    set_attribute(model, "mip_heuristic_effort", 0.0)
    set_attribute(model, "mip_detect_symmetry", false)
    set_attribute(model, "mip_rel_gap", 1.0e-2)

    # ---- CONTINUOUS decision variables: power flows (kW) ----
    @variable(model, P_ch_MCS[M, N, K] >= 0)
    @variable(model, P_dch_MCS[M, N, K] >= 0)
    @variable(model, P_MCS_CEV[M, N_c, E, K] >= 0)
    @variable(model, P_work[N_c, E, K] >= 0)
    @variable(model, P_ch_tot[M, K] >= 0)
    @variable(model, P_dch_tot[M, K] >= 0)
    # Missed-work slack, ONE per (site, day-block). Cumulative target -> a shortfall
    # automatically ROLLS OVER into the next day (and is penalised again).
    @variable(model, s_miss_dig[N_c, blockdays] >= 0)
    @variable(model, s_miss_load[N_c, blockdays] >= 0)
    @variable(model, s_prec[N_c, K] >= 0)
    @variable(model, s_pace_hi[E, K] >= 0)
    @variable(model, s_pace_lo[E, K] >= 0)

    # ---- travel energy (kWh) ----
    @variable(model, L_trv[M, N, N, K] >= 0)
    @variable(model, L_trv_tot[M, K] >= 0)

    # ---- state of energy, indexed at interval BOUNDARIES ----
    @variable(model, SOE_MCS[M, Tb] >= 0)
    @variable(model, SOE_CEV[E, Tb] >= 0)

    # ---- BINARY decision variables ----
    @variable(model, u[E, N, B, K], Bin)
    @variable(model, mu[N, E, K], Bin)
    @variable(model, rho[M, N, E, K], Bin)
    @variable(model, z[M, N, K], Bin)
    @variable(model, g_ch[M, N_g, K], Bin)
    @variable(model, x[M, N, N, K], Bin)
    @variable(model, y_trv[M, N, N, K], Bin)
    @variable(model, beta_arr[M, N, K], Bin)
    @variable(model, beta_dep[M, N, K], Bin)
    @variable(model, P_peak_NC >= 0)
    @variable(model, P_peak_OP >= 0)
    @variable(model, s_term_cev[E, blockdays] >= 0)   # per (CEV, day): daily neutrality slack

    # ---- OBJECTIVE: total operating cost ----
    obj = @expression(model,
        sum(price_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        sum((d.carbon_price_per_ton / 1000.0) * co2_k(k) * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        d.rho_miss * (sum(s_miss_dig[i, dy] for i in N_c, dy in blockdays) +
                      sum(s_miss_load[i, dy] for i in N_c, dy in blockdays)) +
        d.lambda_demand_NC * P_peak_NC +
        d.lambda_demand_OP * P_peak_OP +
        d.rho_labor * delta_T * sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K))

    W_prec = 8.0e2; W_pace = 1.0e2; W_term = 1.5e2
    soft_prec || @constraint(model, [i in N_c, k in K], s_prec[i, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_hi[e, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_lo[e, k] == 0)
    soft_term || @constraint(model, [e in E, dy in blockdays], s_term_cev[e, dy] == 0)
    @objective(model, Min, obj +
        (soft_prec ? W_prec * sum(s_prec[i, k] for i in N_c, k in K) : AffExpr(0.0)) +
        (soft_pace ? W_pace * sum(s_pace_hi[e, k] + s_pace_lo[e, k] for e in E, k in K) : AffExpr(0.0)) +
        (soft_term ? W_term * sum(s_term_cev[e, dy] for e in E, dy in blockdays) : AffExpr(0.0)))

    # ---- power aggregation & where power may flow ----
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

    # plug-level and excavator-acceptance limits
    @constraint(model, [m in M, i in N_c, e in E, k in K],
        P_MCS_CEV[m, i, e, k] <= d.DCH_MCS_plug[m] * rho[m, i, e, k])
    @constraint(model, [i in N_c, e in E, k in K],
        sum(P_MCS_CEV[m, i, e, k] for m in M) <= d.CH_CEV[e] * mu[i, e, k])

    # peak-demand trackers
    @constraint(model, P_peak_NC >= peak_nc0)
    @constraint(model, P_peak_OP >= peak_op0)
    @constraint(model, [k in K], P_peak_NC >= sum(P_ch_tot[m, k] for m in M))
    @constraint(model, [k in K_peak], P_peak_OP >= sum(P_ch_tot[m, k] for m in M))
    if peak_demand_limit !== nothing
        @constraint(model, [k in K], sum(P_ch_tot[m, k] for m in M) <= peak_demand_limit)
    end

    # ---- travel energy bookkeeping ----
    for m in M, i in N, j in N, k in K
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k] == 1)
        else
            @constraint(model, y_trv[m, i, j, k] == sum(x[m, i, j, tau]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end
    @constraint(model, [m in M, i in N, j in N, k in K],
        L_trv[m, i, j, k] == d.k_trv * delta_T * y_trv[m, i, j, k])
    @constraint(model, [m in M, k in K],
        L_trv_tot[m, k] == sum(L_trv[m, i, j, k] for i in N, j in N))

    # ---- battery dynamics ----
    @constraint(model, [m in M], SOE_MCS[m, first(Tb)] == soe_mcs0[m])
    @constraint(model, [e in E], SOE_CEV[e, first(Tb)] == soe_cev0[e])
    # MCS flow WITHIN each day (skipped across a night boundary: overnight recharge).
    @constraint(model, [m in M, k in K; !(k in night_eve)],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
            d.eta_ch_dch[m] * P_ch_tot[m, k] * delta_T -
            (P_dch_tot[m, k] * delta_T) / d.eta_ch_dch[m] -
            L_trv_tot[m, k])
    # Overnight bridge: each MCS starts the next day recharged to its start-of-day level.
    @constraint(model, [m in M, k in night_eve], SOE_MCS[m, k + 1] == d.SOE_MCS_ini[m])
    # CEV battery carries over continuously across nights (single link, all intervals).
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
            sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -
            sum(P_work[i, e, k] for i in N_c) * delta_T)

    @constraint(model, [m in M, t in Tb], d.SOE_MCS_min[m] <= SOE_MCS[m, t] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb], d.SOE_CEV_min[e] <= SOE_CEV[e, t] <= d.SOE_CEV_max[e])

    # ---- CEV energy neutrality at the END OF EVERY DAY (daily realignment) ----
    # Each excavator must return to its START-OF-DAY SOE by every 18:00 present in the
    # window (the boundary just after each evening interval). This makes each reported day
    # ENERGY-NEUTRAL (start SOE == end SOE) instead of letting the battery drift across
    # days. (`is_global_terminal` is no longer used to gate this; it is kept only as an
    # accepted kwarg for API compatibility.)
    if enforce_cev_terminal
        for ke in eve_k
            dy = dayof(ke)
            if soft_term
                @constraint(model, [e in E],  SOE_CEV[e, ke + 1] - d.SOE_CEV_ini[e] <= s_term_cev[e, dy])
                @constraint(model, [e in E], -(SOE_CEV[e, ke + 1] - d.SOE_CEV_ini[e]) <= s_term_cev[e, dy])
            else
                @constraint(model, [e in E], SOE_CEV[e, ke + 1] >= d.SOE_CEV_ini[e] - term_tol)
            end
        end

        # ---- PROACTIVE KEEP-UP RESERVE, PER DAY (keeps each daily terminal recursively
        # feasible) ----------------------------------------------------------------------
        # Built backward from EACH day's evening deadline Gd. A CEV is only charged while
        # the MCS is parked at its site; to honour the evening end-at-grid rule the MCS
        # must depart `tgrid` intervals before Gd, so the LAST interval it can charge is
        # `Lc`. After that the CEV idles and, because idling itself draws power, its SOE
        # strictly DRAINS. We lower-bound the CEV SOE at every boundary of that day by the
        # least level from which the day's terminal is still reachable. The bound only
        # binds in each day's late tail, so it never distorts productive hours; applied
        # every step it makes the daily hard terminal recursively feasible.
        plug_cap = maximum(d.DCH_MCS_plug)
        idle_a   = B[4]
        for e in E
            site_e = findfirst(i -> d.A[i, e] == 1, N)
            site_e === nothing && continue
            tgrid = minimum(travel_steps[site_e, g] for g in N_g) + 1   # +1: departure interval is transit
            idle_drain = p_activity[idle_a] * delta_T
            chg_net    = max(min(d.CH_CEV[e], plug_cap) * delta_T - idle_drain, 1.0e-6)
            target_e   = d.SOE_CEV_ini[e] - term_tol
            for ke in eve_k
                Gd = ke                                          # this day's evening (deadline) interval
                Lc = Gd - tgrid                                  # last interval the MCS can charge before leaving
                Lc < first(K) && continue                        # can't charge in-window for this deadline; skip
                n_tail = Gd - Lc
                S_star = target_e + idle_drain * n_tail
                day_lo = max(first(K), Gd - n_day + 1)           # first in-window boundary of this day-block
                for t in day_lo:(Gd + 1)
                    lb = t <= Lc + 1 ? S_star - chg_net * ((Lc + 1) - t) :
                                       target_e + idle_drain * (Gd + 1 - t)
                    lb = min(lb, d.SOE_CEV_max[e])
                    lb > d.SOE_CEV_min[e] && @constraint(model, SOE_CEV[e, t] >= lb)
                end
            end
        end
    end

    # ---- plugging / presence logic ----
    @constraint(model, [m in M, i in N_c, k in K], sum(rho[m, i, e, k] for e in E) <= d.C_MCS_plug[m])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= d.A[i, e])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])
    @constraint(model, [m in M, i in N, k in K], x[m, i, i, k] == 0)

    @constraint(model, [m in M, k in K],
        sum(z[m, i, k] for i in N) + sum(y_trv[m, i, j, k] for i in N, j in N if i != j) == 1)

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
            @constraint(model, beta_arr[m, j, k] == 1)
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

    for m in M, i in N
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k] for k in K) - sum(beta_dep[m, i, k] for k in K) ==
            z[m, i, last(K)] - start_here)
    end

    # terminal position: parked at a grid node at EVERY evening (each day's 18:00).
    @constraint(model, [m in M, k in eve_k], sum(z[m, i, k] for i in N_g) == 1)

    if require_site_visit
        @constraint(model, [m in M], sum(beta_arr[m, i, k] for i in N_c, k in K) >= 1)
    end
    if single_visit_per_site
        @constraint(model, [m in M, i in N_c], sum(beta_arr[m, i, k] for k in K) <= 1)
        @constraint(model, [m in M, i in N_c], sum(beta_dep[m, i, k] for k in K) <= 1)
    end

    # ---- activity scheduling ----
    @constraint(model, [i in N_c, e in E, k in K],
        sum(u[e, i, a, k] for a in B) == d.A[i, e])
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= d.A[i, e])
    @constraint(model, [i in N_c, e in E, k in K],
        sum(p_activity[a] * u[e, i, a, k] for a in (B[1], B[2], B[3])) <=
        Rwork(i, e, k) * d.A[i, e] * (1 - mu[i, e, k]))
    @constraint(model, [i in N_c, e in E, k in K], mu[i, e, k] <= u[e, i, B[4], k])
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- daily work quota: CUMULATIVE target, pinned from BOTH sides ----
    # Work is a PER-DAY schedule: each morning after the window's first day adds that
    # day's own quota (dig_by_day[dd] / load_by_day[dd]). For every day-block dy in the
    # window we bound the CUMULATIVE work done through the END of dy against the cumulative
    # target `tgt_*` (window-start rem_*, which already holds the first day's fresh quota,
    # PLUS each subsequent morning's own quota):
    #   * LOWER (soft): shortfall s_miss_* >= 0 is penalised and ROLLS OVER to the next day;
    #   * UPPER (HARD): NO working ahead -- cumulative work through end of dy may not exceed
    #     the cumulative quota. Because the cap is per-day-cumulative, unfinished work from
    #     an earlier day can still be CAUGHT UP later (it is inside the same cumulative
    #     budget), but a day can never borrow work from a FUTURE day. Working-less is always
    #     feasible, so the hard cap can never make a window infeasible.
    qd(dd, i) = dig_by_day  === nothing ? d.hours_digging[i]          :
                (1 <= dd <= length(dig_by_day)  ? dig_by_day[dd][i]  : 0.0)
    ql(dd, i) = load_by_day === nothing ? d.hours_loading_swinging[i] :
                (1 <= dd <= length(load_by_day) ? load_by_day[dd][i] : 0.0)
    for dy in blockdays
        Kupto = [k for k in K if dayof(k) <= dy]
        extra_dig(i)  = sum((qd(dd, i) for dd in blockdays if firstday < dd <= dy); init = 0.0)
        extra_load(i) = sum((ql(dd, i) for dd in blockdays if firstday < dd <= dy); init = 0.0)
        tgt_dig(i)  = max(rem_dig[i], 0.0)  + extra_dig(i)
        tgt_load(i) = max(rem_load[i], 0.0) + extra_load(i)
        done_dig(i)  = delta_T * sum(u[e, i, B[1], k] for e in E, k in Kupto)
        done_load(i) = delta_T * sum(u[e, i, B[2], k] for e in E, k in Kupto)
        @constraint(model, [i in N_c], s_miss_dig[i, dy]  >= tgt_dig(i)  - done_dig(i))
        @constraint(model, [i in N_c], s_miss_load[i, dy] >= tgt_load(i) - done_load(i))
        @constraint(model, [i in N_c], done_dig(i)  <= tgt_dig(i))     # hard: no working ahead
        @constraint(model, [i in N_c], done_load(i) <= tgt_load(i))
    end

    # precedence: cumulative loading <= scale * cumulative digging, WITHIN each day-block
    # (counters restart each morning; carried realized work seeds only the current day).
    bstart(k) = first(block_ks(dayof(k)))
    @constraint(model, [i in N_c, k in K],
        (((dayof(k) == firstday) ? cum_load_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[2], tau] for tau in bstart(k):k, e in E)) <=
        d.scale * (((dayof(k) == firstday) ? cum_dig_site(i) : 0.0) +
            delta_T * sum(u[e, i, B[1], tau] for tau in bstart(k):k, e in E)) +
        s_prec[i, k])

    # rest rule: <= rest_cap work intervals in any (rest_cap+1) window, WITHIN a day-block
    # (a night is a long break, so the count restarts each morning). Two parts:
    #   (a) within-window: every (rest_cap+1)-window lying fully inside one day-block of K;
    #   (b) SEAM: windows straddling the window start, seeded with the applied Work/Break
    #       flags of the CURRENT day (work_hist). Without (b) a work-run could leak across
    #       the every-15-min re-solves (rest_cap at the tail of one window + rest_cap at the
    #       head of the next). The o=rest_cap seam is the binding one.
    rest_cap = Int(round(d.t_limit_rest / delta_T))
    rest_win = rest_cap + 1
    Wc(e, i, k) = sum(u[e, i, a, k] for a in (B[1], B[2], B[3]))
    if length(K) >= rest_win
        rest_starts = [k0 for k0 in first(K):(last(K) - rest_win + 1)
                       if dayof(k0) == dayof(k0 + rest_win - 1)]
        @constraint(model, [i in N_c, e in E, k0 in rest_starts],
            sum(Wc(e, i, k) for k in k0:(k0 + rest_win - 1)) <= rest_cap)
    end
    if work_hist !== nothing
        for e in E
            h = work_hist[e]; Lh = length(h)
            for o in 1:min(rest_cap, Lh)
                nfut = rest_win - o
                ks = [first(K) + t for t in 0:(nfut - 1)]
                all(k -> k in K && dayof(k) == firstday, ks) || continue
                hsum = sum(h[(Lh - o + 1):Lh])
                @constraint(model, [i in N_c], hsum + sum(Wc(e, i, k) for k in ks) <= rest_cap)
            end
        end
    end

    # travel pacing: keep cumulative travel ~ proportional to cumulative work (per day).
    kappa = d.kappa_wt
    for e in E, kk in K
        carry = (dayof(kk) == firstday)
        bs = bstart(kk)
        trv_cum  = (carry ? cum_trv_e[e] / delta_T : 0.0) +
                   sum(u[e, i, B[3], tau] for i in N_c, tau in bs:kk)
        work_cum = (carry ? (cum_dig_e[e] + cum_load_e[e]) / delta_T : 0.0) +
                   sum(u[e, i, a, tau] for i in N_c, a in (B[1], B[2]), tau in bs:kk)
        @constraint(model, kappa * trv_cum <= work_cum + s_pace_hi[e, kk])
        @constraint(model, kappa * trv_cum >= work_cum - kappa - s_pace_lo[e, kk])
    end

    # Solve (crash-tolerant: a rare native HiGHS memory fault degrades to no-solution).
    try
        optimize!(model)
    catch err
        @warn "MCSModel: solver threw during optimize!; treating window as no-solution (hold state)." exception = err
    end
    return model
end

# =============================================================================
# PHASE 2 — OVERNIGHT SMART-CHARGE  (deterministic; per night)
# =============================================================================
function phase2_overnight_charge(d, soe_mcs_end)
    dt   = d.delta_T
    ov_k = (d.n_day + 1):d.n_int
    nov  = length(ov_k)
    P_ov = zeros(length(d.M), nov)
    soe_path = [fill(float(soe_mcs_end[m]), nov + 1) for m in d.M]

    for m in d.M
        eta  = d.eta_ch_dch[m]
        rate = d.CH_MCS[m]
        deficit = d.SOE_MCS_ini[m] - soe_mcs_end[m]
        if deficit > 1e-9
            order = sort(collect(1:nov); by = j -> d.lambda_whl_elec[ov_k[j]])
            remaining = deficit
            for j in order
                remaining <= 1e-9 && break
                gain = min(eta * rate * dt, remaining)
                P_ov[m, j] = gain / (eta * dt)
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
                   clock = [clock_label(d.t_start, d.delta_T, k) for k in ov_k],
                   price = [d.lambda_whl_elec[k] for k in ov_k])
    for m in d.M
        df[!, Symbol("MCS$(m)_charge_kW")] = P_ov[m, :]
        df[!, Symbol("MCS$(m)_soe_kWh")]   = soe_path[m][2:end]
        df[!, Symbol("MCS$(m)_charging")]  = [P_ov[m, j] > 1e-6 ? "Yes" : "No" for j in 1:nov]
    end
    return df, P_ov, ov_k
end

end # module MCSModel
