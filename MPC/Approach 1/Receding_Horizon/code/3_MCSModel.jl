# #############################################################################
# MCSModel.jl  —  module MCSModel  (RECEDING, fixed-length 24h rolling window)
# -----------------------------------------------------------------------------
# The "optimise" half of the controller. `build_window_model` builds and solves a
# MILP over exactly the next 24h (n_day intervals) from NOW - no more, no less,
# whatever time of day "now" is. The MPC loop calls it once every 15 minutes; the
# window slides forward by one interval each time. The MCS/CEV energy-neutral
# recovery target (Eq. 10a/10b, per the arXiv paper's numbering) is pinned to
# the next fixed calendar day-boundary
# (t_start, e.g. 8am) rather than to the window's own endpoint, and is enforced on
# EVERY solve - guaranteeing recursive feasibility on a rolling basis. Any hours the
# window can see PAST that recovery point (only when "now" isn't itself the
# boundary) are free/unconstrained context for planning, not a second deadline.
#
# NOMENCLATURE is identical to MCS_OPTIMAL_v4_real.jl so a reviewer sees the same
# decision variables:
#   P_ch_MCS, P_dch_MCS, P_MCS_CEV, P_work, P_ch_tot, P_dch_tot   (power flows)
#   SOE_MCS, SOE_CEV                                              (state of energy;
#       MCS uses eta_ch_dch_mcs, CEV charging uses eta_ch_dch_cev)
#   u, mu, rho, z, x, y_trv, beta_arr, beta_dep                   (binaries)
#   P_peak_NC, P_peak_OP, s_miss_work                             (peaks / slack)
#
# The MCS/CEV terminal energy-neutral rule (Eq. 10a/10b) is enforced inside this
# single MILP at the fixed next-8am point, so the overnight MCS recharge is
# scheduled by the optimiser itself (no separate deterministic phase).
# #############################################################################
module MCSModel

using JuMP
using HiGHS
using DataFrames

using ..Common: normalize_travel_steps, in_peak, clock_label

export build_window_model

# =============================================================================
# WINDOW MILP
# =============================================================================
# The arguments after `d` are the CLOSED-LOOP CARRY-IN — the real state handed
# from the previous step:
#   soe_mcs0/soe_cev0 : measured battery levels now (kWh)
#   mcs_node0         : node the MCS is parked at (0 if mid-drive)
#   mcs_transit0      : nothing, or (i,j,r) = mid-drive on arc i->j, r steps left
#   rem_dig/rem_load  : work hours still remaining at each site
#   hist              : the SHARED per-CEV applied-activity history. `hist[e]` is the
#                       chronological list of COMPLETED intervals for excavator e, each
#                       a tuple (act, hrs): act = applied activity index (1=dig, 2=load,
#                       3=travel, 4=idle); hrs = realized [dig,load,travel,idle] hours.
#                       Every history-dependent rule reads from this one object:
#                       precedence & pacing use the summed hours; the rest rule uses the
#                       recent Work/Break pattern. (Whether a rule uses it is decided here.)
#   peak_nc0/peak_op0 : biggest grid draw seen so far today (demand charges)
#   pvec              : the current per-activity power estimate (est.mu)
function build_window_model(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, hist,
                            peak_nc0, peak_op0, pvec;
                            # PER-DAY work schedule (absolute-day indexed vectors of
                            # node-length quotas). `nothing` => repeat d.hours_* each day.
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true)
    # Frequently-used sets/scalars.
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
    Tb = vcat(K, last(K) + 1)                                   # boundary indices
    K_peak = [k for k in K if in_peak(k, delta_T, d.t_start)]   # on-peak subset
    firstday   = dayof(first(K))
    # ---- FIXED recovery deadline: the next calendar day-boundary (t_start, e.g. 8am) ----
    # The window is always exactly one day-block (n_day intervals) long, so it always
    # contains the next occurrence of the day-boundary somewhere inside it - at the
    # window's own last interval only when "now" (first(K)) itself IS the boundary;
    # otherwise a bit before the end, with a few extra hours of the day-after still
    # visible for planning context but with NO recovery requirement attached to them
    # (those hours are "free" until the FOLLOWING day-boundary, one window later).
    k_term = firstday * n_day        # last interval before the next day-boundary
    b_term = k_term + 1               # boundary state AT the next day-boundary
    # Activity index -> its (estimated) power draw.
    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))

    # ---- read the SHARED applied-activity history (Option-2 unification) ----
    # All three history-dependent rules derive what they need from `hist`:
    #   precedence -> summed realized HOURS per activity, over the WHOLE run
    #                 (unchanged -- loading-vs-digging still needs true energy/time
    #                 accounting and is not a calendar-day-scoped rule);
    #   pacing     -> summed applied INTERVAL COUNTS off the u indicator, scoped to
    #                 the CURRENT CALENDAR DAY only (resets at each day boundary,
    #                 since `hist` itself never resets across the multi-day run --
    #                 see `today_start` below);
    #   rest rule  -> the recent Work(1)/Break(0) pattern (travel = work), unchanged.
    cum_dig_e  = [sum((r[2][1] for r in hist[e]); init = 0.0) for e in E]
    cum_load_e = [sum((r[2][2] for r in hist[e]); init = 0.0) for e in E]
    # `hist[e]` has one entry per applied GLOBAL interval, in order, starting at k=1,
    # so entry index == global k. today_start is the global k of this calendar day's
    # first interval; only entries from there to "now" feed the pacing seed.
    today_start = (firstday - 1) * n_day + 1
    cum_trv_cnt_e  = [count(r -> r[1] == 3, @view hist[e][min(today_start, length(hist[e]) + 1):end]) for e in E]
    cum_work_cnt_e = [count(r -> r[1] in (1, 2), @view hist[e][min(today_start, length(hist[e]) + 1):end]) for e in E]
    work_hist  = [Int[(r[1] in (1, 2, 3)) ? 1 : 0 for r in hist[e]] for e in E]

    # Cumulative site work already done (seeds precedence).
    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    # Helpers for a drive already underway when the window began.
    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    # ---- model + solver configuration ----
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    # SOLVER TIME LIMIT (control point #3, the innermost one). This caps how long
    # HiGHS spends on EACH window MILP. Pass time_limit_sec = Inf (from run_scenario_1
    # / run_mpc) to REMOVE the cap entirely and let HiGHS solve every window to the
    # mip_rel_gap tolerance below. HiGHS rejects a non-finite limit, so only set it
    # when the value is finite; Inf simply leaves HiGHS on its default (no limit).
    isfinite(time_limit_sec) && set_time_limit_sec(model, time_limit_sec)
    # Force single-threaded, deterministic solving.
    set_attribute(model, "threads", 1)
    set_attribute(model, "parallel", "off")
    # Disable HiGHS's sub-MIP primal heuristics (RENS/RINS) and root-node symmetry
    # detection: both launch internal sub-solvers that spin up HiGHS's parallel
    # task deque even when the OUTER model is serial, which intermittently
    # segfaults on Windows (EXCEPTION_ACCESS_VIOLATION in HighsSplitDeque). Turning
    # them off keeps HiGHS on the stable serial branch-and-cut path; the small
    # per-window MILPs solve fine without them.
    set_attribute(model, "mip_heuristic_effort", 0.0)
    set_attribute(model, "mip_detect_symmetry", false)
    # Accept a solution within 1% of optimal (MPC only applies the first interval).
    set_attribute(model, "mip_rel_gap", 1.0e-2)

    # ---- CONTINUOUS decision variables: power flows (kW) ----
    @variable(model, P_ch_MCS[M, N, K] >= 0)       # grid -> MCS charge power, per node
    @variable(model, P_dch_MCS[M, N, K] >= 0)      # MCS -> site discharge power, per node
    @variable(model, P_MCS_CEV[M, N_c, E, K] >= 0) # MCS -> specific excavator power
    @variable(model, P_work[N_c, E, K] >= 0)       # power an excavator spends working
    @variable(model, P_ch_tot[M, K] >= 0)          # total grid draw by the MCS
    @variable(model, P_dch_tot[M, K] >= 0)         # total discharge out of the MCS
    @variable(model, s_miss_work[N_c, B] >= 0)     # UNFINISHED work (hours) — penalised slack (Eq. 12c)  

    # ---- state of energy, indexed at interval BOUNDARIES ----
    @variable(model, SOE_MCS[M, Tb] >= 0)
    @variable(model, SOE_CEV[E, Tb] >= 0)

    # ---- BINARY decision variables ----
    @variable(model, u[E, N, B, K], Bin)           # which activity each excavator does
    @variable(model, mu[N, E, K], Bin)             # is the excavator charging?
    @variable(model, rho[M, N, E, K], Bin)         # is the excavator plugged into the MCS?
    @variable(model, z[M, N, K], Bin)              # is the MCS parked at this node?
    @variable(model, x[M, N, N, K], Bin)           # does the MCS depart i -> j this interval?
    @variable(model, y_trv[M, N, N, K], Bin)       # is the MCS in transit on arc i -> j?
    @variable(model, beta_arr[M, N, K], Bin)       # MCS arrival indicator at a node
    @variable(model, beta_dep[M, N, K], Bin)       # MCS departure indicator at a node
    @variable(model, P_peak_NC >= 0)               # tracked whole-day peak grid draw
    @variable(model, P_peak_OP >= 0)               # tracked on-peak peak grid draw

    # ---- small time-index tie-break penalty on CEV charging ----
    # When multiple schedules are cost-tied under the deterministic mean-case
    # forecast (e.g. "charge the CEV now" vs "wait, then charge" -- same total
    # energy, same price, so mathematically identical in the real-cost objective
    # below), the solver has no preference between them, even though waiting
    # quietly spends down the CEV's safety margin in the real uncertain world.
    # The penalty targets `mu[i,e,k]` (is CEV e accepting power at site i,
    # interval k -- see the P_MCS_CEV <= CH_CEV * mu constraint below), the
    # binary that governs whether the CEV is charging. idx is the window's own
    # LOCAL position (1..length(K)), not the global/day-relative wd(k) used for
    # price/CO2 above -- earlier WITHIN THIS WINDOW is preferred when tied.
    Kvec = collect(K)
    early_charge_term = sum(idx * mu[i, e, Kvec[idx]] for i in N_c, e in E, idx in eachindex(Kvec))

    # ---- OBJECTIVE (Eq. 1): total operating cost. All constraints are HARD;
    # the only slack is s_miss_work (Eq. 12c). ----
    @objective(model, Min,
        sum(d.lambda_whl_elec[wd(k)] * P_ch_tot[m, k] * delta_T for m in M, k in K) +                             # energy cost: price x grid kWh
        sum((d.carbon_price_per_ton / 1000.0) * d.lambda_CO2[wd(k)] * P_ch_tot[m, k] * delta_T for m in M, k in K) +  # carbon cost of that grid energy
        d.rho_miss * sum(s_miss_work[i, a] for i in N_c, a in B) +                                             # SOFT penalty for unfinished work
        d.lambda_demand_NC * P_peak_NC +                                                                      # non-coincident demand charge
        d.lambda_demand_OP * P_peak_OP +                                                                      # on-peak demand charge
        d.rho_labor * delta_T * sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K) +                    # towing labour: cost of time in transit
        1e-6 * early_charge_term)                                                                              # small earlier-charging tie-break

    # ---- power aggregation & where power may flow ----
    # Total grid draw of an MCS = sum of its per-grid-node charge power.
    @constraint(model, [m in M, k in K], P_ch_tot[m, k]  == sum(P_ch_MCS[m, i, k]  for i in N_g))
    # Total discharge of an MCS = sum of its per-site discharge power.
    @constraint(model, [m in M, k in K], P_dch_tot[m, k] == sum(P_dch_MCS[m, i, k] for i in N_c))
    # No discharging at a grid node (the MCS only CHARGES from the grid) ...
    @constraint(model, [m in M, i in N_g, k in K], P_dch_MCS[m, i, k] == 0)
    # ... and no charging at a site node (the MCS only DISCHARGES to CEVs on site).
    @constraint(model, [m in M, i in N_c, k in K], P_ch_MCS[m, i, k]  == 0)
    # Site discharge is fully accounted for by what is delivered to the CEVs there.
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] == sum(P_MCS_CEV[m, i, e, k] for e in E))
    # Discharge is only possible where the MCS is actually parked (z=1), capped by DCH_MCS.
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] <= d.DCH_MCS[m] * z[m, i, k])

    # Grid charging power is capped by CH_MCS and flows only when the MCS is
    # physically parked at that node (z=1).
    @constraint(model, [m in M, i in N_g, k in K], P_ch_MCS[m, i, k] <= d.CH_MCS[m] * z[m, i, k])

    # plug-level and excavator-acceptance limits
    # Power into one CEV via one plug is capped by the per-plug rate and needs rho=1 (plugged in).
    @constraint(model, [m in M, i in N_c, e in E, k in K],
        P_MCS_CEV[m, i, e, k] <= d.DCH_MCS_plug[m] * rho[m, i, e, k])
    # Total power a CEV accepts is capped by its own charge rate and needs mu=1 (charging).
    @constraint(model, [i in N_c, e in E, k in K],
        sum(P_MCS_CEV[m, i, e, k] for m in M) <= d.CH_CEV[e] * mu[i, e, k])
    # Charging-mode consistency: CEV e is in charging mode at site i (mu=1) exactly
    # when it is plugged into an MCS there (sum of rho over M).
    @constraint(model, [i in N_c, e in E, k in K],
        mu[i, e, k] == sum(rho[m, i, e, k] for m in M))

    # peak-demand trackers (carry the peak already seen earlier today)
    # Whole-day peak is at least the biggest grid draw already realised before this window.
    @constraint(model, P_peak_NC >= peak_nc0)
    # On-peak peak likewise carries in the biggest on-peak draw seen so far.
    @constraint(model, P_peak_OP >= peak_op0)
    # ... and is at least the total grid draw in EVERY interval of this window.
    @constraint(model, [k in K], P_peak_NC >= sum(P_ch_tot[m, k] for m in M))
    # On-peak tracker only bounds the on-peak intervals (K_peak).
    @constraint(model, [k in K_peak], P_peak_OP >= sum(P_ch_tot[m, k] for m in M))
    if peak_demand_limit !== nothing
        @constraint(model, [k in K], sum(P_ch_tot[m, k] for m in M) <= peak_demand_limit)
    end

    # ---- travel energy bookkeeping ----
    # y_trv[m,i,j,k] = 1 iff MCS m is in transit on arc i->j during interval k. A trip
    # launched at tau (x=1) occupies the next travel_steps[i,j] intervals, so y at k is
    # the OR of departures within that look-back window (or forced 1 for a carried-in drive).
    for m in M, i in N, j in N, k in K
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k] == 1)                     # drive already underway at window start
        else
            @constraint(model, y_trv[m, i, j, k] == sum(x[m, i, j, tau]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end
    # MCS transit cost is captured as time via the rho_labor term in the objective;
    # y_trv is used there and in the routing/presence bookkeeping below.

    # ---- battery dynamics ----
    # Pin the first boundary of each battery to the measured carried-in SOE (MPC initial condition).
    @constraint(model, [m in M], SOE_MCS[m, first(Tb)] == soe_mcs0[m])
    @constraint(model, [e in E], SOE_CEV[e, first(Tb)] == soe_cev0[e])
    # MCS SOE recursion: previous + charge*(eta_mcs) - discharge/(eta_mcs). Transit
    # does not draw from the battery (see travel-energy bookkeeping above).
    @constraint(model, [m in M, k in K],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
            d.eta_ch_dch_mcs[m] * P_ch_tot[m, k] * delta_T -
            (P_dch_tot[m, k] * delta_T) / d.eta_ch_dch_mcs[m])
    # CEV SOE recursion: previous + energy RECEIVED from the MCS (scaled by the CEV's
    # own charge-acceptance efficiency, eta_ch_dch_cev) - energy spent working this step.
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
            d.eta_ch_dch_cev[e] * sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -
            sum(P_work[i, e, k] for i in N_c) * delta_T)

    # SOE operating ranges (Eq. 8c, 8d).
    @constraint(model, [m in M, t in Tb], d.SOE_MCS_min[m] <= SOE_MCS[m, t] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb], d.SOE_CEV_min[e] <= SOE_CEV[e, t] <= d.SOE_CEV_max[e])

    # ---- Terminal energy targets (Eq. 10a, 10b), pinned to the next 8am, always ----
    # MCS: EXACT equality to its initial SOE (Eq. 10a) so it is fully ready by the next
    # calendar day-start; the objective (TOU price x kWh) then naturally schedules the
    # recharge in whichever hours before that deadline are cheapest (usually overnight)
    # - no separate deterministic "phase 2" is needed for that.
    # CEV: a lower bound at its initial SOE (Eq. 10b as a FLOOR, >=). OVERCHARGING IS
    # ALLOWED — the CEV may finish the day at or above its start level. This removes
    # the overcharge knife-edge: since a CEV cannot discharge, a hard equality would
    # be unrecoverable whenever the stochastic plant lets its SOE drift above the
    # target; the floor keeps the terminal reachable while still guaranteeing the
    # fleet ends at least as charged as it began.
    @constraint(model, [m in M], SOE_MCS[m, b_term] == d.SOE_MCS_ini[m])   # Eq. 10a (exact)
    @constraint(model, [e in E], SOE_CEV[e, b_term] >= d.SOE_CEV_ini[e])   # Eq. 10b (floor; overcharge OK)

    # ---- plugging / presence logic ----
    # No more plugged-in CEVs than the MCS has plugs (C_MCS_plug) at any site/interval.
    @constraint(model, [m in M, i in N_c, k in K], sum(rho[m, i, e, k] for e in E) <= d.C_MCS_plug[m])
    # A CEV can only be plugged into the MCS at ITS OWN assigned site (A[i,e]=1).
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= d.A[i, e])
    # ... and only when the MCS is actually parked there (z=1).
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])
    # Forbid a self-loop "trip" i -> i.
    @constraint(model, [m in M, i in N, k in K], x[m, i, i, k] == 0)

    # presence partition: parked at exactly one node OR in transit on one arc.
    @constraint(model, [m in M, k in K],
        sum(z[m, i, k] for i in N) + sum(y_trv[m, i, j, k] for i in N, j in N if i != j) == 1)

    # initial position (unless mid-drive, handled above)
    for m in M
        if mcs_transit0[m] === nothing
            p = mcs_node0[m]
            @constraint(model, z[m, p, first(K)] + sum(x[m, p, j, first(K)] for j in N if j != p) == 1)
        end
    end

    # departures / arrivals
        # beta_dep[m,i,k] = 1 iff the MCS departs node i this interval (any outgoing trip x).
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
    # Presence bookkeeping: arriving minus departing = change in "parked here" between steps.
    @constraint(model, [m in M, i in N, k in K[2:end]],
        beta_arr[m, i, k] - beta_dep[m, i, k] == z[m, i, k] - z[m, i, k - 1])
    # Cannot arrive and depart the same node in the same interval.
    @constraint(model, [m in M, i in N, k in K],
        beta_arr[m, i, k] + beta_dep[m, i, k] <= 1)

    # flow conservation (works with a mid-drive / site start and a grid end)
    for m in M, i in N
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k] for k in K) - sum(beta_dep[m, i, k] for k in K) ==
            z[m, i, last(K)] - start_here)
    end

    # The terminal energy target above (Eq. 10a) governs recovery on its own: since
    # transit does not draw from the battery, the MCS can reach the exact SOE target
    # while charging and then depart for a site afterward at no energy cost, so no
    # separate requirement to remain physically parked at the grid node is needed.

    # optional site-visit rules
    if require_site_visit
        @constraint(model, [m in M], sum(beta_arr[m, i, k] for i in N_c, k in K) >= 1)
    end
    if single_visit_per_site
        @constraint(model, [m in M, i in N_c], sum(beta_arr[m, i, k] for k in K) <= 1)
        @constraint(model, [m in M, i in N_c], sum(beta_dep[m, i, k] for k in K) <= 1)
    end

    # ---- activity scheduling ----
    # Each assigned CEV does EXACTLY one activity per interval (dig/load/travel/idle)
    # during working hours only (Eq. 8b, restricted to is_working; wd(k) since
    # is_working is a single-day profile repeated across the multi-day window).
    @constraint(model, [i in N_c, e in E, k in K; d.is_working[i, e, wd(k)]],
        sum(u[e, i, a, k] for a in B) == d.A[i, e])
    # (Eq. 8b): outside working hours every activity is forced off -- including idle --
    # replacing the old R_work big-M power cap (see 2_DataLoader.jl for why).
    @constraint(model, [i in N_c, e in E, a in B, k in K; !d.is_working[i, e, wd(k)]],
        u[e, i, a, k] == 0)
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= d.A[i, e])
    # A CEV may charge (mu=1) only in an idle interval (the 4-activity encoding of the
    # PDF's work-or-charge exclusivity; idle draws 0 kW so it is a true "do nothing").
    @constraint(model, [i in N_c, e in E, k in K],
        mu[i, e, k] <= (d.is_working[i, e, wd(k)] ? u[e, i, B[4], k] : 1))
    # (5e): work power = the chosen activity's constant draw. Idle (B[4]) has p_idle = 0,
    # so an idling CEV consumes no power (no time-varying power, no shutdown state).
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- required work (or the miss penalty) ----
    # Digging done this window (hours) + unfinished slack == remaining digging requirement.
    # The slack s_miss_work is penalised in the objective, so any shortfall costs rho_miss.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[1], k] for e in E, k in K if k <= k_term) + s_miss_work[i, B[1]] == max(rem_dig[i], 0.0))
    # Same balance for loading/swinging.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[2], k] for e in E, k in K if k <= k_term) + s_miss_work[i, B[2]] == max(rem_load[i], 0.0))

    # precedence (Eq. 12d): cumulative loading <= scale * cumulative digging, in raw
    # interval counts EXACTLY as in Avik (MCS_OPTIMAL_v4_real.jl):
    #   sum u[B2] <= scale * sum u[B1].
    # MPC SEAM (Avik has none, single-shot): seed with the work already APPLIED in
    # earlier windows (cum_*_site, converted hours -> interval counts by /delta_T).
    @constraint(model, [i in N_c, k in K],
        cum_load_site(i) / delta_T + sum(u[e, i, B[2], tau] for tau in first(K):k, e in E) <=
        d.scale * (cum_dig_site(i) / delta_T + sum(u[e, i, B[1], tau] for tau in first(K):k, e in E)))

    # ---- rest rule: <= rest_cap work intervals in any (rest_cap + 1) window ----
    # Equivalently "no (rest_cap+1)-th consecutive WORK interval" (travel counts as
    # work). In closed loop the window has no memory of the intervals already applied,
    # so a purely within-window rule lets work-runs leak across every re-solve. We fix
    # that by seeding the rule from `work_hist` (the completed Work/Break flags):
    #   (a) within-window: every (rest_cap+1)-window lying fully inside K;
    #   (b) SEAM: windows straddling the window start k, using the last `o` applied
    #       flags as known constants. The o = rest_cap seam (last rest_cap flags + the
    #       current interval) is the BINDING one — it guarantees the realized trajectory
    #       never runs a (rest_cap+1)-th consecutive work, INCLUDING at end of day.
    rest_cap = Int(round(d.t_limit_rest / delta_T))
    rest_win = rest_cap + 1
    Wc(e, i, k) = sum(u[e, i, a, k] for a in (B[1], B[2], B[3]))
    # (a) within-window full windows (start at k .. so it also covers the current step)
    if length(K) >= rest_win
        @constraint(model, [i in N_c, e in E, k0 in first(K):(last(K) - rest_win + 1)],
            sum(Wc(e, i, k) for k in k0:(k0 + rest_win - 1)) <= rest_cap)
    end
    # (b) seam windows straddling the boundary, seeded with applied history.
    for e in E
        h = work_hist[e]
        Lh = length(h)
        for o in 1:min(rest_cap, Lh)
            nfut = rest_win - o
            ks = [first(K) + t for t in 0:(nfut - 1)]
            all(k -> k in K, ks) || continue          # only enforce REAL (in-day) windows
            hsum = sum(h[(Lh - o + 1):Lh])
            @constraint(model, [i in N_c], hsum + sum(Wc(e, i, k) for k in ks) <= rest_cap)
        end
    end

    # travel pacing (Eq. 13a, 13b): one travel per `work_per_travel` intervals of useful
    # work (dig or load), EXACTLY as in Avik (MCS_OPTIMAL_v4_real.jl, work_per_travel = 4).
    # Two-sided band on cumulative travel V(k) vs cumulative useful work W(k):
    #   W(k) - work_per_travel <= work_per_travel * V(k) <= W(k).
    # Indexed per (site, CEV) exactly like Avik. V and W are raw APPLIED INTERVAL COUNTS
    # (whether the u indicator fired), scoped to the CURRENT CALENDAR DAY only (see
    # cum_trv_cnt_e / cum_work_cnt_e above) -- a battery-shortage-capped interval still
    # counts as one full travel/work interval, so no tolerance is needed. The A-guard
    # restricts to each CEV's assigned site so the nonzero seed cannot create spurious
    # constraints on unassigned (i,e) pairs (in Avik's single shot the seeds are 0, so
    # those rows are trivially 0 <= 0 and need no guard).
    work_per_travel = 4
    for i in N_c, e in E
        d.A[i, e] == 1 || continue
        for k in K
            V = cum_trv_cnt_e[e] + sum(u[e, i, B[3], tau] for tau in first(K):k)
            W = cum_work_cnt_e[e] +
                sum(u[e, i, a, tau] for a in (B[1], B[2]), tau in first(K):k)
            @constraint(model, work_per_travel * V <= W)
            @constraint(model, work_per_travel * V >= W - work_per_travel)
        end
    end

    # Solve. HiGHS's native MIP path can, rarely and non-deterministically on
    # Windows, throw a memory fault on a particular window. We catch it so a single
    # bad solve does NOT kill a multi-hour run: the caller checks has_values(model)
    # and, finding none, treats this interval as infeasible and HOLDS state.
    try
        optimize!(model)
    catch err
        @warn "MCSModel: solver threw during optimize!; treating window as no-solution (hold state)." exception = err
    end
    return model
end

end # module MCSModel
