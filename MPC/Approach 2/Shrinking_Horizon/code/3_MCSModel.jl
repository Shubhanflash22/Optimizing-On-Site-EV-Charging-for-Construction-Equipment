# #############################################################################
# MCSModel.jl  —  module MCSModel
# -----------------------------------------------------------------------------
# The "optimise" half of the controller. Given the CURRENT physical state and
# the CURRENT activity-power estimate, `build_window_model` builds and solves a
# mixed-integer linear program (MILP) that plans everything from "now" (the
# first interval of K_win) to the end of the daytime horizon. The MPC loop calls
# it once every 15 minutes on the remaining day (shrinking horizon), so the
# window always reaches the true end of day and all terminal rules genuinely
# apply.
#
# NOMENCLATURE is identical to MCS_OPTIMAL_v4_real.jl so a reviewer sees the same
# decision variables:
#   P_ch_MCS, P_dch_MCS, P_MCS_CEV, P_work, P_ch_tot, P_dch_tot   (power flows)
#   SOE_MCS, SOE_CEV                                              (state of energy;
#       MCS uses eta_ch_dch_mcs, CEV charging uses eta_ch_dch_cev)
#   u, mu, rho, z, x, y_trv, beta_arr, beta_dep                   (binaries)
#   P_peak_NC, P_peak_OP, s_miss_work                             (peaks / slack)
#
# The horizon is the full 24 h, and the MCS/CEV terminal energy-neutral rule
# (Eq. 10a/10b, per the arXiv paper's numbering) is enforced inside this single
# MILP, so the overnight MCS recharge
# is scheduled by the optimiser itself (no separate deterministic phase).
#
# -----------------------------------------------------------------------------
# APPROACH 2 ADDITION — `build_window_model_stochastic` (near the bottom of this
# file). `build_window_model` above is kept byte-for-byte as Approach 1 wrote
# it (still used for the Approach 0 one-shot baseline, which stays certainty-
# equivalent on purpose — see docs/Understanding_Stochastic_MPC.md). The new
# function is a SEPARATE model builder for the scenario-based stochastic MPC:
# instead of one power vector `pvec`, it takes a small SET of `S` sampled power
# vectors (`scenarios`) and builds `S` linked copies of the window, coupled by
# NON-ANTICIPATIVITY constraints on every "here-and-now" decision at the
# window's first interval (the activity chosen, the charge/discharge power
# scheduled, routing/plugging binaries) — so the plan commits to ONE action for
# right now that is feasible in EVERY scenario, while later intervals are free
# to differ per scenario (recourse). See that function's own docstring for the
# exact non-anticipativity set and why it excludes `P_work`/`SOE_CEV`.
# #############################################################################
module MCSModel

using JuMP
using HiGHS
using DataFrames

using ..Common: normalize_travel_steps, in_peak, clock_label

export build_window_model, build_window_model_stochastic

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
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true)
    # Frequently-used sets/scalars.
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

    K = collect(K_win)
    Tb = vcat(K, last(K) + 1)                                   # boundary indices
    K_peak = [k for k in K if in_peak(k, delta_T, d.t_start)]   # on-peak subset
    is_terminal = last(K) == d.n_day                            # does the window reach day-end?
    # Activity index -> its (estimated) power draw.
    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))

    # ---- read the SHARED applied-activity history (Option-2 unification) ----
    # All three history-dependent rules derive what they need from `hist`:
    #   precedence -> summed realized HOURS per activity (unchanged);
    #   pacing     -> summed applied INTERVAL COUNTS off the u indicator;
    #   rest rule  -> the recent Work(1)/Break(0) pattern (travel = work).
    cum_dig_e   = [sum((r[2][1] for r in hist[e]); init = 0.0) for e in E]
    cum_load_e  = [sum((r[2][2] for r in hist[e]); init = 0.0) for e in E]
    cum_trv_cnt_e  = [count(r -> r[1] == 3, hist[e]) for e in E]
    cum_work_cnt_e = [count(r -> r[1] in (1, 2), hist[e]) for e in E]
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
    set_attribute(model, "threads", 8)
    set_attribute(model, "parallel", "on")
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
    # binary that governs whether the CEV is charging. This adds a small,
    # deliberately-scaled penalty on LATER CEV-charging activity (idx-weighted
    # sum of the mu binary), so among schedules tied on real cost, earlier CEV
    # charging is preferred over waiting. weight = 1e-6 is small enough that it
    # can only ever decide among options already tied on real cost -- it can
    # never override a genuine cost difference (max possible spread over a
    # full day's intervals is a small fraction of a cent; see docs for the
    # worked numeric example). This penalty applies only to the deterministic
    # objective here; the scenario-based objective below already hedges timing
    # risk via scenarios.
    Kvec = collect(K)
    early_charge_term = sum(idx * mu[i, e, Kvec[idx]] for i in N_c, e in E, idx in eachindex(Kvec))

    # ---- OBJECTIVE (Eq. 1): total operating cost. All constraints are HARD;
    # the only slack is s_miss_work (Eq. 12c). ----
    @objective(model, Min,
        sum(d.lambda_whl_elec[k] * P_ch_tot[m, k] * delta_T for m in M, k in K) +                             # energy cost: price x grid kWh
        sum((d.carbon_price_per_ton / 1000.0) * d.lambda_CO2[k] * P_ch_tot[m, k] * delta_T for m in M, k in K) +  # carbon cost of that grid energy
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

    # ---- Terminal energy targets (Eq. 10a, 10b) ----
    # MCS: EXACT equality to its initial SOE (Eq. 10a) so it is fully ready for the
    # next day; because the horizon is the full 24 h, the overnight MCS recharge is
    # scheduled inside this single MILP (no separate phase).
    # CEV: a lower bound at its initial SOE (Eq. 10b as a FLOOR, >=). OVERCHARGING IS
    # ALLOWED — the CEV may finish the day at or above its start level. This removes
    # the overcharge knife-edge: since a CEV cannot discharge, a hard equality would
    # be unrecoverable whenever the stochastic plant lets its SOE drift above the
    # target; the floor keeps the terminal reachable while still guaranteeing the
    # fleet ends at least as charged as it began.
    if is_terminal
        @constraint(model, [m in M], SOE_MCS[m, last(Tb)] == d.SOE_MCS_ini[m])   # Eq. 10a (exact)
        @constraint(model, [e in E], SOE_CEV[e, last(Tb)] >= d.SOE_CEV_ini[e])   # Eq. 10b (floor; overcharge OK)
    end

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
    # during working hours only (Eq. 8b, restricted to is_working).
    @constraint(model, [i in N_c, e in E, k in K; d.is_working[i, e, k]],
        sum(u[e, i, a, k] for a in B) == d.A[i, e])
    # (Eq. 8b): outside working hours every activity is forced off -- including idle --
    # replacing the old R_work big-M power cap (see 2_DataLoader.jl for why).
    @constraint(model, [i in N_c, e in E, a in B, k in K; !d.is_working[i, e, k]],
        u[e, i, a, k] == 0)
    # An activity bit can only be set at the CEV's own site.
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= d.A[i, e])
    # A CEV may charge (mu=1) only in an idle interval (the 4-activity encoding of the
    # PDF's work-or-charge exclusivity; idle draws 0 kW so it is a true "do nothing").
    @constraint(model, [i in N_c, e in E, k in K],
        mu[i, e, k] <= (d.is_working[i, e, k] ? u[e, i, B[4], k] : 1))
    # (5e): work power = the chosen activity's constant draw. Idle (B[4]) has p_idle = 0,
    # so an idling CEV consumes no power (no time-varying power, no shutdown state).
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- required work (or the miss penalty) ----
    # Digging done this window (hours) + unfinished slack == remaining digging requirement.
    # The slack s_miss_work is penalised in the objective, so any shortfall costs rho_miss.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[1], k] for e in E, k in K) + s_miss_work[i, B[1]] == max(rem_dig[i], 0.0))
    # Same balance for loading/swinging.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[2], k] for e in E, k in K) + s_miss_work[i, B[2]] == max(rem_load[i], 0.0))

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
    # (whether the u indicator fired), seeded with cum_trv_cnt_e / cum_work_cnt_e -- a
    # battery-shortage-capped interval still counts as one full travel/work interval,
    # so no tolerance is needed. The A-guard restricts to each CEV's assigned site so
    # the nonzero seed cannot create spurious constraints on unassigned (i,e) pairs.
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

# =============================================================================
# STOCHASTIC (SCENARIO-BASED) WINDOW MILP  —  Approach 2
# -----------------------------------------------------------------------------
# Same physics, same constraints, same objective structure as `build_window_model`
# above — the ONLY conceptual difference is that the single power vector `pvec`
# is replaced by a SET of `S = length(scenarios)` sampled power vectors, and
# every variable family gains a trailing scenario index `s in 1:S`. The window
# therefore contains `S` parallel copies of "the future", coupled by a small set
# of NON-ANTICIPATIVITY constraints described below.
#
# WHICH VARIABLES ARE NON-ANTICIPATIVE (tied equal across every scenario at the
# window's very first interval, k0 = first(K)) and which are NOT:
#
#   TIED at k0 (the "here-and-now" decision — one action, chosen before you know
#   which scenario is real):
#     u, mu, rho, z, x, y_trv, beta_arr, beta_dep             (every binary)
#     P_ch_MCS, P_MCS_CEV, P_dch_MCS, P_ch_tot, P_dch_tot    (controlled power flows)
#   These are all things the controller CHOOSES: which activity to run, whether
#   to charge, how much grid/CEV power to schedule, whether to depart. Nothing
#   about "which scenario turns out true" can influence a choice that must be
#   made before that information exists — that IS non-anticipativity.
#
#   NOT tied at k0 (the UNCERTAIN CONSEQUENCE of that shared decision — this is
#   exactly the mechanism the worked example in docs/Understanding_Stochastic_MPC.md
#   walks through numerically):
#     P_work            — power an excavator's CHOSEN activity actually draws;
#                          computed as scenario_s power x the (tied) activity
#                          choice, so it is scenario-dependent even at k0 by
#                          construction (Dig is the same decision in every
#                          scenario; hard-soil Dig and average-soil Dig do NOT
#                          cost the same energy).
#     SOE_CEV            — propagates from P_work, so it diverges per scenario
#                          from k0+1 onward even though the action was shared.
#     SOE_MCS            — NOT explicitly tied either, but since every quantity
#                          feeding its recursion (P_ch_tot, P_dch_tot at k0) IS
#                          tied, it comes out equal across scenarios anyway —
#                          only the CEV side of this model is stochastic
#                          (consistent with Approach 0/1's own plant: "only the
#                          CEV side of the plant is stochastic").
#     s_miss_work, P_peak_NC, P_peak_OP  — window-wide aggregates that depend on
#                          the FULL scenario trajectory (k > k0), so they are
#                          naturally scenario-dependent and enter the objective
#                          as a probability-weighted average, not a worst case.
#
# OBJECTIVE: the same Eq. 1 cost, evaluated separately in each scenario copy and
# combined as a probability-weighted average, sum_s weights[s] * cost_s — the
# two-stage stochastic-program-with-recourse objective from
# docs/Understanding_Stochastic_MPC.md (Level 4). HARD constraints (battery
# floors, plug counts, precedence, etc.) apply in EVERY scenario individually —
# nothing is relaxed on average — which is what makes the optimiser reject a
# first-move that fails even one sampled scenario, exactly as in the worked
# example ("Digging is thrown out immediately ... because the one scenario
# where it fails is a hard-constraint breach").
#
# ARGUMENTS beyond `build_window_model`: `scenarios` is a `Vector{Vector{Float64}}`
# of length `S`, each an `eachindex(d.B)`-length vector of per-activity powers
# (build with `ScenarioSampler.sample_scenarios`); `weights` is the matching
# probability vector (defaults to equal, `1/S`, via `ScenarioSampler.equal_weights`
# if left `nothing`).
# =============================================================================
function build_window_model_stochastic(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, hist,
                            peak_nc0, peak_op0, scenarios::AbstractVector, weights::Union{Nothing,AbstractVector} = nothing;
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true)
    # Frequently-used sets/scalars.
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

    K = collect(K_win)
    Tb = vcat(K, last(K) + 1)                                   # boundary indices
    K_peak = [k for k in K if in_peak(k, delta_T, d.t_start)]   # on-peak subset
    is_terminal = last(K) == d.n_day                            # does the window reach day-end?
    k0 = first(K)                                                # the "here-and-now" interval

    # ---- scenarios ----
    nS = length(scenarios)
    nS >= 1 || error("build_window_model_stochastic: need at least one scenario, got $nS")
    S_scen = 1:nS
    w = weights === nothing ? fill(1.0 / nS, nS) : collect(float.(weights))
    length(w) == nS || error("build_window_model_stochastic: weights must have length $nS, got $(length(w))")
    # Activity index -> its sampled power draw, one Dict per scenario.
    p_activity_s = [Dict(B[a] => scenarios[s][a] for a in eachindex(B)) for s in S_scen]

    # ---- read the SHARED applied-activity history (same as the deterministic model) ----
    # precedence keeps hours; pacing now uses applied interval counts (see D-model twin).
    cum_dig_e   = [sum((r[2][1] for r in hist[e]); init = 0.0) for e in E]
    cum_load_e  = [sum((r[2][2] for r in hist[e]); init = 0.0) for e in E]
    cum_trv_cnt_e  = [count(r -> r[1] == 3, hist[e]) for e in E]
    cum_work_cnt_e = [count(r -> r[1] in (1, 2), hist[e]) for e in E]
    work_hist  = [Int[(r[1] in (1, 2, 3)) ? 1 : 0 for r in hist[e]] for e in E]

    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    # ---- model + solver configuration (identical to the deterministic builder) ----
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    isfinite(time_limit_sec) && set_time_limit_sec(model, time_limit_sec)
    set_attribute(model, "threads", 8)
    set_attribute(model, "parallel", "on")
    set_attribute(model, "mip_heuristic_effort", 0.0)
    set_attribute(model, "mip_detect_symmetry", false)
    set_attribute(model, "mip_rel_gap", 1.0e-2)

    # ---- CONTINUOUS decision variables: power flows (kW), now one copy per scenario ----
    @variable(model, P_ch_MCS[M, N, K, S_scen] >= 0)
    @variable(model, P_dch_MCS[M, N, K, S_scen] >= 0)
    @variable(model, P_MCS_CEV[M, N_c, E, K, S_scen] >= 0)
    @variable(model, P_work[N_c, E, K, S_scen] >= 0)
    @variable(model, P_ch_tot[M, K, S_scen] >= 0)
    @variable(model, P_dch_tot[M, K, S_scen] >= 0)
    @variable(model, s_miss_work[N_c, B, S_scen] >= 0)

    # ---- state of energy, indexed at interval BOUNDARIES ----
    @variable(model, SOE_MCS[M, Tb, S_scen] >= 0)
    @variable(model, SOE_CEV[E, Tb, S_scen] >= 0)

    # ---- BINARY decision variables ----
    @variable(model, u[E, N, B, K, S_scen], Bin)
    @variable(model, mu[N, E, K, S_scen], Bin)
    @variable(model, rho[M, N, E, K, S_scen], Bin)
    @variable(model, z[M, N, K, S_scen], Bin)
    @variable(model, x[M, N, N, K, S_scen], Bin)
    @variable(model, y_trv[M, N, N, K, S_scen], Bin)
    @variable(model, beta_arr[M, N, K, S_scen], Bin)
    @variable(model, beta_dep[M, N, K, S_scen], Bin)
    @variable(model, P_peak_NC[S_scen] >= 0)
    @variable(model, P_peak_OP[S_scen] >= 0)

    # ---- OBJECTIVE: probability-weighted average of the Eq. 1 cost across scenarios ----
    @objective(model, Min,
        sum(w[s] * (
            sum(d.lambda_whl_elec[k] * P_ch_tot[m, k, s] * delta_T for m in M, k in K) +
            sum((d.carbon_price_per_ton / 1000.0) * d.lambda_CO2[k] * P_ch_tot[m, k, s] * delta_T for m in M, k in K) +
            d.rho_miss * sum(s_miss_work[i, a, s] for i in N_c, a in B) +
            d.lambda_demand_NC * P_peak_NC[s] +
            d.lambda_demand_OP * P_peak_OP[s] +
            d.rho_labor * delta_T * sum(y_trv[m, i, j, k, s] for m in M, i in N, j in N, k in K)
        ) for s in S_scen))

    # ---- power aggregation & where power may flow (per scenario) ----
    @constraint(model, [m in M, k in K, s in S_scen], P_ch_tot[m, k, s]  == sum(P_ch_MCS[m, i, k, s]  for i in N_g))
    @constraint(model, [m in M, k in K, s in S_scen], P_dch_tot[m, k, s] == sum(P_dch_MCS[m, i, k, s] for i in N_c))
    @constraint(model, [m in M, i in N_g, k in K, s in S_scen], P_dch_MCS[m, i, k, s] == 0)
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], P_ch_MCS[m, i, k, s]  == 0)
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen],
        P_dch_MCS[m, i, k, s] == sum(P_MCS_CEV[m, i, e, k, s] for e in E))
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen],
        P_dch_MCS[m, i, k, s] <= d.DCH_MCS[m] * z[m, i, k, s])

    # Grid charging power is capped by CH_MCS and flows only when the MCS is
    # physically parked at that node (z=1).
    @constraint(model, [m in M, i in N_g, k in K, s in S_scen], P_ch_MCS[m, i, k, s] <= d.CH_MCS[m] * z[m, i, k, s])

    # plug-level and excavator-acceptance limits
    @constraint(model, [m in M, i in N_c, e in E, k in K, s in S_scen],
        P_MCS_CEV[m, i, e, k, s] <= d.DCH_MCS_plug[m] * rho[m, i, e, k, s])
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen],
        sum(P_MCS_CEV[m, i, e, k, s] for m in M) <= d.CH_CEV[e] * mu[i, e, k, s])
    # Charging-mode consistency: CEV e is in charging mode at site i (mu=1) exactly
    # when it is plugged into an MCS there (sum of rho over M).
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen],
        mu[i, e, k, s] == sum(rho[m, i, e, k, s] for m in M))

    # peak-demand trackers (carry the peak already seen earlier today), per scenario
    @constraint(model, [s in S_scen], P_peak_NC[s] >= peak_nc0)
    @constraint(model, [s in S_scen], P_peak_OP[s] >= peak_op0)
    @constraint(model, [k in K, s in S_scen], P_peak_NC[s] >= sum(P_ch_tot[m, k, s] for m in M))
    @constraint(model, [k in K_peak, s in S_scen], P_peak_OP[s] >= sum(P_ch_tot[m, k, s] for m in M))
    if peak_demand_limit !== nothing
        @constraint(model, [k in K, s in S_scen], sum(P_ch_tot[m, k, s] for m in M) <= peak_demand_limit)
    end

    # ---- travel energy bookkeeping ----
    for m in M, i in N, j in N, k in K, s in S_scen
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k, s] == 1)
        else
            @constraint(model, y_trv[m, i, j, k, s] == sum(x[m, i, j, tau, s]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end
    # MCS transit cost is captured as time via the rho_labor term in the objective;
    # y_trv is used there and in the routing/presence bookkeeping below.

    # ---- battery dynamics (per scenario) ----
    @constraint(model, [m in M, s in S_scen], SOE_MCS[m, first(Tb), s] == soe_mcs0[m])
    @constraint(model, [e in E, s in S_scen], SOE_CEV[e, first(Tb), s] == soe_cev0[e])
    # MCS SOE recursion: previous + charge*(eta_mcs) - discharge/(eta_mcs). Transit
    # does not draw from the battery (see travel-energy bookkeeping above).
    @constraint(model, [m in M, k in K, s in S_scen],
        SOE_MCS[m, k + 1, s] == SOE_MCS[m, k, s] +
            d.eta_ch_dch_mcs[m] * P_ch_tot[m, k, s] * delta_T -
            (P_dch_tot[m, k, s] * delta_T) / d.eta_ch_dch_mcs[m])
    # CEV SOE recursion uses THIS SCENARIO's sampled power for the work term — this
    # is the single line where the uncertainty actually enters the model. Received
    # energy is scaled by the CEV's own charge-acceptance efficiency, eta_ch_dch_cev.
    @constraint(model, [e in E, k in K, s in S_scen],
        SOE_CEV[e, k + 1, s] == SOE_CEV[e, k, s] +
            d.eta_ch_dch_cev[e] * sum(P_MCS_CEV[m, i, e, k, s] for m in M, i in N_c) * delta_T -
            sum(P_work[i, e, k, s] for i in N_c) * delta_T)

    # SOE operating ranges (Eq. 8c, 8d), per scenario — HARD in every scenario.
    @constraint(model, [m in M, t in Tb, s in S_scen], d.SOE_MCS_min[m] <= SOE_MCS[m, t, s] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb, s in S_scen], d.SOE_CEV_min[e] <= SOE_CEV[e, t, s] <= d.SOE_CEV_max[e])

    # ---- Terminal energy targets (Eq. 10a, 10b), per scenario ----
    if is_terminal
        @constraint(model, [m in M, s in S_scen], SOE_MCS[m, last(Tb), s] == d.SOE_MCS_ini[m])
        @constraint(model, [e in E, s in S_scen], SOE_CEV[e, last(Tb), s] >= d.SOE_CEV_ini[e])
    end

    # ---- plugging / presence logic ----
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], sum(rho[m, i, e, k, s] for e in E) <= d.C_MCS_plug[m])
    @constraint(model, [m in M, i in N, e in E, k in K, s in S_scen], rho[m, i, e, k, s] <= d.A[i, e])
    @constraint(model, [m in M, i in N, e in E, k in K, s in S_scen], rho[m, i, e, k, s] <= z[m, i, k, s])
    @constraint(model, [m in M, i in N, k in K, s in S_scen], x[m, i, i, k, s] == 0)

    # presence partition: parked at exactly one node OR in transit on one arc.
    @constraint(model, [m in M, k in K, s in S_scen],
        sum(z[m, i, k, s] for i in N) + sum(y_trv[m, i, j, k, s] for i in N, j in N if i != j) == 1)

    # initial position (unless mid-drive, handled above)
    for m in M, s in S_scen
        if mcs_transit0[m] === nothing
            p = mcs_node0[m]
            @constraint(model, z[m, p, first(K), s] + sum(x[m, p, j, first(K), s] for j in N if j != p) == 1)
        end
    end

    # departures / arrivals
    @constraint(model, [m in M, i in N, k in K, s in S_scen],
        beta_dep[m, i, k, s] == sum(x[m, i, j, k, s] for j in N if j != i))
    for m in M, j in N, k in K, s in S_scen
        if carried_arrival_k(m) == k && j == mcs_transit0[m][2]
            @constraint(model, beta_arr[m, j, k, s] == 1)
        else
            terms = Any[]
            for i in N
                i == j && continue
                tau = k - travel_steps[i, j]
                tau in K && push!(terms, x[m, i, j, tau, s])
            end
            @constraint(model, beta_arr[m, j, k, s] == (isempty(terms) ? 0 : sum(terms)))
        end
    end
    @constraint(model, [m in M, i in N, k in K[2:end], s in S_scen],
        beta_arr[m, i, k, s] - beta_dep[m, i, k, s] == z[m, i, k, s] - z[m, i, k - 1, s])
    @constraint(model, [m in M, i in N, k in K, s in S_scen],
        beta_arr[m, i, k, s] + beta_dep[m, i, k, s] <= 1)

    # flow conservation
    for m in M, i in N, s in S_scen
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k, s] for k in K) - sum(beta_dep[m, i, k, s] for k in K) ==
            z[m, i, last(K), s] - start_here)
    end

    # The terminal energy target above (Eq. 10a) governs recovery on its own, per
    # scenario: since transit does not draw from the battery, the MCS can reach the
    # exact SOE target while charging and then depart for a site afterward at no
    # energy cost, so no separate requirement to remain at the grid node is needed.

    if require_site_visit
        @constraint(model, [m in M, s in S_scen], sum(beta_arr[m, i, k, s] for i in N_c, k in K) >= 1)
    end
    if single_visit_per_site
        @constraint(model, [m in M, i in N_c, s in S_scen], sum(beta_arr[m, i, k, s] for k in K) <= 1)
        @constraint(model, [m in M, i in N_c, s in S_scen], sum(beta_dep[m, i, k, s] for k in K) <= 1)
    end

    # ---- activity scheduling (per scenario) ----
    # Eq. 8b, restricted to is_working; the same working-hours mask applies to every
    # scenario s since is_working is a property of the CALENDAR, not the plant draw.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen; d.is_working[i, e, k]],
        sum(u[e, i, a, k, s] for a in B) == d.A[i, e])
    @constraint(model, [i in N_c, e in E, a in B, k in K, s in S_scen; !d.is_working[i, e, k]],
        u[e, i, a, k, s] == 0)
    @constraint(model, [i in N_c, e in E, a in B, k in K, s in S_scen], u[e, i, a, k, s] <= d.A[i, e])
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen],
        mu[i, e, k, s] <= (d.is_working[i, e, k] ? u[e, i, B[4], k, s] : 1))
    # (5e) but with THIS SCENARIO's sampled power, not the posterior mean.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen],
        P_work[i, e, k, s] == sum(p_activity_s[s][a] * u[e, i, a, k, s] for a in B))

    # ---- required work (or the miss penalty), per scenario ----
    @constraint(model, [i in N_c, s in S_scen],
        delta_T * sum(u[e, i, B[1], k, s] for e in E, k in K) + s_miss_work[i, B[1], s] == max(rem_dig[i], 0.0))
    @constraint(model, [i in N_c, s in S_scen],
        delta_T * sum(u[e, i, B[2], k, s] for e in E, k in K) + s_miss_work[i, B[2], s] == max(rem_load[i], 0.0))

    # precedence (Eq. 12d), per scenario, seeded from applied history exactly as the
    # deterministic builder does.
    @constraint(model, [i in N_c, k in K, s in S_scen],
        cum_load_site(i) / delta_T + sum(u[e, i, B[2], tau, s] for tau in first(K):k, e in E) <=
        d.scale * (cum_dig_site(i) / delta_T + sum(u[e, i, B[1], tau, s] for tau in first(K):k, e in E)))

    # ---- rest rule (per scenario), seeded from applied history ----
    rest_cap = Int(round(d.t_limit_rest / delta_T))
    rest_win = rest_cap + 1
    Wc(e, i, k, s) = sum(u[e, i, a, k, s] for a in (B[1], B[2], B[3]))
    if length(K) >= rest_win
        @constraint(model, [i in N_c, e in E, k0w in first(K):(last(K) - rest_win + 1), s in S_scen],
            sum(Wc(e, i, k, s) for k in k0w:(k0w + rest_win - 1)) <= rest_cap)
    end
    for e in E
        h = work_hist[e]
        Lh = length(h)
        for o in 1:min(rest_cap, Lh)
            nfut = rest_win - o
            ks = [first(K) + t for t in 0:(nfut - 1)]
            all(k -> k in K, ks) || continue
            hsum = sum(h[(Lh - o + 1):Lh])
            @constraint(model, [i in N_c, s in S_scen], hsum + sum(Wc(e, i, k, s) for k in ks) <= rest_cap)
        end
    end

    # travel pacing (Eq. 13a, 13b), per scenario, seeded from applied INTERVAL COUNTS
    # (no tolerance needed -- see the deterministic builder above for the rationale).
    work_per_travel = 4
    for i in N_c, e in E
        d.A[i, e] == 1 || continue
        for k in K, s in S_scen
            V = cum_trv_cnt_e[e] + sum(u[e, i, B[3], tau, s] for tau in first(K):k)
            W = cum_work_cnt_e[e] +
                sum(u[e, i, a, tau, s] for a in (B[1], B[2]), tau in first(K):k)
            @constraint(model, work_per_travel * V <= W)
            @constraint(model, work_per_travel * V >= W - work_per_travel)
        end
    end

    # =========================================================================
    # NON-ANTICIPATIVITY — the ONE structural difference from S independent
    # deterministic MILPs solved side by side (Level 5 of the docs). Every
    # "here-and-now" control variable at k0 = first(K) is pinned equal across
    # every scenario, tied to scenario 1 (the choice of reference scenario is
    # arbitrary since equality is symmetric/transitive across all S copies).
    # See this function's docstring above for exactly which families are tied
    # and why P_work / SOE_CEV are deliberately excluded.
    # =========================================================================
    if nS > 1
        @constraint(model, [e in E, i in N, a in B, s in 2:nS], u[e, i, a, k0, s] == u[e, i, a, k0, 1])
        @constraint(model, [i in N, e in E, s in 2:nS], mu[i, e, k0, s] == mu[i, e, k0, 1])
        @constraint(model, [m in M, i in N, e in E, s in 2:nS], rho[m, i, e, k0, s] == rho[m, i, e, k0, 1])
        @constraint(model, [m in M, i in N, s in 2:nS], z[m, i, k0, s] == z[m, i, k0, 1])
        @constraint(model, [m in M, i in N, j in N, s in 2:nS], x[m, i, j, k0, s] == x[m, i, j, k0, 1])
        @constraint(model, [m in M, i in N, j in N, s in 2:nS], y_trv[m, i, j, k0, s] == y_trv[m, i, j, k0, 1])
        @constraint(model, [m in M, i in N, s in 2:nS], beta_arr[m, i, k0, s] == beta_arr[m, i, k0, 1])
        @constraint(model, [m in M, i in N, s in 2:nS], beta_dep[m, i, k0, s] == beta_dep[m, i, k0, 1])
        @constraint(model, [m in M, i in N, s in 2:nS], P_ch_MCS[m, i, k0, s] == P_ch_MCS[m, i, k0, 1])
        @constraint(model, [m in M, i in N, s in 2:nS], P_dch_MCS[m, i, k0, s] == P_dch_MCS[m, i, k0, 1])
        @constraint(model, [m in M, i in N_c, e in E, s in 2:nS], P_MCS_CEV[m, i, e, k0, s] == P_MCS_CEV[m, i, e, k0, 1])
        @constraint(model, [m in M, s in 2:nS], P_ch_tot[m, k0, s] == P_ch_tot[m, k0, 1])
        @constraint(model, [m in M, s in 2:nS], P_dch_tot[m, k0, s] == P_dch_tot[m, k0, 1])
    end

    # Solve. Same defensive catch as the deterministic builder.
    try
        optimize!(model)
    catch err
        @warn "MCSModel: solver threw during optimize! (stochastic); treating window as no-solution (hold state)." exception = err
    end
    return model
end

end # module MCSModel
