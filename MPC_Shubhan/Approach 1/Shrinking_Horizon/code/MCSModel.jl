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
#   L_trv, L_trv_tot                                              (travel energy)
#   SOE_MCS, SOE_CEV                                              (state of energy)
#   u, mu, rho, z, g_ch, x, y_trv, beta_arr, beta_dep            (binaries)
#   P_peak_NC, P_peak_OP, s_miss_work                             (peaks / slack)
#
# `phase2_overnight_charge` is the deterministic (non-MILP) overnight smart
# charge that restores each MCS to its start level using the cheapest slots.
# #############################################################################
module MCSModel

using JuMP
using HiGHS
using DataFrames

using ..Common: normalize_travel_steps, in_peak, clock_label

export build_window_model, phase2_overnight_charge

# =============================================================================
# WINDOW MILP
# =============================================================================
# The arguments after `d` are the CLOSED-LOOP CARRY-IN — the real state handed
# from the previous step:
#   soe_mcs0/soe_cev0 : measured battery levels now (kWh)
#   mcs_node0         : node the MCS is parked at (0 if mid-drive)
#   mcs_transit0      : nothing, or (i,j,r) = mid-drive on arc i->j, r steps left
#   rem_dig/rem_load  : work hours still remaining at each site
#   cum_*_e           : hours each excavator has already done (precedence/pacing)
#   peak_nc0/peak_op0 : biggest grid draw seen so far today (demand charges)
#   pvec              : the current per-activity power estimate (est.mu)
function build_window_model(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, cum_dig_e, cum_load_e, cum_trv_e,
                            peak_nc0, peak_op0, pvec;
                            require_site_visit::Bool = false,
                            single_visit_per_site::Bool = false,
                            peak_demand_limit = nothing,
                            time_limit_sec::Float64 = 30.0, silent::Bool = true,
                            soft_prec::Bool = false,
                            soft_pace::Bool = false,
                            soft_term::Bool = false,
                            term_tol::Float64 = 0.0)
    # Frequently-used sets/scalars.
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

    K = collect(K_win)
    Tb = vcat(K, last(K) + 1)                                   # boundary indices
    K_peak = [k for k in K if in_peak(k, delta_T, d.t_start)]   # on-peak subset
    is_terminal = last(K) == d.n_day                            # does the window reach day-end?
    productive_k = Dict(k => any(d.R_work[i, e, k] > 0 for i in N_c, e in E) for k in K)

    # Activity index -> its (estimated) power draw.
    p_activity = Dict(B[a] => pvec[a] for a in eachindex(B))

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
    set_time_limit_sec(model, time_limit_sec)
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
    @variable(model, s_miss_work[N_c, B] >= 0)     # UNFINISHED work (hours) — penalised slack
    @variable(model, s_prec[N_c, K] >= 0)          # precedence slack (soft mode)
    @variable(model, s_pace_hi[E, K] >= 0)         # travel-pacing upper-band slack
    @variable(model, s_pace_lo[E, K] >= 0)         # travel-pacing lower-band slack

    # ---- travel energy (kWh) ----
    @variable(model, L_trv[M, N, N, K] >= 0)
    @variable(model, L_trv_tot[M, K] >= 0)

    # ---- state of energy, indexed at interval BOUNDARIES ----
    @variable(model, SOE_MCS[M, Tb] >= 0)
    @variable(model, SOE_CEV[E, Tb] >= 0)

    # ---- BINARY decision variables ----
    @variable(model, u[E, N, B, K], Bin)           # which activity each excavator does
    @variable(model, mu[N, E, K], Bin)             # is the excavator charging?
    @variable(model, rho[M, N, E, K], Bin)         # is the excavator plugged into the MCS?
    @variable(model, z[M, N, K], Bin)              # is the MCS parked at this node?
    @variable(model, g_ch[M, N_g, K], Bin)         # is the MCS actively grid-charging here?
    @variable(model, x[M, N, N, K], Bin)           # does the MCS depart i -> j this interval?
    @variable(model, y_trv[M, N, N, K], Bin)       # is the MCS in transit on arc i -> j?
    @variable(model, beta_arr[M, N, K], Bin)       # MCS arrival indicator at a node
    @variable(model, beta_dep[M, N, K], Bin)       # MCS departure indicator at a node
    @variable(model, P_peak_NC >= 0)               # tracked whole-day peak grid draw
    @variable(model, P_peak_OP >= 0)               # tracked on-peak peak grid draw
    @variable(model, s_term_cev[E] >= 0)           # CEV end-level slack (soft mode)

    # ---- OBJECTIVE: total operating cost ----
    obj = @expression(model,
        sum(d.lambda_whl_elec[k] * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        sum((d.carbon_price_per_ton / 1000.0) * d.lambda_CO2[k] * P_ch_tot[m, k] * delta_T for m in M, k in K) +
        d.rho_miss * sum(s_miss_work[i, a] for i in N_c, a in B) +
        d.lambda_demand_NC * P_peak_NC +
        d.lambda_demand_OP * P_peak_OP +
        d.rho_labor * delta_T * sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K))

    # HARD MODE (default): pin optional slacks to zero. Soft mode penalises them.
    W_prec = 8.0e2; W_pace = 1.0e2; W_term = 1.5e2
    soft_prec || @constraint(model, [i in N_c, k in K], s_prec[i, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_hi[e, k] == 0)
    soft_pace || @constraint(model, [e in E, k in K], s_pace_lo[e, k] == 0)
    soft_term || @constraint(model, [e in E], s_term_cev[e] == 0)
    @objective(model, Min, obj +
        (soft_prec ? W_prec * sum(s_prec[i, k] for i in N_c, k in K) : AffExpr(0.0)) +
        (soft_pace ? W_pace * sum(s_pace_hi[e, k] + s_pace_lo[e, k] for e in E, k in K) : AffExpr(0.0)) +
        (soft_term ? W_term * sum(s_term_cev[e] for e in E) : AffExpr(0.0)))

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

    # peak-demand trackers (carry the peak already seen earlier today)
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
    @constraint(model, [m in M, k in K],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
            d.eta_ch_dch[m] * P_ch_tot[m, k] * delta_T -
            (P_dch_tot[m, k] * delta_T) / d.eta_ch_dch[m] -
            L_trv_tot[m, k])
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
            sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -
            sum(P_work[i, e, k] for i in N_c) * delta_T)

    @constraint(model, [m in M, t in Tb], d.SOE_MCS_min[m] <= SOE_MCS[m, t] <= d.SOE_MCS_max[m])
    @constraint(model, [e in E, t in Tb], d.SOE_CEV_min[e] <= SOE_CEV[e, t] <= d.SOE_CEV_max[e])

    # NOTE: no "MCS must end the day full" rule — the MCS is refilled overnight
    # in the cheap Phase-2 charge, not during the daytime MILP.

    # ---- CEV end-of-day energy neutrality (only on the terminal window) ----
    if is_terminal
        if soft_term
            @constraint(model, [e in E],  SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e] <= s_term_cev[e])
            @constraint(model, [e in E], -(SOE_CEV[e, last(Tb)] - d.SOE_CEV_ini[e]) <= s_term_cev[e])
        else
            @constraint(model, [e in E], SOE_CEV[e, last(Tb)] >= d.SOE_CEV_ini[e] - term_tol)
        end
    end

    # ---- plugging / presence logic ----
    @constraint(model, [m in M, i in N_c, k in K], sum(rho[m, i, e, k] for e in E) <= d.C_MCS_plug[m])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= d.A[i, e])
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])
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

    # flow conservation (works with a mid-drive / site start and a grid end)
    for m in M, i in N
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k] for k in K) - sum(beta_dep[m, i, k] for k in K) ==
            z[m, i, last(K)] - start_here)
    end

    # terminal position: end parked at a grid node (ready for overnight refill).
    if is_terminal
        @constraint(model, [m in M], sum(z[m, i, last(K)] for i in N_g) == 1)
    end

    # optional site-visit rules
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
        d.R_work[i, e, k] * d.A[i, e] * (1 - mu[i, e, k]))
    @constraint(model, [i in N_c, e in E, k in K], mu[i, e, k] <= u[e, i, B[4], k])
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B))

    # ---- required work (or the miss penalty) ----
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[1], k] for e in E, k in K) + s_miss_work[i, B[1]] == max(rem_dig[i], 0.0))
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[2], k] for e in E, k in K) + s_miss_work[i, B[2]] == max(rem_load[i], 0.0))

    # precedence: cumulative loading <= scale * cumulative digging (seeded).
    @constraint(model, [i in N_c, k in K],
        (cum_load_site(i) + delta_T * sum(u[e, i, B[2], tau] for tau in first(K):k, e in E)) <=
        d.scale * (cum_dig_site(i) + delta_T * sum(u[e, i, B[1], tau] for tau in first(K):k, e in E)) +
        s_prec[i, k])

    # rest rule: <= t_limit_rest hours of work per rolling (t_limit_rest + step) window.
    rest_cap = Int(round(d.t_limit_rest / delta_T))
    rest_win = rest_cap + 1
    if length(K) >= rest_win
        @constraint(model, [i in N_c, e in E, k0 in first(K):(last(K) - rest_win + 1)],
            sum(u[e, i, a, k] for a in (B[1], B[2], B[3]), k in k0:(k0 + rest_win - 1)) <= rest_cap)
    end

    # travel pacing: keep cumulative travel ~ proportional to cumulative work.
    kappa = d.kappa_wt
    for e in E, kk in K
        trv_cum  = cum_trv_e[e] / delta_T +
                   sum(u[e, i, B[3], tau] for i in N_c, tau in first(K):kk)
        work_cum = (cum_dig_e[e] + cum_load_e[e]) / delta_T +
                   sum(u[e, i, a, tau] for i in N_c, a in (B[1], B[2]), tau in first(K):kk)
        @constraint(model, kappa * trv_cum <= work_cum + s_pace_hi[e, kk])
        @constraint(model, kappa * trv_cum >= work_cum - kappa - s_pace_lo[e, kk])
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
# PHASE 2 — OVERNIGHT SMART-CHARGE  (deterministic; NOT an optimisation)
# =============================================================================
# After the daytime horizon the MCS is parked at the grid with some energy. The
# overnight job buys back exactly the energy it is short by, using the CHEAPEST
# overnight 15-min slots first, capped at its charge rate and capacity. Because
# energy only rises and the target is <= capacity, greedy "cheapest slots first"
# is provably optimal — no MILP needed. Returns (df, P_ov, ov_k).
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
