# #############################################################################
# MCSModel.jl  -  module MCSModel
# -----------------------------------------------------------------------------
# Builds and solves the MILP for a single MPC window: given the current state
# (SOE, MCS node/transit status, remaining work, activity history) and a fixed
# window of intervals K_win, constructs the JuMP model implementing the paper's
# objective function (4) and constraints (5)-(14), solves it with HiGHS, and
# returns the solved (or failed) model for the caller to read decisions from.
#
# This file has one exported function:
#
#   build_window_model_stochastic(d, K_win, soe_mcs0, soe_cev0, mcs_node0,
#                       mcs_transit0, rem_dig, rem_load, hist, peak_nc0,
#                       peak_op0, scenarios, weights; ...)
#   -- takes the problem data `d` (from DataLoader), the window's interval
#      range, the state carried in from outside the window (current SOE,
#      MCS location/in-transit status, remaining digging/loading work per
#      site, per-CEV activity history, and running NC/OP demand peaks), a
#      set of sampled power vectors (scenarios) and their probability
#      weights (equal weights if none are given). It builds one copy of every decision variable and
#      constraint per scenario, minimises the weighted average of the paper's
#      objective function (4) over the scenarios, and ties the decisions of
#      the window's first interval to be equal in every scenario, so that one
#      action is chosen now that works in all of them. The scenario copies
#      and the ties between them are not part of the paper's MILP formulation.
#      Also accepts solver options (time_limit_sec, silent).
#      Called repeatedly by 4_MPCLoop.jl, once per re-solve, each time with a
#      shifted window, updated carry-in state and a set of
#      scenarios from 2b_ScenarioSampler.jl -- this file has no notion of
#      "the whole day" or of re-solving itself, that logic lives entirely in
#      4_MPCLoop.jl.
# #############################################################################
module MCSModel

# external packages used across this file
using JuMP
using HiGHS
using DataFrames

using ..Common: normalize_travel_steps, in_peak, clock_label

# everything below that other files are allowed to use
export build_window_model_stochastic

# Builds and solves the MILP for one MPC window, with one copy of the window per scenario (see file header for the overall role of this function).
# d is the problem data from DataLoader, K_win is this window's interval range, and the remaining arguments carry in state from outside the window: current SOE, MCS node/transit status, remaining dig/load work per site, per-CEV activity history, running NC/OP demand peaks, the sampled activity power vectors scenarios, and their probability weights.
function build_window_model_stochastic(d, K_win, soe_mcs0, soe_cev0, mcs_node0, mcs_transit0,
                            rem_dig, rem_load, hist,
                            peak_nc0, peak_op0, scenarios::AbstractVector, weights::Union{Nothing,AbstractVector} = nothing;
                            time_limit_sec::Float64 = 30.0, silent::Bool = true)

    # Unpacks the index sets and base parameters needed throughout, and converts the raw travel-time matrix into integer interval-step counts.
    M, E, N, N_g, N_c, B = d.M, d.E, d.N, d.N_g, d.N_c, d.B
    delta_T = d.delta_T
    travel_steps = normalize_travel_steps(d.tau_trv, N)

     # Builds this window's interval set K and boundary set Tb, flags which intervals in K fall in the on-peak window, checks whether this window reaches the end of the day, and records the window's first interval k0, which is the interval whose decisions are tied across scenarios.
    K = collect(K_win)
    Tb = vcat(K, last(K) + 1)                                   
    K_peak = [k for k in K if in_peak(k, delta_T, d.t_start)]   
    is_terminal = last(K) == d.n_day                            
    k0 = first(K)                                                

    # Counts the scenarios and builds their index set S_scen, errors if there is none, takes the probability weights w (1/nS each if none are given) and errors unless there is one per scenario, and maps each activity index to its sampled power in every scenario.
    nS = length(scenarios)
    nS >= 1 || error("build_window_model_stochastic: need at least one scenario, got $nS")
    S_scen = 1:nS
    w = weights === nothing ? fill(1.0 / nS, nS) : collect(float.(weights))
    length(w) == nS || error("build_window_model_stochastic: weights must have length $nS, got $(length(w))")
    p_activity_s = [Dict(B[a] => scenarios[s][a] for a in eachindex(B)) for s in S_scen]

    # Aggregates each CEV's activity history (from before this window) into cumulative digging/loading hours and travel/work counts, plus a flat work-or-not history vector, used later by the precedence and rest constraints.
    cum_dig_e   = [sum((r[2][1] for r in hist[e]); init = 0.0) for e in E]
    cum_load_e  = [sum((r[2][2] for r in hist[e]); init = 0.0) for e in E]
    cum_trv_cnt_e  = [count(r -> r[1] == 3, hist[e]) for e in E]
    cum_work_cnt_e = [count(r -> r[1] in (1, 2), hist[e]) for e in E]
    work_hist  = [Int[(r[1] in (1, 2, 3)) ? 1 : 0 for r in hist[e]] for e in E]

    # Rolls the per-CEV cumulative dig/load history up to a per-site total, using the assignment matrix A, for use in the precedence constraint below.
    cum_dig_site(i)  = sum(cum_dig_e[e]  * d.A[i, e] for e in E)
    cum_load_site(i) = sum(cum_load_e[e] * d.A[i, e] for e in E)

    # Helper predicates for an MCS that entered this window already mid-transit (carried over from the previous window's solve): is_carried_trv marks the intervals it's still forced to be traveling, and carried_arrival_k gives the interval it's forced to arrive.
    # Basically this is how "still mid-journey" state gets carried forward into the new window's solve
    is_carried_trv(m, i, j, k) = (mcs_transit0[m] !== nothing &&
        (i, j) == (mcs_transit0[m][1], mcs_transit0[m][2]) &&
        k <= K[min(mcs_transit0[m][3], length(K))])
    carried_arrival_k(m) = mcs_transit0[m] === nothing ? nothing :
        (mcs_transit0[m][3] + 1 <= length(K) ? K[mcs_transit0[m][3] + 1] : nothing)

    # Creates the JuMP model on the HiGHS solver and sets solver options: 8 threads in parallel mode, symmetry detection on, MIP heuristics of 0.2, a 0.1% relative gap tolerance, and an optional time limit.
    # The 0.1% gap is a deliberate speed trade-off, so reported costs can sit up to about 0.1% above the true optimum, which is far below the effect sizes compared (a single extra travel or missed-work interval costs several times more).
    # The gap actually achieved is recorded per day in solve_log.gap_percent.
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    isfinite(time_limit_sec) && set_time_limit_sec(model, time_limit_sec)
    set_attribute(model, "threads", 8)
    set_attribute(model, "parallel", "on")
    set_attribute(model, "mip_heuristic_effort", 0.2)
    set_attribute(model, "mip_detect_symmetry", true)
    set_attribute(model, "mip_rel_gap", 1.0e-3)

    # Declares the continuous power, SOE, and missed-work decision variables from set D in the paper (P^ch,MCS, P^dch,MCS, P^MCS->CEV, P^work, P^ch,tot, P^dch,tot, s^miss, SOE^MCS, SOE^CEV), with one copy per scenario.
    @variable(model, P_ch_MCS[M, N, K, S_scen] >= 0)
    @variable(model, P_dch_MCS[M, N, K, S_scen] >= 0)
    @variable(model, P_MCS_CEV[M, N_c, E, K, S_scen] >= 0)
    @variable(model, P_work[N_c, E, K, S_scen] >= 0)
    @variable(model, P_ch_tot[M, K, S_scen] >= 0)
    @variable(model, P_dch_tot[M, K, S_scen] >= 0)
    @variable(model, s_miss_work[N_c, B, S_scen] >= 0)
    @variable(model, SOE_MCS[M, Tb, S_scen] >= 0)
    @variable(model, SOE_CEV[E, Tb, S_scen] >= 0)

    # Declares the binary decision variables from set D: activity assignment (u), charge-connection status (mu, rho), MCS node presence (z), MCS travel decisions (x, y_trv), and arrival/departure indicators (beta_arr, beta_dep), plus the NC/OP demand peak variables, with one copy per scenario.
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

    # early_charge_term is a small tie-breaking penalty (weight 1e-6, negligible next to the real cost terms) that nudges the solver toward charging CEVs earlier in the window when multiple schedules are otherwise equally good, weighted over the scenarios in the same way as the cost terms.
    Kvec = collect(K)
    early_charge_term = sum(w[s] * idx * mu[i, e, Kvec[idx], s] for i in N_c, e in E, idx in eachindex(Kvec), s in S_scen)

    # Objective function (4) of the paper's MILP formulation, taken as the probability-weighted average over the scenarios: electricity cost, carbon cost, missed-work penalty, NC/OP demand charges, and MCS travel labor cost, plus the early_charge_term tie-break.
    @objective(model, Min,
        sum(w[s] * (
            sum(d.lambda_whl_elec[k] * P_ch_tot[m, k, s] * delta_T for m in M, k in K) +
            sum((d.carbon_price_per_ton / 1000.0) * d.lambda_CO2[k] * P_ch_tot[m, k, s] * delta_T for m in M, k in K) +
            d.rho_miss * sum(s_miss_work[i, a, s] for i in N_c, a in B) +
            d.lambda_demand_NC * P_peak_NC[s] +
            d.lambda_demand_OP * P_peak_OP[s] +
            d.rho_labor * delta_T * sum(y_trv[m, i, j, k, s] for m in M, i in N, j in N, k in K)
        ) for s in S_scen) +
        1e-6 * early_charge_term)

    # Every constraint from here up to the tie constraints at the end of the function is repeated once for every scenario s in S_scen, on that scenario's own copy of the variables.
    # The constraint numbers cited below are those of the paper's MILP formulation, which has no scenario index.

    # Defines each MCS's total grid-charging power for this interval as the sum of what it draws across every grid node.
    # Constraint 5a of the paper's MILP formulation.
    @constraint(model, [m in M, k in K, s in S_scen], P_ch_tot[m, k, s]  == sum(P_ch_MCS[m, i, k, s]  for i in N_g))
    # Defines each MCS's total discharging power for this interval as the sum of what it sends out across every construction site.
    # Constraint 5b of the paper's MILP formulation.
    @constraint(model, [m in M, k in K, s in S_scen], P_dch_tot[m, k, s] == sum(P_dch_MCS[m, i, k, s] for i in N_c))
    # Forces an MCS to never discharge while sitting at a grid (charging) node.
    # Constraint 5c of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_g, k in K, s in S_scen], P_dch_MCS[m, i, k, s] == 0)
    # Forces an MCS to never charge from the grid while sitting at a construction site.
    # Constraint 5d of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], P_ch_MCS[m, i, k, s]  == 0)
    # Ensures the power an MCS sends out at a site exactly matches the sum of what it individually sends to each connected CEV there.
    # Constraint 5e of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], P_dch_MCS[m, i, k, s] == sum(P_MCS_CEV[m, i, e, k, s] for e in E))
    # Caps how much an MCS can discharge at a site by its hardware discharge capacity, only while it's physically present there.
    # Constraint 6b of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], P_dch_MCS[m, i, k, s] <= d.DCH_MCS[m] * z[m, i, k, s])
    # Caps how much an MCS can charge from the grid by its hardware charge capacity, only while it's physically present at that grid node.
    # Constraint 6a of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_g, k in K, s in S_scen], P_ch_MCS[m, i, k, s] <= d.CH_MCS[m] * z[m, i, k, s])
    # Caps the power through a single MCS-to-CEV plug by its hardware limit, only while that MCS and CEV are actually connected.
    # Constraint 7a of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_c, e in E, k in K, s in S_scen], P_MCS_CEV[m, i, e, k, s] <= d.DCH_MCS_plug[m] * rho[m, i, e, k, s])
    # Caps the total power a CEV receives from all MCSs combined by its own charging acceptance rate, only while it's marked ready to accept charge.
    # Constraint 7b of the paper's MILP formulation.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen], sum(P_MCS_CEV[m, i, e, k, s] for m in M) <= d.CH_CEV[e] * mu[i, e, k, s])
    # Defines whether a CEV is charging at all (mu) as the sum of its individual per-MCS connection flags (rho).
    # Constraint 7c of the paper's MILP formulation.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen], mu[i, e, k, s] == sum(rho[m, i, e, k, s] for m in M))
    # Seeds this window's running NC demand peak with whatever peak was already reached before this window began.
    # Carries forward the running max behind the P^NC term in objective function (4); not itself a numbered constraint.
    @constraint(model, [s in S_scen], P_peak_NC[s] >= peak_nc0)
    # Seeds this window's running OP demand peak with whatever peak was already reached before this window began.
    # Carries forward the running max behind the P^OP term in objective function (4); not itself a numbered constraint.
    @constraint(model, [s in S_scen], P_peak_OP[s] >= peak_op0)
    # Forces the NC peak variable to be at least as large as every interval's total charging power, which the objective's minimization then pins to the true maximum.
    # Implements the P^NC = max{...} definition from objective function (4).
    @constraint(model, [k in K, s in S_scen], P_peak_NC[s] >= sum(P_ch_tot[m, k, s] for m in M))
    # Same idea as above, restricted to only the on-peak intervals (4-9pm).
    # Implements the P^OP = max{...} definition from objective function (4).
    @constraint(model, [k in K_peak, s in S_scen], P_peak_OP[s] >= sum(P_ch_tot[m, k, s] for m in M))

    # Defines the travel-status indicator y_trv from the departure decision x over the travel-time window.
    # Includes a special case forcing y_trv=1 for an MCS already mid-transit before this window started, so a trip already underway can't be silently cancelled at the next re-solve.
    # Constraint 12b of the paper's MILP formulation.
    for m in M, i in N, j in N, k in K, s in S_scen
        i == j && continue
        if is_carried_trv(m, i, j, k)
            @constraint(model, y_trv[m, i, j, k, s] == 1)
        else
            @constraint(model, y_trv[m, i, j, k, s] == sum(x[m, i, j, tau, s]
                for tau in max(first(K), k - travel_steps[i, j] + 1):k if tau in K))
        end
    end

    # Fixes each MCS's starting SOE for this window to whatever it actually was at the end of the previous window's solve.
    # Initial condition feeding the SOE recursion in constraint 9a.
    @constraint(model, [m in M, s in S_scen], SOE_MCS[m, first(Tb), s] == soe_mcs0[m])
    # Fixes each CEV's starting SOE for this window to whatever it actually was at the end of the previous window's solve.
    # Initial condition feeding the SOE recursion in constraint 9b.
    @constraint(model, [e in E, s in S_scen], SOE_CEV[e, first(Tb), s] == soe_cev0[e])
    # Updates each MCS's battery level interval by interval: adds the energy charged from the grid (with efficiency losses), subtracts the energy discharged to CEVs (with efficiency losses).
    # Constraint 9a of the paper's MILP formulation. Physically equivalent to the paper's literal equation: this code's SOE index j corresponds to the paper's real clock boundary (j-1), since it reuses the same integers as the interval labels (K/Tb) rather than a separate 0..n boundary count -- so SOE_MCS[k+1] (state after interval k) driven by P_ch_tot[k] is the same physical recursion as the paper's SOE_{t+1} driven by P_{t+1}, just offset in how the boundary is numbered. Confirmed self-consistent with how 4_MPCLoop.jl reads these values (e.g. it logs SOE_CEV[e, k+1] as "the state after interval k").
    @constraint(model, [m in M, k in K, s in S_scen], SOE_MCS[m, k + 1, s] == SOE_MCS[m, k, s] + d.eta_ch_dch_mcs[m] * P_ch_tot[m, k, s] * delta_T - (P_dch_tot[m, k, s] * delta_T) / d.eta_ch_dch_mcs[m])
    # Updates each CEV's battery level interval by interval: adds the energy received from all connected MCSs (with efficiency losses), subtracts the energy spent working.
    # Constraint 9b of the paper's MILP formulation. Same indexing-convention note as the SOE_MCS recursion above applies here -- physically equivalent to the paper's literal equation, not a discrepancy.
    @constraint(model, [e in E, k in K, s in S_scen], SOE_CEV[e, k + 1, s] == SOE_CEV[e, k, s] + d.eta_ch_dch_cev[e] * sum(P_MCS_CEV[m, i, e, k, s] for m in M, i in N_c) * delta_T - sum(P_work[i, e, k, s] for i in N_c) * delta_T)
    # Keeps every MCS's battery level within its allowed minimum and maximum at every point in time, including the window's boundaries.
    # Constraint 10c of the paper's MILP formulation.
    @constraint(model, [m in M, t in Tb, s in S_scen], d.SOE_MCS_min[m] <= SOE_MCS[m, t, s] <= d.SOE_MCS_max[m])
    # Keeps every CEV's battery level within its allowed minimum and maximum at every point in time, including the window's boundaries.
    # Constraint 10d of the paper's MILP formulation.
    @constraint(model, [e in E, t in Tb, s in S_scen], d.SOE_CEV_min[e] <= SOE_CEV[e, t, s] <= d.SOE_CEV_max[e])

    # Only on the final window of the day, forces every MCS's ending SOE back to its starting value and every CEV's ending SOE to at least its starting value.
    # Prevents the day-long schedule from ending with batteries the optimizer has no incentive to refill.
    # Constraints 10a and 10b of the paper's MILP formulation.
    # The CEV condition is a >= inequality where the paper's (10b) is an equality; the two are equivalent whenever SOE_CEV_ini equals SOE_CEV_max (as in the paper's Table IX), and the relaxation is kept deliberately.
    if is_terminal
        @constraint(model, [m in M, s in S_scen], SOE_MCS[m, last(Tb), s] == d.SOE_MCS_ini[m])
        @constraint(model, [e in E, s in S_scen], SOE_CEV[e, last(Tb), s] >= d.SOE_CEV_ini[e])
    end

    # Caps how many CEVs a single MCS can be simultaneously connected to, by its number of physical outlet plugs.
    # Constraint 7d of the paper's MILP formulation.
    @constraint(model, [m in M, i in N_c, k in K, s in S_scen], sum(rho[m, i, e, k, s] for e in E) <= d.C_MCS_plug[m])
    # Only allows a CEV to connect to an MCS at a node if the CEV is actually assigned to that node.
    # Constraint 11a of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, e in E, k in K, s in S_scen], rho[m, i, e, k, s] <= d.A[i, e])
    # Only allows a CEV to connect to an MCS if that MCS is physically present at the same node at the same time.
    # Constraint 11b of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, e in E, k in K, s in S_scen], rho[m, i, e, k, s] <= z[m, i, k, s])
    # Forbids an MCS from "traveling" from a node to itself.
    # Constraint 12a of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, k in K, s in S_scen], x[m, i, i, k, s] == 0)
    # Forces every MCS to be in exactly one of two states each interval: parked at some node, or actively traveling on some route.
    # Constraint 12c of the paper's MILP formulation.
    @constraint(model, [m in M, k in K, s in S_scen], sum(z[m, i, k, s] for i in N) + sum(y_trv[m, i, j, k, s] for i in N, j in N if i != j) == 1)

    # For an MCS that starts this window already parked (not mid-transit), forces it to either stay parked at its known starting node or depart from it in the window's first interval.
    # Adapts the MCS initial-position condition from Appendix E of the paper for a rolling MPC window that doesn't necessarily start at the beginning of the day.
    for m in M, s in S_scen
        if mcs_transit0[m] === nothing
            p = mcs_node0[m]
            @constraint(model, z[m, p, first(K), s] + sum(x[m, p, j, first(K), s] for j in N if j != p) == 1)
        end
    end

    # Defines the departure indicator for an MCS at a node as whether it left for anywhere else this interval.
    # Constraint 13a of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, k in K, s in S_scen], beta_dep[m, i, k, s] == sum(x[m, i, j, k, s] for j in N if j != i))

    # Defines the arrival indicator for an MCS at a node as whether a trip that departed travel_steps intervals ago lands here this interval.
    # Includes a special case forcing the arrival exactly when a trip carried over from before this window is due to complete.
    # Constraint 13b of the paper's MILP formulation, extended for MPC window carry-over.
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

    # Ties arrivals and departures to the change in whether the MCS is present at a node: presence turns on because it arrived, and turns off because it departed.
    # Constraint 13c of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, k in K[2:end], s in S_scen], beta_arr[m, i, k, s] - beta_dep[m, i, k, s] == z[m, i, k, s] - z[m, i, k - 1, s])
    # Forbids an MCS from arriving at and departing from the same node in the same interval.
    # Constraint 13e of the paper's MILP formulation.
    @constraint(model, [m in M, i in N, k in K, s in S_scen], beta_arr[m, i, k, s] + beta_dep[m, i, k, s] <= 1)

    # Balances total arrivals against total departures at each node over the whole window, adjusted for the node the MCS is already sitting at when the window begins.
    # The adjustment (start_here) accounts for the fact that starting presence isn't itself an "arrival".
    # Constraint 13d of the paper's MILP formulation, adapted for a window that doesn't start at time zero.
    # As written, this is implied by constraint 13c plus the starting-position condition, so unlike the paper's version it does not force the MCS to finish where it started.
    # This is deliberate: the MCS may end the day at whichever node suits the plan, and in practice the terminal MCS SOE condition and the cost of extra travel bring it back to the grid node to charge overnight.
    for m in M, i in N, s in S_scen
        start_here = (mcs_transit0[m] === nothing && mcs_node0[m] == i) ? 1 : 0
        @constraint(model,
            sum(beta_arr[m, i, k, s] for k in K) - sum(beta_dep[m, i, k, s] for k in K) ==
            z[m, i, last(K), s] - start_here)
    end

    # When a CEV is scheduled to be on shift this interval, forces it to be doing exactly one activity (dig, load, travel, or idle) if it's assigned to this site, and none if it isn't.
    # Constraint 14a of the paper's MILP formulation, tightened to exactly-one-activity while on shift.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen; d.is_working[i, e, k]], sum(u[e, i, a, k, s] for a in B) == d.A[i, e])
    # When a CEV is not scheduled to be on shift this interval, forces every one of its activities to zero.
    # Implements Re,t = 0 outside work hours from constraint 8b.
    @constraint(model, [i in N_c, e in E, a in B, k in K, s in S_scen; !d.is_working[i, e, k]], u[e, i, a, k, s] == 0)
    # A CEV can only be assigned any activity at the node it's actually stationed at.
    # Part of the assignment restriction in constraint 8a.
    @constraint(model, [i in N_c, e in E, a in B, k in K, s in S_scen], u[e, i, a, k, s] <= d.A[i, e])
    # While on shift, a CEV can only be charging (mu=1) if it's simultaneously marked idle, so it can't charge and work at the same time.
    # Off shift, charging is left unrestricted, since off-shift time isn't governed by the paper's work/charge exclusivity rule.
    # Constraint 8c of the paper's MILP formulation.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen], mu[i, e, k, s] <= (d.is_working[i, e, k] ? u[e, i, B[4], k, s] : 1))
    # Defines a CEV's work power draw for this interval as whichever single activity it's doing, times that activity's sampled power in scenario s.
    # This is the only constraint where the scenarios' power values enter the model.
    # Constraint 8d of the paper's MILP formulation.
    @constraint(model, [i in N_c, e in E, k in K, s in S_scen], P_work[i, e, k, s] == sum(p_activity_s[s][a] * u[e, i, a, k, s] for a in B))
    # Forces the total digging hours scheduled in this window, plus whatever digging is logged as missed, to exactly equal the digging work still outstanding coming into the window.
    # Constraint 14b of the paper's MILP formulation, applied to the digging subactivity.
    @constraint(model, [i in N_c, s in S_scen], delta_T * sum(u[e, i, B[1], k, s] for e in E, k in K) + s_miss_work[i, B[1], s] == max(rem_dig[i], 0.0))
    # Same idea as above, but for the loading+swinging subactivity.
    # Constraint 14b of the paper's MILP formulation, applied to the loading+swinging subactivity.
    @constraint(model, [i in N_c, s in S_scen], delta_T * sum(u[e, i, B[2], k, s] for e in E, k in K) + s_miss_work[i, B[2], s] == max(rem_load[i], 0.0))
    # Caps how much loading+swinging work can be scheduled at a site relative to how much digging has been done there so far, scaled by d.scale, combining history from before this window with the running total inside it.
    # Prevents the loading step from being scheduled ahead of having dug enough material for it.
    # Constraint 14c of the paper's MILP formulation.
    @constraint(model, [i in N_c, k in K, s in S_scen], cum_load_site(i) / delta_T + sum(u[e, i, B[2], tau, s] for tau in first(K):k, e in E) <= d.scale * (cum_dig_site(i) / delta_T + sum(u[e, i, B[1], tau, s] for tau in first(K):k, e in E)))

    # Converts the rest-limit parameter t_limit_rest (in hours) into a whole number of intervals, rest_cap, and defines rest_win as one interval longer -- the window size over which that rest cap is checked.
    # Wc sums a CEV's construction-related activity flags (dig, load, travel -- everything except idle) for one interval, used as the per-interval "was working" indicator by both rest-limit checks below.
    # Supports constraint 14d of the paper's MILP formulation.
    rest_cap = Int(round(d.t_limit_rest / delta_T))
    rest_win = rest_cap + 1
    Wc(e, i, k, s) = sum(u[e, i, a, k, s] for a in (B[1], B[2], B[3]))

    # Limits each CEV to at most rest_cap intervals of construction-related work (dig, load, or travel) within any rolling rest_win-interval window fully inside this window.
    # Constraint 14d of the paper's MILP formulation.
    if length(K) >= rest_win
        @constraint(model, [i in N_c, e in E, k0w in first(K):(last(K) - rest_win + 1), s in S_scen],
            sum(Wc(e, i, k, s) for k in k0w:(k0w + rest_win - 1)) <= rest_cap)
    end

    # Same rest-limit idea as above, but checked against work carried over from just before this window started, for the intervals right at the window's start.
    # Constraint 14d of the paper's MILP formulation, adapted for MPC window carry-over.
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

    # Keeps each CEV's travel evenly interspersed with its productive work: at most one travel interval for every work_per_travel productive work intervals, and at least one required once that many work intervals have piled up.
    # Tracked cumulatively from before this window through the end of it.
    # Constraints 14e and 14f of the paper's MILP formulation.
    work_per_travel = d.kappa_wt
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

    # Ties the here-and-now decisions of the window's first interval k0 to be equal in every scenario, so that a single action is chosen now that is feasible in all of them, while later intervals are free to differ per scenario.
    # Not part of the paper's MILP formulation: these non-anticipativity constraints exist only because the window is copied once per scenario, and each one sets scenario s equal to scenario 1 for s in 2:nS.
    # P_work, SOE_CEV and s_miss_work are deliberately not tied, since they are the uncertain consequence of the shared decision and differ by scenario.
    # SOE_MCS after the first interval needs no tie, because it already follows from the tied charge and discharge power.
    if nS > 1
        # Ties each CEV's activity choice (u) in the first interval to be the same in every scenario.
        @constraint(model, [e in E, i in N, a in B, s in 2:nS], u[e, i, a, k0, s] == u[e, i, a, k0, 1])
        # Ties each CEV's charging status (mu) in the first interval to be the same in every scenario.
        @constraint(model, [i in N, e in E, s in 2:nS], mu[i, e, k0, s] == mu[i, e, k0, 1])
        # Ties each MCS-to-CEV connection flag (rho) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, e in E, s in 2:nS], rho[m, i, e, k0, s] == rho[m, i, e, k0, 1])
        # Ties each MCS's node presence (z) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, s in 2:nS], z[m, i, k0, s] == z[m, i, k0, 1])
        # Ties each MCS's departure decision (x) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, j in N, s in 2:nS], x[m, i, j, k0, s] == x[m, i, j, k0, 1])
        # Ties each MCS's travel status (y_trv) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, j in N, s in 2:nS], y_trv[m, i, j, k0, s] == y_trv[m, i, j, k0, 1])
        # Ties each MCS's arrival indicator (beta_arr) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, s in 2:nS], beta_arr[m, i, k0, s] == beta_arr[m, i, k0, 1])
        # Ties each MCS's departure indicator (beta_dep) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, s in 2:nS], beta_dep[m, i, k0, s] == beta_dep[m, i, k0, 1])
        # Ties each MCS's grid charging power at each node (P_ch_MCS) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, s in 2:nS], P_ch_MCS[m, i, k0, s] == P_ch_MCS[m, i, k0, 1])
        # Ties each MCS's discharging power at each node (P_dch_MCS) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N, s in 2:nS], P_dch_MCS[m, i, k0, s] == P_dch_MCS[m, i, k0, 1])
        # Ties the power each MCS sends to each CEV (P_MCS_CEV) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, i in N_c, e in E, s in 2:nS], P_MCS_CEV[m, i, e, k0, s] == P_MCS_CEV[m, i, e, k0, 1])
        # Ties each MCS's total grid-charging power (P_ch_tot) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, s in 2:nS], P_ch_tot[m, k0, s] == P_ch_tot[m, k0, 1])
        # Ties each MCS's total discharging power (P_dch_tot) in the first interval to be the same in every scenario.
        @constraint(model, [m in M, s in 2:nS], P_dch_tot[m, k0, s] == P_dch_tot[m, k0, 1])
    end

    # Solves the model, catching (rather than crashing on) any solver-level exception so the caller can treat a failed window as "hold current state" instead of the whole run dying.
    try
        optimize!(model)
    catch err
        @warn "MCSModel: solver threw during optimize! (stochastic); treating window as no-solution (hold state)." exception = err
    end
    return model
end

end