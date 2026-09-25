# #############################################################################
# run_one_shot_detailed.jl  —  included inside `module A0App`
# -----------------------------------------------------------------------------
# A near-verbatim copy of Approach 1's `MPCLoop.run_one_shot` (Approach 0: the
# one-shot 8:00 whole-day plan, executed open-loop, no replanning), with
# logging added so every 15-minute decision A0 makes is captured on disk, for
# BOTH the CEVs and the MCS.
#
#   * CEV plan     — reuses Common's own `DetailedPlanLog` / `log_plan_row!`,
#                    the SAME struct A1S/A2S already write into (see
#                    4_MPCLoop.jl's `run_mpc`), so A0_CEV_plan_full.csv lines
#                    up column-for-column with A1_plan_full.csv /
#                    A2_plan_full.csv. A0 solves the WHOLE day in one solve,
#                    so `resolve_step` is always 1 for every day and
#                    `offset_step` runs 1..nKd — this is the one and only
#                    plan A0 ever makes for that day, logged in full (every
#                    step, not just the one that gets executed first).
#   * CEV realized — reuses Common's `RealizedTupleLog` / `log_realized_row!`,
#                    again the same struct A1S/A2S use, fed the exact same
#                    `step.p_true[e]` 4-tuple `apply_and_simulate!` already
#                    computes (nothing new derived, just captured).
#   * MCS plan / realized — A1S/A2S never logged the MCS's own planned or
#                    realized trajectory in detail (only folded
#                    `soe_mcs_planned_kWh` into each CEV row). This file adds
#                    two new, equally simple NamedTuple-vector logs for it
#                    (`_mcs_plan_rows!` / push into `mcs_realized_rows`
#                    below) so the MCS's own status/location/charging story
#                    is on disk too, not just inferred from the CEV rows.
#
# WHY-EXPLANATION COLUMNS: every plan/realized row also gets that interval's
# electricity price and CO2 factor joined on afterwards (`_add_price_co2`
# below) — the two signals that most directly drive the MILP's
# charging/discharging/timing choices — so a reviewer can see, for any
# 15-minute slot, both WHAT was decided and the price/carbon context that
# pushed the optimizer toward it.
#
# Nothing about `build_window_model`, `apply_and_simulate!`, `activity_label`,
# `mcs_status_label`, or `_terminal_soe_shortfall` is duplicated or
# reimplemented — this file calls the SAME functions Approach 1's own tested
# `run_one_shot` / `run_mpc` call (via `MPCLoop.<name>`, fully qualified since
# they are internal, unexported helpers), just adds bookkeeping around them.
# #############################################################################

# Small local helper -- NOT exported by Common, kept private to this file.
# One row of the MCS's OWN planned trajectory for interval k, for MCS unit m,
# at the single resolve (day-start) that produced `model`.
function _mcs_plan_row(model, d, day::Int, k::Int, m::Int)
    node = findfirst(i -> value(model[:z][m, i, k]) > 0.5, d.N)
    node_label = node === nothing ? "Transit" : string(node)
    grid_charge_kW    = value(model[:P_ch_tot][m, k])
    grid_discharge_kW = value(model[:P_dch_tot][m, k])
    soe_mcs_planned   = value(model[:SOE_MCS][m, k + 1])
    status = MPCLoop.mcs_status_label(model, d, k)   # whole-MCS-fleet status rule, same one A1S/A2S use
    return (; day, resolve_step = 1, offset_step = k,
              offset_clock = clock_label(d.t_start, d.delta_T, k),
              mcs = m,
              mcs_status_planned = status,
              mcs_node_planned = node_label,
              grid_charge_kW_planned = grid_charge_kW,
              grid_discharge_kW_planned = grid_discharge_kW,
              soe_mcs_planned_kWh = soe_mcs_planned)
end

# One row of the MCS's OWN realized trajectory for the interval ACTUALLY
# executed (gidx is the GLOBAL index into the real_* arrays; k0 is day-local,
# used for the clock label and the day-local `real_mcs_act` lookup).
function _mcs_realized_row(d, day::Int, k0::Int, gidx::Int, m::Int,
                            real_P_ch, real_P_dch, real_loc, soe_mcs, mcs_status_label_str)
    node = real_loc[m, gidx]
    node_label = node == 0 ? "Transit" : string(node)
    return (; day, step = k0, clock = clock_label(d.t_start, d.delta_T, k0),
              mcs = m,
              mcs_status_realized = mcs_status_label_str,
              mcs_node_realized = node_label,
              grid_charge_kW_realized = real_P_ch[m, gidx],
              grid_discharge_kW_realized = real_P_dch[m, gidx],
              soe_mcs_kWh = soe_mcs[m])
end

# Joins price (USD/kWh) and CO2 factor onto a detailed CEV plan/realized
# DataFrame, matched on the day-local step column (`offset_step` for the plan
# log, `step` for the realized log) -- both index the SAME day-local
# `d.lambda_whl_elec` / `d.lambda_CO2` arrays, since conditions repeat
# identically every day (see 4_MPCLoop.jl's CHANGE 5 notes).
function _add_price_co2(df::DataFrame, d, step_col::Symbol, nKd::Int)
    isempty(df) && return df
    lut = DataFrame(step_col => 1:nKd,
                     :price_USD_per_kWh => d.lambda_whl_elec[1:nKd],
                     :co2_factor => d.lambda_CO2[1:nKd])
    return leftjoin(df, lut, on = step_col)
end

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Same signature/semantics as `MPCLoop.run_one_shot` (see that function's own
# header in 4_MPCLoop.jl for the full description of Approach 0's execution
# model); the only addition is that this ALWAYS records the detailed
# plan/realized logs (there is no `detailed_output` off-switch here since the
# whole point of this standalone A0 runner is to always produce them).
function run_one_shot_detailed(d, pool::ActivityPowerPool; time_limit_sec::Float64 = Inf,
                                multi_activity::Bool = false,
                                require_site_visit::Bool = false,
                                single_visit_per_site::Bool = false,
                                plant::Symbol = :sampled,
                                n_day_run::Int = 1,
                                seed::Int = 1)
    plant in (:sampled, :mean) ||
        error("run_one_shot_detailed: plant must be :sampled or :mean, got :$plant")
    n_day_run >= 1 || error("run_one_shot_detailed: n_day_run must be >= 1, got $n_day_run")
    Random.seed!(seed)
    K_all = collect(d.K)
    nKd = length(K_all)
    n_kept = n_day_run * nKd
    time_labels = n_day_run == 1 ? build_time_labels(d.t_start, d.delta_T, nKd) :
                                    build_time_labels_days(d.t_start, d.delta_T, n_day_run, nKd)
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
    solve_log = DataFrame(day = Int[], status = String[], objective = Float64[],
                          gap_percent = Float64[], solve_time_s = Float64[])

    nM = length(d.M); nE = length(d.E); nN = length(d.N)
    real_P_ch  = zeros(nM, n_kept)
    real_P_dch = zeros(nM, n_kept)
    real_L_trv = zeros(nM, n_kept)
    real_SOE_MCS = zeros(nM, n_kept + 1)
    real_SOE_CEV = zeros(nE, n_kept + 1)
    real_P_work  = zeros(nN, nE, n_kept)
    real_loc     = zeros(Int, nM, n_kept)
    real_cev_act = [fill("", n_kept) for _ in d.E]
    real_mcs_act = fill("", n_kept)

    # ---- DETAILED OUTPUT LOGS (always on for this standalone A0 runner) ----
    plan_log_cev     = DetailedPlanLog()
    realized_log_cev = RealizedTupleLog()
    mcs_plan_rows     = NamedTuple[]
    mcs_realized_rows = NamedTuple[]
    day_snapshot_rows = NamedTuple[]

    pmode_txt = plant === :mean ?
        ":mean (DETERMINISTIC -- realized power pinned to mu; realized == planned)" :
        ":sampled (stochastic -- realized power drawn from the shared pool)"
    println("Running Approach 0 (one-shot 8:00 plan per day, executed open-loop, no replanning) -- DETAILED: $n_kept steps ($n_day_run day(s))")
    println("  plant                  : ", pmode_txt)
    println("  planning power (mu)    : ", round.(pool.mu, digits = 2), " kW")
    plant === :sampled && println("  plant sampling sd      : ", round.(pool.sd, digits = 2), " kW")
    println("  solver time limit      : ",
            isfinite(time_limit_sec) ? "$(time_limit_sec) s" : "none (solve to the MIP gap)")
    t0 = time()
    n_obs_total = 0
    n_capped_total = 0

    for day in 1:n_day_run
        rem_dig  .+= float.(d.hours_digging)
        rem_load .+= float.(d.hours_loading_swinging)

        day_t0 = time()
        model = build_window_model(d, K_all, soe_mcs, soe_cev, mcs_node, mcs_transit,
                                   rem_dig, rem_load, hist,
                                   peak_nc, peak_op, pool.mu;
                                   require_site_visit = require_site_visit,
                                   single_visit_per_site = single_visit_per_site,
                                   time_limit_sec = time_limit_sec)
        day_solve_s = time() - day_t0
        stat = string(termination_status(model))
        has_values(model) || error("Approach 0 (one-shot, detailed): day $day's 8:00 whole-day MILP was INFEASIBLE ",
                                   "(status=$stat); there is no fixed plan to execute.")

        push!(solve_log, (day, stat, objective_value(model),
                          100 * (try relative_gap(model) catch; NaN end),
                          try solve_time(model) catch; day_solve_s end))

        # ---- FULL-DAY PLAN CAPTURE (the one and only resolve for this day) ----
        # CEV rows -- into the SAME DetailedPlanLog/log_plan_row! struct A1S/A2S use.
        for k in 1:nKd
            for e in d.E
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                site === nothing && continue
                activity_planned = MPCLoop.activity_label(model, d, e, site, k)
                planned_power_kW = value(model[:P_work][site, e, k])
                p_into = sum(value(model[:P_MCS_CEV][m, site, e, k]) for m in d.M)
                soe_cev_planned  = value(model[:SOE_CEV][e, k + 1])
                soe_mcs_planned  = value(model[:SOE_MCS][1, k + 1])
                log_plan_row!(plan_log_cev, d; day, resolve_step = 1, offset_step = k,
                              activity_planned, planned_power_kW,
                              planned_charging = p_into > 1e-6,
                              soe_cev_planned_kWh = soe_cev_planned,
                              soe_mcs_planned_kWh = soe_mcs_planned,
                              cev = e)
            end
            # MCS rows -- new log, one row per (day, offset_step, mcs unit).
            for m in d.M
                push!(mcs_plan_rows, _mcs_plan_row(model, d, day, k, m))
            end
        end

        for k0 in 1:nKd
            gidx = (day - 1) * nKd + k0

            for m in d.M; real_SOE_MCS[m, gidx] = soe_mcs[m]; end
            for e in d.E; real_SOE_CEV[e, gidx] = soe_cev[e]; end

            step = MPCLoop.apply_and_simulate!(model, k0, nKd, d, pool, cursor, rng, multi_activity,
                                       soe_mcs, soe_cev, mcs_node, mcs_transit, rem_dig, rem_load, hist,
                                       real_P_ch, real_P_dch, real_L_trv, real_loc, real_P_work;
                                       plant_mode = plant, gidx = gidx)
            n_obs_total += step.n_obs_added
            n_capped_total += step.n_capped

            for e in d.E
                site = findfirst(i -> d.A[i, e] == 1, d.N)
                site !== nothing && (real_cev_act[e][gidx] = MPCLoop.activity_label(model, d, e, site, k0))
            end
            real_mcs_act[gidx] = MPCLoop.mcs_status_label(model, d, k0)

            # ---- REALIZED CAPTURE -- CEV rows into RealizedTupleLog ----
            for e in d.E
                log_realized_row!(realized_log_cev, d; day, step = k0, cev = e,
                                  p_tuple = step.p_true[e], activity_executed = real_cev_act[e][gidx],
                                  soe_cev_kWh = safe_get(soe_cev, e), soe_mcs_kWh = soe_mcs[1],
                                  infeasible_flag = false)
            end
            # ---- REALIZED CAPTURE -- MCS rows into the new mcs_realized_rows log ----
            for m in d.M
                push!(mcs_realized_rows, _mcs_realized_row(d, day, k0, gidx, m,
                                                            real_P_ch, real_P_dch, real_loc, soe_mcs,
                                                            real_mcs_act[gidx]))
            end

            peak_nc = max(peak_nc, step.grid_kW)
            in_peak(k0, d.delta_T, d.t_start) && (peak_op = max(peak_op, step.grid_kW))

            push!(log, (day, k0, clock_label(d.t_start, d.delta_T, k0), d.lambda_whl_elec[k0], d.lambda_CO2[k0],
                        step.grid_kW, step.dch_kW, step.work_kW,
                        soe_mcs[1], safe_get(soe_cev, 1), safe_get(soe_cev, 2), step.cur_node,
                        pool.mu[1], pool.mu[2], pool.mu[3], pool.mu[4],
                        pool.sd[1], pool.sd[2], pool.sd[3], pool.sd[4], n_obs_total))
        end

        # ---- END-OF-DAY SNAPSHOT (for the per-day KPI table) ----
        # Cumulative-to-date figures -- backlog and terminal shortfall are, by
        # this model's own design (CHANGE 5), carried forward day to day, not
        # reset -- so "as of end of day X" is the only meaningful reading.
        (; shortfall_kWh, shortfall_hours, shortfall_penalty_cost) =
            MPCLoop._terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, day)
        push!(day_snapshot_rows, (; day,
                                     rem_dig_total_h = sum(rem_dig),
                                     rem_load_total_h = sum(rem_load),
                                     missed_work_cumulative_h = sum(rem_dig) + sum(rem_load),
                                     shortfall_kWh_cumulative = shortfall_kWh,
                                     shortfall_hours_cumulative = shortfall_hours,
                                     shortfall_penalty_cost_cumulative = shortfall_penalty_cost))
    end

    for m in d.M; real_SOE_MCS[m, n_kept + 1] = soe_mcs[m]; end
    for e in d.E; real_SOE_CEV[e, n_kept + 1] = soe_cev[e]; end

    elapsed = time() - t0
    @printf("Approach 0 one-shot DETAILED (plant = :%s) done in %.1f s (%d plant realizations, %d day(s))\n",
            plant, elapsed, n_obs_total, n_day_run)
    n_capped_total > 0 && @printf("  NOTE: %d intervals had work CAPPED by available CEV energy.\n", n_capped_total)

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
        MPCLoop._terminal_soe_shortfall(d, soe_cev, rem_dig, rem_load, n_day_run)

    res = (; d, time_labels, log,
              real_P_ch, real_P_dch, real_L_trv, real_SOE_MCS, real_SOE_CEV,
              real_P_work, real_loc, real_cev_act, real_mcs_act,
              nK = n_kept, nKd, n_day_run, ACT_NAME = MPCLoop.ACT_NAME,
              total_energy, total_cost, total_co2, nc_peak, op_peak, missed,
              labour_cost, transit_intervals,
              soe_cev_end = copy(soe_cev), soe_mcs_end = copy(soe_mcs),
              shortfall_kWh, shortfall_hours, shortfall_penalty_cost,
              n_obs_total, n_infeasible = 0, elapsed,
              approach = 0, plant, n_capped = n_capped_total)

    detailed_plan_df_cev     = _add_price_co2(to_dataframe(plan_log_cev),     d, :offset_step, nKd)
    detailed_realized_df_cev = _add_price_co2(to_dataframe(realized_log_cev), d, :step,        nKd)
    mcs_plan_df     = _add_price_co2(DataFrame(mcs_plan_rows),     d, :offset_step, nKd)
    mcs_realized_df = _add_price_co2(DataFrame(mcs_realized_rows), d, :step,        nKd)
    day_snapshots_df = DataFrame(day_snapshot_rows)

    return (; res, solve_log, day_snapshots_df,
              detailed_plan_df_cev, detailed_realized_df_cev,
              mcs_plan_df, mcs_realized_df)
end
