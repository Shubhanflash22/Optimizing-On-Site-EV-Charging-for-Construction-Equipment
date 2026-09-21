# #############################################################################
# 5_Output.jl  —  module Output   (Approach 0)
# -----------------------------------------------------------------------------
# Approach 0 has no replanning loop, so there is no replan-grid / plan-vs-actual
# figure set to draw the way Approach 1/2's own 5_Output.jl does. What IS worth
# writing to disk is exactly what a reviewer needs to audit a one-shot run:
#   - the full day's plan and what actually got realized, for the CEV(s) AND
#     the MCS (see 4_OneShot.jl's `run_one_shot` docstring for what each field
#     means)
#   - a full KPI table, the SAME 17 metrics and formulas
#     Comparison_A0_A1_A2/Code/8_ComparisonOutput.jl's `cost_components` uses,
#     so Approach 0 numbers are directly comparable to a 3-way comparison run
#   - for multi-day runs, the SAME table broken out Day1..DayN + Overall
# No plots are produced by this module. If you want figures, read the detailed
# CSVs into whatever plotting tool you like -- they carry everything a figure
# would need.
# #############################################################################
module Output

using DataFrames
using CSV
using Printf

using ..Common: in_peak

export write_detailed_output, write_kpi_summary, build_overall_kpi_table, build_daily_kpi_table,
       print_kpis

# =============================================================================
# DETAILED OUTPUT (CEV + MCS, plan + realized) -- always available on `res`
# when `run_one_shot(...; detailed_output = true)` was used; a no-op otherwise.
# =============================================================================
function write_detailed_output(res, out_dir::AbstractString)
    mkpath(out_dir)
    if res.detailed_plan_df !== nothing
        CSV.write(joinpath(out_dir, "A0_plan_full.csv"), res.detailed_plan_df)
    end
    if res.detailed_realized_df !== nothing
        CSV.write(joinpath(out_dir, "A0_realized_tuple.csv"), res.detailed_realized_df)
    end
    if res.detailed_mcs_plan_df !== nothing
        CSV.write(joinpath(out_dir, "A0_MCS_plan_full.csv"), res.detailed_mcs_plan_df)
    end
    if res.detailed_mcs_realized_df !== nothing
        CSV.write(joinpath(out_dir, "A0_MCS_realized_tuple.csv"), res.detailed_mcs_realized_df)
    end
    return out_dir
end

# =============================================================================
# KPI TABLE  -- same metric list + formulas as
# Comparison_A0_A1_A2/Code/8_ComparisonOutput.jl's `cost_components`, kept
# duplicated here (rather than depended on) so this module has no dependency
# on the Comparison driver and can be used completely standalone.
# =============================================================================
const KPI_METRIC_NAMES = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
    "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
    "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh",
    "Total_CO2_Emissions_kg", "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour",
    "MCS_Transit_hour", "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "Solve_time_s"]

function _cost_components(d; total_cost, total_co2, nc_peak, op_peak, missed,
                           labour_cost, shortfall_penalty_cost)
    energy_cost = total_cost
    carbon_cost = (d.carbon_price_per_ton / 1000.0) * total_co2
    ncd_cost    = d.lambda_demand_NC * nc_peak
    opd_cost    = d.lambda_demand_OP * op_peak
    missed_cost = d.rho_miss * missed
    travel_cost = labour_cost
    total = energy_cost + carbon_cost + ncd_cost + opd_cost + missed_cost + travel_cost + shortfall_penalty_cost
    return (; energy_cost, carbon_cost, ncd_cost, opd_cost, missed_cost, travel_cost,
              shortfall_cost = shortfall_penalty_cost, total)
end

function _kpi_column_values(d; total_cost, total_co2, nc_peak, op_peak, missed, labour_cost,
                             shortfall_penalty_cost, total_energy, transit_hours,
                             shortfall_kWh, n_infeasible, solve_time_s)
    c = _cost_components(d; total_cost, total_co2, nc_peak, op_peak, missed,
                          labour_cost, shortfall_penalty_cost)
    return Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
               round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
               round(c.travel_cost, digits = 2), round(c.shortfall_cost, digits = 2),
               round(total_energy, digits = 2), round(total_co2, digits = 2),
               round(nc_peak, digits = 2), round(op_peak, digits = 2), round(missed, digits = 2),
               round(transit_hours, digits = 2), round(shortfall_kWh, digits = 3),
               n_infeasible, round(solve_time_s, digits = 2)]
end

# ---- OVERALL table (one column, "A0") -- every run gets this one. ----
function build_overall_kpi_table(res, d)
    vals = _kpi_column_values(d; total_cost = res.total_cost, total_co2 = res.total_co2,
        nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
        labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
        total_energy = res.total_energy, transit_hours = res.transit_intervals * d.delta_T,
        shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible, solve_time_s = res.elapsed)
    return DataFrame(Metric = KPI_METRIC_NAMES, A0 = vals)
end

# ---- PER-DAY table (Day1..DayN + Overall columns) -- only meaningful for
# n_day_run > 1 runs. Cost/energy/CO2/peaks are THAT day's own contribution;
# Missed_Work_hour and Terminal_SOE_Shortfall_kWh are cumulative-as-of-that-day
# figures, because the simulator carries backlog and terminal shortfall
# forward day to day by design rather than resetting them -- "as of end of
# day X" is the only physically meaningful reading for those two rows.
# `day_snapshots_df` must have one row per day with columns:
#   day, missed_work_cumulative_h, shortfall_kWh_cumulative,
#   shortfall_penalty_cost_cumulative
# and `solve_log` one row per day with columns: day, solve_time_s.
# See 4_OneShot.jl's `run_one_shot` for where both are built.
# =============================================================================
function build_daily_kpi_table(res, day_snapshots_df, solve_log, d)
    days = sort(unique(res.log.day))
    cols = Dict{String, Vector{Any}}()
    for day in days
        daymask = res.log.day .== day
        daylog  = res.log[daymask, :]
        total_energy_day = sum(daylog.grid_kW) * d.delta_T
        total_cost_day   = sum(daylog.grid_kW .* daylog.price) * d.delta_T
        total_co2_day    = sum(daylog.grid_kW .* daylog.co2)  * d.delta_T
        nc_peak_day      = isempty(daylog.grid_kW) ? 0.0 : maximum(daylog.grid_kW)
        op_mask_day      = [in_peak(k, d.delta_T, d.t_start) for k in daylog.k]
        op_peak_day      = any(op_mask_day) ? maximum(daylog.grid_kW[op_mask_day]) : 0.0

        snap_row  = day_snapshots_df[day_snapshots_df.day .== day, :][1, :]
        solve_row = solve_log[solve_log.day .== day, :][1, :]

        vals = _kpi_column_values(d; total_cost = total_cost_day, total_co2 = total_co2_day,
            nc_peak = nc_peak_day, op_peak = op_peak_day, missed = snap_row.missed_work_cumulative_h,
            labour_cost = snap_row.labour_cost_day,
            shortfall_penalty_cost = snap_row.shortfall_penalty_cost_cumulative,
            total_energy = total_energy_day, transit_hours = snap_row.transit_hours_day,
            shortfall_kWh = snap_row.shortfall_kWh_cumulative, n_infeasible = 0,
            solve_time_s = solve_row.solve_time_s)
        cols["Day$(day)"] = vals
    end
    overall_vals = _kpi_column_values(d; total_cost = res.total_cost, total_co2 = res.total_co2,
        nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
        labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
        total_energy = res.total_energy, transit_hours = res.transit_intervals * d.delta_T,
        shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible, solve_time_s = res.elapsed)

    df = DataFrame(Metric = KPI_METRIC_NAMES)
    for day in days
        df[!, "Day$(day)"] = cols["Day$(day)"]
    end
    df[!, "Overall"] = overall_vals
    return df
end

# Writes A0_kpi_summary.csv always, and A0_kpi_summary_by_day.csv additionally
# whenever `day_snapshots_df`/`solve_log` are supplied (multi-day runs).
function write_kpi_summary(res, d, out_dir::AbstractString;
                            day_snapshots_df = nothing, solve_log = nothing)
    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "A0_kpi_summary.csv"), build_overall_kpi_table(res, d))
    if res.n_day_run > 1 && day_snapshots_df !== nothing && solve_log !== nothing
        CSV.write(joinpath(out_dir, "A0_kpi_summary_by_day.csv"),
                  build_daily_kpi_table(res, day_snapshots_df, solve_log, d))
    end
    return out_dir
end

# Human-readable console KPI block (mirrors Approach 1/2's own `_print_kpis`).
function print_kpis(res)
    d = res.d
    println("\n==== Approach 0 (one-shot) KPIs — $(res.n_day_run) day(s), $(res.nK) intervals ====")
    @printf("Total grid energy   : %.2f kWh\n", res.total_energy)
    @printf("Total energy cost   : \$%.2f\n", res.total_cost)
    res.total_co2 > 1e-9 && @printf("Total CO2 emissions : %.2f kg\n", res.total_co2)
    @printf("NC peak demand      : %.2f kW\n", res.nc_peak)
    @printf("On-peak demand      : %.2f kW\n", res.op_peak)
    @printf("Missed work (hours) : %.2f\n", res.missed)
    @printf("Labour (towing)     : \$%.2f  (%.2f h in transit)\n",
            res.labour_cost, res.transit_intervals * d.delta_T)
    @printf("CEV SOE at horizon  : %s kWh (target %s)\n",
            string(round.(res.soe_cev_end, digits = 2)), string(round.(d.SOE_CEV_ini, digits = 2)))
    @printf("MCS SOE at horizon  : %s kWh (target %s)\n",
            string(round.(res.soe_mcs_end, digits = 2)), string(round.(d.SOE_MCS_ini, digits = 2)))
end

end # module Output
