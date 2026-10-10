# #############################################################################
# Output.jl  -  module Output
# -----------------------------------------------------------------------------
# Turns the NamedTuple returned by run_one_shot (4_OneShot.jl) into the files
# and console summary a person actually reads: the detailed CSVs, the KPI
# summary table(s), and a short printed report. Computes nothing new about the
# run itself -- every number here is read straight from res, d, res.log,
# res.day_snapshots_df, or res.solve_log, then formatted, summed, or reshaped
# into a table. Four groups:
#
#   1. DETAILED CSV EXPORT
#      write_detailed_output
#      -- writes the four plan/realized CSVs (A0_plan_full, A0_realized_tuple,
#      A0_MCS_plan_full, A0_MCS_realized_tuple) built in Common.jl's logging
#      structs, when detailed_output was turned on for the run. Writes nothing
#      for a log that's nothing (detailed_output = false).
#
#   2. KPI TABLE ASSEMBLY
#      KPI_METRIC_NAMES, _cost_components, _kpi_column_values,
#      build_overall_kpi_table, build_daily_kpi_table
#      -- KPI_METRIC_NAMES is the fixed, ordered list of 17 metric names every
#      KPI table uses as its row labels (this must stay in sync with whatever
#      script compares Approach 0 against Approaches 1 and 2).
#      _cost_components re-adds up the paper's 6 objective-function (4) cost
#      terms (energy, carbon, NC demand, OP demand, missed work, travel) plus
#      the one approved addition (terminal SOE shortfall penalty) into a total.
#      _kpi_column_values packages that total plus the remaining KPI rows into
#      one ordered vector matching KPI_METRIC_NAMES.
#      build_overall_kpi_table builds the single-column whole-run KPI table;
#      build_daily_kpi_table builds the same KPIs recomputed per day (from
#      res.log's per-day rows and res.day_snapshots_df's cumulative snapshots)
#      alongside an Overall column for the whole run.
#
#   3. FILE WRITING
#      write_kpi_summary
#      -- writes the whole-run KPI CSV, the full solver log (A0_solve_log,
#      including the achieved MIP gap per day), and the plain per-interval log
#      (A0_interval_log) on every run; writes the by-day KPI CSV only when the
#      run covered more than one day.
#
#   4. CONSOLE SUMMARY
#      print_kpis
#      -- prints a short human-readable summary of the run's energy, cost,
#      CO2, demand peaks, missed work, MCS transit/labour, end-of-run SOE for
#      every CEV and MCS, and the terminal SOE shortfall penalty.
# #############################################################################
module Output

# external packages used across this file
using DataFrames
using CSV
using Printf

using ..Common: in_peak

# everything below that other files are allowed to use
export write_detailed_output, write_kpi_summary, build_overall_kpi_table, build_daily_kpi_table,
       print_kpis

# Writes the four detailed CSVs (plan and realized, for both CEVs and MCSs) built by Common.jl's logging structs, one file per non-nothing DataFrame in res.
# Writes nothing for whichever ones are nothing, which happens when the run had detailed_output = false.
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

# Fixed, ordered list of the 17 KPI row names every KPI table (overall and by-day) uses.
# The order here must match the order _kpi_column_values returns its values in, and both must match whatever script compares Approach 0's output against Approaches 1 and 2 by name.
const KPI_METRIC_NAMES = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
    "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
    "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh",
    "Total_CO2_Emissions_kg", "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour",
    "MCS_Transit_hour", "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "Solve_time_s"]

# Recomputes the paper's 6 objective function (4) cost terms from already-aggregated inputs (total_cost is actually just the grid energy cost), and adds the one approved non-paper term, the terminal SOE shortfall penalty.
# energy_cost/carbon_cost/ncd_cost/opd_cost/missed_cost/travel_cost correspond in order to objective (4)'s 6 terms: electricity cost, carbon cost, NC demand charge, OP demand charge, missed-work penalty, and MCS travel labour.
# carbon_price_per_ton is divided by 1000 to convert $/ton into $/kg, matching total_co2 being in kg.
function _cost_components(d; total_cost, total_co2, nc_peak, op_peak, missed,
                           labour_cost, shortfall_penalty_cost)
    energy_cost = total_cost
    carbon_cost = (d.carbon_price_per_ton / 1000.0) * total_co2
    ncd_cost    = d.lambda_demand_NC * nc_peak
    opd_cost    = d.lambda_demand_OP * op_peak
    missed_cost = d.rho_miss * missed
    travel_cost = labour_cost
    total = energy_cost + carbon_cost + ncd_cost + opd_cost + missed_cost + travel_cost + shortfall_penalty_cost
    return (; energy_cost, carbon_cost, ncd_cost, opd_cost, missed_cost, travel_cost, shortfall_cost = shortfall_penalty_cost, total)
end

# Packages one full KPI column: recomputes the 6 objective-function (4) cost terms plus the shortfall penalty via _cost_components, then returns all 17 values in the exact order KPI_METRIC_NAMES lists them.
# Costs are rounded to 2 decimals, the SOE shortfall to 3 (it's a small kWh quantity where 2 decimals would lose precision), and n_infeasible is left as a plain integer.
function _kpi_column_values(d; total_cost, total_co2, nc_peak, op_peak, missed, labour_cost,
                             shortfall_penalty_cost, total_energy, transit_hours,
                             shortfall_kWh, n_infeasible, solve_time_s)
    c = _cost_components(d; total_cost, total_co2, nc_peak, op_peak, missed, labour_cost, shortfall_penalty_cost)
    return Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
               round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
               round(c.travel_cost, digits = 2), round(c.shortfall_cost, digits = 2),
               round(total_energy, digits = 2), round(total_co2, digits = 2),
               round(nc_peak, digits = 2), round(op_peak, digits = 2), round(missed, digits = 2),
               round(transit_hours, digits = 2), round(shortfall_kWh, digits = 3),
               n_infeasible, round(solve_time_s, digits = 2)]
end

# Builds the single-column, whole-run KPI table: pulls every input straight from res, converts transit_intervals into hours (transit_intervals is a raw interval count summed over every MCS, from the real_loc-based fix), and uses the true total solve time (summed across every day) rather than wall-clock elapsed time.
function build_overall_kpi_table(res, d)
    vals = _kpi_column_values(d; total_cost = res.total_cost, total_co2 = res.total_co2,
        nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
        labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
        total_energy = res.total_energy, transit_hours = res.transit_intervals * d.delta_T,
        shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible, solve_time_s = sum(res.solve_log.solve_time_s))
    return DataFrame(Metric = KPI_METRIC_NAMES, A0 = vals)
end

# Builds the by-day KPI table: one column per day, plus an Overall column for the whole run.
# Energy, cost (excluding missed-work/shortfall), CO2, and the two demand peaks are recomputed fresh from daylog = res.log filtered to that day, so they are genuinely that day's own numbers.
# missed, shortfall_kWh, and shortfall_penalty_cost are day-over-day deltas: day_snapshots_df's *_cumulative fields are running totals as of that day (rem_dig/rem_load and the shortfall accumulate across days in 4_OneShot.jl), so each day's value here is that day's cumulative figure minus the previous day's, giving the change that happened on that specific day rather than the running total.
# A delta can come out negative for missed work if a day's CEVs pay down more of the earlier backlog than they add that day; the shortfall penalty can come out negative if a day's CEVs pay down more of the earlier shortfall than they add that day. Both are valid and expected.
# Because every column is now genuinely per-day, each Day N's Total_Cost_USD is the cost incurred that day alone (except for the two demand peaks, which are each day's own peak and are not additive across days the way the paper's monthly NC/OP charges would be).
# transit_hours/labour_cost were already genuinely per-day (from that day's own transit_hours_day/labour_cost_day) and are unchanged here.
# n_infeasible is hardcoded to 0 for every day, consistent with run_one_shot erroring out immediately on the first infeasible day rather than continuing.
function build_daily_kpi_table(res, day_snapshots_df, solve_log, d)
    days = sort(unique(res.log.day))
    cols = Dict{String, Vector{Any}}()
    prev_missed           = 0.0
    prev_shortfall_kWh    = 0.0
    prev_shortfall_cost   = 0.0
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

        missed_day         = snap_row.missed_work_cumulative_h - prev_missed
        shortfall_kWh_day  = snap_row.shortfall_kWh_cumulative - prev_shortfall_kWh
        shortfall_cost_day = snap_row.shortfall_penalty_cost_cumulative - prev_shortfall_cost

        vals = _kpi_column_values(d; total_cost = total_cost_day, total_co2 = total_co2_day,
            nc_peak = nc_peak_day, op_peak = op_peak_day, missed = missed_day,
            labour_cost = snap_row.labour_cost_day,
            shortfall_penalty_cost = shortfall_cost_day,
            total_energy = total_energy_day, transit_hours = snap_row.transit_hours_day,
            shortfall_kWh = shortfall_kWh_day, n_infeasible = 0,
            solve_time_s = solve_row.solve_time_s)
        cols["Day$(day)"] = vals

        prev_missed         = snap_row.missed_work_cumulative_h
        prev_shortfall_kWh  = snap_row.shortfall_kWh_cumulative
        prev_shortfall_cost = snap_row.shortfall_penalty_cost_cumulative
    end
    # Overall column uses the same whole-run values as build_overall_kpi_table, so the two are always consistent with each other.
    overall_vals = _kpi_column_values(d; total_cost = res.total_cost, total_co2 = res.total_co2,
        nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
        labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
        total_energy = res.total_energy, transit_hours = res.transit_intervals * d.delta_T,
        shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible, solve_time_s = sum(res.solve_log.solve_time_s))

    df = DataFrame(Metric = KPI_METRIC_NAMES)
    for day in days
        df[!, "Day$(day)"] = cols["Day$(day)"]
    end
    df[!, "Overall"] = overall_vals
    return df
end

# Writes the whole-run KPI CSV, the full per-day solver log (including each day's achieved MIP gap), and the plain per-interval log, unconditionally on every run.
# Writes the by-day KPI CSV only when the run covered more than one day, since a single-day run has nothing meaningful to break out by day beyond what the overall table already shows.
function write_kpi_summary(res, d, out_dir::AbstractString;
                            day_snapshots_df = nothing, solve_log = nothing)
    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "A0_kpi_summary.csv"), build_overall_kpi_table(res, d))
    CSV.write(joinpath(out_dir, "A0_solve_log.csv"), res.solve_log)
    CSV.write(joinpath(out_dir, "A0_interval_log.csv"), res.log)
    if res.n_day_run > 1 && day_snapshots_df !== nothing && solve_log !== nothing
        CSV.write(joinpath(out_dir, "A0_kpi_summary_by_day.csv"), build_daily_kpi_table(res, day_snapshots_df, solve_log, d))
    end
    return out_dir
end

# Prints a short human-readable summary of the whole run: grid energy, energy cost, CO2 (skipped if effectively zero), the two demand peaks, missed work, MCS transit time and labour cost, every CEV's and MCS's end-of-run SOE against its target, and the terminal shortfall penalty.
function print_kpis(res)
    d = res.d
    println("\n==== Approach 0 (one-shot) KPIs — $(res.n_day_run) day(s), $(res.nK) intervals ====")
    @printf("Total grid energy   : %.2f kWh\n", res.total_energy)
    @printf("Total energy cost   : \$%.2f\n", res.total_cost) 
    res.total_co2 > 1e-9 && @printf("Total CO2 emissions : %.2f kg\n", res.total_co2)
    @printf("NC peak demand      : %.2f kW\n", res.nc_peak)
    @printf("On-peak demand      : %.2f kW\n", res.op_peak)
    @printf("Missed work (hours) : %.2f\n", res.missed)
    @printf("Labour (towing)     : \$%.2f  (%.2f h in transit)\n", res.labour_cost, res.transit_intervals * d.delta_T)
    @printf("CEV SOE at horizon  : %s kWh (target %s)\n", string(round.(res.soe_cev_end, digits = 2)), string(round.(d.SOE_CEV_ini, digits = 2)))
    @printf("MCS SOE at horizon  : %s kWh (target %s)\n", string(round.(res.soe_mcs_end, digits = 2)), string(round.(d.SOE_MCS_ini, digits = 2)))
    @printf("Terminal SOE shortfall penalty : \$%.2f\n", res.shortfall_penalty_cost)
end

end 