# #############################################################################
# kpi_tables.jl  —  included inside `module A0App`
# -----------------------------------------------------------------------------
# The A0-only equivalent of Comparison_A0_A1_A2/Code/8_ComparisonOutput.jl's
# `write_kpi_csv` / `cost_components` -- SAME metric list, SAME cost-component
# formulas (copied verbatim from that file so the numbers are directly
# comparable to the existing 08_cost_kpi_metrics.csv), just with a single
# "A0" column instead of one column per approach, and (new) a per-day
# breakdown for multi-day runs.
# #############################################################################

const _KPI_METRIC_NAMES = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
    "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
    "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh",
    "Total_CO2_Emissions_kg", "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour",
    "MCS_Transit_hour", "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "Solve_time_s"]

# Same formulas as 8_ComparisonOutput.jl's `cost_components(res)`, just taking
# the raw numbers as keyword args instead of a full `res` struct, so this
# works identically for a whole-run total AND for a single day's slice.
function _cost_components_a0(d; total_cost, total_co2, nc_peak, op_peak, missed,
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
    c = _cost_components_a0(d; total_cost, total_co2, nc_peak, op_peak, missed,
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
    return DataFrame(Metric = _KPI_METRIC_NAMES, A0 = vals)
end

# ---- PER-DAY table (Day1..DayN + Overall columns) -- only written for
# n_day_run > 1 runs (the 5-day block). Cost/energy/CO2/peaks are THAT day's
# own contribution; Missed_Work_hour and Terminal_SOE_Shortfall_kWh are
# cumulative-as-of-that-day figures, because the underlying simulator carries
# backlog and terminal shortfall forward day to day by design (see
# run_one_shot_detailed.jl's end-of-day snapshot) rather than resetting them
# -- so "as of end of day X" is the only physically meaningful reading. ----
function build_daily_kpi_table(resd, d)
    res = resd.res
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
        transit_day      = count(==(0), daylog.mcs_node)
        labour_cost_day  = d.rho_labor * d.delta_T * transit_day

        snap_row  = resd.day_snapshots_df[resd.day_snapshots_df.day .== day, :][1, :]
        solve_row = resd.solve_log[resd.solve_log.day .== day, :][1, :]

        vals = _kpi_column_values(d; total_cost = total_cost_day, total_co2 = total_co2_day,
            nc_peak = nc_peak_day, op_peak = op_peak_day,
            missed = snap_row.missed_work_cumulative_h, labour_cost = labour_cost_day,
            shortfall_penalty_cost = snap_row.shortfall_penalty_cost_cumulative,
            total_energy = total_energy_day, transit_hours = transit_day * d.delta_T,
            shortfall_kWh = snap_row.shortfall_kWh_cumulative, n_infeasible = 0,
            solve_time_s = solve_row.solve_time_s)
        cols["Day$(day)"] = vals
    end
    overall_vals = _kpi_column_values(d; total_cost = res.total_cost, total_co2 = res.total_co2,
        nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
        labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
        total_energy = res.total_energy, transit_hours = res.transit_intervals * d.delta_T,
        shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible, solve_time_s = res.elapsed)

    df = DataFrame(Metric = _KPI_METRIC_NAMES)
    for day in days
        df[!, "Day$(day)"] = cols["Day$(day)"]
    end
    df[!, "Overall"] = overall_vals
    return df
end
