# #############################################################################
# Reporting.jl  —  module Reporting
# -----------------------------------------------------------------------------
# Owns the TABULAR / KPI outputs (the CSV + HTML companions of the figures), so
# the file set matches the v4_real reference plus the MPC-specific artefacts:
#
#   08_cost_emissions_timeseries.csv / 08_cost_emissions_summary.png
#   09_cost_kpi_metrics.csv          / 09_kpi_metrics_summary.png
#   10_mip_convergence.csv           (per-window solver diagnostics)
#   closed_loop_trajectory.csv       (detailed analyst log)
#   overnight_mcs_charge.csv         (Phase-2 overnight schedule)
#   worker_schedule.csv              (plain-words site instructions)
#   replan_grids/*.csv + *.html      (per-step forward plans + replanning view)
#
# Cost/KPI breakdowns are derived from the realized trajectory using the same
# component definitions as MCS_OPTIMAL_v4_real.jl (energy, carbon, NCD, OPD,
# missed work, travel/labour, total).
# #############################################################################
module Reporting

using DataFrames
using CSV
using Plots
using Printf

using ..Common: clock_label, in_peak

export write_reports

gr()

# -----------------------------------------------------------------------------
# 08 — per-interval grid energy / cost / CO2 with running cumulatives.
# -----------------------------------------------------------------------------
function _cost_emissions_timeseries(res)
    d = res.d; log = res.log
    e_kwh = log.grid_kW .* d.delta_T
    cost  = e_kwh .* log.price
    co2   = e_kwh .* log.co2
    return DataFrame(
        Time_Period = log.k,
        Time_Label  = log.clock,
        Grid_Energy_kWh = e_kwh,
        Energy_Cost_USD = cost,
        CO2_Emissions_kg = co2,
        Cumulative_Energy_Cost_USD = cumsum(cost),
        Cumulative_CO2_Emissions_kg = cumsum(co2))
end

# The six objective cost components (same definitions as the reference).
function _cost_components(res)
    d = res.d
    energy_cost = res.total_cost
    carbon_cost = (d.carbon_price_per_ton / 1000.0) * res.total_co2
    ncd_cost    = d.lambda_demand_NC * res.nc_peak
    opd_cost    = d.lambda_demand_OP * res.op_peak
    missed_cost = d.rho_miss * res.missed
    travel_cost = res.labour_cost
    total       = energy_cost + carbon_cost + ncd_cost + opd_cost + missed_cost + travel_cost
    return (; energy_cost, carbon_cost, ncd_cost, opd_cost, missed_cost, travel_cost, total)
end

function _write_cost_emissions(res, out_dir)
    ts = _cost_emissions_timeseries(res)
    CSV.write(joinpath(out_dir, "08_cost_emissions_timeseries.csv"), ts)

    # 08 figure: cumulative cost (left) + cumulative CO2 (right).
    nK = nrow(ts); x = 1:nK
    p = plot(x, ts.Cumulative_Energy_Cost_USD, title = "", xlabel = "Interval",
             ylabel = "Cumulative Cost (USD)", label = nothing, color = :blue, linewidth = 2,
             size = (900, 500), guidefontsize = 16, tickfontsize = 14,
             bottom_margin = 14Plots.mm, left_margin = 16Plots.mm, right_margin = 14Plots.mm,
             legend = :topleft)
    if any(ts.Cumulative_CO2_Emissions_kg .> 0)
        pt = twinx(p)
        plot!(pt, x, ts.Cumulative_CO2_Emissions_kg, ylabel = "Cumulative CO₂ (kg)", label = nothing,
              color = :red, linewidth = 2, guidefontsize = 16, tickfontsize = 14)
        plot!(p, [NaN], [NaN], color = :red, linewidth = 2, label = "Cumulative CO₂ (kg)")
    end
    plot!(p, [NaN], [NaN], color = :blue, linewidth = 2, label = "Cumulative Cost (USD)")
    savefig(p, joinpath(out_dir, "08_cost_emissions_summary.png"))
end

# -----------------------------------------------------------------------------
# 09 — KPI totals CSV + a two-panel bar summary (costs + operations).
# -----------------------------------------------------------------------------
function _write_kpi_metrics(res, out_dir)
    c = _cost_components(res)
    totals = DataFrame(
        Metric = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
                  "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
                  "Travel_Labour_USD", "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg",
                  "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour", "MCS_Transit_hour",
                  "Overnight_Recharge_kWh", "Overnight_Cost_USD", "Infeasible_windows", "MPC_loop_time_s"],
        Value = Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
                    round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
                    round(c.travel_cost, digits = 2), round(res.total_energy, digits = 2), round(res.total_co2, digits = 2),
                    round(res.nc_peak, digits = 2), round(res.op_peak, digits = 2), round(res.missed, digits = 2),
                    round(res.transit_intervals * res.d.delta_T, digits = 2),
                    round(res.overnight_energy, digits = 2), round(res.overnight_cost, digits = 2),
                    res.n_infeasible, round(res.elapsed, digits = 2)])
    CSV.write(joinpath(out_dir, "09_cost_kpi_metrics.csv"), totals)

    cost_labels = ["Energy", "CO₂", "NCD", "OPD", "Missed Work", "Travel", "Total"]
    cost_values = [c.energy_cost, c.carbon_cost, c.ncd_cost, c.opd_cost, c.missed_cost, c.travel_cost, c.total]
    cost_colors = [:steelblue, :forestgreen, :darkorange, :purple, :firebrick, :teal, :black]
    cost_ymax = max(maximum(cost_values), 1.0)
    p_costs = plot(title = "", xlabel = "(a)", ylabel = "Cost (USD)",
                   xticks = (1:length(cost_labels), cost_labels), xlims = (0.5, length(cost_labels) + 0.5),
                   ylims = (0, 1.35 * cost_ymax), legend = false, xrotation = 25,
                   guidefontsize = 16, tickfontsize = 14, size = (1100, 450),
                   bottom_margin = 14Plots.mm, left_margin = 14Plots.mm, right_margin = 12Plots.mm)
    for i in eachindex(cost_labels)
        bar!(p_costs, [i], [cost_values[i]], color = cost_colors[i], label = false, bar_width = 0.65)
        annotate!(p_costs, i, cost_values[i] + 0.10 * cost_ymax,
                  text(@sprintf("\$%.1f", cost_values[i]), :black, 12, :center))
    end

    peak_ymax = max(res.nc_peak, res.op_peak, 1.0)
    p_ops = plot(title = "", xlabel = "(b)", ylabel = "Demand Peak (kW)",
                 xticks = (1:2, ["NCD Peak", "OPD Peak"]), xlims = (0.5, 2.5),
                 ylims = (0, 1.2 * peak_ymax), legend = false,
                 guidefontsize = 16, tickfontsize = 14, size = (1100, 450),
                 bottom_margin = 14Plots.mm, left_margin = 14Plots.mm, right_margin = 14Plots.mm)
    bar!(p_ops, [1], [res.nc_peak], color = :darkorange, label = false, bar_width = 0.55)
    bar!(p_ops, [2], [res.op_peak], color = :purple, label = false, bar_width = 0.55)

    p_summary = plot(p_costs, p_ops, layout = (2, 1), size = (1200, 900), plot_title = "KPI Metrics Summary")
    savefig(p_summary, joinpath(out_dir, "09_kpi_metrics_summary.png"))
end

# -----------------------------------------------------------------------------
# 10 — per-window solver diagnostics (the MPC analogue of a single MIP log).
# -----------------------------------------------------------------------------
_write_mip_convergence(res, out_dir) =
    CSV.write(joinpath(out_dir, "10_mip_convergence.csv"), res.solve_log)

# -----------------------------------------------------------------------------
# Worker-facing schedule (plain words) + detailed trajectory + overnight table.
# -----------------------------------------------------------------------------
function _write_schedules(res, out_dir)
    d = res.d
    CSV.write(joinpath(out_dir, "closed_loop_trajectory.csv"), res.log)
    CSV.write(joinpath(out_dir, "overnight_mcs_charge.csv"), res.ov_df)
    fe = DataFrame(time = res.fe_time)
    for e in d.E
        fe[!, Symbol("CEV$(e)_activity")]       = res.fe_act[e]
        fe[!, Symbol("CEV$(e)_plug_in_charge")] = res.fe_chg[e]
    end
    fe[!, :MCS_charge_from_grid] = res.fe_mcs
    CSV.write(joinpath(out_dir, "worker_schedule.csv"), fe)
end

# ---- replanning-grid cell formatting + CSV/HTML writers ---------------------
_cell(v::AbstractString) = v
_cell(v::Real) = isnan(v) ? "" : round(v, digits = 3)

function _write_replan_grid(path, mat, res)
    d = res.d; nK = res.nK
    df = DataFrame(replan_at = [clock_label(d.t_start, d.delta_T, k0) for k0 in 1:nK])
    for k in 1:nK
        df[!, Symbol(clock_label(d.t_start, d.delta_T, k))] =
            Any[_cell(k < k0 ? mat[k, k] : mat[k0, k]) for k0 in 1:nK]
    end
    CSV.write(path, df)
    _write_replan_grid_html(replace(path, r"\.csv$" => ".html"), mat, res)
end

function _write_replan_grid_html(path, mat, res)
    d = res.d; nK = res.nK
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif}")
    println(io, "table{border-collapse:collapse;font-size:11px}")
    println(io, "th,td{border:1px solid #ccc;padding:2px 6px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".done{background:#c6efce}")
    println(io, ".pend{background:#ffeb9c}")
    println(io, "</style></head><body>")
    println(io, "<p><b>How to read this grid.</b> Every cell is a <i>PLANNED</i> value.<br>",
                "&nbsp;&nbsp;\u2022 <b>Each ROW</b> = one 15-min re-plan step (labelled by the clock time the plan was made at).<br>",
                "&nbsp;&nbsp;\u2022 <b>Each COLUMN</b> = the interval being planned for (labelled by its clock time).<br>",
                "&nbsp;&nbsp;\u2022 The <b>diagonal</b> (row time == column time) is the decision applied to the plant that step.</p>")
    println(io, "<p><b>Colour:</b> <span class=\"done\">&nbsp;&nbsp;&nbsp;</span> complete (past, fixed) &nbsp;&nbsp; ",
                "<span class=\"pend\">&nbsp;&nbsp;&nbsp;</span> pending (current step + forward plan)</p>")
    println(io, "<table><tr><th>re-plan made at &darr; &nbsp;\\&nbsp; interval &rarr;</th>")
    for k in 1:nK
        print(io, "<th>", clock_label(d.t_start, d.delta_T, k), "</th>")
    end
    println(io, "</tr>")
    for k0 in 1:nK
        print(io, "<tr><th>", clock_label(d.t_start, d.delta_T, k0), "</th>")
        for k in 1:nK
            cell = _cell(k < k0 ? mat[k, k] : mat[k0, k])
            cls  = cell == "" ? "" : (k < k0 ? "done" : "pend")
            print(io, "<td class=\"", cls, "\">", cell, "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

function _write_replan_grids(res, out_dir)
    d = res.d
    grid_dir = joinpath(out_dir, "replan_grids"); mkpath(grid_dir)
    _write_replan_grid(joinpath(grid_dir, "plan_grid_kW.csv"), res.plan_grid_kW, res)
    _write_replan_grid(joinpath(grid_dir, "plan_mcs_soe.csv"), res.plan_mcs_soe, res)
    for e in d.E
        _write_replan_grid(joinpath(grid_dir, "plan_cev$(e)_soe.csv"),      res.plan_cev_soe[e], res)
        _write_replan_grid(joinpath(grid_dir, "plan_cev$(e)_activity.csv"), res.plan_cev_act[e], res)
    end
end

# =============================================================================
# Write every tabular / KPI report into out_dir.
# =============================================================================
function write_reports(res, out_dir)
    mkpath(out_dir)
    _write_cost_emissions(res, out_dir)
    _write_kpi_metrics(res, out_dir)
    _write_mip_convergence(res, out_dir)
    _write_schedules(res, out_dir)
    _write_replan_grids(res, out_dir)
    return nothing
end

end # module Reporting
