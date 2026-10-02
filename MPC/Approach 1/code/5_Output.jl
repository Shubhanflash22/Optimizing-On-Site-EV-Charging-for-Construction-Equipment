# #############################################################################
# 5_Output.jl  -  module Output
# -----------------------------------------------------------------------------
# Writes the figures, CSV tables, HTML reports and cost KPIs of one run_mpc result into the output folder, and every file name starts with A1_.
# It only reads the result of run_mpc and never changes it.
# The cost KPIs price the terms of objective function (4) of the paper's MILP formulation, and add the terminal SOE shortfall penalty outside it.
# Seven groups:
#
#   1. SHARED SETUP AND PLOT STYLE
#      COLORS, _base_plot
#      -- the plotting backend, the unit colors and the common plot settings.
#
#   2. TRAJECTORY FIGURES AND CSVs
#      fig_total_grid_power, fig_work_by_site, _fig_soe, fig_mcs_soe, fig_cev_soe, fig_price_emission, fig_location, figs_individual_mcs, csv_mcs_cev_soe, write_trajectory_figures
#      -- the realized grid power, work power, SOE, price, MCS location and per-MCS power figures with their CSVs, and the combined summary figure.
#
#   3. COST KPIs
#      _cost_components, _run_quantities, _day_quantities, KPI_METRIC_NAMES, _kpi_column_values, _write_kpi_metrics
#      -- price a whole run or a single day into its cost components, and write the KPI table, the per-day KPI table and the cost and peak bar charts.
#
#   4. RE-PLAN GRIDS
#      _cell, _write_replan_grid, _write_replan_grid_html, _write_replan_grids
#      -- write one grid per planned quantity with one row per re-solve and one column per interval, as CSV and HTML.
#
#   5. PLANNED VERSUS REALIZED KPIs
#      _first_plan_row, _planned_kpis, _overall_planned_kpis, _plan_vs_actual_rows, _plan_vs_actual_tables, _write_plan_vs_actual_chart, _write_plan_vs_actual_day, _write_plan_vs_actual_overall, _write_plan_vs_actual_all, _write_plan_vs_actual_html
#      -- price one day's first plan, or the Overall plan across every day, and compare it with the matching realized values.
#      For a run of several days, the Overall comparison is written at the top level and each day's own comparison into its own day<N> folder; a single-day run writes straight into the top level.
#
#   6. PLANNED VERSUS ACTUAL ACTIVITY
#      _base_label, _act_code, _act_short, _act_bg, _activity_panel, _activity_legend_panel, _write_plan_vs_actual_activity_day, _write_plan_vs_actual_activity_all, _write_side_by_side_html, _write_by_entity_html
#      -- compare the planned and executed label of every CEV and MCS in each interval of one day, as a heatmap figure and two HTML tables, written into that day's own folder for a run of several days.
#      There is no Overall version of this, since an activity label has no meaningful average or sum across days.
#
#   7. ENTRY POINTS
#      _write_logs, write_reports, write_outputs, write_important_outputs, write_detailed_output
#      -- write the two logs, the reports, the figures and reports together, only the eight most important files, and the four detailed logs when they were built.
# #############################################################################
module Output

# external packages used across this file
using Plots
using DataFrames
using CSV
using Printf

using ..Common: create_fixed_2hour_xticks, stepify_interval_values,
                  stepify_boundary_values, interval_time_dataframe,
                  clock_label, in_peak

# everything below that other files are allowed to use
export write_outputs, write_detailed_output, write_important_outputs

# Selects the GR plotting backend.
gr()

# Colors used in turn for the MCSs, CEVs and sites in the figures.
const COLORS = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

# Returns a new plot with the font, margin and size settings shared by all the figures.
# Any settings passed in override these defaults.
_base_plot(; kw...) = plot(; size = (900, 500), xrotation = 45,
    guidefontsize = 18, tickfontsize = 18, legendfontsize = 12,
    bottom_margin = 18Plots.mm, left_margin = 16Plots.mm, right_margin = 14Plots.mm, kw...)

# Returns the plot and the CSV of the total grid charging power and the total MCS discharge power into the CEVs, summed over all MCSs, for every interval of the run.
# Discharging is drawn below zero, and the net power column is charging minus discharging.
function fig_total_grid_power(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    charging    = [sum(res.real_P_ch[m, k]  for m in d.M) for k in K]
    discharging = [sum(res.real_P_dch[m, k] for m in d.M) for k in K]
    p = _base_plot(title = "", xlabel = "Time", ylabel = "Power (kW)",
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    xc, yc = stepify_interval_values(K, charging)
    xd, yd = stepify_interval_values(K, -discharging)
    plot!(p, xc, yc, label = "Total Charging (Grid)", alpha = 0.8, linewidth = 2)
    plot!(p, xd, yd, label = "Total Discharging (CEVs)", alpha = 0.6, linewidth = 2)
    hline!(p, [0.0], color = :black, linestyle = :dash, alpha = 0.5, label = nothing)
    ymax = max(maximum(charging), maximum(discharging), 1.0)
    ylims!(p, (-1.1 * ymax, 1.1 * ymax))
    csv = interval_time_dataframe(K, res.time_labels)
    csv[!, "Total_Charging_Power_kW"]    = charging
    csv[!, "Total_Discharging_Power_kW"] = discharging
    csv[!, "Net_Power_kW"]               = charging .- discharging
    return p, csv
end

# Returns three things: a plot with one panel per construction site, a plot overlaying the work power total of every site, and the CSV of both.
# Each site panel shows the realized work power of each CEV working there and the site total, which is the sum over its CEVs.
function fig_work_by_site(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    site_totals = Dict(i => [sum(res.real_P_work[i, e, k] for e in d.E) for k in K] for i in d.N_c)
    ymax = maximum(vcat(values(site_totals)...); init = 0.0)
    ylim = ymax > 0 ? (0, 1.1 * ymax) : (0, 1)

    p_overlay = _base_plot(title = "", xlabel = "Time", ylabel = "Power (kW)",
                           xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    csv = interval_time_dataframe(K, res.time_labels)
    site_plots = Any[]
    for (idx, i) in enumerate(d.N_c)
        p_site = _base_plot(title = "Site $i", titlefontsize = 18, xlabel = "Time",
                            ylabel = "Power (kW)", xticks = (rT, rL),
                            xlims = (first(Tplot), last(Tplot)), ylims = ylim, legend = :topright)
        for (e_idx, e) in enumerate(d.E)
            cev_work = [res.real_P_work[i, e, k] for k in K]
            if !isempty(cev_work) && maximum(cev_work) > 0
                xs, ys = stepify_interval_values(K, cev_work)
                plot!(p_site, xs, ys, label = "CEV $e", color = COLORS[mod1(e_idx, length(COLORS))], linewidth = 2)
            end
        end
        site_work = site_totals[i]
        if !isempty(site_work) && maximum(site_work) > 0
            xs, ys = stepify_interval_values(K, site_work)
            plot!(p_site, xs, ys, label = "Site total", color = :black, linewidth = 2, linestyle = :dash)
            plot!(p_overlay, xs, ys, label = "Site $i", color = COLORS[mod1(idx, length(COLORS))], linewidth = 2)
        end
        csv[!, "Site_$(i)_Work_Power_kW"] = site_work
        push!(site_plots, p_site)
    end
    csv[!, "Total_Work_Power_kW"] = [sum(res.real_P_work[i, e, k] for i in d.N_c, e in d.E) for k in K]
    n = length(site_plots)
    p_multi = n == 0 ? plot(title = "") :
        plot(site_plots...; layout = (n, 1), size = (900, 400 * n),
             plot_title = "Work Power Profiles by Site", plot_titlevspan = 0.13)
    return p_multi, p_overlay, csv
end

# Returns the plot and CSV of the state of energy in kWh of the units in unit_set over the run, with each unit's maximum and minimum SOE as dashed lines.
# soe, soe_max and soe_min are the SOE array and the bounds for that kind of unit, and label_prefix names it.
function _fig_soe(res, unit_set, soe, soe_max, soe_min, label_prefix)
    d = res.d; T = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(T, d.t_start)
    p = _base_plot(title = "", xlabel = "Time", ylabel = "State of Energy (kWh)",
                   xticks = (rT, rL), xlims = (first(T), last(T)))
    csv = DataFrame(Time_Period = collect(T), Time_Label = res.time_labels)
    for (idx, u) in enumerate(unit_set)
        vals = [soe[u, t] for t in T]
        xs, ys = stepify_boundary_values(T, vals)
        plot!(p, xs, ys, label = "$label_prefix $u", color = COLORS[mod1(idx, length(COLORS))], linewidth = 2)
        csv[!, "$(label_prefix)_$(u)_SOE_kWh"]     = vals
        csv[!, "$(label_prefix)_$(u)_Max_SOE_kWh"] = fill(soe_max[u], length(T))
        csv[!, "$(label_prefix)_$(u)_Min_SOE_kWh"] = fill(soe_min[u], length(T))
    end
    hline!(p, [soe_max[u] for u in unit_set], color = :black, linestyle = :dash, label = "Max Energy")
    hline!(p, [soe_min[u] for u in unit_set], color = :gray,  linestyle = :dash, label = "Min Energy")
    return p, csv
end

# Plots the SOE of every MCS.
fig_mcs_soe(res) = _fig_soe(res, res.d.M, res.real_SOE_MCS, res.d.SOE_MCS_max, res.d.SOE_MCS_min, "MCS")
# Plots the SOE of every CEV.
fig_cev_soe(res) = _fig_soe(res, res.d.E, res.real_SOE_CEV, res.d.SOE_CEV_max, res.d.SOE_CEV_min, "CEV")

# Returns the plot and CSV of the electricity price (left axis) and the grid CO2 emission factor (right axis) for every interval of the run.
function fig_price_emission(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    csv = interval_time_dataframe(K, res.time_labels)
    csv[!, "Electricity_Price_USD_per_kWh"]        = [d.lambda_whl_elec[k] for k in K]
    csv[!, "CO2_Emission_Factor_kg_CO2_per_kWh"]   = [d.lambda_CO2[k] for k in K]
    p = _base_plot(title = "", xlabel = "Time", ylabel = "Electricity Price (\$/kWh)",
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)),
                   top_margin = 24Plots.mm, legend = (0.01, 1.26), grid = true, color = :blue)
    xs, ys = stepify_interval_values(K, [d.lambda_whl_elec[k] for k in K])
    plot!(p, xs, ys, label = "Electricity Price", linewidth = 2)
    p_twin = twinx(p)
    xc, yc = stepify_interval_values(K, [d.lambda_CO2[k] for k in K])
    plot!(p_twin, xc, yc, ylabel = "CO₂ Emission Factor (kg CO₂/kWh)", label = nothing,
          linewidth = 2, xlims = (first(Tplot), last(Tplot)), xticks = (rT, rL),
          color = :red, guidefontsize = 18, tickfontsize = 18)
    plot!(p, [NaN], [NaN], label = "CO₂ Emission Factor", linewidth = 2)
    return p, csv
end

# Returns the plot and CSV of where each MCS is in every interval, as a node index, with 0 meaning it is on the road.
function fig_location(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    node_labels = [node in d.N_g ? "Grid $node" : "Site $node" for node in d.N]
    yt_pos = vcat(0, collect(d.N)); yt_lab = vcat("Travel", node_labels)
    p = _base_plot(title = "", xlabel = "Time", ylabel = "Node Type",
                   yticks = (yt_pos, yt_lab), xticks = (rT, rL),
                   xlims = (first(Tplot), last(Tplot)), grid = true)
    csv = interval_time_dataframe(K, res.time_labels)
    for (idx, m) in enumerate(d.M)
        locs = [res.real_loc[m, k] for k in K]
        csv[!, "MCS_$(m)_Location"] = locs
        csv[!, "MCS_$(m)_Location_Type"] =
            [i == 0 ? "Travel" : (i in d.N_g ? "Grid" : "Construction") for i in locs]
        xs, ys = stepify_interval_values(K, locs)
        plot!(p, xs, ys, label = "MCS $m", linewidth = 2, marker = :circle, markersize = 4,
              color = COLORS[mod1(idx, length(COLORS))])
    end
    return p, csv
end

# Returns one plot and one CSV per MCS, each showing that MCS's grid charging power and its discharge power into the CEVs for every interval.
function figs_individual_mcs(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    plots = Any[]; csvs = DataFrame[]
    for m in d.M
        p = _base_plot(title = "MCS $m", titlefontsize = 18, xlabel = "Time", ylabel = "Power (kW)",
                       xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
        charging    = [res.real_P_ch[m, k]  for k in K]
        discharging = [res.real_P_dch[m, k] for k in K]
        xc, yc = stepify_interval_values(K, charging)
        xd, yd = stepify_interval_values(K, -discharging)
        plot!(p, xc, yc, label = "Charging", alpha = 0.8, linewidth = 2)
        plot!(p, xd, yd, label = "Discharging", alpha = 0.6, linewidth = 2)
        hline!(p, [0.0], color = :black, linestyle = :dash, alpha = 0.5, label = nothing)
        ymax = max(maximum(charging), maximum(discharging), 1.0)
        ylims!(p, (-1.1 * ymax, 1.1 * ymax))
        csv = interval_time_dataframe(K, res.time_labels)
        csv[!, "Charging_Power_kW"]    = charging
        csv[!, "Discharging_Power_kW"] = discharging
        csv[!, "Net_Power_kW"]         = charging .- discharging
        push!(plots, p); push!(csvs, csv)
    end
    return plots, csvs
end

# Returns one CSV with a row per interval, holding for every MCS its grid charging, discharging and start and end SOE, and for every CEV its work power and start and end SOE.
# The MCS traveling column is always zero, because an MCS draws no battery energy while it is on the road.
function csv_mcs_cev_soe(res)
    d = res.d; K = collect(1:res.nK)
    csv = DataFrame(Time_Interval = K, Time_Period = K,
                    Start_Time_Label = res.time_labels[K],
                    End_Time_Label   = res.time_labels[K .+ 1])
    for m in d.M
        csv[!, "MCS_$(m)_Charging_kW"]    = [res.real_P_ch[m, k]  for k in K]
        csv[!, "MCS_$(m)_Discharging_kW"] = [res.real_P_dch[m, k] for k in K]
        csv[!, "MCS_$(m)_Traveling_kW"]   = zeros(length(K)) 
        csv[!, "MCS_$(m)_SOE_Start_kWh"]  = [res.real_SOE_MCS[m, k]     for k in K]
        csv[!, "MCS_$(m)_SOE_End_kWh"]    = [res.real_SOE_MCS[m, k + 1] for k in K]
    end
    for e in d.E
        csv[!, "CEV_$(e)_Working_kW"]    = [sum(res.real_P_work[i, e, k] for i in d.N_c) for k in K]
        csv[!, "CEV_$(e)_SOE_Start_kWh"] = [res.real_SOE_CEV[e, k]     for k in K]
        csv[!, "CEV_$(e)_SOE_End_kWh"]   = [res.real_SOE_CEV[e, k + 1] for k in K]
    end
    return csv
end

# Writes the numbered trajectory figures and CSVs into out_dir: 01 total grid power, 02 work by site, 03 MCS SOE, 04 CEV SOE, 05 price and emissions, 06 MCS location, 07 the combined summary figure with its SOE CSV, and 09 one power profile per MCS.
# The combined figure puts the price, grid power, MCS SOE, site work, CEV SOE and location plots on one page, next to a text summary of the run size.
function write_trajectory_figures(res, out_dir)
    mkpath(out_dir)

    p01, c01 = fig_total_grid_power(res)
    savefig(p01, joinpath(out_dir, "A1_01_total_grid_power_profile.png"))
    CSV.write(joinpath(out_dir, "A1_01_total_grid_power_profile.csv"), c01)

    p02_multi, p02_overlay, c02 = fig_work_by_site(res)
    savefig(p02_multi, joinpath(out_dir, "A1_02_work_profiles_by_site.png"))
    CSV.write(joinpath(out_dir, "A1_02_work_profiles_by_site.csv"), c02)

    p03, c03 = fig_mcs_soe(res)
    savefig(p03, joinpath(out_dir, "A1_03_mcs_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "A1_03_mcs_state_of_energy.csv"), c03)

    p04, c04 = fig_cev_soe(res)
    savefig(p04, joinpath(out_dir, "A1_04_cev_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "A1_04_cev_state_of_energy.csv"), c04)

    p05, c05 = fig_price_emission(res)
    savefig(p05, joinpath(out_dir, "A1_05_electricity_prices_emissions.png"))
    CSV.write(joinpath(out_dir, "A1_05_electricity_prices.csv"), c05)

    p06, c06 = fig_location(res)
    savefig(p06, joinpath(out_dir, "A1_06_mcs_location_trajectory.png"))
    CSV.write(joinpath(out_dir, "A1_06_mcs_location_trajectory.csv"), c06)

    summary_text = """
    Optimization Summary
    -------------------
    Number of MCSs: $(length(res.d.M))
    Number of CEVs: $(length(res.d.E))
    Number of nodes: $(length(res.d.N)) (Grid: $(length(res.d.N_g)), Construction: $(length(res.d.N_c)))
    Time interval: $(res.d.delta_T) h
    Number of intervals: $(res.nK) ($(res.n_day_run) day(s))
    """
    p_summary = plot(legend = false, grid = false, framestyle = :none, xticks = false, yticks = false,
                     left_margin = 16Plots.mm, right_margin = 14Plots.mm)
    annotate!(p_summary, 0, 0.5, text(summary_text, :black, 12, :left))
    p_combined = plot(p05, p01, p03, p02_overlay, p04, p06, p_summary,
                      layout = (4, 2), size = (1800, 2200), left_margin = 16Plots.mm)
    savefig(p_combined, joinpath(out_dir, "A1_07_mcs_optimization_summary.png"))
    CSV.write(joinpath(out_dir, "A1_07_mcs_cev_soe.csv"), csv_mcs_cev_soe(res))

    mcs_plots, mcs_csvs = figs_individual_mcs(res)
    for (m_idx, mp) in enumerate(mcs_plots)
        savefig(mp, joinpath(out_dir, "A1_09_mcs_$(m_idx)_power_profile.png"))
        CSV.write(joinpath(out_dir, "A1_09_mcs_$(m_idx)_power_profile.csv"), mcs_csvs[m_idx])
    end
    return nothing
end

# Prices a run into its cost components in USD: grid energy, carbon, the non-coincident and on-peak demand charges, missed work, MCS travel labour and the terminal SOE shortfall penalty, plus their total.
# The first six are the terms of objective function (4) of the paper's MILP formulation, with carbon converted from dollars per ton to dollars per kg, and the shortfall penalty is added outside it.
function _cost_components(d, q)
    energy_cost = q.total_cost
    carbon_cost = (d.carbon_price_per_ton / 1000.0) * q.total_co2
    ncd_cost    = d.lambda_demand_NC * q.nc_peak
    opd_cost    = d.lambda_demand_OP * q.op_peak
    missed_cost = d.rho_miss * q.missed
    travel_cost = q.labour_cost
    shortfall_cost = q.shortfall_penalty_cost
    total       = energy_cost + carbon_cost + ncd_cost + opd_cost + missed_cost + travel_cost + shortfall_cost
    return (; energy_cost, carbon_cost, ncd_cost, opd_cost, missed_cost, travel_cost, shortfall_cost, total)
end

# Returns the whole-run quantities from which the cost components and the KPI table are built.
# q is a named tuple that holds the energy cost, CO2, the two demand peaks, missed work, travel labour, the terminal shortfall, MCS transit hours, the number of infeasible windows and the total solver time.
function _run_quantities(res)
    return (; total_cost = res.total_cost, total_co2 = res.total_co2,
              nc_peak = res.nc_peak, op_peak = res.op_peak, missed = res.missed,
              labour_cost = res.labour_cost, shortfall_penalty_cost = res.shortfall_penalty_cost,
              total_energy = res.total_energy, transit_hours = res.transit_intervals * res.d.delta_T,
              shortfall_kWh = res.shortfall_kWh, n_infeasible = res.n_infeasible,
              solve_time_s = sum(filter(!isnan, res.solve_log.solve_time_s)))
end

# Returns the same quantities as _run_quantities, but for one day of the run.
# Energy, cost, CO2, the peaks and the solver numbers come from that day's rows of the logs.
# Missed work and the terminal shortfall are the change from the end of the previous day, so the days add up to the whole-run value.
function _day_quantities(res, day)
    d = res.d; dt = d.delta_T
    daylog = res.log[res.log.day .== day, :]
    op_mask = [in_peak(k, dt, d.t_start) for k in daylog.k]
    snaps = res.day_snapshots_df
    snap = snaps[snaps.day .== day, :][1, :]
    prev_missed = 0.0; prev_kWh = 0.0; prev_cost = 0.0
    if day > 1
        p = snaps[snaps.day .== day - 1, :][1, :]
        prev_missed = p.missed_work_cumulative_h
        prev_kWh    = p.shortfall_kWh_cumulative
        prev_cost   = p.shortfall_penalty_cost_cumulative
    end
    solve_mask = res.solve_log.day .== day
    return (; total_cost = sum(daylog.grid_kW .* daylog.price) * dt,
              total_co2 = sum(daylog.grid_kW .* daylog.co2) * dt,
              nc_peak = isempty(daylog.grid_kW) ? 0.0 : maximum(daylog.grid_kW),
              op_peak = any(op_mask) ? maximum(daylog.grid_kW[op_mask]) : 0.0,
              missed = snap.missed_work_cumulative_h - prev_missed,
              labour_cost = snap.labour_cost_day,
              shortfall_penalty_cost = snap.shortfall_penalty_cost_cumulative - prev_cost,
              total_energy = sum(daylog.grid_kW) * dt,
              transit_hours = snap.transit_hours_day,
              shortfall_kWh = snap.shortfall_kWh_cumulative - prev_kWh,
              n_infeasible = count(isnan, res.solve_log.objective[solve_mask]),
              solve_time_s = sum(filter(!isnan, res.solve_log.solve_time_s[solve_mask])))
end

# The metric names of the KPI table, in the order of the values returned by _kpi_column_values.
const KPI_METRIC_NAMES = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
                          "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
                          "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg",
                          "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour", "MCS_Transit_hour",
                          "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "MPC_loop_time_s", "Total_solve_time_s"]

# Returns the KPI values of one column of the KPI table, rounded and in the order of KPI_METRIC_NAMES.
# q holds the whole-run or one-day quantities, and loop_time_s is the wall time of the loop, or missing when it is not known for that column.
function _kpi_column_values(d, q; loop_time_s)
    c = _cost_components(d, q)
    return Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
               round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
               round(c.travel_cost, digits = 2), round(c.shortfall_cost, digits = 2),
               round(q.total_energy, digits = 2), round(q.total_co2, digits = 2),
               round(q.nc_peak, digits = 2), round(q.op_peak, digits = 2), round(q.missed, digits = 2),
               round(q.transit_hours, digits = 2),
               round(q.shortfall_kWh, digits = 3),
               q.n_infeasible, ismissing(loop_time_s) ? missing : round(loop_time_s, digits = 2),
               round(q.solve_time_s, digits = 2)]
end

# Writes the KPI table to A1_kpi_summary.csv, and when the run has several days also the per-day KPI table to A1_kpi_summary_by_day.csv.
# With figures set to false it stops there, and otherwise it also writes the cost and demand peak bar charts to A1_08_kpi_metrics_summary.png.
# The tables hold the cost components, grid energy, CO2, the two demand peaks, missed work, MCS transit hours, the terminal SOE shortfall, the number of infeasible windows, the loop run time and the total solver time.
function _write_kpi_metrics(res, out_dir; figures::Bool = true)
    d = res.d
    q = _run_quantities(res)
    c = _cost_components(d, q)
    totals = DataFrame(Metric = KPI_METRIC_NAMES, Value = _kpi_column_values(d, q; loop_time_s = res.elapsed))
    CSV.write(joinpath(out_dir, "A1_kpi_summary.csv"), totals)

    if res.n_day_run > 1
        by_day = DataFrame(Metric = KPI_METRIC_NAMES)
        for day in 1:res.n_day_run
            by_day[!, "Day$(day)"] = _kpi_column_values(d, _day_quantities(res, day); loop_time_s = missing)
        end
        by_day[!, "Overall"] = totals.Value
        CSV.write(joinpath(out_dir, "A1_kpi_summary_by_day.csv"), by_day)
    end
    figures || return nothing

    cost_labels = ["Energy", "CO₂", "NCD", "OPD", "Missed Work", "Travel", "Terminal Shortfall", "Total"]
    cost_values = [c.energy_cost, c.carbon_cost, c.ncd_cost, c.opd_cost, c.missed_cost, c.travel_cost, c.shortfall_cost, c.total]
    cost_colors = [:steelblue, :forestgreen, :darkorange, :purple, :firebrick, :teal, :sienna, :black]
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
    savefig(p_summary, joinpath(out_dir, "A1_08_kpi_metrics_summary.png"))
end

# Formats one cell of a re-plan table: text is kept as it is, numbers are rounded to 3 decimals, and NaN becomes an empty cell.
_cell(v::AbstractString) = v
_cell(v::Real) = isnan(v) ? "" : round(v, digits = 3)

# Writes the re-plan grid mat to a CSV at path, and the same grid as an HTML table next to it.
# Rows are the re-solve intervals and columns are the planned intervals.
# For a column before the row's own interval, the cell shows the value planned and applied at that column's step, so each row reads as the part already carried out followed by the forward plan.
function _write_replan_grid(path, mat, res, nK)
    d = res.d
    df = DataFrame(replan_at = [clock_label(d.t_start, d.delta_T, k0) for k0 in 1:nK])
    for k in 1:nK
        df[!, Symbol(clock_label(d.t_start, d.delta_T, k))] =
            Any[_cell(k < k0 ? mat[k, k] : mat[k0, k]) for k0 in 1:nK]
    end
    CSV.write(path, df)
    _write_replan_grid_html(replace(path, r"\.csv$" => ".html"), mat, res, nK)
end

# Writes the grid as an HTML table with a short explanation, coloring past cells green and current and future plan cells yellow.
function _write_replan_grid_html(path, mat, res, nK)
    d = res.d
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

# Writes the re-plan grids of every day into out_dir/replan_grids, with one sub-folder per day when the run has several days.
# For each day it writes the planned grid power, the planned SOE and status of every MCS, and the planned SOE and activity of every CEV, each as a CSV and an HTML table.
function _write_replan_grids(res, out_dir)
    d = res.d; nKd = res.nKd
    for day in 1:res.n_day_run
        g = res.replan_by_day[day]
        gdir = res.n_day_run == 1 ? joinpath(out_dir, "A1_replan_grids") :
                                     joinpath(out_dir, "A1_replan_grids", "day$(day)")
        mkpath(gdir)
        _write_replan_grid(joinpath(gdir, "A1_plan_grid_kW.csv"), g.plan_grid_kW, res, nKd)
        for m in d.M
            _write_replan_grid(joinpath(gdir, "A1_plan_mcs$(m)_soe.csv"),      g.plan_mcs_soe[m], res, nKd)
            _write_replan_grid(joinpath(gdir, "A1_plan_mcs$(m)_activity.csv"), g.plan_mcs_act[m], res, nKd)
        end
        for e in d.E
            _write_replan_grid(joinpath(gdir, "A1_plan_cev$(e)_soe.csv"),      g.plan_cev_soe[e], res, nKd)
            _write_replan_grid(joinpath(gdir, "A1_plan_cev$(e)_activity.csv"), g.plan_cev_act[e], res, nKd)
        end
    end
end

# Returns the first re-solve interval of the given day that has a plan, which is interval 1 unless the first windows of that day were infeasible.
function _first_plan_row(res, day::Int)
    g1 = res.replan_by_day[day]
    for r in 1:res.nKd
        any(!isnan(g1.plan_grid_kW[r, k]) for k in r:res.nKd) && return r
    end
    return 1
end

# Prices the first plan of the given day with the same cost terms as the realized run, so the plan and the realized totals can be compared.
# The planned grid power, MCS transit intervals and digging and loading+swinging hours come from that plan's grid and labels, and intervals before the plan starts count as zero grid power.
# Missed work is that day's required hours minus the planned hours, and the planned total leaves out the terminal SOE shortfall penalty because the plan itself has no shortfall.
function _planned_kpis(res, day::Int)
    d = res.d; nKd = res.nKd; dt = d.delta_T
    r = _first_plan_row(res, day)
    g1 = res.replan_by_day[day]
    g = [ (v = g1.plan_grid_kW[r, k]; isnan(v) ? 0.0 : v) for k in 1:nKd ]
    price = [d.lambda_whl_elec[k] for k in 1:nKd]
    co2f  = [d.lambda_CO2[k]      for k in 1:nKd]
    energy = sum(g) * dt
    ecost  = sum(g .* price) * dt
    co2kg  = sum(g .* co2f)  * dt
    carbon = (d.carbon_price_per_ton / 1000.0) * co2kg
    ncpk   = isempty(g) ? 0.0 : maximum(g)
    opmask = [in_peak(k, dt, d.t_start) for k in 1:nKd]
    oppk   = any(opmask) ? maximum(g[opmask]) : 0.0
    ncd    = d.lambda_demand_NC * ncpk
    opd    = d.lambda_demand_OP * oppk
    transit = sum(count(k -> g1.plan_mcs_act[m][r, k] == "Traveling", 1:nKd) for m in d.M)
    labour  = d.rho_labor * dt * transit
    pdig = zeros(length(d.N)); pload = zeros(length(d.N))
    for e in d.E
        site = findfirst(i -> d.A[i, e] == 1, d.N); site === nothing && continue
        for k in 1:nKd
            lab = g1.plan_cev_act[e][r, k]
            lab == res.ACT_NAME[1] && (pdig[site]  += dt)    
            lab == res.ACT_NAME[2] && (pload[site] += dt)   
        end
    end
    missed = sum((max(d.hours_digging[i]          - pdig[i],  0.0) for i in d.N_c); init = 0.0) +
             sum((max(d.hours_loading_swinging[i] - pload[i], 0.0) for i in d.N_c); init = 0.0)
    missed_cost = d.rho_miss * missed
    total = ecost + carbon + ncd + opd + missed_cost + labour
    return (; r, g, energy, ecost, co2kg, carbon, ncpk, oppk, ncd, opd,
              transit, labour, missed, missed_cost, total)
end

# Prices the Overall plan across every day of the run: the additive quantities (grid energy, costs, CO2, missed work, MCS transit, labour) are summed over each day's first plan, and the two demand peaks are the maximum of each day's planned peak, with their charges recomputed from that maximum, the same way the realized peaks are a single value for the whole run rather than a sum across days.
# The terminal SOE shortfall is planned as zero, the same as for a single day.
function _overall_planned_kpis(res)
    d = res.d; dt = d.delta_T
    per_day = [_planned_kpis(res, day) for day in 1:res.n_day_run]
    energy = sum(p.energy for p in per_day)
    ecost  = sum(p.ecost  for p in per_day)
    co2kg  = sum(p.co2kg  for p in per_day)
    carbon = (d.carbon_price_per_ton / 1000.0) * co2kg
    ncpk   = maximum(p.ncpk for p in per_day)
    oppk   = maximum(p.oppk for p in per_day)
    ncd    = d.lambda_demand_NC * ncpk
    opd    = d.lambda_demand_OP * oppk
    transit = sum(p.transit for p in per_day)
    labour  = sum(p.labour  for p in per_day)
    missed      = sum(p.missed      for p in per_day)
    missed_cost = d.rho_miss * missed
    total = ecost + carbon + ncd + opd + missed_cost + labour
    return (; energy, ecost, co2kg, carbon, ncpk, oppk, ncd, opd,
              transit, labour, missed, missed_cost, total)
end

# Returns the plan-vs-actual summary rows shared by the day view and the Overall view: one tuple per KPI of (name, planned value, realized value).
function _plan_vs_actual_rows(dt, p, c, q)
    return [
        ("Grid energy (kWh)",         p.energy,          q.total_energy),
        ("Energy cost (USD)",         p.ecost,           c.energy_cost),
        ("CO2 emissions (kg)",        p.co2kg,           q.total_co2),
        ("CO2 cost (USD)",            p.carbon,          c.carbon_cost),
        ("NCD peak (kW)",             p.ncpk,            q.nc_peak),
        ("NCD charge (USD)",          p.ncd,             c.ncd_cost),
        ("OPD peak (kW)",             p.oppk,            q.op_peak),
        ("OPD charge (USD)",          p.opd,             c.opd_cost),
        ("Missed work (h)",           p.missed,          q.missed),
        ("Missed work penalty (USD)", p.missed_cost,     c.missed_cost),
        ("MCS transit (h)",           p.transit * dt,    q.transit_hours),
        ("Terminal shortfall (USD)",  0.0,               c.shortfall_cost),
        ("Travel labour (USD)",       p.labour,          c.travel_cost),
        ("TOTAL cost (USD)",          p.total,           c.total),
    ]
end

# Builds the summary DataFrame and, when byint is not nothing, the per-interval grid power DataFrame, from the rows of _plan_vs_actual_rows.
function _plan_vs_actual_tables(dt, p, c, q)
    rows = _plan_vs_actual_rows(dt, p, c, q)
    summ = DataFrame(
        Metric              = [x[1] for x in rows],
        Planned_at_start    = [round(x[2], digits = 3) for x in rows],
        Realized_end_of_day = [round(x[3], digits = 3) for x in rows],
        Delta_real_minus_plan = [round(x[3] - x[2], digits = 3) for x in rows],
        Pct_change = [abs(x[2]) < 1e-9 ? (abs(x[3]) < 1e-9 ? 0.0 : NaN) :
                      round(100 * (x[3] - x[2]) / x[2], digits = 1) for x in rows])
    return summ
end

# Writes the cost bar chart (plan vs realized) to path from the same rows as the summary table.
function _write_plan_vs_actual_chart(path, p, c, plan_label)
    labels = ["Energy", "CO₂", "NCD", "OPD", "Missed", "Labour", "Shortfall", "TOTAL"]
    planned = [p.ecost, p.carbon, p.ncd, p.opd, p.missed_cost, p.labour, 0.0, p.total]
    realized = [c.energy_cost, c.carbon_cost, c.ncd_cost, c.opd_cost, c.missed_cost, c.travel_cost, c.shortfall_cost, c.total]
    ymax = max(maximum(planned), maximum(realized), 1.0)
    pbar = plot(title = "$plan_label vs Realised — cost components",
                xlabel = "", ylabel = "Cost (USD)", titlefontsize = 14,
                xticks = (1:length(labels), labels), xlims = (0.5, length(labels) + 0.5),
                ylims = (0, 1.25 * ymax), xrotation = 20, legend = :topleft,
                guidefontsize = 14, tickfontsize = 12, size = (1150, 500),
                bottom_margin = 12Plots.mm, left_margin = 14Plots.mm)
    for i in eachindex(labels)
        bar!(pbar, [i - 0.19], [planned[i]],  bar_width = 0.36, color = :goldenrod,
             label = i == 1 ? plan_label : "")
        bar!(pbar, [i + 0.19], [realized[i]], bar_width = 0.36, color = :forestgreen,
             label = i == 1 ? "Realised" : "")
    end
    savefig(pbar, path)
    return nothing
end

# Compares the first plan of one day with that day's realized values, and writes that day's A1_plan_vs_actual.html and A1_plan_vs_actual_costs.png into out_dir.
# It also builds a per-interval table of the planned and realized grid power for that day.
function _write_plan_vs_actual_day(res, day::Int, out_dir; figures::Bool = true)
    d = res.d; nKd = res.nKd; dt = d.delta_T
    p = _planned_kpis(res, day)
    q = _day_quantities(res, day)
    c = _cost_components(d, q)
    r = p.r
    plan_clock = clock_label(d.t_start, d.delta_T, r)
    plan_label = "Planned @ $plan_clock"

    summ = _plan_vs_actual_tables(dt, p, c, q)

    realized_g = [sum(res.real_P_ch[m, (day - 1) * nKd + k] for m in d.M) for k in 1:nKd]
    byint = DataFrame(
        k = collect(1:nKd),
        clock = [clock_label(d.t_start, d.delta_T, k) for k in 1:nKd],
        price = [d.lambda_whl_elec[k] for k in 1:nKd],
        co2_factor = [d.lambda_CO2[k] for k in 1:nKd],
        on_peak = [in_peak(k, dt, d.t_start) ? "Yes" : "No" for k in 1:nKd],
        planned_grid_kW  = round.(p.g, digits = 3),
        realized_grid_kW = round.(realized_g, digits = 3),
        delta_kW = round.(realized_g .- p.g, digits = 3))

    header_note = "<b>Yellow</b> = the FIRST optimisation made at $plan_clock (the whole-day forward plan, " *
                  "every interval still \"pending\").&nbsp;&nbsp;<b>Green</b> = the REALISED trajectory " *
                  "after closed-loop re-planning against the stochastic plant."
    _write_plan_vs_actual_html(joinpath(out_dir, "A1_plan_vs_actual.html"),
                               summ, byint, plan_label, "Plan (@ $plan_clock) vs Realised — day $day", header_note)
    figures || return nothing
    _write_plan_vs_actual_chart(joinpath(out_dir, "A1_plan_vs_actual_costs.png"), p, c, plan_label)
    return nothing
end

# Compares the Overall plan (the sum/max of every day's first plan, see _overall_planned_kpis) with the realized values of the whole run, and writes A1_plan_vs_actual.html and A1_plan_vs_actual_costs.png into out_dir.
# There is no single day of intervals to show, so this has no per-interval grid power table.
function _write_plan_vs_actual_overall(res, out_dir; figures::Bool = true)
    d = res.d; dt = d.delta_T
    p = _overall_planned_kpis(res)
    q = _run_quantities(res)
    c = _cost_components(d, q)
    plan_label = "Planned (Days 1–$(res.n_day_run), summed)"

    summ = _plan_vs_actual_tables(dt, p, c, q)
    header_note = "<b>Yellow</b> = the sum (or, for the two demand peaks, the maximum) of every day's first plan.&nbsp;&nbsp;" *
                  "<b>Green</b> = the REALISED values over the whole run."
    _write_plan_vs_actual_html(joinpath(out_dir, "A1_plan_vs_actual.html"),
                               summ, nothing, plan_label, "Plan (Overall) vs Realised", header_note)
    figures || return nothing
    _write_plan_vs_actual_chart(joinpath(out_dir, "A1_plan_vs_actual_costs.png"), p, c, plan_label)
    return nothing
end

# Writes the plan vs realized cost comparison for the whole run into out_dir.
# For a single-day run this is just that one day's comparison, written straight into out_dir.
# For a run of several days, the Overall comparison is written into out_dir, and each day's own comparison is written into out_dir/day<N>/.
function _write_plan_vs_actual_all(res, out_dir; figures::Bool = true)
    if res.n_day_run == 1
        _write_plan_vs_actual_day(res, 1, out_dir; figures)
        return nothing
    end
    _write_plan_vs_actual_overall(res, out_dir; figures)
    for day in 1:res.n_day_run
        daydir = joinpath(out_dir, "day$(day)")
        mkpath(daydir)
        _write_plan_vs_actual_day(res, day, daydir; figures)
    end
    return nothing
end

# Writes the plan versus realized report to an HTML page at path, with the summary table and, when byint is not nothing, the per-interval grid power table.
# plan_label names the planned column, title is the page heading, and header_note is the explanatory paragraph under it.
# Planned cells are shaded yellow and realized cells green, and differences are colored red when the realized value is higher and green when it is lower.
function _write_plan_vs_actual_html(path, summ, byint, plan_label, title, header_note)
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "table{border-collapse:collapse;font-size:12px;margin-bottom:20px}")
    println(io, "th,td{border:1px solid #ccc;padding:3px 8px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".plan{background:#ffeb9c}")
    println(io, ".real{background:#c6efce}")
    println(io, ".pos{color:#b00}.neg{color:#070}")
    println(io, "</style></head><body>")
    println(io, "<h2>", title, "</h2>")
    println(io, "<p>", header_note, "</p>")

    println(io, "<h3>Financial &amp; operational summary</h3><table>")
    println(io, "<tr><th>Metric</th><th class=\"plan\">", plan_label, "</th>",
                "<th class=\"real\">Realised</th><th>Δ (real − plan)</th><th>% change</th></tr>")
    for i in 1:nrow(summ)
        dv = summ.Delta_real_minus_plan[i]
        cls = dv > 0 ? "pos" : (dv < 0 ? "neg" : "")
        println(io, "<tr><th>", summ.Metric[i], "</th>",
                "<td class=\"plan\">", summ.Planned_at_start[i], "</td>",
                "<td class=\"real\">", summ.Realized_end_of_day[i], "</td>",
                "<td class=\"", cls, "\">", dv, "</td>",
                "<td class=\"", cls, "\">", isnan(summ.Pct_change[i]) ? "—" : string(summ.Pct_change[i], "%"), "</td></tr>")
    end
    println(io, "</table>")

    if byint !== nothing
        nK = nrow(byint)
        println(io, "<h3>Grid power (kW) per 15-min interval</h3><table>")
        print(io, "<tr><th>interval &rarr;</th>")
        for k in 1:nK; print(io, "<th>", byint.clock[k], "</th>"); end
        println(io, "</tr>")
        print(io, "<tr><th class=\"plan\">", plan_label, "</th>")
        for k in 1:nK; print(io, "<td class=\"plan\">", round(byint.planned_grid_kW[k], digits = 1), "</td>"); end
        println(io, "</tr>")
        print(io, "<tr><th class=\"real\">Realised</th>")
        for k in 1:nK; print(io, "<td class=\"real\">", round(byint.realized_grid_kW[k], digits = 1), "</td>"); end
        println(io, "</tr>")
        println(io, "</table>")
    end
    println(io, "</body></html>")
    write(path, String(take!(io)))
end

# Strips the realized-minutes suffix, for example " (10 min)", from an activity label.
_base_label(l) = replace(l, r" \(\d+ min\)$" => "")

# Returns the code of an activity or status label, ignoring a trailing minutes suffix: 1 Idle, 2 Digging, 3 Loading/Swinging, 4 Traveling, 5 Charging, 6 Charging (grid), 7 Serving CEV, 8 Off.
# An empty or unknown label gets code 0.
function _act_code(l)
    b = _base_label(l)
    b == "Idle" ? 1 : b == "Digging" ? 2 : b == "Loading/Swinging" ? 3 :
    b == "Traveling" ? 4 : b == "Charging" ? 5 : b == "Charging (grid)" ? 6 :
    b == "Serving CEV" ? 7 : b == "Off" ? 8 : 0
end
# The colors of the activity codes 0 to 8, as plot color names and as HTML color codes, and the names of the codes 1 to 8.
const _ACT_COLORS_SYM = [:white, :gray85, :lightskyblue, :darkseagreen, :sandybrown,
                         :khaki, :goldenrod, :mediumpurple, :gray60]
const _ACT_COLORS_HEX = ["#ffffff", "#e8e8e8", "#9ecae1", "#a1d99b", "#fdae6b",
                         "#fee391", "#f6c744", "#bcbddc", "#999999"]
const _ACT_NAMES = ["Idle", "Digging", "Loading/Swinging", "Traveling",
                    "Charging", "Charging (grid)", "Serving CEV", "Off"]
# Returns the short letter code of a label for the HTML legend, or an empty text for an unknown label.
function _act_short(l)
    b = _base_label(l)
    b == "Digging" ? "D" : b == "Loading/Swinging" ? "L" : b == "Traveling" ? "T" :
    b == "Idle" ? "I" : b == "Charging" ? "C" : b == "Charging (grid)" ? "Cg" :
    b == "Serving CEV" ? "S" : b == "Off" ? "O" : ""
end

# Returns the HTML background color of a label.
_act_bg(l) = _ACT_COLORS_HEX[_act_code(l) + 1]

# Returns a two-row heatmap of one unit's actual and planned labels over the day, and the number of intervals where they differ.
# Intervals where the plan and the actual label differ are outlined in red.
function _activity_panel(res, planned, actual, plan_lbl, title)
    d = res.d; nK = res.nKd   
    rT, rL = create_fixed_2hour_xticks(1:(nK + 1), d.t_start)
    cp = [_act_code(planned[k]) for k in 1:nK]
    ca = [_act_code(actual[k])  for k in 1:nK]
    Z  = [reshape(ca, 1, nK); reshape(cp, 1, nK)]  
    p = heatmap(1:nK, 1:2, Z; color = cgrad(_ACT_COLORS_SYM, categorical = true),
                clims = (-0.5, 8.5), colorbar = false, title = title, titlefontsize = 13,
                yticks = ([1, 2], ["Actual", plan_lbl]), xticks = (rT, rL),
                xlims = (0.5, nK + 0.5), ylims = (0.5, 2.5), xrotation = 45,
                tickfontsize = 10, legend = false,
                left_margin = 26Plots.mm, right_margin = 6Plots.mm, bottom_margin = 10Plots.mm)
    changed = [k for k in 1:nK if cp[k] != ca[k]]
    for k in changed  
        plot!(p, Shape([k - 0.5, k + 0.5, k + 0.5, k - 0.5], [0.5, 0.5, 2.5, 2.5]);
              fillalpha = 0.0, linecolor = :red, linewidth = 2, label = "")
    end
    return p, length(changed)
end

# Returns a plot with only the legend: the color of every activity or status and the red outline for a changed interval.
function _activity_legend_panel()
    p = plot(; framestyle = :none, grid = false, xticks = false, yticks = false,
             legend = :inside, legendcolumns = 4, legendfontsize = 10)
    for c in eachindex(_ACT_NAMES)
        scatter!(p, [NaN], [NaN]; markershape = :rect, markersize = 9,
                 color = _ACT_COLORS_SYM[c + 1], markerstrokecolor = :gray, label = _ACT_NAMES[c])
    end
    plot!(p, [NaN], [NaN]; linecolor = :red, linewidth = 3, label = "Changed (plan != actual)")
    return p
end

# Writes one day's plan_vs_actual_activity.png, with one heatmap panel per CEV and per MCS comparing the labels of that day's first plan with the labels actually executed that day, and the two HTML versions of the same comparison, into out_dir.
function _write_plan_vs_actual_activity_day(res, day::Int, out_dir)
    d = res.d; nKd = res.nKd
    r = _first_plan_row(res, day)
    plan_clk = clock_label(d.t_start, d.delta_T, r)
    plan_lbl = "Planned @ $plan_clk"
    times = [clock_label(d.t_start, d.delta_T, k) for k in 1:nKd]

    g1 = res.replan_by_day[day]
    gidx = ((day - 1) * nKd + 1):(day * nKd)
    cev_plan = [[g1.plan_cev_act[e][r, k] for k in 1:nKd] for e in d.E]
    cev_act  = [res.real_cev_act[e][gidx]                 for e in d.E]
    mcs_plan = [[g1.plan_mcs_act[m][r, k] for k in 1:nKd] for m in d.M]
    mcs_act  = [res.real_mcs_act[m][gidx]                 for m in d.M]

    entities = Any[]
    for (ei, e) in enumerate(d.E)
        push!(entities, ("CEV $e", cev_plan[ei], cev_act[ei]))
    end
    for (mi, m) in enumerate(d.M)
        push!(entities, ("MCS $m", mcs_plan[mi], mcs_act[mi]))
    end
    panels = Any[]
    for (title, plnd, act) in entities
        pnl, _ = _activity_panel(res, plnd, act, plan_lbl, title)
        push!(panels, pnl)
    end
    push!(panels, _activity_legend_panel())
    n = length(panels)
    heights = vcat(fill(0.9 / (n - 1), n - 1), [0.1])
    combined = plot(panels...; layout = grid(n, 1, heights = heights),
                    size = (1500, 230 * (n - 1) + 130),
                    plot_title = "Plan (@ $plan_clk) vs Actual activity, day $day  —  red = changed intervals",
                    plot_titlefontsize = 15)
    savefig(combined, joinpath(out_dir, "A1_plan_vs_actual_activity.png"))

    _write_side_by_side_html(joinpath(out_dir, "A1_plan_vs_actual_side_by_side.html"),
                             res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    _write_by_entity_html(joinpath(out_dir, "A1_plan_vs_actual_by_entity.html"),
                          res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    return nothing
end

# Writes the plan vs actual activity comparison for the whole run.
# There is no Overall version of this, because an activity label has no meaningful average or sum across days, so for a run of several days it is written only into each out_dir/day<N>/ folder.
# For a single-day run it is written straight into out_dir, the same as before.
function _write_plan_vs_actual_activity_all(res, out_dir)
    if res.n_day_run == 1
        _write_plan_vs_actual_activity_day(res, 1, out_dir)
        return nothing
    end
    for day in 1:res.n_day_run
        daydir = joinpath(out_dir, "day$(day)")
        mkpath(daydir)
        _write_plan_vs_actual_activity_day(res, day, daydir)
    end
    return nothing
end

# Writes an HTML table with the planned and actual label of every CEV and every MCS side by side for each interval of the day.
# The planned labels come from the first plan of day 1, and actual cells that differ from the plan are outlined in red.
# The page header gives the color legend and the number of changed intervals per unit.
function _write_side_by_side_html(path, res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    d = res.d; nK = res.nKd; nE = length(d.E); nM = length(d.M)
    cev_chg = [_base_label.(cev_plan[ei]) .!= _base_label.(cev_act[ei]) for ei in 1:nE]
    mcs_chg = [_base_label.(mcs_plan[mi]) .!= _base_label.(mcs_act[mi]) for mi in 1:nM]
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "table{border-collapse:collapse;font-size:12px}")
    println(io, "th,td{border:1px solid #ccc;padding:3px 9px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".planh{background:#fff2cc}.acth{background:#d9ead3}")
    println(io, ".chg{outline:2px solid #d00;outline-offset:-2px;font-weight:bold}")
    println(io, ".sw{display:inline-block;width:15px;height:15px;line-height:15px;border:1px solid #999;text-align:center;font-size:10px;vertical-align:middle}")
    println(io, "</style></head><body>")
    println(io, "<h2>Plan (@ $plan_clk) vs Actual activity &mdash; side by side</h2>")
    println(io, "<p><b>Planned</b> = the activity the first whole-day optimisation (made @ $plan_clk) ",
                "assigned to each interval. <b>Actual</b> = what the closed loop executed after re-planning ",
                "every step. <b>Actual</b> cells <span class=\"chg\">outlined in red</span> differ from the plan.</p>")

    print(io, "<p><b>Activity:</b>&nbsp;&nbsp;")
    for c in eachindex(_ACT_NAMES)
        print(io, "<span class=\"sw\" style=\"background:", _ACT_COLORS_HEX[c + 1], "\">",
              _act_short(_ACT_NAMES[c]), "</span> ", _ACT_NAMES[c], "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    print(io, "<p><b>Changed intervals:</b>&nbsp;&nbsp;")
    for (ei, e) in enumerate(d.E)
        print(io, "CEV $e = ", count(cev_chg[ei]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    for (mi, m) in enumerate(d.M)
        print(io, "MCS $m = ", count(mcs_chg[mi]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    println(io, "<table>")
    print(io, "<tr><th rowspan=\"2\">Time</th>",
              "<th colspan=\"", nE + nM, "\" class=\"planh\">Planned @ $plan_clk</th>",
              "<th colspan=\"", nE + nM, "\" class=\"acth\">Actual</th></tr>")
    print(io, "<tr>")
    for e in d.E; print(io, "<th class=\"planh\">CEV $e</th>"); end
    for m in d.M; print(io, "<th class=\"planh\">MCS $m</th>"); end
    for e in d.E; print(io, "<th class=\"acth\">CEV $e</th>"); end
    for m in d.M; print(io, "<th class=\"acth\">MCS $m</th>"); end
    println(io, "</tr>")
    for k in 1:nK
        print(io, "<tr><th>", times[k], "</th>")
        for ei in 1:nE
            l = cev_plan[ei][k]; print(io, "<td style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        for mi in 1:nM
            l = mcs_plan[mi][k]; print(io, "<td style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        for ei in 1:nE
            l = cev_act[ei][k]
            print(io, "<td class=\"", cev_chg[ei][k] ? "chg" : "", "\" style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        for mi in 1:nM
            l = mcs_act[mi][k]
            print(io, "<td class=\"", mcs_chg[mi][k] ? "chg" : "", "\" style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# Writes an HTML table with a planned and an actual column for each CEV and each MCS next to each other, grouped by unit, for each interval of the day.
# The planned labels come from the first plan of day 1, and actual cells that differ from the plan are outlined in red.
# The page header gives the color legend and the number of changed intervals per unit.
function _write_by_entity_html(path, res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    d = res.d; nK = res.nKd; nE = length(d.E); nM = length(d.M)
    cev_chg = [_base_label.(cev_plan[ei]) .!= _base_label.(cev_act[ei]) for ei in 1:nE]
    mcs_chg = [_base_label.(mcs_plan[mi]) .!= _base_label.(mcs_act[mi]) for mi in 1:nM]
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "table{border-collapse:collapse;font-size:12px}")
    println(io, "th,td{border:1px solid #ccc;padding:3px 9px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".planh{background:#fff2cc}.acth{background:#d9ead3}")
    println(io, ".chg{outline:2px solid #d00;outline-offset:-2px;font-weight:bold}")
    println(io, ".sw{display:inline-block;width:15px;height:15px;line-height:15px;border:1px solid #999;text-align:center;font-size:10px;vertical-align:middle}")
    println(io, "</style></head><body>")
    println(io, "<h2>Plan (@ $plan_clk) vs Actual activity &mdash; grouped by unit</h2>")
    println(io, "<p>For each CEV and each MCS, the <b>Planned @ $plan_clk</b> column (the first whole-day plan) ",
                "sits next to the <b>Actual</b> column (what the closed loop executed). ",
                "<b>Actual</b> cells <span class=\"chg\">outlined in red</span> differ from the plan.</p>")

    print(io, "<p><b>Activity:</b>&nbsp;&nbsp;")
    for c in eachindex(_ACT_NAMES)
        print(io, "<span class=\"sw\" style=\"background:", _ACT_COLORS_HEX[c + 1], "\">",
              _act_short(_ACT_NAMES[c]), "</span> ", _ACT_NAMES[c], "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    print(io, "<p><b>Changed intervals:</b>&nbsp;&nbsp;")
    for (ei, e) in enumerate(d.E)
        print(io, "CEV $e = ", count(cev_chg[ei]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    for (mi, m) in enumerate(d.M)
        print(io, "MCS $m = ", count(mcs_chg[mi]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    println(io, "<table>")
    print(io, "<tr><th rowspan=\"2\">Time</th>")
    for e in d.E; print(io, "<th colspan=\"2\">CEV $e</th>"); end
    for m in d.M; print(io, "<th colspan=\"2\">MCS $m</th>"); end
    println(io, "</tr>")
    print(io, "<tr>")
    for _ in 1:(nE + nM)
        print(io, "<th class=\"planh\">Planned @ $plan_clk</th><th class=\"acth\">Actual</th>")
    end
    println(io, "</tr>")
    for k in 1:nK
        print(io, "<tr><th>", times[k], "</th>")
        for ei in 1:nE
            lp = cev_plan[ei][k]; la = cev_act[ei][k]
            print(io, "<td class=\"planh\" style=\"background:", _act_bg(lp), "\">", lp, "</td>")
            print(io, "<td class=\"", cev_chg[ei][k] ? "chg" : "acth", "\" style=\"background:", _act_bg(la), "\">", la, "</td>")
        end
        for mi in 1:nM
            lp = mcs_plan[mi][k]; la = mcs_act[mi][k]
            print(io, "<td class=\"planh\" style=\"background:", _act_bg(lp), "\">", lp, "</td>")
            print(io, "<td class=\"", mcs_chg[mi][k] ? "chg" : "acth", "\" style=\"background:", _act_bg(la), "\">", la, "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# Writes the plain per-interval log and the per-solve log of one run into out_dir as CSV files.
# The interval log has one row per interval, and the solve log has one row per window solve.
function _write_logs(res, out_dir)
    CSV.write(joinpath(out_dir, "A1_interval_log.csv"), res.log)
    CSV.write(joinpath(out_dir, "A1_solve_log.csv"), res.solve_log)
    return nothing
end


# Writes the report files into out_dir: the interval and solve logs, the KPI tables and charts, the re-plan grids, the plan versus realized cost comparison (Overall plus one per day for a run of several days), and the plan versus actual activity comparison (one per day).
function write_reports(res, out_dir)
    mkpath(out_dir)
    _write_logs(res, out_dir)
    _write_kpi_metrics(res, out_dir)
    _write_replan_grids(res, out_dir)
    _write_plan_vs_actual_all(res, out_dir)
    _write_plan_vs_actual_activity_all(res, out_dir)
    return nothing
end

# Writes all the figures and all the reports of one run into out_dir.
function write_outputs(res, out_dir)
    mkpath(out_dir)
    write_trajectory_figures(res, out_dir)
    write_reports(res, out_dir)
    return nothing
end

# Writes the four detailed logs of one run into out_dir as CSV files: plan_full, realized_tuple, MCS_plan_full and MCS_realized_tuple.
# A log is skipped when it was not built because detailed_output was off, and the function returns out_dir.
function write_detailed_output(res, out_dir::AbstractString)
    mkpath(out_dir)
    if res.detailed_plan_df !== nothing
        CSV.write(joinpath(out_dir, "A1_plan_full.csv"), res.detailed_plan_df)
    end
    if res.detailed_realized_df !== nothing
        CSV.write(joinpath(out_dir, "A1_realized_tuple.csv"), res.detailed_realized_df)
    end
    if res.detailed_mcs_plan_df !== nothing
        CSV.write(joinpath(out_dir, "A1_MCS_plan_full.csv"), res.detailed_mcs_plan_df)
    end
    if res.detailed_mcs_realized_df !== nothing
        CSV.write(joinpath(out_dir, "A1_MCS_realized_tuple.csv"), res.detailed_mcs_realized_df)
    end
    return out_dir
end

# Writes only the eight most important files of one run into out_dir, with no figures: the interval log, the solve log, the KPI summary, the four detailed plan and realized logs, and the plan versus realized report.
# For a run of several days it also writes the per-day KPI table.
# The four detailed logs are only available when the run was made with detailed_output on, and otherwise they are skipped.
function write_important_outputs(res, out_dir)
    mkpath(out_dir)
    _write_logs(res, out_dir)
    _write_kpi_metrics(res, out_dir; figures = false)
    _write_plan_vs_actual_all(res, out_dir; figures = false)
    write_detailed_output(res, out_dir)
    return nothing
end

end 