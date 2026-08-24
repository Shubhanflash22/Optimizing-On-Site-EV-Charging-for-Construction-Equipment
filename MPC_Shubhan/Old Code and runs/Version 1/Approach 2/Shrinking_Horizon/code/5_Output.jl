# #############################################################################
# Output.jl  —  module Output   (COMBINED plotting + reporting)
# -----------------------------------------------------------------------------
# One module that owns EVERY on-disk artefact of a run: the v4_real-style STEP
# figures (PNG) and their data (CSV), PLUS the tabular / KPI reports (cost &
# emissions, KPI metrics, per-window solver diagnostics, worker schedule, the
# detailed trajectory and the replanning grids).
#
# It merges what used to live in two files (Plotting.jl + Reporting.jl) behind a
# single entry point, `write_outputs(res, out_dir)`, called once by the main.
#
# FIGURES (built from the REALIZED closed-loop trajectory captured by MPCLoop;
# every figure is drawn with STEP helpers so nothing is a smooth line):
#   01_total_grid_power_profile   charging(+)/discharging(-) summed over MCS
#   02_work_profiles_by_site      per-site work power (one panel per site)
#   03_mcs_state_of_energy        MCS SOE with min/max guide lines
#   04_cev_state_of_energy        CEV SOE with min/max guide lines
#   05_electricity_prices_emissions  price (left) + CO2 factor (right)
#   06_mcs_location_trajectory    MCS node index over time
#   07_mcs_optimization_summary   combined multi-panel overview
#   08_kpi_metrics_summary        cost + demand-peak bar summary
#   09_mcs_<m>_power_profile      per-MCS charging/discharging
#
# REPORTS:
#   08_cost_kpi_metrics.csv          (KPI totals)
#   replan_grids/*.csv + *.html      (per-step forward plans + replanning view)
#   plan_vs_actual.html + plan_vs_actual_costs.png  (08:00 plan vs realised, financial)
#   plan_vs_actual_activity.png      (planned vs realised ACTIVITY timeline heatmap)
#   plan_vs_actual_side_by_side.html (ACTIVITY: all Planned cols, then all Actual cols)
#   plan_vs_actual_by_entity.html    (ACTIVITY grouped per unit: Planned beside Actual)
# #############################################################################
module Output

using Plots
using DataFrames
using CSV
using Printf

using ..Common: create_fixed_2hour_xticks, stepify_interval_values,
                  stepify_boundary_values, interval_time_dataframe,
                  clock_label, in_peak

export write_outputs, write_approach_comparison

gr()

const COLORS = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

# Shared plot styling to match the reference figures.
_base_plot(; kw...) = plot(; size = (900, 500), xrotation = 45,
    guidefontsize = 18, tickfontsize = 18, legendfontsize = 12,
    bottom_margin = 18Plots.mm, left_margin = 16Plots.mm, right_margin = 14Plots.mm, kw...)

# =============================================================================
# FIGURES
# =============================================================================

# 01 — total grid-side charging (+) and site-side discharging (-) over the MCSs.
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

# 02 — per-site work power (multi-panel), plus a flat overlay for the summary.
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

# Generic SOE figure (MCS or CEV) with min/max guide lines. `soe` is (unit x T).
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

fig_mcs_soe(res) = _fig_soe(res, res.d.M, res.real_SOE_MCS, res.d.SOE_MCS_max, res.d.SOE_MCS_min, "MCS")
fig_cev_soe(res) = _fig_soe(res, res.d.E, res.real_SOE_CEV, res.d.SOE_CEV_max, res.d.SOE_CEV_min, "CEV")

# 05 — electricity price (left axis) + CO2 emission factor (right axis), stepped.
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

# 06 — MCS node index over time (0 = in transit).
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

# Per-MCS charging(+)/discharging(-) profile figures + CSVs.
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

# 07 — per-interval MCS+CEV operating quantities (long-form CSV).
function csv_mcs_cev_soe(res)
    d = res.d; K = collect(1:res.nK)
    csv = DataFrame(Time_Interval = K, Time_Period = K,
                    Start_Time_Label = res.time_labels[K],
                    End_Time_Label   = res.time_labels[K .+ 1])
    for m in d.M
        csv[!, "MCS_$(m)_Charging_kW"]    = [res.real_P_ch[m, k]  for k in K]
        csv[!, "MCS_$(m)_Discharging_kW"] = [res.real_P_dch[m, k] for k in K]
        csv[!, "MCS_$(m)_Traveling_kW"]   = [res.real_L_trv[m, k] / d.delta_T for k in K]
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

# Write every trajectory figure (PNG) + its CSV into out_dir.
function write_trajectory_figures(res, out_dir)
    mkpath(out_dir)

    p01, c01 = fig_total_grid_power(res)
    savefig(p01, joinpath(out_dir, "01_total_grid_power_profile.png"))
    CSV.write(joinpath(out_dir, "01_total_grid_power_profile.csv"), c01)

    p02_multi, p02_overlay, c02 = fig_work_by_site(res)
    savefig(p02_multi, joinpath(out_dir, "02_work_profiles_by_site.png"))
    CSV.write(joinpath(out_dir, "02_work_profiles_by_site.csv"), c02)

    p03, c03 = fig_mcs_soe(res)
    savefig(p03, joinpath(out_dir, "03_mcs_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "03_mcs_state_of_energy.csv"), c03)

    p04, c04 = fig_cev_soe(res)
    savefig(p04, joinpath(out_dir, "04_cev_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "04_cev_state_of_energy.csv"), c04)

    p05, c05 = fig_price_emission(res)
    savefig(p05, joinpath(out_dir, "05_electricity_prices_emissions.png"))
    CSV.write(joinpath(out_dir, "05_electricity_prices.csv"), c05)

    p06, c06 = fig_location(res)
    savefig(p06, joinpath(out_dir, "06_mcs_location_trajectory.png"))
    CSV.write(joinpath(out_dir, "06_mcs_location_trajectory.csv"), c06)

    # 07 — combined multi-panel overview (reuse the panels above).
    summary_text = """
    Optimization Summary
    -------------------
    Number of MCSs: $(length(res.d.M))
    Number of CEVs: $(length(res.d.E))
    Number of nodes: $(length(res.d.N)) (Grid: $(length(res.d.N_g)), Construction: $(length(res.d.N_c)))
    Time interval: $(res.d.delta_T) h
    Number of intervals: $(res.nK) (full 24 h)
    """
    p_summary = plot(legend = false, grid = false, framestyle = :none, xticks = false, yticks = false,
                     left_margin = 16Plots.mm, right_margin = 14Plots.mm)
    annotate!(p_summary, 0, 0.5, text(summary_text, :black, 12, :left))
    p_combined = plot(p05, p01, p03, p02_overlay, p04, p06, p_summary,
                      layout = (4, 2), size = (1800, 2200), left_margin = 16Plots.mm)
    savefig(p_combined, joinpath(out_dir, "07_mcs_optimization_summary.png"))
    CSV.write(joinpath(out_dir, "07_mcs_cev_soe.csv"), csv_mcs_cev_soe(res))

    # 09 — per-MCS charging/discharging power profiles.
    mcs_plots, mcs_csvs = figs_individual_mcs(res)
    for (m_idx, mp) in enumerate(mcs_plots)
        savefig(mp, joinpath(out_dir, "09_mcs_$(m_idx)_power_profile.png"))
        CSV.write(joinpath(out_dir, "09_mcs_$(m_idx)_power_profile.csv"), mcs_csvs[m_idx])
    end
    return nothing
end

# =============================================================================
# REPORTS (tabular / KPI)
# =============================================================================

# The six objective cost components (same definitions as the reference).
# The six objective cost components (same definitions as the reference), PLUS
# Change 3's terminal SOE_CEV shortfall penalty (see _terminal_soe_shortfall in
# 4_MPCLoop.jl) -- a pure end-of-day check against SOE_CEV_ini, independent of
# `missed`/`missed_cost` above (which only ever reflects live physical-floor
# capping).
function _cost_components(res)
    d = res.d
    energy_cost = res.total_cost
    carbon_cost = (d.carbon_price_per_ton / 1000.0) * res.total_co2
    ncd_cost    = d.lambda_demand_NC * res.nc_peak
    opd_cost    = d.lambda_demand_OP * res.op_peak
    missed_cost = d.rho_miss * res.missed
    travel_cost = res.labour_cost
    shortfall_cost = res.shortfall_penalty_cost
    total       = energy_cost + carbon_cost + ncd_cost + opd_cost + missed_cost + travel_cost + shortfall_cost
    return (; energy_cost, carbon_cost, ncd_cost, opd_cost, missed_cost, travel_cost, shortfall_cost, total)
end

# 08 — KPI totals CSV + a two-panel bar summary (costs + operations).
function _write_kpi_metrics(res, out_dir)
    c = _cost_components(res)
    totals = DataFrame(
        Metric = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
                  "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
                  "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg",
                  "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour", "MCS_Transit_hour",
                  "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "MPC_loop_time_s"],
        Value = Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
                    round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
                    round(c.travel_cost, digits = 2), round(c.shortfall_cost, digits = 2), round(res.total_energy, digits = 2), round(res.total_co2, digits = 2),
                    round(res.nc_peak, digits = 2), round(res.op_peak, digits = 2), round(res.missed, digits = 2),
                    round(res.transit_intervals * res.d.delta_T, digits = 2),
                    round(res.shortfall_kWh, digits = 3),
                    res.n_infeasible, round(res.elapsed, digits = 2)])
    CSV.write(joinpath(out_dir, "08_cost_kpi_metrics.csv"), totals)

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
    savefig(p_summary, joinpath(out_dir, "08_kpi_metrics_summary.png"))
end

# ---- replanning-grid cell formatting + CSV/HTML writers ---------------------
_cell(v::AbstractString) = v
_cell(v::Real) = isnan(v) ? "" : round(v, digits = 3)

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

# CHANGE 5 FIX -- res.plan_grid_kW etc no longer exist as flat fields (they were
# silently overwritten every day for n_day_run > 1); run_mpc now returns
# res.replan_by_day[day], one full set of DAY-LOCAL grids per day, mirroring
# Receding_Horizon's established multi-day pattern exactly. Writes one grid
# folder per KEPT day.
function _write_replan_grids(res, out_dir)
    d = res.d; nKd = res.nKd
    for day in 1:res.n_day_run
        g = res.replan_by_day[day]
        gdir = res.n_day_run == 1 ? joinpath(out_dir, "replan_grids") :
                                     joinpath(out_dir, "replan_grids", "day$(day)")
        mkpath(gdir)
        _write_replan_grid(joinpath(gdir, "plan_grid_kW.csv"), g.plan_grid_kW, res, nKd)
        _write_replan_grid(joinpath(gdir, "plan_mcs_soe.csv"), g.plan_mcs_soe, res, nKd)
        _write_replan_grid(joinpath(gdir, "plan_mcs_activity.csv"), g.plan_mcs_act, res, nKd)
        for e in d.E
            _write_replan_grid(joinpath(gdir, "plan_cev$(e)_soe.csv"),      g.plan_cev_soe[e], res, nKd)
            _write_replan_grid(joinpath(gdir, "plan_cev$(e)_activity.csv"), g.plan_cev_act[e], res, nKd)
        end
    end
end

# =============================================================================
# PLAN-vs-ACTUAL: the FIRST optimisation (08:00, the all-"pending"/yellow row of
# the replan grid) vs the REALISED day (what actually happened, the "done"/green
# diagonal). The 08:00 plan is the MILP's whole-day forecast made before any
# stochastic-plant disturbance; the realised trajectory is the closed-loop result
# after re-planning every 15 min. This report quantifies the gap financially.
# =============================================================================

# First replan step that actually produced a forward plan (row 1 unless step 1
# was infeasible, in which case the earliest feasible row).
function _first_plan_row(res)
    g1 = res.replan_by_day[1]
    for r in 1:res.nKd
        any(!isnan(g1.plan_grid_kW[r, k]) for k in r:res.nKd) && return r
    end
    return 1
end

# Financial + operational KPIs implied by the 08:00 whole-day plan (row r of the
# replan grids). Grid draw, price, carbon and peaks come from the planned grid
# power; travel labour from the planned MCS status; missed work from the planned
# per-CEV activity vs each site's requirement.
function _planned_kpis(res)
    d = res.d; nKd = res.nKd; dt = d.delta_T
    r = _first_plan_row(res)
    g1 = res.replan_by_day[1]
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
    transit = count(k -> g1.plan_mcs_act[r, k] == "Traveling", 1:nKd)
    labour  = d.rho_labor * dt * transit
    # planned missed work from the per-CEV activity grid (row r)
    pdig = zeros(length(d.N)); pload = zeros(length(d.N))
    for e in d.E
        site = findfirst(i -> d.A[i, e] == 1, d.N); site === nothing && continue
        for k in 1:nKd
            lab = g1.plan_cev_act[e][r, k]
            lab == res.ACT_NAME[1] && (pdig[site]  += dt)   # Digging
            lab == res.ACT_NAME[2] && (pload[site] += dt)   # Loading/Swinging
        end
    end
    missed = sum((max(d.hours_digging[i]          - pdig[i],  0.0) for i in d.N_c); init = 0.0) +
             sum((max(d.hours_loading_swinging[i] - pload[i], 0.0) for i in d.N_c); init = 0.0)
    missed_cost = d.rho_miss * missed
    total = ecost + carbon + ncd + opd + missed_cost + labour
    return (; r, g, energy, ecost, co2kg, carbon, ncpk, oppk, ncd, opd,
              transit, labour, missed, missed_cost, total)
end

function _write_plan_vs_actual(res, out_dir)
    d = res.d; nKd = res.nKd; dt = d.delta_T
    p = _planned_kpis(res)
    c = _cost_components(res)
    r = p.r
    plan_clock = clock_label(d.t_start, d.delta_T, r)

    # ---- (a) headline summary table: planned@08:00 (day 1) vs realised (whole
    # run) -- matches Receding_Horizon's own established convention of comparing
    # day 1's plan against the run's overall realized totals ----
    rows = [
        ("Grid energy (kWh)",         p.energy,          res.total_energy),
        ("Energy cost (USD)",         p.ecost,           c.energy_cost),
        ("CO2 emissions (kg)",        p.co2kg,           res.total_co2),
        ("CO2 cost (USD)",            p.carbon,          c.carbon_cost),
        ("NCD peak (kW)",             p.ncpk,            res.nc_peak),
        ("NCD charge (USD)",          p.ncd,             c.ncd_cost),
        ("OPD peak (kW)",             p.oppk,            res.op_peak),
        ("OPD charge (USD)",          p.opd,             c.opd_cost),
        ("Missed work (h)",           p.missed,          res.missed),
        ("Missed work penalty (USD)", p.missed_cost,     c.missed_cost),
        ("MCS transit (h)",           p.transit * dt,    res.transit_intervals * dt),
        ("Travel labour (USD)",       p.labour,          c.travel_cost),
        ("TOTAL cost (USD)",          p.total,           c.total),
    ]
    summ = DataFrame(
        Metric              = [x[1] for x in rows],
        Planned_at_start    = [round(x[2], digits = 3) for x in rows],
        Realized_end_of_day = [round(x[3], digits = 3) for x in rows],
        Delta_real_minus_plan = [round(x[3] - x[2], digits = 3) for x in rows],
        Pct_change = [abs(x[2]) < 1e-9 ? (abs(x[3]) < 1e-9 ? 0.0 : NaN) :
                      round(100 * (x[3] - x[2]) / x[2], digits = 1) for x in rows])

    # ---- (b) per-interval planned (day 1) vs realised (day 1's slice, since
    # real_P_ch is globally indexed and day 1 occupies columns 1:nKd) ----
    realized_g = [sum(res.real_P_ch[m, k] for m in d.M) for k in 1:nKd]
    byint = DataFrame(
        k = collect(1:nKd),
        clock = [clock_label(d.t_start, d.delta_T, k) for k in 1:nKd],
        price = [d.lambda_whl_elec[k] for k in 1:nKd],
        co2_factor = [d.lambda_CO2[k] for k in 1:nKd],
        on_peak = [in_peak(k, dt, d.t_start) ? "Yes" : "No" for k in 1:nKd],
        planned_grid_kW  = round.(p.g, digits = 3),
        realized_grid_kW = round.(realized_g, digits = 3),
        delta_kW = round.(realized_g .- p.g, digits = 3))

    # ---- (c) grouped-bar PNG of the cost components ----
    labels = ["Energy", "CO₂", "NCD", "OPD", "Missed", "Labour", "TOTAL"]
    planned = [p.ecost, p.carbon, p.ncd, p.opd, p.missed_cost, p.labour, p.total]
    realized = [c.energy_cost, c.carbon_cost, c.ncd_cost, c.opd_cost, c.missed_cost, c.travel_cost, c.total]
    ymax = max(maximum(planned), maximum(realized), 1.0)
    pbar = plot(title = "Planned @ $plan_clock vs Realised — cost components",
                xlabel = "", ylabel = "Cost (USD)", titlefontsize = 14,
                xticks = (1:length(labels), labels), xlims = (0.5, length(labels) + 0.5),
                ylims = (0, 1.25 * ymax), xrotation = 20, legend = :topleft,
                guidefontsize = 14, tickfontsize = 12, size = (1150, 500),
                bottom_margin = 12Plots.mm, left_margin = 14Plots.mm)
    for i in eachindex(labels)
        bar!(pbar, [i - 0.19], [planned[i]],  bar_width = 0.36, color = :goldenrod,
             label = i == 1 ? "Planned @ $plan_clock" : "")
        bar!(pbar, [i + 0.19], [realized[i]], bar_width = 0.36, color = :forestgreen,
             label = i == 1 ? "Realised" : "")
    end
    savefig(pbar, joinpath(out_dir, "plan_vs_actual_costs.png"))

    # ---- (d) HTML: the full "yellow" (08:00 plan) row vs the full "green"
    #          (realised) row of grid power, plus the summary table ----
    _write_plan_vs_actual_html(joinpath(out_dir, "plan_vs_actual.html"),
                               res, p, summ, byint, plan_clock)
    return nothing
end

function _write_plan_vs_actual_html(path, res, p, summ, byint, plan_clock)
    d = res.d; nK = res.nKd   # byint/p are DAY-1-SCOPED (nKd rows), not global
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "table{border-collapse:collapse;font-size:12px;margin-bottom:20px}")
    println(io, "th,td{border:1px solid #ccc;padding:3px 8px;text-align:center;white-space:nowrap}")
    println(io, "th{background:#f4f4f4}")
    println(io, ".plan{background:#ffeb9c}")   # yellow = the 08:00 forward plan (all pending)
    println(io, ".real{background:#c6efce}")   # green  = what was actually realised
    println(io, ".pos{color:#b00}.neg{color:#070}")
    println(io, "</style></head><body>")
    println(io, "<h2>Plan (@ $plan_clock) vs Realised — full-day comparison</h2>")
    println(io, "<p><b>Yellow</b> = the FIRST optimisation made at $plan_clock (the whole-day forward plan, ",
                "every interval still \"pending\").&nbsp;&nbsp;<b>Green</b> = the REALISED trajectory ",
                "after closed-loop re-planning against the stochastic plant.</p>")

    # summary table
    println(io, "<h3>Financial &amp; operational summary</h3><table>")
    println(io, "<tr><th>Metric</th><th class=\"plan\">Planned @ $plan_clock</th>",
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

    # the two rows of grid power over the day
    println(io, "<h3>Grid power (kW) per 15-min interval</h3><table>")
    print(io, "<tr><th>interval &rarr;</th>")
    for k in 1:nK; print(io, "<th>", clock_label(d.t_start, d.delta_T, k), "</th>"); end
    println(io, "</tr>")
    print(io, "<tr><th class=\"plan\">Planned @ $plan_clock</th>")
    for k in 1:nK; print(io, "<td class=\"plan\">", round(byint.planned_grid_kW[k], digits = 1), "</td>"); end
    println(io, "</tr>")
    print(io, "<tr><th class=\"real\">Realised</th>")
    for k in 1:nK; print(io, "<td class=\"real\">", round(byint.realized_grid_kW[k], digits = 1), "</td>"); end
    println(io, "</tr>")
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# =============================================================================
# PLAN-vs-ACTUAL: ACTIVITY. For every 15-min interval the FIRST whole-day plan
# (@ 08:00) assigned an activity to each CEV and the MCS; the closed loop then
# executed a possibly different activity each step. Two artefacts are emitted:
#   * plan_vs_actual_activity.png       — timeline heatmap (one Planned/Actual
#                                         band per CEV + MCS; changed intervals
#                                         boxed in red).
#   * plan_vs_actual_side_by_side.html  — the same laid out as a side-by-side
#                                         table (Planned block | Actual block),
#                                         with changed ACTUAL cells outlined red.
# =============================================================================

# Activity label -> integer code (0 = none). Colours indexed by code+1 in two
# parallel palettes: named symbols for Plots (robust in cgrad) + hex for HTML.
_act_code(l) = l == "Idle" ? 1 : l == "Digging" ? 2 : l == "Loading/Swinging" ? 3 :
               l == "Traveling" ? 4 : l == "Charging" ? 5 : l == "Charging (grid)" ? 6 :
               l == "Serving CEV" ? 7 : 0
const _ACT_COLORS_SYM = [:white, :gray85, :lightskyblue, :darkseagreen, :sandybrown,
                         :khaki, :goldenrod, :mediumpurple]
const _ACT_COLORS_HEX = ["#ffffff", "#e8e8e8", "#9ecae1", "#a1d99b", "#fdae6b",
                         "#fee391", "#f6c744", "#bcbddc"]
const _ACT_NAMES = ["Idle", "Digging", "Loading/Swinging", "Traveling",
                    "Charging", "Charging (grid)", "Serving CEV"]
_act_short(l) = l == "Digging" ? "D" : l == "Loading/Swinging" ? "L" : l == "Traveling" ? "T" :
                l == "Idle" ? "I" : l == "Charging" ? "C" : l == "Charging (grid)" ? "Cg" :
                l == "Serving CEV" ? "S" : ""
_act_bg(l) = _ACT_COLORS_HEX[_act_code(l) + 1]

# One heatmap panel (2 rows: Actual on top-index 1, Planned on 2) for one entity.
function _activity_panel(res, planned, actual, plan_lbl, title)
    d = res.d; nK = res.nKd   # planned/actual are DAY-1-SCOPED (length nKd), not global
    rT, rL = create_fixed_2hour_xticks(1:(nK + 1), d.t_start)
    cp = [_act_code(planned[k]) for k in 1:nK]
    ca = [_act_code(actual[k])  for k in 1:nK]
    Z  = [reshape(ca, 1, nK); reshape(cp, 1, nK)]   # row 1 = Actual, row 2 = Planned
    p = heatmap(1:nK, 1:2, Z; color = cgrad(_ACT_COLORS_SYM, categorical = true),
                clims = (-0.5, 7.5), colorbar = false, title = title, titlefontsize = 13,
                yticks = ([1, 2], ["Actual", plan_lbl]), xticks = (rT, rL),
                xlims = (0.5, nK + 0.5), ylims = (0.5, 2.5), xrotation = 45,
                tickfontsize = 10, legend = false,
                left_margin = 26Plots.mm, right_margin = 6Plots.mm, bottom_margin = 10Plots.mm)
    changed = [k for k in 1:nK if cp[k] != ca[k]]
    for k in changed   # outline every changed interval in red
        plot!(p, Shape([k - 0.5, k + 0.5, k + 0.5, k - 0.5], [0.5, 0.5, 2.5, 2.5]);
              fillalpha = 0.0, linecolor = :red, linewidth = 2, label = "")
    end
    return p, length(changed)
end

# A colour-swatch legend rendered as its own (frameless) panel.
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

function _write_plan_vs_actual_side_by_side(res, out_dir)
    d = res.d; nKd = res.nKd
    r = _first_plan_row(res)
    plan_clk = clock_label(d.t_start, d.delta_T, r)
    plan_lbl = "Planned @ $plan_clk"
    times = [clock_label(d.t_start, d.delta_T, k) for k in 1:nKd]

    # PLANNED = day 1's 08:00 forward plan (row r). ACTUAL = day 1's applied
    # diagonal -- real_cev_act/real_mcs_act are globally indexed, and day 1
    # occupies indices 1:nKd, so slicing to that range picks out day 1's actual.
    g1 = res.replan_by_day[1]
    cev_plan = [[g1.plan_cev_act[e][r, k] for k in 1:nKd] for e in d.E]
    cev_act  = [res.real_cev_act[e][1:nKd]                for e in d.E]
    mcs_plan = [g1.plan_mcs_act[r, k] for k in 1:nKd]
    mcs_act  = res.real_mcs_act[1:nKd]

    # ---- (a) PNG timeline heatmap (one Planned/Actual band per CEV + MCS + legend) ----
    entities = Any[]
    for (ei, e) in enumerate(d.E)
        push!(entities, ("CEV $e", cev_plan[ei], cev_act[ei]))
    end
    push!(entities, ("MCS", mcs_plan, mcs_act))
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
                    plot_title = "Plan (@ $plan_clk) vs Actual activity  —  red = changed intervals",
                    plot_titlefontsize = 15)
    savefig(combined, joinpath(out_dir, "plan_vs_actual_activity.png"))

    # ---- (b) HTML, two layouts ----
    #   side_by_side : all Planned columns, then all Actual columns.
    _write_side_by_side_html(joinpath(out_dir, "plan_vs_actual_side_by_side.html"),
                             res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    #   by_entity    : per unit, Planned column immediately beside its Actual column.
    _write_by_entity_html(joinpath(out_dir, "plan_vs_actual_by_entity.html"),
                          res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    return nothing
end

function _write_side_by_side_html(path, res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    d = res.d; nK = res.nKd; nE = length(d.E)   # inputs are DAY-1-SCOPED, not global
    cev_chg = [cev_plan[ei] .!= cev_act[ei] for ei in 1:nE]
    mcs_chg = mcs_plan .!= mcs_act
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

    # activity legend
    print(io, "<p><b>Activity:</b>&nbsp;&nbsp;")
    for c in eachindex(_ACT_NAMES)
        print(io, "<span class=\"sw\" style=\"background:", _ACT_COLORS_HEX[c + 1], "\">",
              _act_short(_ACT_NAMES[c]), "</span> ", _ACT_NAMES[c], "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    # change counts
    print(io, "<p><b>Changed intervals:</b>&nbsp;&nbsp;")
    for (ei, e) in enumerate(d.E)
        print(io, "CEV $e = ", count(cev_chg[ei]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "MCS = ", count(mcs_chg), "/", nK, "</p>")

    # table: Time | <planned block> | <actual block>
    println(io, "<table>")
    print(io, "<tr><th rowspan=\"2\">Time</th>",
              "<th colspan=\"", nE + 1, "\" class=\"planh\">Planned @ $plan_clk</th>",
              "<th colspan=\"", nE + 1, "\" class=\"acth\">Actual</th></tr>")
    print(io, "<tr>")
    for e in d.E; print(io, "<th class=\"planh\">CEV $e</th>"); end
    print(io, "<th class=\"planh\">MCS</th>")
    for e in d.E; print(io, "<th class=\"acth\">CEV $e</th>"); end
    println(io, "<th class=\"acth\">MCS</th></tr>")
    for k in 1:nK
        print(io, "<tr><th>", times[k], "</th>")
        for ei in 1:nE
            l = cev_plan[ei][k]; print(io, "<td style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        print(io, "<td style=\"background:", _act_bg(mcs_plan[k]), "\">", mcs_plan[k], "</td>")
        for ei in 1:nE
            l = cev_act[ei][k]
            print(io, "<td class=\"", cev_chg[ei][k] ? "chg" : "", "\" style=\"background:", _act_bg(l), "\">", l, "</td>")
        end
        print(io, "<td class=\"", mcs_chg[k] ? "chg" : "", "\" style=\"background:", _act_bg(mcs_act[k]), "\">", mcs_act[k], "</td>")
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# Same data as side_by_side, but grouped per unit: each CEV/MCS shows its
# "Planned @ plan_clk" column immediately next to its "Actual" column.
function _write_by_entity_html(path, res, times, cev_plan, cev_act, mcs_plan, mcs_act, plan_clk)
    d = res.d; nK = res.nKd; nE = length(d.E)   # inputs are DAY-1-SCOPED, not global
    cev_chg = [cev_plan[ei] .!= cev_act[ei] for ei in 1:nE]
    mcs_chg = mcs_plan .!= mcs_act
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
    println(io, "<p>For each CEV and the MCS, the <b>Planned @ $plan_clk</b> column (the first whole-day plan) ",
                "sits next to the <b>Actual</b> column (what the closed loop executed). ",
                "<b>Actual</b> cells <span class=\"chg\">outlined in red</span> differ from the plan.</p>")

    # activity legend
    print(io, "<p><b>Activity:</b>&nbsp;&nbsp;")
    for c in eachindex(_ACT_NAMES)
        print(io, "<span class=\"sw\" style=\"background:", _ACT_COLORS_HEX[c + 1], "\">",
              _act_short(_ACT_NAMES[c]), "</span> ", _ACT_NAMES[c], "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "</p>")

    # change counts
    print(io, "<p><b>Changed intervals:</b>&nbsp;&nbsp;")
    for (ei, e) in enumerate(d.E)
        print(io, "CEV $e = ", count(cev_chg[ei]), "/", nK, "&nbsp;&nbsp;&nbsp;")
    end
    println(io, "MCS = ", count(mcs_chg), "/", nK, "</p>")

    # table: Time | (Planned CEVe | Actual CEVe).. | Planned MCS | Actual MCS
    println(io, "<table>")
    print(io, "<tr><th rowspan=\"2\">Time</th>")
    for e in d.E; print(io, "<th colspan=\"2\">CEV $e</th>"); end
    println(io, "<th colspan=\"2\">MCS</th></tr>")
    print(io, "<tr>")
    for _ in d.E
        print(io, "<th class=\"planh\">Planned @ $plan_clk</th><th class=\"acth\">Actual</th>")
    end
    println(io, "<th class=\"planh\">Planned @ $plan_clk</th><th class=\"acth\">Actual</th></tr>")
    for k in 1:nK
        print(io, "<tr><th>", times[k], "</th>")
        for ei in 1:nE
            lp = cev_plan[ei][k]; la = cev_act[ei][k]
            print(io, "<td class=\"planh\" style=\"background:", _act_bg(lp), "\">", lp, "</td>")
            print(io, "<td class=\"", cev_chg[ei][k] ? "chg" : "acth", "\" style=\"background:", _act_bg(la), "\">", la, "</td>")
        end
        print(io, "<td class=\"planh\" style=\"background:", _act_bg(mcs_plan[k]), "\">", mcs_plan[k], "</td>")
        print(io, "<td class=\"", mcs_chg[k] ? "chg" : "acth", "\" style=\"background:", _act_bg(mcs_act[k]), "\">", mcs_act[k], "</td>")
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

# =============================================================================
# 10 — SIDE-BY-SIDE TIMELINE COMPARISON (Approach 0 vs Approach 2), styled after
# the reference 3-row figure:
#   (a) MCS power     — charging(+)/discharging(-), with NCDP & OPDP annotated
#   (b) CEV state of energy — one line per CEV, dashed min/max guide lines
#   (c) CEV work power — bars coloured by activity (Digging / Traveling /
#                         Loading+Swinging), zero-power intervals left blank
# The on-peak window is shaded gray in every panel. Two columns: Approach 0
# (one-shot, left) and Approach 2 (stochastic closed-loop, right), sharing the row axes.
# =============================================================================

const _WORK_ACT_COLORS = Dict(
    "Digging"          => :steelblue,
    "Traveling"        => :sandybrown,
    "Loading/Swinging" => :mediumseagreen,
)

# Shade the on-peak window(s) behind whatever is already on the panel. Must be
# called AFTER ylims are fixed on `p` so the rectangle spans the full height.
function _shade_on_peak!(p, res)
    d = res.d; nK = res.nK; dt = d.delta_T
    peak_idx = [k for k in 1:nK if in_peak(k, dt, d.t_start)]
    isempty(peak_idx) && return p
    ylo, yhi = Plots.ylims(p)
    lo, hi = minimum(peak_idx), maximum(peak_idx) + 1
    plot!(p, Shape([lo, hi, hi, lo], [ylo, yhi, yhi, ylo]);
          fillalpha = 0.15, fillcolor = :gray, linealpha = 0, label = "")
    ylims!(p, (ylo, yhi))
    return p
end

# (a) MCS power: charging(+)/discharging(-), NCDP & OPDP annotated as text +
# dotted guide lines (the peaks res.nc_peak / res.op_peak are already the
# realised non-coincident / on-peak demand peaks for this run).
function _panel_mcs_power(res, title)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    charging    = [sum(res.real_P_ch[m, k]  for m in d.M) for k in K]
    discharging = [sum(res.real_P_dch[m, k] for m in d.M) for k in K]
    ymax = max(maximum(charging; init = 0.0), maximum(discharging; init = 0.0),
               res.nc_peak, res.op_peak, 1.0)
    p = _base_plot(title = title, titlefontsize = 16, xlabel = "Time of day",
                   ylabel = "MCS power (kW)", xticks = (rT, rL),
                   xlims = (first(Tplot), last(Tplot)), ylims = (-1.2 * ymax, 1.35 * ymax),
                   legend = :bottomright, legendfontsize = 9)
    _shade_on_peak!(p, res)
    xc, yc = stepify_interval_values(K, charging)
    xd, yd = stepify_interval_values(K, -discharging)
    plot!(p, xc, yc, label = "MCS charge", color = :darkorange, linewidth = 2)
    plot!(p, xd, yd, label = "MCS discharge", color = :magenta, linewidth = 2)
    hline!(p, [res.nc_peak], color = :darkorange, linestyle = :dot, linewidth = 2, label = "NCDP")
    hline!(p, [res.op_peak], color = :firebrick,  linestyle = :dot, linewidth = 2, label = "OPDP")
    total_grid_kwh = sum(charging) * d.delta_T
    xa = first(Tplot) + 0.62 * (last(Tplot) - first(Tplot))
    annotate!(p, xa, 1.18 * ymax,
              text("NCDP $(round(res.nc_peak, digits = 2)) kW", :darkorange, 11, :left))
    annotate!(p, xa, 1.00 * ymax,
              text("OPDP $(round(res.op_peak, digits = 2)) kW", :firebrick, 11, :left))
    annotate!(p, xa, 0.82 * ymax,
              text("Total grid charging = $(round(total_grid_kwh, digits = 2)) kWh", :black, 11, :left))
    return p
end

# (b) CEV state of energy — one line per CEV, dashed min/max guide lines.
function _panel_cev_soe(res, title)
    d = res.d; T = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(T, d.t_start)
    ymax = maximum(values(d.SOE_CEV_max); init = 1.0)
    p = _base_plot(title = title, titlefontsize = 16, xlabel = "Time of day",
                   ylabel = "CEV state of energy (kWh)", xticks = (rT, rL),
                   xlims = (first(T), last(T)), ylims = (0, 1.1 * ymax),
                   legend = :bottomright, legendfontsize = 9)
    _shade_on_peak!(p, res)
    for (idx, e) in enumerate(d.E)
        vals = [res.real_SOE_CEV[e, t] for t in T]
        plot!(p, collect(T), vals, label = "CEV $e", color = COLORS[mod1(idx, length(COLORS))], linewidth = 2)
    end
    hline!(p, [d.SOE_CEV_max[e] for e in d.E], color = :black, linestyle = :dash, linewidth = 1, label = "SOE limits")
    hline!(p, [d.SOE_CEV_min[e] for e in d.E], color = :black, linestyle = :dash, linewidth = 1, label = "")
    return p
end

# Realised missed work, split by activity, converted from hours to kWh using
# each activity's power rate (same site/activity accounting as _planned_kpis,
# but read off the REALISED per-CEV activity trace instead of the 08:00 plan).
function _realised_missed_kwh(res)
    d = res.d; nK = res.nK; dt = d.delta_T
    rdig = zeros(length(d.N)); rload = zeros(length(d.N))
    for e in d.E
        site = findfirst(i -> d.A[i, e] == 1, d.N); site === nothing && continue
        for k in 1:nK
            lab = res.real_cev_act[e][k]
            lab == res.ACT_NAME[1] && (rdig[site]  += dt)   # Digging
            lab == res.ACT_NAME[2] && (rload[site] += dt)   # Loading/Swinging
        end
    end
    # Total REQUIRED hours across the whole run is n_day_run copies of one day's
    # requirement (CHANGE 5 -- same work every day), matching
    # _terminal_soe_shortfall's identical fix in 4_MPCLoop.jl.
    miss_dig_h  = sum((max(res.n_day_run * d.hours_digging[i]          - rdig[i],  0.0) for i in d.N_c); init = 0.0)
    miss_load_h = sum((max(res.n_day_run * d.hours_loading_swinging[i] - rload[i], 0.0) for i in d.N_c); init = 0.0)
    return miss_dig_h * d.p_digging, miss_load_h * d.p_loading_swinging
end

# (c) CEV work power — bars coloured by activity (Digging / Traveling /
# Loading+Swinging); zero-power intervals are left blank. When several CEVs
# are present, each gets its own narrower bar within the interval so they
# don't overlap.
function _panel_cev_work(res, title)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    nE = max(length(d.E), 1)
    work = Dict(e => [sum(res.real_P_work[i, e, k] for i in d.N_c) for k in K] for e in d.E)
    ymax = maximum(vcat(0.0, values(work)...); init = 0.0)
    ymax = ymax > 0 ? ymax : 1.0
    p = _base_plot(title = title, titlefontsize = 16, xlabel = "Time of day",
                   ylabel = "CEV work power (kW)", xticks = (rT, rL),
                   xlims = (first(Tplot), last(Tplot)), ylims = (0, 1.3 * ymax),
                   legend = :topright, legendfontsize = 9)
    _shade_on_peak!(p, res)
    seen = Set{String}()
    bw = 0.85 / nE
    for (idx, e) in enumerate(d.E)
        for k in K
            pw = work[e][k]
            pw <= 1e-9 && continue
            lab = res.real_cev_act[e][k]
            col = get(_WORK_ACT_COLORS, lab, :gray)
            lbl = lab in seen ? "" : lab
            push!(seen, lab)
            x0 = k - 0.85 / 2 + (idx - 1) * bw
            bar!(p, [x0 + bw / 2], [pw]; bar_width = bw, color = col, linecolor = col, label = lbl)
        end
    end
    miss_dig_kwh, miss_load_kwh = _realised_missed_kwh(res)
    miss_txt = (miss_dig_kwh <= 1e-9 && miss_load_kwh <= 1e-9) ?
        "Missed work = 0 kWh" :
        "Missed work = " * join(
            filter(!isempty, [
                miss_dig_kwh  > 1e-9 ? "$(round(miss_dig_kwh,  digits = 2)) kWh digging" : "",
                miss_load_kwh > 1e-9 ? "$(round(miss_load_kwh, digits = 2)) kWh loading+swinging" : "",
            ]), " + ")
    xa = first(Tplot) + 0.02 * (last(Tplot) - first(Tplot))
    annotate!(p, xa, 1.22 * ymax, text(miss_txt, :firebrick, 11, :left))
    return p
end

# Build the full 3-row x 2-col comparison figure: Approach 0 (left column) vs
# Approach 2 (right column), rows = MCS power / CEV SOE / CEV work power.
function fig_approach_timeline_comparison(res0, res1)
    p_a0 = _panel_mcs_power(res0, "Approach 0 (one-shot)")
    p_a1 = _panel_mcs_power(res1, "Approach 2 (stochastic, closed-loop)")
    p_b0 = _panel_cev_soe(res0, "")
    p_b1 = _panel_cev_soe(res1, "")
    p_c0 = _panel_cev_work(res0, "")
    p_c1 = _panel_cev_work(res1, "")
    combined = plot(p_a0, p_a1, p_b0, p_b1, p_c0, p_c1, layout = (3, 2),
                    size = (1700, 1500), left_margin = 16Plots.mm, bottom_margin = 16Plots.mm)
    return combined
end

# =============================================================================
# APPROACH 0 (one-shot 8:00 plan, executed open-loop) vs APPROACH 2 (existing
# stochastic scenario-based closed-loop MPC). Unlike plan_vs_actual.html (which
# compares the DETERMINISTIC 8:00 forecast to Approach 2's realised trajectory),
# both columns here are FULLY REALISED outcomes over 08:00 -> next-day 08:00,
# built from the SAME shared ActivityPowerPool, so the gap reflects only the
# value of re-planning.
# Additive-only: does not modify or replace any existing report.
# =============================================================================
# Every headline KPI of ONE realized run, as an ordered (label, value) list. Used to
# build each column of the comparison table, so adding a metric adds it everywhere.
function _approach_kpi_rows(res)
    c = _cost_components(res)
    return [
        ("Grid energy (kWh)",         res.total_energy),
        ("Energy cost (USD)",         c.energy_cost),
        ("CO2 emissions (kg)",        res.total_co2),
        ("CO2 cost (USD)",            c.carbon_cost),
        ("NCD peak (kW)",             res.nc_peak),
        ("NCD charge (USD)",          c.ncd_cost),
        ("OPD peak (kW)",             res.op_peak),
        ("OPD charge (USD)",          c.opd_cost),
        ("Missed work (h)",           res.missed),
        ("Missed work penalty (USD)", c.missed_cost),
        ("Terminal SOE shortfall (kWh)",       res.shortfall_kWh),
        ("Terminal shortfall penalty (USD)",   c.shortfall_cost),
        ("MCS transit (h)",           res.transit_intervals * res.d.delta_T),
        ("Travel labour (USD)",       c.travel_cost),
        ("TOTAL cost (USD)",          c.total),
    ]
end

# res0 : Approach 0 -- the whole-day MILP solved ONCE at 08:00 and replayed open-loop.
#        Which baseline it is depends on the plant mode it was run under (res0.plant):
#          :mean     realized == planned exactly, so the column IS the whole-day MILP's
#                    own optimum (deterministic reference, nothing sampled anywhere).
#          :sampled  the same fixed plan drifting under the stochastic pool with no
#                    feedback -- what re-planning is being credited against.
#        Exactly one of the two runs per call; the header and the delta label below
#        adapt so the report always says which comparison it is actually showing.
# res1 : Approach 2, the stochastic closed loop (always :sampled).
function write_approach_comparison(res0, res1, out_dir)
    mkpath(out_dir)

    # 10 — side-by-side timeline figure (MCS power / CEV SOE / CEV work power)
    p_timeline = fig_approach_timeline_comparison(res0, res1)
    savefig(p_timeline, joinpath(out_dir, "10_approach0_vs_approach1_timeline.png"))
    CSV.write(joinpath(out_dir, "10_approach0_timeline.csv"), csv_mcs_cev_soe(res0))
    CSV.write(joinpath(out_dir, "10_approach1_timeline.csv"), csv_mcs_cev_soe(res1))

    r0 = _approach_kpi_rows(res0)
    r1 = _approach_kpi_rows(res1)
    is_mean = res0.plant === :mean
    a0_head = is_mean ? "Approach 0 &mdash; mean plant<br>(whole-day MILP optimum)" :
                        "Approach 0 &mdash; sampled plant<br>(one-shot, open loop)"
    blurb = is_mean ?
        string("<b>Approach 0</b> was replayed under the <b>deterministic</b> plant: realized power ",
               "is pinned to the planning mean &mu; and each interval realizes its single planned ",
               "activity in full, so realized equals planned exactly and the column is the whole-day ",
               "MILP's own optimum. <b>Approach 2</b> re-plans every 15 min, hedging across sampled ",
               "pool. The &Delta; therefore mixes TWO effects &mdash; the drift the stochastic plant ",
               "introduces, and the value of re-planning against it. Re-run with ",
               "<code>approach0_plant = :sampled</code> to isolate the second.") :
        string("Both columns are fully realized over 08:00 &rarr; next-day 08:00 and draw from the ",
               "SAME pre-generated per-(excavator, activity) power samples, so the &Delta; reflects ",
               "only the value of re-planning, not different randomness. Re-run with ",
               "<code>approach0_plant = :mean</code> to see the deterministic MILP optimum instead.")

    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif}")
    println(io, "table{border-collapse:collapse;font-size:13px}")
    println(io, "th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}")
    println(io, "th{background:#f4f4f4}")
    println(io, "td:first-child,th:first-child{text-align:left}")
    println(io, "tr:last-child td{font-weight:bold;background:#fafafa}")
    println(io, "</style></head><body>")
    println(io, "<h2>Approach 0 (one-shot 8:00 plan) vs Approach 2 (stochastic closed-loop MPC)</h2>")
    println(io, "<p>", blurb, "</p>")
    println(io, "<table><tr><th>Metric</th><th>", a0_head,
                "</th><th>Approach 2 &mdash; closed loop</th><th>&Delta; (A2 &minus; A0)</th></tr>")
    for idx in eachindex(r0)
        name = r0[idx][1]; v0 = r0[idx][2]; v1 = r1[idx][2]
        println(io, "<tr><td>", name, "</td><td>", @sprintf("%.3f", v0), "</td><td>",
                    @sprintf("%.3f", v1), "</td><td>", @sprintf("%+.3f", v1 - v0), "</td></tr>")
    end
    println(io, "</table>")
    # Diagnostics that qualify the numbers above rather than being costs themselves.
    println(io, "<h3>Run diagnostics</h3><table><tr><th>Run</th><th>Plant</th>",
                "<th>Infeasible windows</th><th>Energy caps</th><th>Solve time (s)</th></tr>")
    for (lbl, r) in [("Approach 0 (one-shot)", res0), ("Approach 2 (closed loop, stochastic)", res1)]
        println(io, "<tr><td>", lbl, "</td><td>:", r.plant, "</td><td>",
                    r.n_infeasible, "</td><td>", r.n_capped, "</td><td>",
                    @sprintf("%.1f", r.elapsed), "</td></tr>")
    end
    println(io, "</table></body></html>")
    write(joinpath(out_dir, "approach0_vs_approach1.html"), String(take!(io)))
    return nothing
end

function write_reports(res, out_dir)
    mkpath(out_dir)
    _write_kpi_metrics(res, out_dir)
    _write_replan_grids(res, out_dir)
    _write_plan_vs_actual(res, out_dir)
    _write_plan_vs_actual_side_by_side(res, out_dir)
    return nothing
end

# =============================================================================
# SINGLE ENTRY POINT: write the FULL figure + report set into out_dir.
# =============================================================================
function write_outputs(res, out_dir)
    mkpath(out_dir)
    write_trajectory_figures(res, out_dir)
    write_reports(res, out_dir)
    return nothing
end

end # module Output
