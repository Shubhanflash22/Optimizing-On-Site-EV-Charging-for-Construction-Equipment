# #############################################################################
# Output.jl  —  module Output   (COMBINED plotting + reporting, MULTI-DAY)
# -----------------------------------------------------------------------------
# One module that owns EVERY on-disk artefact of a run: the v4_real-style STEP
# figures (PNG) and their data (CSV), PLUS the tabular / KPI reports (cost &
# emissions, KPI metrics, per-window solver diagnostics, worker schedule, the
# detailed trajectory, overnight charge tables and the per-day replanning grids).
#
# Everything is rendered over the KEPT concatenated multi-day horizon (the buffer
# day is already dropped by MPCLoop). Figures use the multi-day x-ticks in
# `res.xticks`; per-interval grid quantities that live on the DAILY profile
# (price, CO2) are indexed by within-day position so day 2+ shows the same daily
# curve rather than reading into the overnight tail.
#
# FIGURES (from the REALIZED closed-loop trajectory; every figure uses STEP
# helpers so nothing is a smooth line):
#   01_total_grid_power_profile   charging(+)/discharging(-) summed over MCS
#   02_work_profiles_by_site      per-site work power (one panel per site)
#   03_mcs_state_of_energy        MCS SOE with min/max guide lines
#   04_cev_state_of_energy        CEV SOE with min/max guide lines
#   05_electricity_prices_emissions  price (left) + CO2 factor (right)
#   06_mcs_location_trajectory    MCS node index over time
#   07_mcs_optimization_summary   combined multi-panel overview
#   mcs_<m>_power_profile         per-MCS charging/discharging
#   11_power_estimate_convergence fixed power estimate vs hidden truth (extra)
#
# REPORTS:
#   08_cost_emissions_timeseries.csv / 08_cost_emissions_summary.png
#   09_cost_kpi_metrics.csv          / 09_kpi_metrics_summary.png
#   10_mip_convergence.csv           (per-window solver diagnostics)
#   closed_loop_trajectory.csv       (detailed analyst log)
#   overnight_mcs_charge_day*.csv    (Phase-2 overnight schedule, per kept day)
#   worker_schedule.csv              (plain-words site instructions)
#   replan_grids/day*/*.csv + *.html (per-step forward plans + replanning view)
# #############################################################################
module Output

using Plots
using DataFrames
using CSV
using Printf

using ..Common: stepify_interval_values, stepify_boundary_values,
                  interval_time_dataframe, clock_label, in_peak

export write_outputs

gr()

const COLORS = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

# Shared plot styling to match the reference figures.
_base_plot(; kw...) = plot(; size = (900, 500), xrotation = 45,
    guidefontsize = 18, tickfontsize = 18, legendfontsize = 12,
    bottom_margin = 18Plots.mm, left_margin = 16Plots.mm, right_margin = 14Plots.mm, kw...)

# Map a concatenated multi-day interval k (1..nK) to its within-day position
# (1..nK_day) so daily profiles (price, CO2) repeat instead of reading overnight.
_within_day(res, k) = mod1(k, res.nK_day)

# =============================================================================
# FIGURES
# =============================================================================

# 01 — total grid-side charging (+) and site-side discharging (-) over the MCSs.
function fig_total_grid_power(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = res.xticks
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
    rT, rL = res.xticks
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
    rT, rL = res.xticks
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
# Both live on the DAILY profile, so index by within-day position.
function fig_price_emission(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = res.xticks
    price = [d.lambda_whl_elec[_within_day(res, k)] for k in K]
    co2   = [d.lambda_CO2[_within_day(res, k)]      for k in K]
    csv = interval_time_dataframe(K, res.time_labels)
    csv[!, "Electricity_Price_USD_per_kWh"]      = price
    csv[!, "CO2_Emission_Factor_kg_CO2_per_kWh"] = co2
    p = _base_plot(title = "", xlabel = "Time", ylabel = "Electricity Price (\$/kWh)",
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)),
                   top_margin = 24Plots.mm, legend = (0.01, 1.26), grid = true, color = :blue)
    xs, ys = stepify_interval_values(K, price)
    plot!(p, xs, ys, label = "Electricity Price", linewidth = 2)
    p_twin = twinx(p)
    xc, yc = stepify_interval_values(K, co2)
    plot!(p_twin, xc, yc, ylabel = "CO₂ Emission Factor (kg CO₂/kWh)", label = nothing,
          linewidth = 2, xlims = (first(Tplot), last(Tplot)), xticks = (rT, rL),
          color = :red, guidefontsize = 18, tickfontsize = 18)
    plot!(p, [NaN], [NaN], label = "CO₂ Emission Factor", linewidth = 2)
    return p, csv
end

# 06 — MCS node index over time (0 = in transit).
function fig_location(res)
    d = res.d; K = 1:res.nK; Tplot = 1:(res.nK + 1)
    rT, rL = res.xticks
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
    rT, rL = res.xticks
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

# 11 (extra) — the power-model figure: each activity's estimate + uncertainty
# ribbon against the hidden true power (dashed). Under FORK B the estimate is
# fixed, so the lines are flat: this shows the calibrated model vs ground truth.
function fig_estimate_convergence(res)
    d = res.d; log = res.log
    x = (:gstep in propertynames(log)) ? log.gstep : log.k
    p = plot(xlabel = "Interval (15 min each)", ylabel = "Estimated power (kW)",
             title = "Fixed power model vs hidden truth", legend = :right, size = (900, 500),
             guidefontsize = 14, tickfontsize = 12)
    names_ = ["Digging", "Loading/Swinging", "Traveling", "Idling"]
    ests   = [log.est_dig, log.est_load, log.est_trv, log.est_idle]
    uncs   = [log.unc_dig, log.unc_load, log.unc_trv, log.unc_idle]
    cols   = [:steelblue, :darkorange, :purple, :seagreen]
    for j in 1:4
        plot!(p, x, ests[j], ribbon = uncs[j], lw = 2, color = cols[j], label = names_[j] * " est.")
        hline!(p, [d.true_powers[j]], lw = 1.5, ls = :dash, color = cols[j], label = names_[j] * " true")
    end
    return p
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
    Kept days: $(res.n_days_keep) (buffer day dropped); steps/day: $(res.nK_day)
    Kept intervals (concatenated): $(res.nK)
    Daytime horizon end (inferred): $(Int(round(res.d.day_end_hour))):00
    """
    p_summary = plot(legend = false, grid = false, framestyle = :none, xticks = false, yticks = false,
                     left_margin = 16Plots.mm, right_margin = 14Plots.mm)
    annotate!(p_summary, 0, 0.5, text(summary_text, :black, 12, :left))
    p_combined = plot(p05, p01, p03, p02_overlay, p04, p06, p_summary,
                      layout = (4, 2), size = (1800, 2200), left_margin = 16Plots.mm)
    savefig(p_combined, joinpath(out_dir, "07_mcs_optimization_summary.png"))
    CSV.write(joinpath(out_dir, "07_mcs_cev_soe.csv"), csv_mcs_cev_soe(res))

    # Per-MCS profiles.
    mcs_plots, mcs_csvs = figs_individual_mcs(res)
    for (m_idx, mp) in enumerate(mcs_plots)
        savefig(mp, joinpath(out_dir, "mcs_$(m_idx)_power_profile.png"))
        CSV.write(joinpath(out_dir, "mcs_$(m_idx)_power_profile.csv"), mcs_csvs[m_idx])
    end

    # 11 — extra power-model figure (specific to the MPC pipeline).
    savefig(fig_estimate_convergence(res), joinpath(out_dir, "11_power_estimate_convergence.png"))
    return nothing
end

# =============================================================================
# REPORTS (tabular / KPI)
# =============================================================================

# 08 — per-interval grid energy / cost / CO2 with running cumulatives.
function _cost_emissions_timeseries(res)
    d = res.d; log = res.log
    e_kwh = log.grid_kW .* d.delta_T
    cost  = e_kwh .* log.price
    co2   = e_kwh .* log.co2
    return DataFrame(
        Time_Period = (:gstep in propertynames(log)) ? log.gstep : log.k,
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

# 09 — KPI totals CSV + a two-panel bar summary (costs + operations).
function _write_kpi_metrics(res, out_dir)
    c = _cost_components(res)
    totals = DataFrame(
        Metric = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
                  "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
                  "Travel_Labour_USD", "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg",
                  "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour", "MCS_Transit_hour",
                  "Overnight_Recharge_kWh", "Overnight_Cost_USD", "Softened_windows", "Infeasible_windows", "MPC_loop_time_s"],
        Value = Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
                    round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
                    round(c.travel_cost, digits = 2), round(res.total_energy, digits = 2), round(res.total_co2, digits = 2),
                    round(res.nc_peak, digits = 2), round(res.op_peak, digits = 2), round(res.missed, digits = 2),
                    round(res.transit_intervals * res.d.delta_T, digits = 2),
                    round(res.overnight_energy, digits = 2), round(res.overnight_cost, digits = 2),
                    res.n_softened, res.n_infeasible, round(res.elapsed, digits = 2)])
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

# 10 — per-window solver diagnostics (the MPC analogue of a single MIP log).
_write_mip_convergence(res, out_dir) =
    CSV.write(joinpath(out_dir, "10_mip_convergence.csv"), res.solve_log)

# Worker-facing schedule (plain words) + detailed trajectory + overnight table.
function _write_schedules(res, out_dir)
    d = res.d
    CSV.write(joinpath(out_dir, "closed_loop_trajectory.csv"), res.log)
    # Overnight Phase-2 smart-charge: one file per KEPT day.
    for day in 1:res.n_days_keep
        CSV.write(joinpath(out_dir, "overnight_mcs_charge_day$(day).csv"), res.overnight_by_day[day])
    end
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

function _write_replan_grids(res, out_dir)
    d = res.d; nKd = res.nK_day
    # One subfolder of nKd x nKd replan grids per KEPT day.
    for day in 1:res.n_days_keep
        g = res.replan_by_day[day]
        gdir = joinpath(out_dir, "replan_grids", "day$(day)"); mkpath(gdir)
        _write_replan_grid(joinpath(gdir, "plan_grid_kW.csv"), g.plan_grid_kW, res, nKd)
        _write_replan_grid(joinpath(gdir, "plan_mcs_soe.csv"), g.plan_mcs_soe, res, nKd)
        _write_replan_grid(joinpath(gdir, "plan_mcs_activity.csv"), g.plan_mcs_act, res, nKd)
        for e in d.E
            _write_replan_grid(joinpath(gdir, "plan_cev$(e)_soe.csv"),      g.plan_cev_soe[e], res, nKd)
            _write_replan_grid(joinpath(gdir, "plan_cev$(e)_activity.csv"), g.plan_cev_act[e], res, nKd)
        end
    end
end

function write_reports(res, out_dir)
    mkpath(out_dir)
    _write_cost_emissions(res, out_dir)
    _write_kpi_metrics(res, out_dir)
    _write_mip_convergence(res, out_dir)
    _write_schedules(res, out_dir)
    _write_replan_grids(res, out_dir)
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
