# #############################################################################
# ComparisonOutput.jl  —  module ComparisonOutput   [GENERALIZED, 2-5 WAY]
# -----------------------------------------------------------------------------
# Produces MERGED, N-way side-by-side artefacts for any subset (2 to 5
# approaches) drawn from: Approach 0 (one-shot), Approach 1 - Shrinking,
# Approach 2 - Shrinking (stochastic), Approach 1 - Receding, Approach 2 -
# Receding (stochastic) — all sharing the SAME ActivityPowerPool (one pool
# object, built once by the driver and passed into every run; see
# 7_Comparison_main.jl). This module does not call any source codebase's own
# Output.jl at all — only these merged files are written, directly into the
# run's `out_dir` (no per-approach subfolders):
#
#   01_total_grid_power_profile.png/.csv   … 09_mcs_<m>_power_profile.png/.csv
#     each figure overlays every approach passed in (05_electricity_prices_
#     emissions is drawn once since price/CO2 are a shared input, identical
#     across approaches by construction)
#   07_mcs_optimization_summary.png        combined multi-panel overview
#     (original 4x2 summary grid: price/CO2, grid power, MCS SOE, work-by-
#     site, CEV SOE, location, plus a text summary panel)
#   07_approach_timeline_comparison.png    3-row x N-col grid: one column per
#     approach passed in, rows = MCS power / CEV SOE / CEV work power — the
#     N-way extension of each source codebase's own 3x2 Approach0-vs-
#     Approach1 timeline figure (5_Output.jl)
#   08_kpi_metrics_summary.png             grouped bars, one bar per approach
#                                           per metric
#   08_cost_kpi_metrics.csv                same KPI rows as each source
#                                           codebase's own report, one column
#                                           per approach
#   <key1>_vs_<key2>[_vs_...].html         the N-way analogue of each source
#                                           codebase's own approach0_vs_approach1.html
#                                           (a Δ column is added only when
#                                           exactly 2 approaches are compared)
#   10_diagnostic_dispatch_trace.csv       per-interval MCS->CEV discharge,
#                                           work power, activity labels, and
#                                           running cumulative-discharge GAP
#                                           vs the first approach passed in
#   11_diagnostic_capacity_summary.csv     per-approach totals: discharge kWh,
#                                           work kWh, capped/infeasible
#                                           interval counts, missed-work $
#   13_solve_status_<key>.csv              one file per approach that has a
#                                           solve_log: raw per-re-solve solver
#                                           record (status/gap/etc, whatever
#                                           columns solve_log actually has)
#
# Every function below takes `apps::Vector{Approach}` and only ever loops
# over it / reads `length(apps)` — nothing is hardcoded to 3, so the SAME
# code path serves the full 5-way run and every 2-way/3-way subset.
# #############################################################################
module ComparisonOutput

using Plots
using DataFrames
using CSV
using Printf

export write_comparison_outputs, write_diagnostic_dispatch_trace, write_solve_status_diagnostic, Approach

gr()

# =============================================================================
# SMALL SHARED HELPERS  (duplicated, on purpose, from Common.jl)
# -----------------------------------------------------------------------------
# RecedingApp.Common and ShrinkingApp.Common are two DIFFERENT modules (each
# nested inside its own namespace -- see 7_Comparison_main.jl), so there is no
# single canonical "Common" this module could depend on without picking a
# side. These five helpers are tiny, dependency-free, and IDENTICAL in both
# source codebases, so they are copied here verbatim rather than importing
# from one app and quietly coupling this module's correctness to it.
# =============================================================================
function clock_label(t_start, delta_T, k)
    m = mod(Int(round(t_start * 60 + (k - 1) * delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

function create_fixed_2hour_xticks(T, t_start::Real = 0)
    Tvec = collect(T)
    n_intervals = length(Tvec) - 1
    ticks = Int[]; labels = String[]
    span_hours = n_intervals * 0.25
    hi = Int(ceil(span_hours))
    for hour_offset in 0:2:hi
        idx = first(Tvec) + Int(round(hour_offset / span_hours * n_intervals))
        idx > last(Tvec) && continue
        push!(ticks, idx)
        clock_hour = Int(mod(t_start + hour_offset, 24))
        push!(labels, lpad(string(clock_hour), 2, '0') * ":00")
    end
    return ticks, labels
end

function stepify_interval_values(K, values)
    Kvec = collect(K)
    x_step = Int[]; y_step = eltype(values)[]
    for (idx, k) in enumerate(Kvec)
        push!(x_step, k);     push!(y_step, values[idx])
        push!(x_step, k + 1); push!(y_step, values[idx])
    end
    return x_step, y_step
end

function stepify_boundary_values(T, values)
    Tvec = collect(T)
    x_step = Int[]; y_step = eltype(values)[]
    isempty(Tvec) && return x_step, y_step
    for idx in 1:(length(Tvec) - 1)
        push!(x_step, Tvec[idx]);     push!(y_step, values[idx])
        push!(x_step, Tvec[idx + 1]); push!(y_step, values[idx])
    end
    push!(x_step, last(Tvec)); push!(y_step, values[end])
    return x_step, y_step
end

function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

# =============================================================================
# APPROACH IDENTITY: name + plot colour, kept together so every figure/table
# uses the same label and colour for the same approach.
# =============================================================================
struct Approach
    key::String     # short machine key, e.g. "approach0"
    label::String   # human label, e.g. "Approach 0 (one-shot)"
    res::NamedTuple
    color::Symbol
end

_base_plot(; kw...) = plot(; size = (1000, 520), xrotation = 45,
    guidefontsize = 16, tickfontsize = 14, legendfontsize = 11,
    bottom_margin = 16Plots.mm, left_margin = 16Plots.mm, right_margin = 14Plots.mm, kw...)

# The six cost components -- SAME definitions/formula as each source
# codebase's private `_cost_components` (5_Output.jl), PLUS Change 3's
# terminal SOE_CEV shortfall penalty (a pure end-of-day check against
# SOE_CEV_ini, independent of `missed`/`missed_cost` above, which only ever
# reflects live physical-floor capping). Duck-typed on `res` fields
# (total_cost, total_co2, nc_peak, op_peak, missed, labour_cost,
# shortfall_penalty_cost) plus `res.d` (carbon_price_per_ton,
# lambda_demand_NC/OP, rho_miss), so it works unmodified on a result produced
# by any of the four solver codebases (A1S/A1R/A2S/A2R) or Approach 0.
function cost_components(res)
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

# =============================================================================
# 01 — total grid power (charging +, discharging -), one pair of step-lines
# per approach.
# =============================================================================
function fig01_total_grid_power(apps)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K = 1:nK; Tplot = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    p = _base_plot(xlabel = "Time", ylabel = "Power (kW)",
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    ymax = 1.0
    csv = DataFrame(Time_Period = collect(K), Clock = [clock_label(d.t_start, d.delta_T, k) for k in K])
    for a in apps
        charging    = [sum(a.res.real_P_ch[m, k]  for m in a.res.d.M) for k in K]
        discharging = [sum(a.res.real_P_dch[m, k] for m in a.res.d.M) for k in K]
        ymax = max(ymax, maximum(charging), maximum(discharging))
        xc, yc = stepify_interval_values(K, charging)
        xd, yd = stepify_interval_values(K, -discharging)
        plot!(p, xc, yc, label = "$(a.label) — Charging", color = a.color, alpha = 0.85, linewidth = 2)
        plot!(p, xd, yd, label = "$(a.label) — Discharging", color = a.color, alpha = 0.45,
              linewidth = 2, linestyle = :dash)
        csv[!, "$(a.key)_Charging_kW"]    = charging
        csv[!, "$(a.key)_Discharging_kW"] = discharging
    end
    hline!(p, [0.0], color = :black, linestyle = :dot, alpha = 0.5, label = nothing)
    ylims!(p, (-1.1 * ymax, 1.1 * ymax))
    return p, csv
end

# =============================================================================
# 02 — work power by site: one panel per site, one line per approach
# (summed over that site's excavators).
# =============================================================================
function fig02_work_by_site(apps)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K = 1:nK; Tplot = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    site_series = Dict(i => Dict{String, Vector{Float64}}() for i in d.N_c)
    ymax = 1.0
    csv = DataFrame(Time_Period = collect(K), Clock = [clock_label(d.t_start, d.delta_T, k) for k in K])
    for a in apps, i in a.res.d.N_c
        tot = [sum(a.res.real_P_work[i, e, k] for e in a.res.d.E) for k in K]
        site_series[i][a.key] = tot
        ymax = max(ymax, maximum(tot; init = 0.0))
        csv[!, "Site_$(i)_$(a.key)_kW"] = tot
    end
    site_plots = Any[]
    for i in sort(collect(d.N_c))
        p = _base_plot(title = "Site $i", titlefontsize = 16, xlabel = "Time", ylabel = "Power (kW)",
                       xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)),
                       ylims = (0, 1.1 * ymax), legend = :topright)
        for a in apps
            haskey(site_series[i], a.key) || continue
            xs, ys = stepify_interval_values(K, site_series[i][a.key])
            plot!(p, xs, ys, label = a.label, color = a.color, linewidth = 2)
        end
        push!(site_plots, p)
    end
    n = length(site_plots)
    p_multi = n == 0 ? plot(title = "") :
        plot(site_plots...; layout = (n, 1), size = (1000, 420 * n),
             plot_title = "Work Power Profiles by Site — " * join([a.label for a in apps], " vs "),
             plot_titlevspan = 0.10)
    return p_multi, csv
end

# =============================================================================
# 03 / 04 — SOE figures (MCS / CEV): one panel per unit, one line per approach,
# plus that unit's min/max guide lines (identical battery data across
# approaches since all three share the SAME ev_data.csv / mcs_data.csv).
# =============================================================================
function _fig_soe_multi(apps, unit_set, soe_field, max_field, min_field, label_prefix)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    T = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(T, d.t_start)
    csv = DataFrame(Time_Period = collect(T))
    panels = Any[]
    soe_max = getproperty(d, max_field); soe_min = getproperty(d, min_field)
    for u in unit_set
        p = _base_plot(title = "$label_prefix $u", titlefontsize = 16,
                       xlabel = "Time", ylabel = "State of Energy (kWh)",
                       xticks = (rT, rL), xlims = (first(T), last(T)))
        for a in apps
            soe = getproperty(a.res, soe_field)
            vals = [soe[u, t] for t in T]
            xs, ys = stepify_boundary_values(T, vals)
            plot!(p, xs, ys, label = a.label, color = a.color, linewidth = 2)
            csv[!, "$(label_prefix)_$(u)_$(a.key)_kWh"] = vals
        end
        hline!(p, [soe_max[u]], color = :black, linestyle = :dash, label = "Max")
        hline!(p, [soe_min[u]], color = :gray,  linestyle = :dash, label = "Min")
        push!(panels, p)
    end
    n = length(panels)
    p_multi = n == 0 ? plot(title = "") :
        plot(panels...; layout = (n, 1), size = (1000, 420 * n))
    return p_multi, csv
end

fig03_mcs_soe(apps) = _fig_soe_multi(apps, first(apps).res.d.M, :real_SOE_MCS, :SOE_MCS_max, :SOE_MCS_min, "MCS")
fig04_cev_soe(apps) = _fig_soe_multi(apps, first(apps).res.d.E, :real_SOE_CEV, :SOE_CEV_max, :SOE_CEV_min, "CEV")

# =============================================================================
# 05 — electricity price + CO2 factor. These are EXOGENOUS inputs (same
# time_data.csv for all three approaches), so there is only ONE curve to draw,
# not three -- the figure says so explicitly.
# =============================================================================
function fig05_price_emission(apps)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K = 1:nK; Tplot = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    # d.lambda_whl_elec / d.lambda_CO2 are ONE day's curve (96 intervals),
    # straight from time_data.csv -- they are NOT tiled out to the full
    # multi-day span. For multi-day runs (nK > length of that array), wrap
    # the index the same way the solver's own objective does
    # (lambda_whl_elec[wd(k)], wd(k) = mod1(k, nKd), in 3_MCSModel.jl) --
    # the price/CO2 pattern simply repeats every kept day.
    wd(k) = mod1(k, length(d.lambda_whl_elec))
    csv = DataFrame(Time_Period = collect(K),
                    Electricity_Price_USD_per_kWh = [d.lambda_whl_elec[wd(k)] for k in K],
                    CO2_Emission_Factor_kg_per_kWh = [d.lambda_CO2[wd(k)] for k in K])
    
    # 1. Standalone 2-panel figure for File 05
    xs, ys = stepify_interval_values(K, [d.lambda_whl_elec[wd(k)] for k in K])
    p1 = _base_plot(ylabel = "Price (\$/kWh)",
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    plot!(p1, xs, ys, label = "Price", linewidth = 2, color = :blue)
    
    xc, yc = stepify_interval_values(K, [d.lambda_CO2[wd(k)] for k in K])
    p2 = _base_plot(xlabel = "Time", ylabel = "CO₂ (kg/kWh)", 
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    plot!(p2, xc, yc, label = "CO₂ Factor", linewidth = 2, color = :red)
    
    p_standalone = plot(p1, p2, layout = (2, 1), size = (1000, 800),
                        plot_title = "Shared input (identical across all $(length(apps)) approaches)",
                        plot_titlefontsize = 13)

    # 2. Flat single-panel figure (kept for any future embedding use)
    p_flat = _base_plot(xlabel = "Time", ylabel = "Price (\$/kWh)",
                        title = "Electricity Price & CO₂", titlefontsize = 14,
                        xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
    plot!(p_flat, xs, ys, label = "Electricity Price", color = :blue, linewidth = 2)

    return p_standalone, p_flat, csv
end

# =============================================================================
# 06 — MCS location trajectory, one line per approach.
# =============================================================================
function fig06_location(apps)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K = 1:nK; Tplot = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    node_labels = [node in d.N_g ? "Grid $node" : "Site $node" for node in d.N]
    yt_pos = vcat(0, collect(d.N)); yt_lab = vcat("Travel", node_labels)
    p = _base_plot(xlabel = "Time", ylabel = "Node", yticks = (yt_pos, yt_lab),
                   xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)), grid = true)
    csv = DataFrame(Time_Period = collect(K), Clock = [clock_label(d.t_start, d.delta_T, k) for k in K])
    for a in apps
        m = first(a.res.d.M)
        locs = [a.res.real_loc[m, k] for k in K]
        xs, ys = stepify_interval_values(K, locs)
        plot!(p, xs, ys, label = a.label, linewidth = 2, marker = :circle, markersize = 3, color = a.color)
        csv[!, "$(a.key)_MCS1_Location"] = locs
    end
    return p, csv
end

# =============================================================================
# 09 — per-MCS charging/discharging profile, one panel per MCS unit, one pair
# of step-lines per approach (same construction as 01, split out per unit).
# =============================================================================
function fig09_mcs_power_profiles(apps)
    d = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K = 1:nK; Tplot = 1:(nK + 1)
    rT, rL = create_fixed_2hour_xticks(Tplot, d.t_start)
    plots = Any[]; csvs = DataFrame[]
    for m in d.M
        p = _base_plot(title = "MCS $m", titlefontsize = 16, xlabel = "Time", ylabel = "Power (kW)",
                       xticks = (rT, rL), xlims = (first(Tplot), last(Tplot)))
        csv = DataFrame(Time_Period = collect(K), Clock = [clock_label(d.t_start, d.delta_T, k) for k in K])
        ymax = 1.0
        for a in apps
            charging    = [a.res.real_P_ch[m, k]  for k in K]
            discharging = [a.res.real_P_dch[m, k] for k in K]
            ymax = max(ymax, maximum(charging), maximum(discharging))
            xc, yc = stepify_interval_values(K, charging)
            xd, yd = stepify_interval_values(K, -discharging)
            plot!(p, xc, yc, label = "$(a.label) — Charging", color = a.color, linewidth = 2, alpha = 0.85)
            plot!(p, xd, yd, label = "$(a.label) — Discharging", color = a.color, linewidth = 2,
                  alpha = 0.45, linestyle = :dash)
            csv[!, "$(a.key)_Charging_kW"]    = charging
            csv[!, "$(a.key)_Discharging_kW"] = discharging
        end
        hline!(p, [0.0], color = :black, linestyle = :dot, alpha = 0.5, label = nothing)
        ylims!(p, (-1.1 * ymax, 1.1 * ymax))
        push!(plots, p); push!(csvs, csv)
    end
    return plots, csvs
end

# =============================================================================
# 08 — KPI metrics summary: grouped bars (cost components + demand peaks),
# 3 bars per group (one per approach).
# =============================================================================
function fig08_kpi_summary(apps)
    labels = ["Energy", "CO₂", "NCD", "OPD", "Missed", "Travel", "Shortfall", "TOTAL"]
    ymax = 1.0
    p_costs = plot(title = "Realised cost components", xlabel = "", ylabel = "Cost (USD)",
                   xticks = (1:length(labels), labels), xlims = (0.5, length(labels) + 0.5),
                   xrotation = 20, legend = :topleft, guidefontsize = 15, tickfontsize = 13,
                   size = (1300, 500), bottom_margin = 14Plots.mm, left_margin = 14Plots.mm)
    nA = length(apps)
    width = 0.8 / nA
    for (ai, a) in enumerate(apps)
        c = cost_components(a.res)
        vals = [c.energy_cost, c.carbon_cost, c.ncd_cost, c.opd_cost, c.missed_cost, c.travel_cost, c.shortfall_cost, c.total]
        ymax = max(ymax, maximum(vals))
        offset = (ai - (nA + 1) / 2) * width
        bar!(p_costs, (1:length(labels)) .+ offset, vals; bar_width = width * 0.92,
             color = a.color, label = a.label)
    end
    ylims!(p_costs, (0, 1.25 * ymax))

    peak_labels = ["NCD Peak (kW)", "OPD Peak (kW)"]
    p_ops = plot(title = "Demand peaks", xlabel = "", ylabel = "Power (kW)",
                xticks = (1:2, peak_labels), xlims = (0.5, 2.5), legend = :topleft,
                guidefontsize = 15, tickfontsize = 13, size = (1300, 500),
                bottom_margin = 14Plots.mm, left_margin = 14Plots.mm)
    peak_max = 1.0
    for (ai, a) in enumerate(apps)
        offset = (ai - (nA + 1) / 2) * width
        vals = [a.res.nc_peak, a.res.op_peak]
        peak_max = max(peak_max, maximum(vals))
        bar!(p_ops, (1:2) .+ offset, vals; bar_width = width * 0.92, color = a.color,
             label = a.label)
    end
    ylims!(p_ops, (0, 1.25 * peak_max))
    p = plot(p_costs, p_ops, layout = (2, 1), size = (1300, 950),
             plot_title = "KPI Metrics Summary — " * join([a.label for a in apps], " vs "))
    return p
end

# =============================================================================
# 07a — combined multi-panel overview (reuses the panels above, same layout
# style as each source codebase's own "07_mcs_optimization_summary"). This is
# the original figure, restored exactly as it was before.
# =============================================================================
function fig07_summary(apps, p05, p01, p03, p02, p04, p06)
    summary_lines = ["Comparison Summary", "-------------------"]
    for a in apps
        push!(summary_lines, "$(a.label):")
        push!(summary_lines, @sprintf("  Grid energy   : %.2f kWh", a.res.total_energy))
        push!(summary_lines, @sprintf("  Energy cost   : \$%.2f", a.res.total_cost))
        push!(summary_lines, @sprintf("  NCD / OPD peak: %.2f / %.2f kW", a.res.nc_peak, a.res.op_peak))
        push!(summary_lines, @sprintf("  Missed work   : %.2f h", a.res.missed))
        push!(summary_lines, @sprintf("  Terminal shortfall: %.3f kWh (\$%.2f)", a.res.shortfall_kWh, a.res.shortfall_penalty_cost))
    end
    p_summary = plot(legend = false, grid = false, framestyle = :none, xticks = false, yticks = false,
                     left_margin = 14Plots.mm, right_margin = 12Plots.mm)
    annotate!(p_summary, 0, 0.5, text(join(summary_lines, "\n"), :black, 10, :left))

    # Add an explicitly blank 8th plot to prevent the EmptyLayout crash
    p_blank = plot(legend = false, grid = false, framestyle = :none, xticks = false, yticks = false)

    # Add p_blank to the end of the list so the 4x2 grid is perfectly filled
    p = plot(p05, p01, p03, p02, p04, p06, p_summary, p_blank,
             layout = (4, 2), size = (2000, 2300), left_margin = 16Plots.mm)
    return p
end

# =============================================================================
# 07b — 3-way side-by-side TIMELINE comparison. A direct port of each source
# codebase's own 3-row x 2-col fig_approach_timeline_comparison (see
# 5_Output.jl), which only ever laid out Approach 0 vs Approach 1 side by
# side. Here there are THREE approaches (approach0 / shrinking / receding),
# so it stays a 3x3 grid — one column per approach, sharing the same three
# rows:
#   row 1: MCS power (charging/discharging, NCDP/OPDP annotated)
#   row 2: CEV state of energy
#   row 3: CEV work power (bars coloured by activity)
# =============================================================================
const COLORS = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

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

# (row 1) MCS power: charging(+)/discharging(-), NCDP & OPDP annotated.
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

# (row 2) CEV state of energy — one line per CEV, dashed min/max guide lines.
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

# Realised missed work, split by activity, converted from hours to kWh —
# same site/activity accounting as each source codebase's own version, read
# off the REALISED per-CEV activity trace.
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
    miss_dig_h  = sum((max(d.hours_digging[i]          - rdig[i],  0.0) for i in d.N_c); init = 0.0)
    miss_load_h = sum((max(d.hours_loading_swinging[i] - rload[i], 0.0) for i in d.N_c); init = 0.0)
    return miss_dig_h * d.p_digging, miss_load_h * d.p_loading_swinging
end

# (row 3) CEV work power — bars coloured by activity (Digging / Traveling /
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

# Build the full 3-row x N-col comparison: one column per approach passed in
# (2 to 5 of them), rows = MCS power / CEV SOE / CEV work power. Column
# titles carry each approach's label (top row only, same as the 3x2 source
# version only titling its top row). Width scales with N (800px/column) so a
# 2-way comparison isn't stretched as wide as the full 5-way one.
function fig07_timeline_comparison(apps)
    nA = length(apps)
    row1 = [_panel_mcs_power(a.res, a.label) for a in apps]
    row2 = [_panel_cev_soe(a.res, "")         for a in apps]
    row3 = [_panel_cev_work(a.res, "")        for a in apps]
    p = plot(row1..., row2..., row3...; layout = (3, nA),
             size = (800 * nA, 1500), left_margin = 16Plots.mm, bottom_margin = 16Plots.mm)
    return p
end

# =============================================================================
# HTML: N-way KPI comparison table  ->  <key1>_vs_<key2>[_vs_...].html
# A Δ column (second minus first) is added ONLY when exactly 2 approaches are
# being compared -- with 3+ it's ambiguous which pair the reader wants
# differenced, so the plain per-approach columns are left to speak for
# themselves (the CSV / bar chart cover cross-approach comparison either way).
# =============================================================================
function write_kpi_html(apps, out_dir)
    rows_def = [
        ("Grid energy (kWh)",         a -> a.res.total_energy),
        ("Energy cost (USD)",         a -> cost_components(a.res).energy_cost),
        ("CO2 emissions (kg)",        a -> a.res.total_co2),
        ("CO2 cost (USD)",            a -> cost_components(a.res).carbon_cost),
        ("NCD peak (kW)",             a -> a.res.nc_peak),
        ("NCD charge (USD)",          a -> cost_components(a.res).ncd_cost),
        ("OPD peak (kW)",             a -> a.res.op_peak),
        ("OPD charge (USD)",          a -> cost_components(a.res).opd_cost),
        ("Missed work (h)",           a -> a.res.missed),
        ("Missed work penalty (USD)", a -> cost_components(a.res).missed_cost),
        ("Terminal SOE shortfall (kWh)",     a -> a.res.shortfall_kWh),
        ("Terminal shortfall penalty (USD)", a -> cost_components(a.res).shortfall_cost),
        ("MCS transit (h)",           a -> a.res.transit_intervals * a.res.d.delta_T),
        ("Travel labour (USD)",       a -> cost_components(a.res).travel_cost),
        ("TOTAL cost (USD)",          a -> cost_components(a.res).total),
    ]
    nA = length(apps)
    add_delta = nA == 2
    fname = join([a.key for a in apps], "_vs_") * ".html"
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "table{border-collapse:collapse;font-size:13px}")
    println(io, "th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}")
    println(io, "th{background:#f4f4f4}")
    println(io, "td:first-child,th:first-child{text-align:left}")
    println(io, "</style></head><body>")
    println(io, "<h2>", join([a.label for a in apps], " vs "), "</h2>")
    println(io, "<p>All ", nA, " column", nA == 1 ? "" : "s", " are FULLY REALISED outcomes, drawn from the ",
                "SAME shared per-(excavator, activity) power sample pool (same seed, same frozen prior ",
                "mu/sigma -- input data is identical across all four codebases apart from the regression-",
                "refreshed <code>parameters.csv</code>, built once and shared). Receding Horizon runs use ",
                "the shared <code>n_days_receding</code> reported day(s) so they are directly comparable to ",
                "the single-day Shrinking Horizon runs.</p>")
    println(io, "<table><tr><th>Metric</th>")
    for a in apps; println(io, "<th>", a.label, "</th>"); end
    add_delta && println(io, "<th>&Delta; (", apps[2].label, " &minus; ", apps[1].label, ")</th>")
    println(io, "</tr>")
    for (name, f) in rows_def
        vals = [f(a) for a in apps]
        print(io, "<tr><td>", name, "</td>")
        for v in vals; print(io, "<td>", @sprintf("%.3f", v), "</td>"); end
        if add_delta
            println(io, "<td>", @sprintf("%.3f", vals[2] - vals[1]), "</td></tr>")
        else
            println(io, "</tr>")
        end
    end
    println(io, "</table></body></html>")
    write(joinpath(out_dir, fname), String(take!(io)))
    return nothing
end

# =============================================================================
# CSV: N-way KPI totals  ->  08_cost_kpi_metrics.csv (one column per approach,
# already fully generic -- no per-N special-casing needed here)
# =============================================================================
function write_kpi_csv(apps, out_dir)
    metric_names = ["Total_Cost_USD", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD",
                    "NC_demand_charge_USD", "OP_demand_charge_USD", "Missed_Work_Penalty_USD",
                    "Travel_Labour_USD", "Terminal_Shortfall_Penalty_USD", "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg",
                    "NCD_Peak_kW", "OPD_Peak_kW", "Missed_Work_hour", "MCS_Transit_hour",
                    "Terminal_SOE_Shortfall_kWh", "Infeasible_windows", "Solve_time_s"]
    df = DataFrame(Metric = metric_names)
    for a in apps
        c = cost_components(a.res)
        vals = Any[round(c.total, digits = 2), round(c.energy_cost, digits = 2), round(c.carbon_cost, digits = 2),
                   round(c.ncd_cost, digits = 2), round(c.opd_cost, digits = 2), round(c.missed_cost, digits = 2),
                   round(c.travel_cost, digits = 2), round(c.shortfall_cost, digits = 2), round(a.res.total_energy, digits = 2), round(a.res.total_co2, digits = 2),
                   round(a.res.nc_peak, digits = 2), round(a.res.op_peak, digits = 2), round(a.res.missed, digits = 2),
                   round(a.res.transit_intervals * a.res.d.delta_T, digits = 2),
                   round(a.res.shortfall_kWh, digits = 3),
                   a.res.n_infeasible, round(a.res.elapsed, digits = 2)]
        df[!, a.key] = vals
    end
    CSV.write(joinpath(out_dir, "08_cost_kpi_metrics.csv"), df)
    return nothing
end

# =============================================================================
# Entry point: write the FULL merged artefact set for whatever `apps` subset
# (2 to 5 Approach objects) is passed in, into `out_dir`. Every helper above
# is N-generic, so this is the SAME code path for the full 5-way run and
# every 2-way/3-way subset -- only `apps` and `out_dir` change per call.
# =============================================================================
function write_comparison_outputs(apps::Vector{Approach}, out_dir::AbstractString)
    length(apps) >= 2 || error("write_comparison_outputs: need at least 2 approaches to compare, got $(length(apps))")
    mkpath(out_dir)

    p01, c01 = fig01_total_grid_power(apps)
    savefig(p01, joinpath(out_dir, "01_total_grid_power_profile.png"))
    CSV.write(joinpath(out_dir, "01_total_grid_power_profile.csv"), c01)

    p02, c02 = fig02_work_by_site(apps)
    savefig(p02, joinpath(out_dir, "02_work_profiles_by_site.png"))
    CSV.write(joinpath(out_dir, "02_work_profiles_by_site.csv"), c02)

    p03, c03 = fig03_mcs_soe(apps)
    savefig(p03, joinpath(out_dir, "03_mcs_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "03_mcs_state_of_energy.csv"), c03)

    p04, c04 = fig04_cev_soe(apps)
    savefig(p04, joinpath(out_dir, "04_cev_state_of_energy.png"))
    CSV.write(joinpath(out_dir, "04_cev_state_of_energy.csv"), c04)

    p05_standalone, p05_flat, c05 = fig05_price_emission(apps)
    savefig(p05_standalone, joinpath(out_dir, "05_electricity_prices_emissions.png"))
    CSV.write(joinpath(out_dir, "05_electricity_prices.csv"), c05)

    p06, c06 = fig06_location(apps)
    savefig(p06, joinpath(out_dir, "06_mcs_location_trajectory.png"))
    CSV.write(joinpath(out_dir, "06_mcs_location_trajectory.csv"), c06)

    # Pass p05_flat into fig07_summary so it nests safely!
    p07a = fig07_summary(apps, p05_flat, p01, p03, p02, p04, p06)
    savefig(p07a, joinpath(out_dir, "07_mcs_optimization_summary.png"))

    p07b = fig07_timeline_comparison(apps)
    savefig(p07b, joinpath(out_dir, "07_approach_timeline_comparison.png"))

    p08 = fig08_kpi_summary(apps)
    savefig(p08, joinpath(out_dir, "08_kpi_metrics_summary.png"))

    mcs_plots, mcs_csvs = fig09_mcs_power_profiles(apps)
    for (m_idx, mp) in enumerate(mcs_plots)
        savefig(mp, joinpath(out_dir, "09_mcs_$(m_idx)_power_profile.png"))
        CSV.write(joinpath(out_dir, "09_mcs_$(m_idx)_power_profile.csv"), mcs_csvs[m_idx])
    end

    write_kpi_html(apps, out_dir)
    write_kpi_csv(apps, out_dir)

    write_diagnostic_dispatch_trace(apps, out_dir)
    write_solve_status_diagnostic(apps, out_dir)

    return nothing
end

# =============================================================================
# 10/11 — DIAGNOSTIC: MCS -> CEV dispatch trace + capacity/infeasibility
# summary.
# -----------------------------------------------------------------------------
# Added to trace WHERE two approaches' realized MCS->CEV discharge schedules
# diverge, WITHOUT touching any source codebase (MPCLoop.jl / Common.jl).
# Everything here reads fields the driver already has in memory on `res`
# (real_P_dch, real_P_work, real_cev_act, real_mcs_act, n_capped,
# n_infeasible, missed) -- nothing new is computed by the solver, this just
# exposes data that was already being thrown away after the existing 01-09
# figures were built.
#
#   10_diagnostic_dispatch_trace.csv   one row per interval, per approach:
#     - Discharge_kW / Discharge_cum_kWh  (MCS -> CEV, summed over MCS units)
#     - Work_kW                            (power delivered to excavators)
#     - CEV<e>_Activity                    (Digging / Loading / Traveling /
#                                            Charging / idle, per excavator)
#     - MCS_Status                         (Charging (grid) / Serving CEV /
#                                            Traveling / Idle)
#     - <key>_minus_<base>_cum_kWh         running discharge GAP against the
#                                           FIRST approach passed in (put A0
#                                           first when comparing against it) --
#                                           the interval where this stops
#                                           being ~0 is where the schedules
#                                           first diverge.
#
#   11_diagnostic_capacity_summary.csv  one column per approach:
#     - Total_Discharge_kWh / Total_Work_kWh  (whole-day totals)
#     - N_Capped_Intervals    how many intervals had work capped by available
#                              CEV energy (battery-floor events; see
#                              apply_and_simulate! in 4_MPCLoop.jl)
#     - N_Infeasible_Windows  how many re-solves were infeasible under the
#                              hard constraints (plant held state that step)
#     - Missed_Work_hour / Missed_Work_Penalty_USD  end-of-run backlog, same
#                              numbers already in 08_cost_kpi_metrics.csv,
#                              repeated here for easy side-by-side reading
#                              against the capped/infeasible counts above.
# =============================================================================
function write_diagnostic_dispatch_trace(apps::Vector{Approach}, out_dir::AbstractString)
    mkpath(out_dir)
    d  = first(apps).res.d
    nK = minimum(a.res.nK for a in apps)
    K  = 1:nK

    csv = DataFrame(Time_Period = collect(K), Clock = [clock_label(d.t_start, d.delta_T, k) for k in K])

    for a in apps
        dch  = [sum(a.res.real_P_dch[m, k] for m in a.res.d.M) for k in K]
        work = [sum(a.res.real_P_work[i, e, k] for i in a.res.d.N_c, e in a.res.d.E) for k in K]

        csv[!, "$(a.key)_Discharge_kW"]      = dch
        csv[!, "$(a.key)_Discharge_cum_kWh"] = cumsum(dch) .* d.delta_T
        csv[!, "$(a.key)_Work_kW"]           = work

        for e in a.res.d.E
            csv[!, "$(a.key)_CEV$(e)_Activity"] = [a.res.real_cev_act[e][k] for k in K]
        end
        csv[!, "$(a.key)_MCS_Status"] = [a.res.real_mcs_act[k] for k in K]
    end

    # running discharge GAP vs the first approach passed in -- scan this
    # column for the first nonzero row to find exactly where two schedules
    # start to differ.
    if length(apps) >= 2
        base_key = first(apps).key
        base_cum = csv[!, "$(base_key)_Discharge_cum_kWh"]
        for a in apps[2:end]
            csv[!, "$(a.key)_minus_$(base_key)_cum_kWh"] =
                csv[!, "$(a.key)_Discharge_cum_kWh"] .- base_cum
        end
    end

    CSV.write(joinpath(out_dir, "10_diagnostic_dispatch_trace.csv"), csv)

    # ---- per-approach capacity / infeasibility summary ----
    metrics = ["Total_Discharge_kWh", "Total_Work_kWh", "N_Capped_Intervals",
               "N_Infeasible_Windows", "Missed_Work_hour", "Missed_Work_Penalty_USD"]
    out = DataFrame(Metric = metrics)
    for a in apps
        dch  = [sum(a.res.real_P_dch[m, k] for m in a.res.d.M) for k in K]
        work = [sum(a.res.real_P_work[i, e, k] for i in a.res.d.N_c, e in a.res.d.E) for k in K]
        col = Any[
            round(sum(dch) * d.delta_T; digits = 3),
            round(sum(work) * d.delta_T; digits = 3),
            a.res.n_capped,
            a.res.n_infeasible,
            round(a.res.missed; digits = 3),
            round(d.rho_miss * a.res.missed; digits = 2),
        ]
        out[!, a.key] = col
    end
    CSV.write(joinpath(out_dir, "11_diagnostic_capacity_summary.csv"), out)

    return csv, out
end

# =============================================================================
# 13 — DIAGNOSTIC: PER-RE-SOLVE SOLVER STATUS
# -----------------------------------------------------------------------------
# Written to help answer "did the MPC solver actually prove this schedule
# optimal, or did it just run out of time on a flat-price / tied-cost
# window and stop wherever it happened to land?" run_mpc already returns a
# `solve_log` field (a DataFrame -- see 4_MPCLoop.jl) with one row per
# re-solve, recording that step's solver termination status and (for
# HiGHS/JuMP) typically a MIP gap and solve time.
#
# IMPORTANT CAVEAT: this function does NOT hardcode solve_log's column
# names. I don't currently have the exact schema in front of me (the
# Receding-Horizon 4_MPCLoop.jl that defines it isn't in what's been shared
# in this conversation), so hardcoding guessed column names risks silently
# assuming the wrong ones. Instead this writes out solve_log AS-IS, whatever
# columns it actually has -- open 13_solve_status_<key>.csv and look at the
# column headers directly to see what's really being tracked (status/
# termination reason, gap, solve time, etc.). If solve_log isn't present on
# a given approach's result (e.g. run_one_shot may not track one the same
# way run_mpc does, since it solves once per day rather than once per
# interval), that approach is skipped with a console note instead of
# erroring.
#
#   13_solve_status_<key>.csv   one file per approach that has a solve_log,
#                                the raw per-re-solve solver record for that
#                                approach, unmodified.
# =============================================================================
function write_solve_status_diagnostic(apps::Vector{Approach}, out_dir::AbstractString)
    mkpath(out_dir)
    for a in apps
        if !hasproperty(a.res, :solve_log) || a.res.solve_log === nothing
            println("  (no solve_log on $(a.key) -- skipping 13_solve_status_$(a.key).csv)")
            continue
        end
        sl = a.res.solve_log
        if !(sl isa DataFrame) || nrow(sl) == 0
            println("  (solve_log on $(a.key) is empty or not a DataFrame -- skipping)")
            continue
        end
        CSV.write(joinpath(out_dir, "13_solve_status_$(a.key).csv"), sl)
    end
    return nothing
end

end # module ComparisonOutput
