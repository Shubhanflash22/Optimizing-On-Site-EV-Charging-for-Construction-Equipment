module MCSOptimizer

using JuMP
using HiGHS
using Plots
using DataFrames
using Dates
using LinearAlgebra
import MathOptInterface as MOI
using Printf

export solve_and_analyze

"""
Create sparse x-axis labels for plotting.
"""
function create_readable_time_labels(T, time_labels)
    if length(T) <= 24
        step = max(1, div(length(T), 8))
    elseif length(T) <= 48
        step = max(1, div(length(T), 12))
    else
        step = max(1, div(length(T), 8))
    end
    readable_indices = 1:step:length(T)
    readable_times = time_labels[readable_indices]
    readable_T = collect(T)[readable_indices]
    return readable_T, readable_times
end

"""
For boundary-indexed time, map 00:00 -> index 1 and 24:00 -> last(T).
For 15-minute data, this gives 00:00 -> 1, 03:00 -> 13, ..., 24:00 -> 97.
"""
function create_fixed_2hour_xticks(T, t_start::Real=0)
    Tvec = collect(T)
    n_intervals = length(Tvec) - 1
    ticks = Int[]
    labels = String[]
    for hour_offset in 0:2:24
        idx = first(Tvec) + Int(round(hour_offset / 24 * n_intervals))
        push!(ticks, idx)
        clock_hour = Int(mod(t_start + hour_offset, 24))
        push!(labels, lpad(string(clock_hour), 2, '0') * ":00")
    end
    return ticks, labels
end

"""
Convert travel-time input into an integer arc travel-step matrix `travel_steps[i,j]`.

Expected interpretation:
- `tau_trv[i,j]` is travel time between nodes i and j in TIME INDICES.

Returned matrix is integer-valued with:
- zero on the diagonal,
- at least one step for positive off-diagonal travel time.
"""
function normalize_travel_steps(tau_trv, N)
    n = length(N)
    travel_steps = zeros(Int, n, n)
    
    if isa(tau_trv, AbstractMatrix)
        for i in N, j in N
            if i == j
                travel_steps[i, j] = 0
            else
                v = tau_trv[i, j]
                travel_steps[i, j] = v <= 0 ? 1 : max(1, Int(round(v)))
            end
        end
    else
        error("tau_trv must be an arc travel-time matrix in time indices.")
    end
    return travel_steps
end

"""
Return a road-energy-consumption matrix k_way[i,j] in kWh/mile.

Accepted forms:
- scalar: same value on every off-diagonal arc
- matrix: arc-specific values
"""
function get_k_way_matrix(k_trv, N)
    n = length(N)
    k_way = zeros(Float64, n, n)

    if isa(k_trv, Number)
        for i in N, j in N
            k_way[i, j] = (i == j) ? 0.0 : Float64(k_trv)
        end
    elseif isa(k_trv, AbstractMatrix)
        for i in N, j in N
            k_way[i, j] = Float64(k_trv[i, j])
        end
    else
        error("k_trv must be a scalar or matrix representing road energy consumption k_way.")
    end

    return k_way
end

"""
Return `value(x)` when available, otherwise 0.0.
"""
safe_value(x) =
    try
        value(x)
    catch
        0.0
    end

"""
For interval-indexed quantities. Value at interval k is drawn over [k, k+1].
Thus K=1:96 is plotted from boundary index 1 through boundary index 97.
"""
function stepify_interval_values(K, values)
    Kvec = collect(K)
    x_step = Int[]
    y_step = eltype(values)[]

    for (idx, k) in enumerate(Kvec)
        # value at interval index k is held over boundary interval [k, k+1]

        push!(x_step, k)
        push!(y_step, values[idx])
        push!(x_step, k + 1)
        push!(y_step, values[idx])
    end

    return x_step, y_step
end

"""
For boundary-indexed states. Value at boundary t is held until the next boundary.
The last value is drawn at last(T) without extending beyond 24:00.
"""
function stepify_boundary_values(T, values)
    Tvec = collect(T)
    x_step = Int[]
    y_step = eltype(values)[]

    if isempty(Tvec)
        return x_step, y_step
    end

    for idx in 1:(length(Tvec)-1)
        push!(x_step, Tvec[idx])
        push!(y_step, values[idx])
        push!(x_step, Tvec[idx+1])
        push!(y_step, values[idx])
    end
    push!(x_step, last(Tvec))
    push!(y_step, values[end])
    return x_step, y_step
end

"""
Build a DataFrame with one row per interval k, carrying both the integer
interval index and human-readable start/end clock labels.

Used as the base table that per-quantity reporters append columns to.
"""
function interval_time_dataframe(K, time_labels)
    Kvec = collect(K)
    return DataFrame(
        Time_Period=Kvec,
        Time_Start_Label=[time_labels[k] for k in Kvec],
        Time_End_Label=[time_labels[k + 1] for k in Kvec],
    )
end

"""
Per-MCS power profile plot and CSV. Shows grid-side charging (positive)
and site-side discharging (negative) for each MCS individually.
"""
function individual_mcs_power_profiles(model, M, K, time_labels; t_start::Real=0)
    Tplot = 1:(length(K) + 1)
    readable_T, readable_times = create_fixed_2hour_xticks(Tplot, t_start)
    mcs_plots = Any[]
    mcs_csv_data = DataFrame[]

    for m in M
        p = plot(
            title="MCS $m",
            titlefontsize=18,
            xlabel="Time",
            ylabel="Power (kW)",
            xticks=(readable_T, readable_times),
            xlims=(first(Tplot), last(Tplot)),
            size=(900, 500),
            xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
            bottom_margin=18Plots.mm,
            left_margin=16Plots.mm,
            right_margin=14Plots.mm
        )

        charging = [safe_value(model[:P_ch_tot][m, k]) for k in K]
        discharging = [safe_value(model[:P_dch_tot][m, k]) for k in K]

        xchg, ychg = stepify_interval_values(K, charging)
        xdch, ydch = stepify_interval_values(K, -discharging)
        plot!(p, xchg, ychg, label="Charging", alpha=0.8, linewidth=2)
        plot!(p, xdch, ydch, label="Discharging", alpha=0.6, linewidth=2)
        hline!(p, [0.0], color=:black, linestyle=:dash, alpha=0.5, label=nothing)
        ymax_abs = max(maximum(charging), maximum(discharging), 1.0)
        ylims!(p, (-1.1 * ymax_abs, 1.1 * ymax_abs))

        csv_data = interval_time_dataframe(K, time_labels)
        csv_data[!, "Charging_Power_kW"] = charging
        csv_data[!, "Discharging_Power_kW"] = discharging
        csv_data[!, "Net_Power_kW"] = charging .- discharging

        push!(mcs_plots, p)
        push!(mcs_csv_data, csv_data)
    end

    return mcs_plots, mcs_csv_data
end

"""
Aggregate grid-side charging and site-side discharging power summed over
all MCSs. Used to inspect the fleet-level grid draw against price/peak windows.
"""
function all_mcs_power_profile(model, M, K, time_labels; t_start::Real=0)
    Tplot = 1:(length(K) + 1)
    readable_T, readable_times = create_fixed_2hour_xticks(Tplot, t_start)
    p = plot(
        title="",
        xlabel="Time",
        ylabel="Power (kW)",
        xticks=(readable_T, readable_times),
        xlims=(first(Tplot), last(Tplot)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
        bottom_margin=18Plots.mm,
        left_margin=16Plots.mm,
        right_margin=14Plots.mm
    )

    total_charging = [sum(safe_value(model[:P_ch_tot][m, k]) for m in M) for k in K]
    total_discharging = [sum(safe_value(model[:P_dch_tot][m, k]) for m in M) for k in K]

    xchg, ychg = stepify_interval_values(K, total_charging)
    xdch, ydch = stepify_interval_values(K, -total_discharging)
    plot!(p, xchg, ychg, label="Total Charging (Grid)", alpha=0.8, linewidth=2)
    plot!(p, xdch, ydch, label="Total Discharging (CEVs)", alpha=0.6, linewidth=2)
    hline!(p, [0.0], color=:black, linestyle=:dash, alpha=0.5, label=nothing)
    ymax_abs = max(maximum(total_charging), maximum(total_discharging), 1.0)
    ylims!(p, (-1.1 * ymax_abs, 1.1 * ymax_abs))

    csv_data = interval_time_dataframe(K, time_labels)
    csv_data[!, "Total_Charging_Power_kW"] = total_charging
    csv_data[!, "Total_Discharging_Power_kW"] = total_discharging
    csv_data[!, "Net_Power_kW"] = total_charging .- total_discharging

    return p, csv_data
end

"""
MCS state-of-energy trajectories with min/max bound lines overlaid.
Boundary-indexed: one value per t in T (= K+1 boundaries).
"""
function mcs_energy_profiles(model, M, K, T, time_labels, SOE_MCS_max, SOE_MCS_min; t_start::Real=0)
    readable_T, readable_times = create_fixed_2hour_xticks(T, t_start)
    p = plot(
        title="",
        xlabel="Time",
        ylabel="State of Energy (kWh)",
        xticks=(readable_T, readable_times),
        xlims=(first(T), last(T)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
        bottom_margin=18Plots.mm,
        left_margin=16Plots.mm,
        right_margin=14Plots.mm
    )

    csv_data = DataFrame(Time_Period=T, Time_Label=time_labels)
    colors = [:blue, :red, :green, :purple, :orange, :brown]

    for (idx, m) in enumerate(M)
        color = colors[mod1(idx, length(colors))]
        soe_values = [safe_value(model[:SOE_MCS][m, t]) for t in T]
        plot!(p, collect(T), soe_values, label="MCS $m", color=color, linewidth=2)

        csv_data[!, "MCS_$(m)_SOE_kWh"] = soe_values
        csv_data[!, "MCS_$(m)_Max_SOE_kWh"] = fill(SOE_MCS_max[m], length(T))
        csv_data[!, "MCS_$(m)_Min_SOE_kWh"] = fill(SOE_MCS_min[m], length(T))
    end

    hline!(p, [SOE_MCS_max[m] for m in M], color=:black, linestyle=:dash, label="Max Energy")
    hline!(p, [SOE_MCS_min[m] for m in M], color=:gray, linestyle=:dash, label="Min Energy")

    return p, csv_data
end

"""
CEV state-of-energy trajectories with min/max bound lines overlaid.
Same boundary indexing as `mcs_energy_profiles`.
"""
function cev_energy_profiles(model, E, K, T, time_labels, SOE_CEV_max, SOE_CEV_min; t_start::Real=0)
    readable_T, readable_times = create_fixed_2hour_xticks(T, t_start)
    p = plot(
        title="",
        xlabel="Time",
        ylabel="State of Energy (kWh)",
        xticks=(readable_T, readable_times),
        xlims=(first(T), last(T)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
        bottom_margin=18Plots.mm,
        left_margin=16Plots.mm,
        right_margin=14Plots.mm
    )

    csv_data = DataFrame(Time_Period=T, Time_Label=time_labels)
    colors = [:blue, :red, :green, :purple, :orange, :brown]

    for (idx, e) in enumerate(E)
        color = colors[mod1(idx, length(colors))]
        soe_values = [safe_value(model[:SOE_CEV][e, t]) for t in T]
        plot!(p, collect(T), soe_values, label="CEV $e", color=color, linewidth=2)

        csv_data[!, "CEV_$(e)_SOE_kWh"] = soe_values
        csv_data[!, "CEV_$(e)_Max_SOE_kWh"] = fill(SOE_CEV_max[e], length(T))
        csv_data[!, "CEV_$(e)_Min_SOE_kWh"] = fill(SOE_CEV_min[e], length(T))
    end

    hline!(p, [SOE_CEV_max[e] for e in E], color=:black, linestyle=:dash, label="CEV Max")
    hline!(p, [SOE_CEV_min[e] for e in E], color=:gray, linestyle=:dash, label="CEV Min")

    return p, csv_data
end

#Work profile plot and csv file — one subplot per site (multi-panel),
# plus a flat single-panel overlay for use inside the combined summary
# (nested multi-panel layouts can't be saved with savefig).
function site_all_work_profiles(model, N_c, E, K, time_labels; t_start::Real=0)
    Tplot = 1:(length(K) + 1)
    readable_T, readable_times = create_fixed_2hour_xticks(Tplot, t_start)
    cev_colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]
    site_colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

    csv_data = interval_time_dataframe(K, time_labels)
    site_plots = Any[]

    site_totals = Dict(i => [sum(safe_value(model[:P_work][i, e, k]) for e in E) for k in K] for i in N_c)
    y_max = maximum(vcat(values(site_totals)...); init=0.0)
    y_lim = y_max > 0 ? (0, 1.1 * y_max) : (0, 1)

    # Flat overlay version (one panel, one line per site) — safe to nest.
    p_overlay = plot(
        title="",
        xlabel="Time",
        ylabel="Power (kW)",
        xticks=(readable_T, readable_times),
        xlims=(first(Tplot), last(Tplot)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
        bottom_margin=18Plots.mm,
        left_margin=16Plots.mm,
        right_margin=14Plots.mm,
    )

    for (idx, i) in enumerate(N_c)
        p_site = plot(
            title="Site $i",
            titlefontsize=18,
            xlabel="Time",
            ylabel="Power (kW)",
            xticks=(readable_T, readable_times),
            xlims=(first(Tplot), last(Tplot)),
            ylims=y_lim,
            xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
            bottom_margin=18Plots.mm,
            left_margin=16Plots.mm,
            right_margin=14Plots.mm,
            top_margin=2Plots.mm,
            legend=:topright
        )

        # Per-CEV breakdown at this site.
        for (e_idx, e) in enumerate(E)
            cev_work = [safe_value(model[:P_work][i, e, k]) for k in K]
            if !isempty(cev_work) && maximum(cev_work) > 0
                color = cev_colors[mod1(e_idx, length(cev_colors))]
                xstep, ystep = stepify_interval_values(K, cev_work)
                plot!(p_site, xstep, ystep, label="CEV $e", color=color, linewidth=2)
            end
        end

        # Site total overlay on the per-site subplot.
        site_work = site_totals[i]
        if !isempty(site_work) && maximum(site_work) > 0
            xstep, ystep = stepify_interval_values(K, site_work)
            plot!(p_site, xstep, ystep, label="Site total", color=:black, linewidth=2, linestyle=:dash)

            # Also add this site's total to the flat overlay.
            site_color = site_colors[mod1(idx, length(site_colors))]
            plot!(p_overlay, xstep, ystep, label="Site $i", color=site_color, linewidth=2)
        end

        csv_data[!, "Site_$(i)_Work_Power_kW"] = site_work
        push!(site_plots, p_site)
    end

    csv_data[!, "Total_Work_Power_kW"] = [sum(safe_value(model[:P_work][i, e, k]) for i in N_c, e in E) for k in K]

    n = length(site_plots)
    p_multi = if n == 0
        plot(title="")
    else
        plot(site_plots...; layout=(n, 1), size=(900, 400 * n), plot_title="Work Power Profiles by Site", plot_titlevspan=0.13)
    end
    return p_multi, p_overlay, csv_data
end

"""
Dual-axis plot of exogenous wholesale electricity price (left, USD/kWh)
and grid CO2 emission factor (right, kg/kWh) over the horizon.
"""
function price_emission_factors(lambda_whl_elec, lambda_CO2, K, time_labels; t_start::Real=0)
    Tplot = 1:(length(K) + 1)
    readable_T, readable_times = create_fixed_2hour_xticks(Tplot, t_start)

    csv_data = interval_time_dataframe(K, time_labels)
    csv_data[!, "Electricity_Price_USD_per_kWh"] = [lambda_whl_elec[k] for k in K]
    csv_data[!, "CO2_Emission_Factor_kg_CO2_per_kWh"] = [lambda_CO2[k] for k in K]

    p = plot(
        title="",
        xlabel="Time",
        ylabel="Electricity Price (\$/kWh)",
        xticks=(readable_T, readable_times),
        xlims=(first(Tplot), last(Tplot)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=11,
        bottom_margin=18Plots.mm,
        left_margin=16Plots.mm,
        right_margin=16Plots.mm,
        top_margin=24Plots.mm,
        legend=(0.01, 1.26),
        grid=true,
        color =:blue
    )
    price_values = [lambda_whl_elec[k] for k in K]
    xstep, ystep = stepify_interval_values(K, price_values)
    plot!(p, xstep, ystep, label="Electricity Price", linewidth=2)

    # Right y-axis: CO2 emission factor
    p_twin = twinx(p)

    co2_values = [lambda_CO2[k] for k in K]
    x_co2, y_co2 = stepify_interval_values(K, co2_values)

    plot!(
        p_twin,
        x_co2,
        y_co2,
        ylabel="CO₂ Emission Factor (kg CO₂/kWh)",
        label=nothing,
        linewidth=2,
        xlims=(first(Tplot), last(Tplot)),
        xticks=(readable_T, readable_times),
        color = :red,
        guidefontsize=18,
        tickfontsize=18
    )

    # Add dummy legend entry for the CO2 curve on the main legend
    plot!(
        p,
        [NaN],
        [NaN],
        label="CO₂ Emission Factor",
        linewidth=2
    )
    return p, csv_data
end

"""
Step plot of each MCS's node index over time.
Y-value 0 is used when no `z[m,*,k]` is set (i.e. the MCS is in transit).
"""
function mcs_location_trajectory(model, M, N, N_g, K, time_labels; t_start::Real=0)
    node_labels = [node in N_g ? "Grid $node" : "Site $node" for node in N]
    ytick_positions = vcat(0, collect(N))
    ytick_labels = vcat("Travel", node_labels)
    Tplot = 1:(length(K) + 1)
    readable_T, readable_times = create_fixed_2hour_xticks(Tplot, t_start)

    p = plot(
        title="",
        xlabel="Time",
        ylabel="Node Type",
        yticks=(ytick_positions, ytick_labels),
        xticks=(readable_T, readable_times),
        xlims=(first(Tplot), last(Tplot)),
        size=(900, 500),
        xrotation=45, guidefontsize=18, tickfontsize=18, legendfontsize=12,
        left_margin=16Plots.mm,
        right_margin=14Plots.mm,
        bottom_margin=18Plots.mm,
        grid=true
    )

    csv_data = interval_time_dataframe(K, time_labels)
    
    colors = [:red, :blue, :green, :purple, :orange]

    for (idx, m) in enumerate(M)
        color = colors[mod1(idx, length(colors))]
        locations = Int[]
        for k in K
            node_here = findfirst(i -> safe_value(model[:z][m, i, k]) > 0.5, N)
            push!(locations, isnothing(node_here) ? 0 : node_here)
        end

        csv_data[!, "MCS_$(m)_Location"] = locations
        csv_data[!, "MCS_$(m)_Location_Type"] = [i == 0 ? "Travel" : (i in N_g ? "Grid" : "Construction") 
        for i in locations]

        xstep, ystep = stepify_interval_values(K, locations)
        plot!(p, xstep, ystep, label="MCS $m", linewidth=2, marker=:circle, markersize=4, color=color)
    end

    return p, csv_data
end

"""
Per-interval long-form export of MCS and CEV operating quantities.
For each MCS: charging, discharging, travel-energy debit, SOE at interval
boundaries, and the sums of routing variables (x, y_trv, z) for debugging.
For each CEV: power received, power consumed for work, and SOE boundaries.
"""
function mcs_cev_energy_profiles(model, N_c, N, M, E, T, K, delta_T, time_labels)
    csv_data = DataFrame(
        Time_Interval=collect(K),
        Time_Period=collect(K),
        Start_Time_Label=time_labels[collect(K)],
        End_Time_Label=time_labels[collect(K) .+ 1],
        Time_Label=time_labels[collect(K) .+ 1]
    )

    for m in M
        mcs_soe_start = [value(model[:SOE_MCS][m, k]) for k in K]
        mcs_soe_end = [value(model[:SOE_MCS][m, k + 1]) for k in K]
        mcs_charging = [value(model[:P_ch_tot][m, k]) for k in K]
        mcs_discharging = [value(model[:P_dch_tot][m, k]) for k in K]
        mcs_traveling = [value(0) for k in K]
        sum_x = [sum(value(model[:x][m, i, j, k]) for i in N, j in N) for k in K]
        sum_y_trv = [sum(value(model[:y_trv][m, i, j, k]) for i in N, j in N) for k in K]
        sum_z = [sum(value(model[:z][m, i, k]) for i in N) for k in K]

        csv_data[!, "MCS_$(m)_Charging_kW"] = mcs_charging
        csv_data[!, "MCS_$(m)_Discharging_kW"] = mcs_discharging
        csv_data[!, "MCS_$(m)_Traveling_kW"] = mcs_traveling
        csv_data[!, "MCS_$(m)_NetPower_kW"] = mcs_charging .- mcs_discharging .- mcs_traveling
        csv_data[!, "MCS_$(m)_SOE_Start_kWh"] = mcs_soe_start
        csv_data[!, "MCS_$(m)_SOE_End_kWh"] = mcs_soe_end
        csv_data[!, "MCS_$(m)_sum_x"] = sum_x
        csv_data[!, "MCS_$(m)_sum_y_trv"] = sum_y_trv
        csv_data[!, "MCS_$(m)_sum_z"] = sum_z
    end

    for e in E
        cev_soe_start = [value(model[:SOE_CEV][e, k]) for k in K]
        cev_soe_end = [value(model[:SOE_CEV][e, k + 1]) for k in K]
        cev_charging = [sum(value(model[:P_MCS_CEV][m, i, e, k]) for m in M, i in N_c) for k in K]
        cev_working = [sum(value(model[:P_work][i, e, k]) for i in N_c) for k in K]

        csv_data[!, "CEV_$(e)_Charging_kW"] = cev_charging
        csv_data[!, "CEV_$(e)_Working_kW"] = cev_working
        csv_data[!, "CEV_$(e)_NetPower_kW"] = cev_charging .- cev_working
        csv_data[!, "CEV_$(e)_SOE_Start_kWh"] = cev_soe_start
        csv_data[!, "CEV_$(e)_SOE_End_kWh"] = cev_soe_end
    end

    return csv_data
end

function solve_and_analyze(
    K_peak, M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
    k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
    SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch_mcs, eta_ch_dch_cev, delta_T, time_labels, work_cap, p_digging, p_loading_swinging, p_traveling, scale, hours_digging, hours_loading_swinging, B,
    peak_demand_limit, lambda_demand_NC, lambda_demand_OP, carbon_price_per_ton;
    require_site_visit::Bool=false,
    single_visit_per_site::Bool=false,
    time_limit_sec::Float64=50.0,
    t_start::Real=0,
    mip_log_path::AbstractString="",
)
    model = Model(HiGHS.Optimizer)
    # Mirror the solver's MIP progress log (gap + elapsed time) to a file so it can be
    # saved as a CSV alongside the run results (parsed by the main driver). Empty path = off.
    if !isempty(mip_log_path)
        set_attribute(model, "log_file", mip_log_path)
    end
    # set_silent(model)
    set_time_limit_sec(model, time_limit_sec)

    # Broadcast scalar parameters into per-MCS / per-CEV vectors so the rest of
    # the formulation can index them by m or e uniformly.
    SOE_MCS_ini_vec = isa(SOE_MCS_ini, Number) ? fill(Float64(SOE_MCS_ini), length(M)) : collect(SOE_MCS_ini)
    SOE_MCS_max_vec = isa(SOE_MCS_max, Number) ? fill(Float64(SOE_MCS_max), length(M)) : collect(SOE_MCS_max)
    SOE_MCS_min_vec = isa(SOE_MCS_min, Number) ? fill(Float64(SOE_MCS_min), length(M)) : collect(SOE_MCS_min)
    CH_MCS_vec       = isa(CH_MCS, Number)       ? fill(Float64(CH_MCS), length(M))       : collect(Float64.(CH_MCS))
    DCH_MCS_vec      = isa(DCH_MCS, Number)      ? fill(Float64(DCH_MCS), length(M))      : collect(Float64.(DCH_MCS))
    DCH_MCS_plug_vec = isa(DCH_MCS_plug, Number) ? fill(Float64(DCH_MCS_plug), length(M)) : collect(Float64.(DCH_MCS_plug))
    C_MCS_plug_vec   = isa(C_MCS_plug, Number)   ? fill(Int(C_MCS_plug), length(M))       : collect(Int.(C_MCS_plug))
    eta_ch_dch_mcs_vec   = isa(eta_ch_dch_mcs, Number)   ? fill(Float64(eta_ch_dch_mcs), length(M))   : collect(Float64.(eta_ch_dch_mcs))
    eta_ch_dch_cev_vec   = isa(eta_ch_dch_cev, Number)   ? fill(Float64(eta_ch_dch_cev), length(E))   : collect(Float64.(eta_ch_dch_cev))
    SOE_CEV_ini_vec = isa(SOE_CEV_ini, Number) ? fill(Float64(SOE_CEV_ini), length(E)) : collect(SOE_CEV_ini)
    SOE_CEV_max_vec = isa(SOE_CEV_max, Number) ? fill(Float64(SOE_CEV_max), length(E)) : collect(SOE_CEV_max)
    SOE_CEV_min_vec = isa(SOE_CEV_min, Number) ? fill(Float64(SOE_CEV_min), length(E)) : collect(SOE_CEV_min)


    B = collect(B)
    @assert length(B) >= 2 "B must contain at least two activity indices: digging and loading/swinging."
    # Map activity index -> nominal power draw (kW). B[1] = digging, B[2] = loading/swinging.
    p_activity = Dict(B[1] => Float64(p_digging), B[2] => Float64(p_loading_swinging), B[3] => Float64(p_traveling))

    # travel_steps[i,j] : integer number of intervals to traverse arc (i,j).
    # k_way[i,j]        : energy (kWh) consumed when traversing arc (i,j).
    travel_steps = normalize_travel_steps(tau_trv, N)
    k_way = get_k_way_matrix(k_trv, N)
    CH_CEV_limit = isa(CH_CEV, AbstractArray) ? collect(CH_CEV) : fill(CH_CEV, length(E))
    # Big-M scale for CEV SOE logic; use each CEV's feasible SOE range.
    cev_soe_range = [max(SOE_CEV_max_vec[e] - SOE_CEV_min_vec[e], 1.0e-6) for e in E]
    # Numerical tolerance for declaring a CEV full at the max-SOE boundary.
    cev_full_tol = [min(1.0e-4, 0.01 * cev_soe_range[e]) for e in E]

    # ---- Power flow variables (interval-indexed, kW) ----
    # Grid -> MCS m at node i during interval k. Non-zero only at grid nodes (i in N_g).
    @variable(model, P_ch_MCS[M, N, K] >= 0)
    # MCS m -> site i during interval k. Non-zero only at construction sites (i in N_c).
    @variable(model, P_dch_MCS[M, N, K] >= 0)
    # MCS m -> CEV e at site i during interval k (port-level allocation of P_dch_MCS).
    @variable(model, P_MCS_CEV[M, N_c, E, K] >= 0)
    # Work power consumed by CEV e at site i during interval k.
    @variable(model, P_work[N_c, E, K] >= 0)
    # Slack on per-site, per-activity work-hours target; penalized in objective.
    @variable(model, s_miss_work[N_c, B] >= 0)
    # Activity indicator: u[e,i,a,k] = 1 iff CEV e performs activity a at site i during interval k.
    @variable(model, u[E, N, B, K], Bin)

    # ---- Travel energy variables (kWh per interval) ----
    # Total grid-side charging power of MCS m in interval k (sum over grid nodes).
    @variable(model, P_ch_tot[M, K] >= 0)
    # Total site-side discharging power of MCS m in interval k (sum over construction sites).
    @variable(model, P_dch_tot[M, K] >= 0)

    # ---- State-of-energy variables (boundary-indexed, kWh) ----
    # Boundary indexing: T = 1..K+1, so SOE[t] is the energy at the start of interval t.
    @variable(model, SOE_MCS[M, T] >= 0)
    # CEV battery state of energy at every time boundary.
    @variable(model, SOE_CEV[E, T] >= 0)

    # ---- Routing & assignment binaries ----
    # rho[m,i,e,k]      : 1 iff CEV e is plugged into MCS m at site i during interval k.
    @variable(model, rho[M, N, E, K], Bin)
    # beta_arr[m,i,k]   : 1 iff MCS m arrives at node i at the start of interval k.
    @variable(model, beta_arr[M, N, K], Bin)
    # beta_dep[m,i,k]  : 1 iff MCS m departs from node i during interval k.
    @variable(model, beta_dep[M, N, K], Bin)
    # x[m,i,j,k]        : 1 iff MCS m departs i toward j at interval k (primitive routing decision).
    @variable(model, x[M, N, N, K], Bin)
    # mu[i,e,k]         : 1 iff CEV e is in charging mode at site i during interval k (mutually exclusive with work).
    @variable(model, mu[N, E, K], Bin)
    # z[m,i,k]          : 1 iff MCS m is parked at node i during interval k.
    @variable(model, z[M, N, K], Bin)
    # y_trv[m,i,j,k]    : in-transit indicator on arc (i,j) during interval k. Equals the count of
    # departures x[m,i,j,*] whose transit window still covers k (see constraint below).
    @variable(model, y_trv[M, N, N, K], Bin)
    # g_ch[m,i,k] = 1 iff MCS m draws grid power at grid node i in interval k (declared here so the objective can use it; grid-exclusivity constraints are below).
    @variable(model, g_ch[M, N_g, K], Bin)
    # Reference cycle helpers: detect CEV charge-cycle starts, full-SOE boundaries, and
    # soft early charge starts before the CEV has reached minimum SOE.
    # charge_start[e,k] is 1 only on the first interval of a CEV charge cycle.
    @variable(model, charge_start[E, K], Bin)
    # cev_full[e,t] is 1 only when CEV e is close enough to SOE_CEV_max at boundary t.
    @variable(model, cev_full[E, T], Bin)
    # early_charge_cev_start_energy penalizes charge starts that occur above min SOE.
    @variable(model, early_charge_cev_start_energy[E, K] >= 0)
    # last_charge_start[e,k] flags the final charge cycle's start (the charge start with no
    # charge start after it). The terminal top-up that returns the CEV to its energy-neutral
    # terminal SOE is necessarily this last cycle, and it legitimately begins well above min
    # SOE when only a little work remains — so it is exempted from the early-charge penalty
    # below. charge_start_after[e,k] = 1 iff a charge start occurs strictly after interval k,
    # and is the bookkeeping used to identify which start is the last one.
    @variable(model, 0 <= charge_start_after[E, K] <= 1)
    @variable(model, 0 <= last_charge_start[E, K] <= 1)

    # Aggregate CEV charge mode from the existing site-indexed mu variables.
    @expression(model, cev_charging[e in E, k in K], sum(mu[i, e, k] for i in N_c))

    # ============================================================================
    # THIS IS THE Reference METHOD.
    #
    # Reference is a rule-based "work-to-min / charge-to-full" cycle baseline. Instead of
    # cost-optimizing against electricity price, carbon intensity, or demand
    # charges, it imposes a simple operating discipline on each CEV:
    #   1. Run the CEV down toward its minimum SOE doing useful work first.
    #   2. Only then start a charge cycle (charge_start), and once a CEV begins
    #      charging it must charge all the way to full (cev_full) before stopping.
    #   3. Starting a charge cycle while still above minimum SOE is softly
    #      penalized (early_charge_cev_start_energy), discouraging premature tops-up.
    # The objective only shapes WHEN work and charging happen — it does not price
    # the grid energy — making Reference a behavioral benchmark to compare against the
    # full cost-optimal method.
    # ============================================================================

    Kvec = collect(K)
    # Total unmet work-hours across all sites and activity types.
    missed_work_term = sum(s_miss_work[i, a] for i in N_c, a in B)
    # Time-weighted work placement: later work receives a larger penalty.
    early_work_term = sum(idx * u[e, i, a, Kvec[idx]] for e in E, i in N_c, a in B, idx in eachindex(Kvec))
    # Penalize later MCS grid charging (idx-weighted, like early_work_term) to push charging earlier.
    early_charge_term = sum(idx * g_ch[m, i, Kvec[idx]] for m in M, i in N_g, idx in eachindex(Kvec))
    # Convert early CEV charge-start energy into equivalent work-hours for the penalty.
    cev_work_power_ref = max(maximum([p_activity[a] for a in B]), 1.0e-6)
    early_charge_cev_start_hours_term = sum(early_charge_cev_start_energy[e, k] / cev_work_power_ref for e in E, k in K)
    #  routing penalty to avoid arbitrary travel when multiple schedules tie.
    travel_tiebreak_term = sum(y_trv[m, i, j, k] for m in M, i in N, j in N, k in K)

    # Minimize the Reference cycle-policy objective with missed work as the dominant term.
    @objective(model, Min,
        rho_miss * missed_work_term +
        early_work_term +
        rho_miss * early_charge_cev_start_hours_term +
        early_charge_term +
        rho_labor * delta_T * travel_tiebreak_term
    )

    # Total MCS grid-side charging power is the sum of per-grid-node charging contributions.
    @constraint(model, [m in M, k in K], P_ch_tot[m, k] == sum(P_ch_MCS[m, i, k] for i in N_g))
    # Total MCS site-side discharging power is the sum of per-site discharging contributions.
    @constraint(model, [m in M, k in K], P_dch_tot[m, k] == sum(P_dch_MCS[m, i, k] for i in N_c))

    # MCS cannot discharge at a grid node — discharging only happens at construction sites.
    @constraint(model, [m in M, i in N_g, k in K], P_dch_MCS[m, i, k] == 0)
    # MCS cannot charge at a construction site — charging only happens at grid nodes.
    @constraint(model, [m in M, i in N_c, k in K], P_ch_MCS[m, i, k] == 0)

    # Energy balance at the discharge port: power flowing out of the MCS at a site equals
    # the sum of power delivered into each CEV plugged in at that site.
    @constraint(model, [m in M, i in N_c, k in K],
        P_dch_MCS[m, i, k] == sum(P_MCS_CEV[m, i, e, k] for e in E)
    )

    # Per-node charging cap — commented out as redundant: for grid nodes the g_ch block below gives
    # P_ch_MCS <= CH_MCS*g_ch with g_ch <= z (hence <= CH_MCS*z), and at construction sites P_ch_MCS == 0 already.
    # @constraint(model, [m in M, i in N, k in K], P_ch_MCS[m, i, k] <= CH_MCS_vec[m] * z[m, i, k])
    # Per-site discharging power is zero unless the MCS is physically at that site (z=1), capped by DCH_MCS.
    # Scoped to N_c: discharging at grid nodes (N_g) is already forced to 0 above, so those rows were vacuous.
    @constraint(model, [m in M, i in N_c, k in K], P_dch_MCS[m, i, k] <= DCH_MCS_vec[m] * z[m, i, k])

    # Grid-connection exclusivity: at most one MCS may CHARGE from each grid node at a time,
    # even though any number may be physically parked there (presence z is not restricted).
    # Grid charging power is nonzero only when the MCS is charge-active at that node (capped by CH_MCS).
    @constraint(model, [m in M, i in N_g, k in K], P_ch_MCS[m, i, k] <= CH_MCS_vec[m] * g_ch[m, i, k])
    # An MCS can only be charge-active where it is physically present.
    @constraint(model, [m in M, i in N_g, k in K], g_ch[m, i, k] <= z[m, i, k])
    

    # Power into each CEV from each MCS is bounded by the plug rate, and is only non-zero when
    # the (MCS,site,CEV) pairing rho is active.
    @constraint(model, [m in M, i in N_c, e in E, k in K],
        P_MCS_CEV[m, i, e, k] <= DCH_MCS_plug_vec[m] * rho[m, i, e, k]
    )

    # CEV acceptance: total power into CEV e (from any MCS at any site) is capped by the CEV's
    # own charging rate and only when CEV e is in charging state (mu=1) at site i.
    @constraint(model, [i in N_c, e in E, k in K],
        sum(P_MCS_CEV[m, i, e, k] for m in M) <= CH_CEV_limit[e] * mu[i, e, k]
    )

    # Charge-mode/pairing link: CEV e is in charging mode at site i (mu=1) exactly when it is
    # paired with one MCS there. Summing rho over M ties mu to the (MCS,site,CEV) assignment and,
    # since mu is binary, also enforces that at most one MCS charges a given CEV at a site/interval.
    @constraint(model, [i in N_c, e in E, k in K],
       mu[i, e, k] == sum(rho[m, i, e, k] for m in M)
    )

    # Reference grid-charge cap: total MCS grid draw cannot exceed aggregate CEV charge acceptance.
    # @constraint(model, [k in K], sum(P_ch_tot[m, k] for m in M) <= sum(CH_CEV_limit[e] for e in E))


    # In-transit indicator on arc (i,j): y_trv[m,i,j,k] = 1 iff a departure x[m,i,j,τ] = 1
    # occurred within the past travel_steps[i,j] intervals (i.e. τ ∈ [k - travel_steps + 1, k]),
    # which means MCS m is currently traversing arc (i,j) during interval k.
    @constraint(model, [m in M, i in N, j in N, k in K; i != j],
    y_trv[m, i, j, k] ==
        sum(x[m, i, j, τ]
            for τ in max(first(K), k - travel_steps[i,j] + 1):k
            if τ in K)
)

    # Combined cap: work is bounded by the required-work profile AND by the not-charging state.
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] <= R_work[i, e, k] * A[i, e] * (1 - mu[i, e, k])
    )

    # Initial state of energy at boundary 1: each MCS starts at its configured initial SOE.
    @constraint(model, [m in M], SOE_MCS[m, first(T)] == SOE_MCS_ini_vec[m])
    # Initial state of energy at boundary 1: each CEV starts at its configured initial SOE.
    @constraint(model, [e in E], SOE_CEV[e, first(T)] == SOE_CEV_ini_vec[e])

    # MCS energy dynamics: SOE at boundary k+1 = SOE at boundary k + (charge in) * eta
    # - (discharge out) / eta - (travel energy loss). Interval k spans [k, k+1].
    @constraint(model, [m in M, k in K],
        SOE_MCS[m, k + 1] == SOE_MCS[m, k] +
                              eta_ch_dch_mcs_vec[m] * P_ch_tot[m, k] * delta_T -
                              (P_dch_tot[m, k] * delta_T) / eta_ch_dch_mcs_vec[m]
    )

    # CEV energy dynamics: SOE at boundary k+1 = SOE at boundary k + (power received from any MCS)
    # - (power consumed doing work) over the interval.
    @constraint(model, [e in E, k in K],
        SOE_CEV[e, k + 1] == SOE_CEV[e, k] +
                             eta_ch_dch_cev_vec[e] * sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c) * delta_T -
                             sum(P_work[i, e, k] for i in N_c) * delta_T
    )

    # Terminal SOE: MCS must end the horizon at its initial SOE (energy-neutral cycle).
    @constraint(model, [m in M], SOE_MCS[m, last(T)] == SOE_MCS_ini_vec[m])
    # Terminal SOE: CEV must end the horizon at its initial SOE (energy-neutral cycle).
    @constraint(model, [e in E], SOE_CEV[e, last(T)] == SOE_CEV_ini_vec[e])

    # MCS lower SOE bound at every boundary point.
    @constraint(model, [m in M, t in T], SOE_MCS[m, t] >= SOE_MCS_min_vec[m])
    # MCS upper SOE bound at every boundary point.
    @constraint(model, [m in M, t in T], SOE_MCS[m, t] <= SOE_MCS_max_vec[m])
    # CEV lower SOE bound at every boundary point.
    @constraint(model, [e in E, t in T], SOE_CEV[e, t] >= SOE_CEV_min_vec[e])
    # CEV upper SOE bound at every boundary point.
    @constraint(model, [e in E, t in T], SOE_CEV[e, t] <= SOE_CEV_max_vec[e])

    # Reference CEV cycle policy:
    # - mu is the only CEV charge-mode variable; it can be active at at most one
    #   assigned construction site for each CEV and interval.
    # - charge_start flags only the first interval of a CEV charge cycle.
    # - cev_full is 1 only when the CEV boundary SOE is within tolerance of max.
    # - once charging starts, it must continue until the CEV reaches max SOE.
    # CEV charging mode is only meaningful at construction sites, not grid nodes.
    @constraint(model, [i in N_g, e in E, k in K], mu[i, e, k] == 0)
    # CEV charging mode can only be active at the construction site assigned to that CEV.
    @constraint(model, [i in N_c, e in E, k in K], mu[i, e, k] <= A[i, e])
    # A CEV can be in charging mode at no more than one construction site per interval.
    @constraint(model, [e in E, k in K], cev_charging[e, k] <= 1)
    # A CEV cannot work and be in aggregate charging mode in the same interval.
    @constraint(model, [e in E, k in K],
        sum(u[e, i, a, k] for i in N_c, a in B) + cev_charging[e, k] <= 1
    )

    # First interval: any active charging mode is the start of a charge cycle.
    @constraint(model, [e in E], charge_start[e, first(K)] == cev_charging[e, first(K)])
    # Later intervals: detect a 0-to-1 transition in aggregate CEV charging mode.
    @constraint(model, [e in E, k in K[2:end]], charge_start[e, k] >= cev_charging[e, k] - cev_charging[e, k - 1])
    # A charge start can only occur in an interval where the CEV is charging.
    @constraint(model, [e in E, k in K[2:end]], charge_start[e, k] <= cev_charging[e, k])
    # A charge start cannot occur if the CEV was already charging in the previous interval.
    @constraint(model, [e in E, k in K[2:end]], charge_start[e, k] <= 1 - cev_charging[e, k - 1])

    # Full-SOE lower implication: if cev_full=1, SOE must be at least max minus tolerance.
    @constraint(model, [e in E, t in T],
        SOE_CEV[e, t] >= SOE_CEV_max_vec[e] - cev_full_tol[e] - cev_soe_range[e] * (1 - cev_full[e, t])
    )
    # Full-SOE upper implication: if SOE is below max minus tolerance, cev_full must be 0.
    @constraint(model, [e in E, t in T],
        SOE_CEV[e, t] <= SOE_CEV_max_vec[e] - cev_full_tol[e] + cev_soe_range[e] * cev_full[e, t]
    )

    # Soft work-to-min rule: charge starts above min SOE create an energy violation,
    # EXCEPT for the last charge start (last_charge_start), which is the terminal top-up
    # and is allowed to begin above min without penalty. Subtracting range*last_charge_start
    # drops this lower bound to <= 0 on the exempt cycle so its penalty energy can be 0.
    @constraint(model, [e in E, k in K],
        early_charge_cev_start_energy[e, k] >= SOE_CEV[e, k] - SOE_CEV_min_vec[e] -
                                              cev_soe_range[e] * (1 - charge_start[e, k]) -
                                              cev_soe_range[e] * last_charge_start[e, k]
    )
    # No early-charge violation is allowed unless a charge cycle starts in that interval.
    @constraint(model, [e in E, k in K],
        early_charge_cev_start_energy[e, k] <= cev_soe_range[e] * charge_start[e, k]
    )

    # Identify the last charge start so the lower bound above can exempt it.
    # charge_start_after[e,k] is back-recursed to be >= any charge start strictly after k;
    # the solver drives it down to exactly that OR. The final interval has nothing after it.
    @constraint(model, [e in E], charge_start_after[e, last(K)] == 0)
    @constraint(model, [e in E, k in K[1:end-1]], charge_start_after[e, k] >= charge_start[e, k + 1])
    @constraint(model, [e in E, k in K[1:end-1]], charge_start_after[e, k] >= charge_start_after[e, k + 1])
    # last_charge_start = charge_start AND NOT charge_start_after: 1 only at an actual charge
    # start that has no later start. The solver pushes it to 1 there to claim the exemption,
    # so exactly one cycle — the last — is exempted whenever the CEV charges at all.
    @constraint(model, [e in E, k in K], last_charge_start[e, k] <= charge_start[e, k])
    @constraint(model, [e in E, k in K], last_charge_start[e, k] <= 1 - charge_start_after[e, k])

    # Strict charge-to-full rule: if charging and not full at the next boundary, keep charging.
    @constraint(model, [e in E, k in K[1:end-1]],
        cev_charging[e, k + 1] >= cev_charging[e, k] - cev_full[e, k + 1]
    )
    # If charging in the final interval, the terminal boundary must be full.
    @constraint(model, [e in E], cev_charging[e, last(K)] <= cev_full[e, last(T)])

    # ---- Cycle-count cuts (bound tightening only; do not change the feasible set) ----
    # The LP relaxation pays ~0 for early_charge_cev_start_energy: charge_start can go
    # fractional behind the range-wide big-M above, so the dual bound never sees the
    # unavoidable cycle penalty and the MIP gap stalls. The three additions below give
    # the solver a single integer handle (n_cycles): one branch on n_cycles (e.g. 2 vs 3)
    # resolves what would otherwise require enumerating combinations of all the
    # per-interval charge_start binaries.

    # Number of charge cycles of CEV e over the horizon, as an explicit integer.
    @variable(model, n_cycles[E] >= 0, Int)
    @constraint(model, [e in E], n_cycles[e] == sum(charge_start[e, k] for k in K))

    # Cycle-count lower bound: a charge-to-full cycle starts at or above min SOE and
    # ends at max SOE, so each cycle delivers at most the usable range (max - min).
    # Total energy delivered to the CEV therefore requires >= delivered/range cycles.
    @constraint(model, [e in E],
        cev_soe_range[e] * n_cycles[e] >=
        delta_T * sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c, k in K)
    )

    # Early-start energy bookkeeping: every cycle ends full (charge-to-full rule), so
    # summed over cycles, (start SOE - min) = n_cycles*range - total delivered energy.
    # Stating that aggregate identity as a >= cut forces the early-start penalty into
    # the dual bound as soon as branching fixes n_cycles, instead of leaving it hidden
    # behind the per-interval big-M rows.
    # The last cycle is exempt from the per-interval penalty, so it can contribute up to
    # (range - tol) of unused depth that no longer needs to appear on the left. Subtracting
    # (range - tol) * sum(last_charge_start) keeps this cut valid (it never cuts off the
    # exempt solution) while still tightening the bound for the remaining cycles.
    @constraint(model, [e in E],
        sum(early_charge_cev_start_energy[e, k] for k in K) >=
        (cev_soe_range[e] - cev_full_tol[e]) * (n_cycles[e] - sum(last_charge_start[e, k] for k in K)) -
        delta_T * sum(P_MCS_CEV[m, i, e, k] for m in M, i in N_c, k in K)
    )

    # Plug count: at any site, the number of CEVs simultaneously plugged into MCS m
    # cannot exceed that MCS's plug count C_MCS_plug.
    @constraint(model, [m in M, i in N_c, k in K],
        sum(rho[m, i, e, k] for e in E) <= C_MCS_plug_vec[m]
    )

    # A CEV can only be plugged into an MCS at a site if the CEV is assigned to that site (A=1).
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= A[i, e])
    # A CEV can only be plugged into MCS m at site i if MCS m is physically present at site i (z=1).
    @constraint(model, [m in M, i in N, e in E, k in K], rho[m, i, e, k] <= z[m, i, k])

    # Routing: no self-loops (cannot "travel" from a node to itself).
    @constraint(model, [m in M, i in N, k in K], x[m, i, i, k] == 0)

    # Presence partition: at every interval, MCS m is either parked at exactly one node (sum z = 1)
    # OR currently in transit on exactly one arc (sum y_trv = 1). Equality (not <=) is enforceable
    # because y_trv is the in-transit indicator that stays 1 for every interval of the trip.
    @constraint(model, [m in M, k in K],
        sum(z[m, i, k] for i in N) + sum(y_trv[m, i, j, k] for i in N, j in N if i != j) == 1
        )

    # Initial position: MCS must start the horizon at a grid node (either parked there or departing from it).
    @constraint(model, [m in M],
    sum(z[m, i, first(K)] for i in N_g) + sum(x[m, i, j, first(K)] for i in N_g, j in N if i != j) == 1
    )


    # Departure indicator: beta_dep[m,i,k] = 1 iff MCS m departs node i along some outgoing arc at interval k.
    @constraint(model, [m in M, i in N, k in K],
        beta_dep[m, i, k] == sum(x[m, i, j, k] for j in N if j != i)
    )

    # Arrival indicator: beta_arr[m,j,k] = 1 iff MCS m arrives at node j at interval k.
    # An arrival at k corresponds to an arc x[m,i,j,τ] that started travel_steps[i,j] intervals earlier
    # (i.e. τ = k - travel_steps[i,j]). If no such τ falls in K, no valid arrival can land at (j,k).
    for m in M, j in N, k in K
        incoming_terms = Any[]
        for i in N
            if i == j
                continue
            end
            τ = k - travel_steps[i, j]
            if τ in K
                push!(incoming_terms, x[m, i, j, τ])
            end
        end

        if isempty(incoming_terms)
            # No feasible prior departure can arrive at node j in interval k.
            @constraint(model, beta_arr[m, j, k] == 0)
        else
            # Arrival indicator equals the sum of departures whose travel time ends at node j in k.
            @constraint(model, beta_arr[m, j, k] == sum(incoming_terms))
        end
    end

    # Node-presence transition: the change in z at node i between intervals k-1 and k equals
    # (arrivals at i in k) - (departures from i in k). I.e. arrival turns z on, departure turns it off.
    @constraint(model, [m in M, i in N, k in K[2:end]],
        beta_arr[m, i, k] - beta_dep[m, i, k] == z[m, i, k] - z[m, i, k-1]
    )

    # Flow conservation: over the full horizon, total arrivals at every node equal total departures.
    @constraint(model, [m in M, i in N],
        sum(beta_arr[m, i, k] for k in K) == sum(beta_dep[m, i, k] for k in K)
    )

    # An MCS cannot arrive at and depart from the same node in the same interval.
    @constraint(model, [m in M, i in N, k in K],
        beta_arr[m, i, k] + beta_dep[m, i, k] <= 1
    )

    # Optional: each MCS must visit at least one construction site somewhere in the horizon.
    if require_site_visit
        # Enforce at least one construction-site arrival for each MCS when requested.
        @constraint(model, [m in M], sum(beta_arr[m, i, k] for i in N_c, k in K) >= 1)
    end

    # Optional: each MCS can arrive at and depart from a given construction site at most once.
    if single_visit_per_site
        # Limit each MCS to one arrival at each construction site.
        @constraint(model, [m in M, i in N_c], sum(beta_arr[m, i, k] for k in K) <= 1)
        # Limit each MCS to one departure from each construction site.
        @constraint(model, [m in M, i in N_c], sum(beta_dep[m, i, k] for k in K) <= 1)
    end

    # Activity scheduling: u[e,i,a,k] = 1 means CEV e performs activity a at site i during interval k.
    # B[1] is digging, B[2] is loading/swinging.
    # Each CEV performs at most one activity at one site per interval.
    @constraint(model, [i in N_c, e in E, k in K], sum(u[e, i, a, k] for a in B) <= 1)
    # At a given site/interval, a CEV is either working (any activity) or charging (mu=1), not both.
    @constraint(model, [i in N_c, e in E, k in K], sum(u[e, i, a, k] for a in B) + mu[i, e, k] <= 1)
    # A CEV can only perform an activity at a site to which it is assigned (A=1).
    @constraint(model, [i in N_c, e in E, a in B, k in K], u[e, i, a, k] <= A[i, e])
    # Work power equals the activity's nominal power (p_digging or p_loading_swinging) when active.
    @constraint(model, [i in N_c, e in E, k in K],
        P_work[i, e, k] == sum(p_activity[a] * u[e, i, a, k] for a in B)
    )

    # Per-site work requirements (vectors indexed over all nodes; only N_c entries are consumed).
    hours_digging_vec          = isa(hours_digging, Number)          ? fill(Float64(hours_digging), length(N))          : collect(Float64.(hours_digging))
    hours_loading_swinging_vec = isa(hours_loading_swinging, Number) ? fill(Float64(hours_loading_swinging), length(N)) : collect(Float64.(hours_loading_swinging))

    # Per-site digging demand: total digging hours (across all CEVs) plus slack equals the site's
    # required hours_digging. Slack s_miss_work captures unmet demand and is penalized in the objective.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[1], k] for e in E, k in K) +
        (s_miss_work[i, B[1]]) == hours_digging_vec[i]
    )
    # Per-site loading/swinging demand: same pattern as digging, against hours_loading_swinging.
    @constraint(model, [i in N_c],
        delta_T * sum(u[e, i, B[2], k] for e in E, k in K) +
        (s_miss_work[i, B[2]]) == hours_loading_swinging_vec[i]
    )

    # Cumulative precedence: loading/swinging cannot get ahead of digging.
    # By interval k, the cumulative loading/swinging time at a site can't exceed cumulative digging time
    # (scaled by `scale`). This enforces the physical order: dig first, then move material.
    @constraint(model, [i in N_c, k in K],
        sum(u[e, i, B[2], τ] for τ in first(K):k, e in E) <=
        scale * sum(u[e, i, B[1], τ] for τ in first(K):k, e in E)
    )

    # Travel pacing: travels are spaced exactly one per 4 intervals of useful work
    # (digging or loading/swinging). Enforced as a two-sided band on cumulative
    # travel V(k) vs cumulative useful work W(k):  W(k) - 4 <= 4*V(k) <= W(k).
    #   - Upper bound (4*V <= W): the v-th travel can't happen until 4*v useful-work
    #     intervals are done, i.e. there are >= 4 useful-work intervals between two travels.
    #   - Lower bound (4*V >= W - 4): useful work can't get more than 4 ahead of travel,
    #     i.e. the CEV must travel after every 4 useful-work intervals before doing a 5th.
    # Together these pin each travel to a 4-useful-work-interval boundary. Rest/charge
    # intervals in between are not counted as useful work, so they don't affect the spacing.
    work_per_travel = 4
    @constraint(model, [i in N_c, e in E, k in K],
        work_per_travel * sum(u[e, i, B[3], τ] for τ in first(K):k) <=
        sum(u[e, i, a, τ] for a in (B[1], B[2]), τ in first(K):k)
    )
    @constraint(model, [i in N_c, e in E, k in K],
        work_per_travel * sum(u[e, i, B[3], τ] for τ in first(K):k) >=
        sum(u[e, i, a, τ] for a in (B[1], B[2]), τ in first(K):k) - work_per_travel
    )

    # # Minimum activity duration: once an activity starts, it must last at least 2 intervals.
    # # First-interval start indicator: start_act equals u itself at k=1.
    # @constraint(model, [e in E, i in N_c, a in B],
    #     start_act[e, i, a, first(K)] == u[e, i, a, first(K)]
    # )
    # # Activity-start detection: start_act[k] >= 1 whenever u toggles from 0 at k-1 to 1 at k.
    # @constraint(model, [e in E, i in N_c, a in B, k in K[2:end]],
    #     start_act[e, i, a, k] >= u[e, i, a, k] - u[e, i, a, k-1]
    # )
    # # Minimum duration: if an activity starts at k, then u must be 1 at both k and k+1
    # # (i.e. at least 2 consecutive intervals of work).
    # @constraint(model, [e in E, i in N_c, a in B, k in K[1:end-1]],
    #     u[e, i, a, k] + u[e, i, a, k+1] >= 2 * start_act[e, i, a, k]
    # )
    # # No activity may start in the final interval (no room for the required 2-interval block).
    # @constraint(model, [e in E, i in N_c, a in B],
    #     start_act[e, i, a, last(K)] == 0
    # )

    # Rest rule:
    # In any 5 consecutive intervals, each CEV can work in at most 4 intervals.
    @constraint(model, [i in N_c, e in E, k in first(K):(last(K)-4)],
    sum(u[e, i, a, tau] for a in B, tau in k:(k+4)) <= 4
    )

    # ---- Solve and report MIP solution quality ----
    optimize!(model)
    status = termination_status(model)
    println("\nSolution Status: ", status)
    if !(status in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.LOCALLY_SOLVED))
        @warn "Solver terminated with status $status"
    end

    # On TIME_LIMIT we still want the best feasible incumbent + duality gap so that
    # results can be reported with a known MIP gap rather than as "optimal".
    if has_values(model)
        primal = objective_value(model)         # best feasible incumbent for the Reference cycle objective
        dual   = objective_bound(model)         # best dual bound from B&B
        gap    = relative_gap(model)            # |primal - dual| / (1e-10 + |primal|)
        @printf("Reference cycle objective  : %.4f\n", primal)
        @printf("Best bound          : %.4f\n", dual)
        @printf("Relative MIP gap    : %.4f %%\n", 100 * gap)
        @printf("Solve time          : %.2f s\n", solve_time(model))
        @printf("Node count          : %d\n",   MOI.get(model, MOI.NodeCount()))
    else
        @warn "No feasible solution found within time limit (status = $status)"
        return
    end

    greedy_obj_val = objective_value(model)

    # ---- Reference objective term breakdown ----
    # Report each term's raw value and its weighted contribution to the objective.
    raw_missed_work       = value(missed_work_term)
    raw_early_work        = value(early_work_term)
    raw_early_charge_cev  = value(early_charge_cev_start_hours_term)
    raw_early_charge      = value(early_charge_term)
    raw_travel_tiebreak   = value(travel_tiebreak_term)

    w_missed_work       = rho_miss * raw_missed_work
    w_early_work        = raw_early_work
    w_early_charge_cev  = rho_miss * raw_early_charge_cev
    w_early_charge      = raw_early_charge
    w_travel_tiebreak   = rho_labor * delta_T * raw_travel_tiebreak

    println("\nReference objective term breakdown (raw value -> weighted contribution):")
    @printf("  missed_work            : %.4f -> %.4f\n", raw_missed_work,      w_missed_work)
    @printf("  early_work             : %.4f -> %.4f\n", raw_early_work,       w_early_work)
    @printf("  early_charge_cev_start : %.4f -> %.4f\n", raw_early_charge_cev, w_early_charge_cev)
    @printf("  early_charge (grid)    : %.4f -> %.4f\n", raw_early_charge,     w_early_charge)
    @printf("  travel_tiebreak        : %.4f -> %.4f\n", raw_travel_tiebreak,  w_travel_tiebreak)
    @printf("  TOTAL (weighted)       : %.4f  (objective_value = %.4f)\n",
            w_missed_work + w_early_work + w_early_charge_cev + w_early_charge + w_travel_tiebreak,
            greedy_obj_val)

    total_electricity_cost = sum(safe_value(P_ch_tot[m, k]) * lambda_whl_elec[k] * delta_T for m in M, k in K)
    total_carbon_cost = sum(carbon_price_per_ton / 1000.0 * lambda_CO2[k] * safe_value(P_ch_tot[m, k]) * delta_T for m in M, k in K)
    missed_work_cost = sum(rho_miss * safe_value(s_miss_work[i, a]) for i in N_c, a in B)
    travel_cost = rho_labor * delta_T * sum(safe_value(y_trv[m,i,j,k]) for m in M, i in N, j in N, k in K)
    
    total_energy_from_grid = sum(safe_value(P_ch_tot[m, k]) * delta_T for m in M, k in K)
    total_carbon_emissions = sum(safe_value(P_ch_tot[m, k]) * lambda_CO2[k] * delta_T for m in M, k in K)
    grid_power_by_k = Dict(k => sum(safe_value(P_ch_tot[m, k]) for m in M) for k in K)
    nc_peak = maximum(values(grid_power_by_k); init=0.0)
    op_peak = maximum((grid_power_by_k[k] for k in K_peak); init=0.0)
    ncdc_cost = lambda_demand_NC * nc_peak
    opdc_cost = lambda_demand_OP * op_peak
    total_missed_work_hour = sum(safe_value(s_miss_work[i, a]) for i in N_c, a in B)
    total_travel_time_hour = delta_T * sum(safe_value(y_trv[m,i,j,k]) for m in M, i in N, j in N, k in K)
    obj_val = total_electricity_cost + total_carbon_cost + ncdc_cost + opdc_cost + missed_work_cost + travel_cost
    
    @printf("Comparable full KPI cost : %.4f\n", obj_val)
    #@printf("Internal Reference cycle objective : %.4f\n", greedy_obj_val)


    now_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

    individual_mcs_power_plots, individual_mcs_power_csv = individual_mcs_power_profiles(model, M, K, time_labels; t_start=t_start)
    mcs_total_grid_plot, mcs_total_grid_csv = all_mcs_power_profile(model, M, K, time_labels; t_start=t_start)
    mcs_energy_plot, mcs_energy_csv = mcs_energy_profiles(model, M, K, T, time_labels, SOE_MCS_max, SOE_MCS_min; t_start=t_start)
    cev_energy_plot, cev_energy_csv = cev_energy_profiles(model, E, K, T, time_labels, SOE_CEV_max, SOE_CEV_min; t_start=t_start)
    site_work_plot, site_work_overlay_plot, site_work_csv = site_all_work_profiles(model, N_c, E, K, time_labels; t_start=t_start)
    price_emission_plot, price_emission_csv = price_emission_factors(lambda_whl_elec, lambda_CO2, K, time_labels; t_start=t_start)
    mcs_location_plot, mcs_location_csv = mcs_location_trajectory(model, M, N, N_g, K, time_labels; t_start=t_start)
    mcs_cev_soe_csv = mcs_cev_energy_profiles(model, N_c, N, M, E, T, K, delta_T, time_labels)
    
     # Prepare summary text
    summary_text = """
    Optimization Summary
    -------------------
    Number of MCSs: $(length(M))
    Number of CEVs: $(length(E))
    Number of nodes: $(length(N)) (Grid: $(length(N_g)), Construction: $(length(N_c)))
    MCS Charging Rate: $(join(CH_MCS_vec, ", ")) kW
    MCS Discharging Rate: $(join(DCH_MCS_vec, ", ")) kW
    Plugs per MCS: $(join(C_MCS_plug_vec, ", "))
    Time interval: $delta_T h
    Number of intervals: $(length(K))
    Number of time boundaries: $(length(T))
    """
   
    # Create the summary as a dummy plot
    p_summary = plot(legend=false, grid=false, framestyle=:none, xticks=false, yticks=false, left_margin=16Plots.mm, right_margin=14Plots.mm)
    annotate!(p_summary, 0, 0.5, text(summary_text, :black, 12, :left))

    # Create an empty plot for the last cell
    p_empty = plot(legend=false, grid=false, framestyle=:none, xticks=false, yticks=false, left_margin=16Plots.mm, right_margin=14Plots.mm)
    
    # Combine all plots (including total grid power profile, excluding individual MCS power plots which are saved separately)
    p_combined = plot(price_emission_plot, mcs_total_grid_plot, mcs_energy_plot,
    site_work_overlay_plot, cev_energy_plot, mcs_location_plot, p_summary, layout=(4, 2), size=(1800, 2200),
    left_margin=16Plots.mm)

    return model, obj_val, total_electricity_cost, total_carbon_cost, ncdc_cost, opdc_cost, missed_work_cost, travel_cost, 
    total_energy_from_grid, total_carbon_emissions, nc_peak, op_peak, total_missed_work_hour, total_travel_time_hour,
    individual_mcs_power_plots, individual_mcs_power_csv, 
    mcs_total_grid_plot, mcs_total_grid_csv, mcs_energy_plot, mcs_energy_csv,cev_energy_plot, 
    cev_energy_csv, site_work_plot, site_work_csv, price_emission_plot, price_emission_csv,
    mcs_location_plot, mcs_location_csv, mcs_cev_soe_csv, p_combined, gap, status
end

end # module
