using JuMP
using HiGHS
using Plots
gr()  # Ensure GR backend is used for stepwise plotting
using DataFrames
using CSV
using Printf
using Dates

# Include necessary modules
include("helper functions/DataLoader_v4_real.jl")

run_index = Dates.format(now(), "yyyymmdd_HHMMSS")

# optimizer_choice = "test"
# optimizer_choice = "OPTIMAL"
optimizer_choice = "B1"
# optimizer_choice = "B1_old"
# optimizer_choice = "B2"
# optimizer_choice = "B3"
# optimizer_choice = "B4"


# Inf = no solver time limit (must stay a Float64 — the solver kwargs are typed
# ::Float64, so `nothing` throws a TypeError; HiGHS reads Inf as "unlimited").
time_limit_sec = 1 * 3600.0

# No hard cap
peak_demand_limit = nothing


if optimizer_choice == "OPTIMAL"
    include("helper functions/MCS_OPTIMAL_v4_real.jl")
elseif optimizer_choice == "B1"
    include("helper functions/MCS_B1_v4_real.jl")
    elseif optimizer_choice == "B1_old"
    include("helper functions/MCS_B1_v4_real_old.jl")
elseif optimizer_choice == "B2"
    include("helper functions/MCS_B2_v4_real.jl")
elseif optimizer_choice == "B3"
    include("helper functions/MCS_B3_v4_real.jl")
elseif optimizer_choice == "B4"
    include("helper functions/MCS_B4_v4_real.jl")
    elseif optimizer_choice == "test"
    include("helper functions/MCS_test_OPTIMAL_v4_real.jl")
else
    error("Unknown optimizer choice: $(ARGS[2]). Use Optimal, B1, B2, or B3.")
end

using .DataLoader
using .MCSOptimizer


"""
    parse_highs_mip_log(path) -> DataFrame

Parse the MIP search-progress table that HiGHS writes to its `log_file` (the same
"BestBound / BestSol / Gap / ... / Time" rows printed to the terminal) into a tidy
DataFrame, one row per logged update. Non-progress lines are ignored, so it is safe to
point at the full solver log. Returns an empty frame if the file is missing/empty.
"""
function parse_highs_mip_log(path::AbstractString)
    cols = (src=String[], nodes=Int[], in_queue=Int[], leaves=Int[],
            explored_pct=Float64[], best_bound=Float64[], best_sol=Float64[],
            gap_pct=Float64[], cuts=Int[], in_lp=Int[], confl=Int[],
            lp_iters=Int[], time_s=Float64[])
    isfile(path) || return DataFrame(cols)
    # HiGHS abbreviates large magnitudes with a k/m/g/t suffix (e.g. "1005k" = 1.005e6)
    # and prints "inf"/"-inf"/"Large" for unbounded bounds/gaps. Returns nothing on any
    # token it cannot interpret, so callers can skip non-progress rows instead of throwing.
    function parsenum(t)
        s = String(strip(t))
        (s == "inf" || s == "Large") && return Inf
        s == "-inf" && return -Inf
        mult = 1.0
        if !isempty(s)
            c = lowercase(last(s))
            c == 'k' && (mult = 1e3; s = chop(s))
            c == 'm' && (mult = 1e6; s = chop(s))
            c == 'g' && (mult = 1e9; s = chop(s))
            c == 't' && (mult = 1e12; s = chop(s))
        end
        v = tryparse(Float64, s)
        return v === nothing ? nothing : v * mult
    end
    asint(x) = (x === nothing || !isfinite(x)) ? nothing : round(Int, x)
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        toks = split(line)
        endswith(toks[end], "s") || continue            # MIP rows end with the time token
        src = ""
        if length(toks) == 13 && length(toks[1]) == 1 && !occursin(r"^[-\d]", toks[1])
            src = String(toks[1]); toks = toks[2:end]    # strip the optional Src flag
        end
        length(toks) == 12 || continue                   # exactly the 12 data columns
        # toks: Proc InQueue Leaves Expl% BestBound BestSol Gap Cuts InLp Confl LpIters Time
        nodes    = asint(parsenum(toks[1]))
        in_queue = asint(parsenum(toks[2]))
        leaves   = asint(parsenum(toks[3]))
        expl     = parsenum(replace(toks[4], "%" => ""))
        bbound   = parsenum(toks[5])
        bsol     = parsenum(toks[6])
        gap      = parsenum(replace(toks[7], "%" => ""))
        cuts     = asint(parsenum(toks[8]))
        in_lp    = asint(parsenum(toks[9]))
        confl    = asint(parsenum(toks[10]))
        lp_iters = asint(parsenum(toks[11]))
        time_s   = parsenum(replace(toks[12], "s" => ""))
        # Skip any line that does not parse as a full numeric progress row (e.g. headers).
        any(x -> x === nothing, (nodes, in_queue, leaves, expl, bbound, bsol, gap,
                                 cuts, in_lp, confl, lp_iters, time_s)) && continue
        push!(cols.src, src);           push!(cols.nodes, nodes)
        push!(cols.in_queue, in_queue); push!(cols.leaves, leaves)
        push!(cols.explored_pct, expl); push!(cols.best_bound, bbound)
        push!(cols.best_sol, bsol);     push!(cols.gap_pct, gap)
        push!(cols.cuts, cuts);         push!(cols.in_lp, in_lp)
        push!(cols.confl, confl);       push!(cols.lp_iters, lp_iters)
        push!(cols.time_s, time_s)
    end
    return DataFrame(cols)
end




"""
Run the optimization with CSV data and save result
"""
function run_optimization_with_logging(dataset_name::String)
    # Construct paths - handle both relative and absolute paths
    if isdir(joinpath(dataset_name, "csv_files"))
        # Relative path (original behavior)
        data_dir = joinpath(dataset_name, "csv_files")
        results_dir = joinpath(dataset_name, "results")
    else
        # Absolute path (new behavior for backend)
        data_dir = joinpath(dataset_name, "csv_files")
        # Create results directory in the same location as the dataset
        results_dir = joinpath(dirname(dataset_name), "results")
    end

    # Debug: Print the paths to understand what's happening
    println("Dataset name: ", dataset_name)
    #println("Data directory: ", data_dir)
    #println("Results directory: ", results_dir)
    #println("Data directory exists: ", isdir(data_dir))
    #println("CSV files in data directory: ", readdir(data_dir))

    # Ensure results directory exists
    mkpath(results_dir)

    #println("Loading data from CSV files in directory: ", data_dir)

    # Determine which data loader to use based on the dataset
    M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
    k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
    SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch, delta_T, work_cap,
    lambda_demand_NC, lambda_demand_OP, carbon_price_per_ton, p_digging, p_loading_swinging, p_traveling,
    hours_digging, hours_loading_swinging = DataLoader.load_all_data(data_dir)

    # Starting time of the simulation (clock hour). The horizon spans
    # length(K) intervals of delta_T hours starting at t_start, wrapping around
    # midnight. With t_start=6 and delta_T=0.25 the run covers 06:00 -> 06:00 next day.
    t_start = 8

    # Build boundary time labels deterministically from t_start + delta_T so the
    # plots and CSV exports show clock times that match the actual simulation window.
    time_labels = [begin
        clock_min = mod(Int(round(t_start * 60 + k * delta_T * 60)), 24 * 60)
        @sprintf("%02d:%02d", div(clock_min, 60), clock_min % 60)
    end for k in 0:length(K)]
    @assert length(time_labels) == length(T) "time_labels must have one label for each boundary index in T"

    println("Data loaded successfully. Running optimization model...")

    # Record start time
    start_time = time()


    # On-peak window: 4 PM - 9 PM (interval k spans [(k-1)*delta_T, k*delta_T] hours from midnight)
    # Interval k covers [t_start + (k-1)*delta_T, t_start + k*delta_T] mod 24
    function in_peak(k, delta_T, t_start)
        start = mod(t_start + (k - 1) * delta_T, 24)
        stop  = mod(t_start + k * delta_T, 24)
        stop_eff = stop == 0 ? 24 : stop
        return start >= 16 && stop_eff <= 21
    end

    K_peak = [k for k in K if in_peak(k, delta_T, t_start)]

    scale = 2
    B = [1, 2, 3]  # activity indices: 1=digging, 2=loading/swinging

    # Solver time limit (seconds). HiGHS will return the best feasible solution
    # found so far if it hits this cap.

    # Path the solver mirrors its MIP progress log to (gap + elapsed time). Parsed into a
    # CSV after the solve. run_index keeps it unique per run; clear any stale file first.
    mip_log_path = joinpath(tempdir(), "highs_mip_$(run_index).log")
    rm(mip_log_path; force=true)

    # Solve the model and analyze results
    model, obj_val, total_electricity_cost, total_carbon_cost, ncdc_cost, opdc_cost, missed_work_cost, travel_cost, 
    total_energy_from_grid, total_carbon_emissions, nc_peak, op_peak, total_missed_work_hour, total_travel_time_hour,
    individual_mcs_power_plots, individual_mcs_power_csv, 
    mcs_total_grid_plot, mcs_total_grid_csv, mcs_energy_plot, mcs_energy_csv,cev_energy_plot, 
    cev_energy_csv, site_work_plot, site_work_csv, price_emission_plot, price_emission_csv,
    mcs_location_plot, mcs_location_csv, mcs_cev_soe_csv, p_combined, gap, status  = Base.invokelatest(MCSOptimizer.solve_and_analyze,
        K_peak, M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
        k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
        SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch, delta_T, time_labels, work_cap, p_digging, p_loading_swinging, p_traveling, scale, hours_digging, hours_loading_swinging, B,
        peak_demand_limit, lambda_demand_NC, lambda_demand_OP, carbon_price_per_ton;
        t_start=t_start,
        time_limit_sec=time_limit_sec,
        mip_log_path=mip_log_path
    )

    # Calculate solve time
    solve_time = time() - start_time
    println("Solve time is: ", round(solve_time, digits=3), " seconds")


    # Calculate work completion percentage
    total_required_work = sum(hours_digging[i] * p_digging + hours_loading_swinging[i] * p_loading_swinging for i in N_c)
    total_completed_work = sum(value.(model[:P_work][i, e, k]) * delta_T for i in N_c, e in E, k in K)
    work_completion_percentage = (total_completed_work / total_required_work) * 100

    # Before saving any results, generate a run-specific directory name.
    run_name = "$(optimizer_choice)_Site_$(length(N_c))_MCS_$(length(M))_CEV_$(length(E))_$(run_index)"
    run_dir = joinpath(results_dir, run_name)
    mkpath(run_dir)

    # Save the solver's MIP convergence trace (best bound, incumbent, gap, elapsed time)
    # as CSV, and keep the raw solver log next to it. Skipped if the solver ran silently.
    if isfile(mip_log_path)
        try
            mip_log_df = parse_highs_mip_log(mip_log_path)
            if nrow(mip_log_df) > 0
                CSV.write(joinpath(run_dir, "10_mip_convergence.csv"), mip_log_df)
                cp(mip_log_path, joinpath(run_dir, "10_mip_convergence.log"); force=true)
            end
        catch err
            @warn "Could not parse/save HiGHS MIP convergence log" exception=err
        end
    end

    # Save individual plots for each of the 8 subplots
    savefig(mcs_total_grid_plot, joinpath(run_dir, "01_total_grid_power_profile.png"))
    savefig(site_work_plot, joinpath(run_dir, "02_work_profiles_by_site.png"))
    savefig(mcs_energy_plot, joinpath(run_dir, "03_mcs_state_of_energy.png"))
    savefig(cev_energy_plot, joinpath(run_dir, "04_cev_state_of_energy.png"))
    savefig(price_emission_plot, joinpath(run_dir, "05_electricity_prices_emissions.png"))
    savefig(mcs_location_plot, joinpath(run_dir, "06_mcs_location_trajectory.png"))
    savefig(p_combined, joinpath(run_dir, "07_mcs_optimization_summary.png"))     # Save the main combined optimization results plot



    # Save CSV data for each plot
    CSV.write(joinpath(run_dir, "01_total_grid_power_profile.csv"), mcs_total_grid_csv)
    CSV.write(joinpath(run_dir, "02_work_profiles_by_site.csv"), site_work_csv)
    CSV.write(joinpath(run_dir, "03_mcs_state_of_energy.csv"), mcs_energy_csv)
    CSV.write(joinpath(run_dir, "04_cev_state_of_energy.csv"), cev_energy_csv)
    CSV.write(joinpath(run_dir, "05_electricity_prices.csv"), price_emission_csv)
    CSV.write(joinpath(run_dir, "06_mcs_location_trajectory.csv"), mcs_location_csv)
    CSV.write(joinpath(run_dir, "07_mcs_cev_soe.csv"), mcs_cev_soe_csv)


    # -------------------------------------------------------------------------
    # Export: electricity cost + CO2 calculations (auto-generated every run)
    # -------------------------------------------------------------------------
    if all(["Total_Charging_Power_kW", "Time_Period"] .∈ Ref(names(mcs_total_grid_csv))) &&
       all(["Electricity_Price_USD_per_kWh", "CO2_Emission_Factor_kg_CO2_per_kWh", "Time_Period"] .∈ Ref(names(price_emission_csv)))
        # Align by Time_Period (should already match)
        price_by_t = Dict(Int(r.Time_Period) => r for r in eachrow(price_emission_csv))

        energy_kwh = Float64[]
        energy_cost_usd = Float64[]
        co2_kg = Float64[]
        cumulative_cost_usd = Float64[]
        cumulative_co2_kg = Float64[]

        running_cost = 0.0
        running_co2 = 0.0

        for r in eachrow(mcs_total_grid_csv)
            t = Int(r.Time_Period)
            p_kw = Float64(r.Total_Charging_Power_kW)
            e_kwh = p_kw * delta_T
            pr = price_by_t[t]
            cost = e_kwh * Float64(pr.Electricity_Price_USD_per_kWh)
            co2 = e_kwh * Float64(pr.CO2_Emission_Factor_kg_CO2_per_kWh)

            running_cost += cost
            running_co2 += co2

            push!(energy_kwh, e_kwh)
            push!(energy_cost_usd, cost)
            push!(co2_kg, co2)
            push!(cumulative_cost_usd, running_cost)
            push!(cumulative_co2_kg, running_co2)
        end

        cost_emissions_ts = DataFrame(
            Time_Period=mcs_total_grid_csv.Time_Period,
            Time_Start_Label=mcs_total_grid_csv.Time_Start_Label,
            Time_End_Label=mcs_total_grid_csv.Time_End_Label,
            Grid_Energy_kWh=energy_kwh,
            Energy_Cost_USD=energy_cost_usd,
            CO2_Emissions_kg=co2_kg,
            Cumulative_Energy_Cost_USD=cumulative_cost_usd,
            Cumulative_CO2_Emissions_kg=cumulative_co2_kg
        )

        CSV.write(joinpath(run_dir, "08_cost_emissions_timeseries.csv"), cost_emissions_ts)

        NC_peak_kw = max(maximum(Float64.(mcs_total_grid_csv.Total_Charging_Power_kW)), 0)
        NC_demand_charge_usd = NC_peak_kw * lambda_demand_NC

        op_peak_mask = in.(mcs_total_grid_csv.Time_Period, Ref(Set(K_peak)))
        op_peak_powers = Float64.(mcs_total_grid_csv.Total_Charging_Power_kW[op_peak_mask])
        OP_peak_kw = isempty(op_peak_powers) ? 0.0 : max(maximum(op_peak_powers), 0)
        OP_demand_charge_usd = OP_peak_kw * lambda_demand_OP


        cost_emissions_totals = DataFrame(
            
            Metric=["Objective_Value", "Total_Energy_Cost_USD", "Total_CO2_Cost_USD", "NC demand charge_USD", "OP demand charge_USD" , "Total_Missed_Work_Penalty_USD", "Total_Travel_Penalty_USD",
            "Total_Grid_Energy_kWh", "Total_CO2_Emissions_kg", "NCD_Peak_kW",  "OPD_Peak_kW", "Total_Missed_Work_hour", "Total_MCS_Travel_hour", "MIP_Gap_percent", "Status", "Solve_Time_s"],
            
            Value=Any[
                round(obj_val, digits=2),
                round(total_electricity_cost, digits=2),
                round(total_carbon_cost, digits=2),
                round(ncdc_cost, digits=2),
                round(opdc_cost, digits=2),
                round(missed_work_cost, digits=2),
                round(travel_cost, digits=2),
                round(total_energy_from_grid, digits=2),
                round(total_carbon_emissions, digits=2),
                round(nc_peak, digits=2),
                round(op_peak, digits=2),
                round(total_missed_work_hour, digits=2),
                round(total_travel_time_hour, digits=2),
                round(100 * gap, digits=2),
                string(status),
                round(JuMP.solve_time(model), digits=2)
            ]
            )


        CSV.write(joinpath(run_dir, "09_cost_kpi_metrics.csv"), cost_emissions_totals)

        cost_labels = ["Energy", "CO₂", "NCD", "OPD", "Missed Work", "Travel", "Total Cost"]
        cost_values = [
            total_electricity_cost,
            total_carbon_cost,
            ncdc_cost,
            opdc_cost,
            missed_work_cost,
            travel_cost,
            obj_val
        ]
        cost_colors = [:steelblue, :forestgreen, :darkorange, :purple, :firebrick, :teal, :black]
        cost_ymax = max(maximum(cost_values), 1.0)
        cost_label_offset = 0.13 * cost_ymax
        energy_ymax = max(total_energy_from_grid, 1.0)
        energy_label_offset = 0.06 * energy_ymax

        p_kpi_costs = plot(
            title="",
            xlabel="(a)",
            ylabel="Cost (USD)",
            xticks=(1:length(cost_labels), cost_labels),
            xlims=(0.5, length(cost_labels) + 0.5),
            ylims=(0, 1.45 * cost_ymax),
            legend=false,
            xrotation=25,
            guidefontsize=18,
            tickfontsize=18,
            size=(1100, 450),
            bottom_margin=14Plots.mm,
            left_margin=14Plots.mm,
            right_margin=12Plots.mm
        )
        for i in eachindex(cost_labels)
            bar!(p_kpi_costs, [i], [cost_values[i]], color=cost_colors[i], label=false, bar_width=0.65)
            annotate!(
                p_kpi_costs,
                i,
                cost_values[i] + cost_label_offset,
                text(@sprintf("\$%.1f", cost_values[i]), :black, 20, :center)
            )
        end
        p_kpi_costs_twin = twinx(p_kpi_costs)
        scatter!(
            p_kpi_costs_twin,
            [1],
            [total_energy_from_grid],
            ylabel="Grid Energy (kWh)",
            color=:navy,
            marker=:diamond,
            markersize=10,
            markerstrokecolor=:white,
            markerstrokewidth=1.5,
            label=false,
            xlims=(0.5, length(cost_labels) + 0.5),
            ylims=(0, 1.25 * energy_ymax),
            xticks=(1:length(cost_labels), cost_labels),
            guidefontsize=18,
            tickfontsize=18
        )
        annotate!(
            p_kpi_costs_twin,
            1,
            total_energy_from_grid + energy_label_offset,
            text(@sprintf("%.1f kWh", total_energy_from_grid), :navy, 18, :center)
        )

        kpi_x = 1:4
        peak_ymax = max(nc_peak, op_peak, 1.0)
        hour_ymax = max(total_missed_work_hour, total_travel_time_hour, 1.0)
        p_kpi_ops = plot(
            title="",
            xlabel="(b)",
            ylabel="Demand Peak (kW)",
            xticks=(kpi_x, ["NCD Peak", "OPD Peak", "Missed Work", "Travel"]),
            xlims=(0.5, 4.5),
            ylims=(0, 1.15 * peak_ymax),
            legend=false,
            guidefontsize=18,
            tickfontsize=18,
            size=(1100, 450),
            bottom_margin=14Plots.mm,
            left_margin=14Plots.mm,
            right_margin=14Plots.mm
        )
        bar!(p_kpi_ops, [1], [nc_peak], color=:darkorange, label=false, bar_width=0.55)
        bar!(p_kpi_ops, [2], [op_peak], color=:purple, label=false, bar_width=0.55)

        p_kpi_ops_twin = twinx(p_kpi_ops)
        plot!(
            p_kpi_ops_twin,
            ylabel="Hours",
            legend=false,
            xlims=(0.5, 4.5),
            ylims=(0, 1.15 * hour_ymax),
            xticks=(kpi_x, ["NCD Peak", "OPD Peak", "Missed Work", "Travel"]),
            guidefontsize=18,
            tickfontsize=18
        )
        bar!(p_kpi_ops_twin, [3], [total_missed_work_hour], color=:firebrick, label=false, bar_width=0.55)
        bar!(p_kpi_ops_twin, [4], [total_travel_time_hour], color=:teal, label=false, bar_width=0.55)

        p_kpi_summary = plot(
            p_kpi_costs,
            p_kpi_ops,
            layout=(2, 1),
            size=(1200, 900),
            plot_title="KPI Metrics Summary"
        )
        savefig(p_kpi_summary, joinpath(run_dir, "09_kpi_metrics_summary.png"))

        # Plot: cumulative cost + cumulative CO2 (two axes)
        nK = nrow(cost_emissions_ts)
        x = 2:(nK+1)  # cumulative quantities are realized at interval-end boundaries

        tick_hours = 0:2:24
        tick_positions = 1 .+ round.(Int, tick_hours ./ 24 .* nK)
        tick_labels = [@sprintf("%02d:00", mod(t_start + h, 24)) for h in tick_hours]

        p_cost = plot(
            x,
            cost_emissions_ts.Cumulative_Energy_Cost_USD,
            title="",
            xlabel="Time",
            ylabel="Cumulative Cost (USD)",
            label=nothing,
            color=:blue,
            linewidth=2,
            xticks=(tick_positions, tick_labels),
            xlims=(1, nK + 1),
            xrotation=45,
            guidefontsize=18,
            tickfontsize=18,
            legendfontsize=12,
            bottom_margin=18Plots.mm,
            left_margin=16Plots.mm,
            right_margin=14Plots.mm,
            size=(900, 500),
            legend=:topleft
        )

        p_cost_twin = twinx(p_cost)
        plot!(
            p_cost_twin,
            x,
            cost_emissions_ts.Cumulative_CO2_Emissions_kg,
            ylabel="Cumulative CO₂ (kg)",
            label=nothing,
            color=:red,
            linewidth=2,
            xlims=(1, nK + 1),
            xticks=(tick_positions, tick_labels),
            guidefontsize=18,
            tickfontsize=18
        )

        plot!(p_cost, [NaN], [NaN], color=:blue, linewidth=2, label="Cumulative Cost (USD)")
        plot!(p_cost, [NaN], [NaN], color=:red, linewidth=2, label="Cumulative CO₂ (kg)")

        savefig(p_cost, joinpath(run_dir, "08_cost_emissions_summary.png"))
    else
        @warn "Skipping cost/CO2 export: required columns missing in CSV dataframes"
    end

    # Save individual MCS power profile plots and CSV data
    for (m_idx, mcs_plot) in enumerate(individual_mcs_power_plots)
        savefig(mcs_plot, joinpath(run_dir, "mcs_$(m_idx)_power_profile.png"))
        # Save CSV data for this MCS
        CSV.write(joinpath(run_dir, "mcs_$(m_idx)_power_profile.csv"), individual_mcs_power_csv[m_idx])
    end

    return nothing
end

#Run optimization for multiple datasets

function run_multiple_datasets(dataset_names::Vector{String})
    results = Dict{String,Any}()

    for dataset in dataset_names
        println("\nProcessing dataset: $dataset")
        try
            run_optimization_with_logging(dataset)
        catch e
            println("Error processing dataset $dataset: ", e)
        end
    end

    return nothing
end

#function main()
if length(ARGS) > 0
    if ARGS[1] == "--all"
        datasets = filter(x -> isdir(x) && x != ".ipynb_checkpoints", readdir())
        run_multiple_datasets(datasets)
    else
        run_optimization_with_logging(ARGS[1])
    end
else
    #run_optimization_with_logging("real_dataset")
    run_optimization_with_logging("simple_dataset")
end
#end
