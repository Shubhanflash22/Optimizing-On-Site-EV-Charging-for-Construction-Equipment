module DataLoader

using CSV
using DataFrames
using Printf
using LinearAlgebra

export load_all_data

"""
Load all required data from CSV files in the specified directory
"""
function load_all_data(data_dir::String)
    # Load parameters
    params_df = CSV.read(joinpath(data_dir, "parameters.csv"), DataFrame)

    # Create a function to safely convert string values to numbers
    function safe_convert(value::Any)
        str_val = string(value)
        # Remove any trailing commas
        str_val = replace(str_val, r",+$" => "")
        try
            # Try to convert to float first
            float_val = parse(Float64, str_val)
            return float_val
        catch e
            # If conversion fails, return the original value
            return value
        end
    end

    # Convert values and create parameters dictionary
    params = Dict{Symbol,Any}()
    for row in eachrow(params_df)
        key = Symbol(row.Parameter)
        value = safe_convert(row.Value)
        params[key] = value
    end


    # Extract parameters
    k_trv = params[:k_trv]
    delta_T = params[:delta_T]
    rho_miss = params[:rho_miss]
    rho_labor = params[:rho_labor]
    lambda_demand_NC = params[:lambda_demand_NC] 
    lambda_demand_OP = params[:lambda_demand_OP]  
    carbon_price_per_ton = params[:carbon_price_per_ton]

    p_digging = params[:p_digging]
    p_loading_swinging = params[:p_loading_swinging]
    p_traveling = params[:p_traveling]



    # Load EV data
    ev_df = CSV.read(joinpath(data_dir, "ev_data.csv"), DataFrame)
    E = 1:nrow(ev_df)
    SOE_CEV_min = ev_df.SOE_min
    SOE_CEV_max = ev_df.SOE_max
    SOE_CEV_ini = ev_df.SOE_ini
    CH_CEV = ev_df.ch_rate
    work_cap = ev_df.work_cap

    # Load MCS data (one row per MCS).
    mcs_df = CSV.read(joinpath(data_dir, "mcs_data.csv"), DataFrame)
    M = 1:nrow(mcs_df)
    SOE_MCS_min = Float64.(mcs_df.SOE_min)
    SOE_MCS_max = Float64.(mcs_df.SOE_max)
    SOE_MCS_ini = Float64.(mcs_df.SOE_ini)
    CH_MCS = Float64.(mcs_df.CH_MCS)
    DCH_MCS = Float64.(mcs_df.DCH_MCS)
    DCH_MCS_plug = Float64.(mcs_df.DCH_MCS_plug)
    C_MCS_plug = Int.(mcs_df.C_MCS_plug)
    eta_ch_dch = Float64.(mcs_df.eta_ch_dch)

    # Load place data
    place_df = CSV.read(joinpath(data_dir, "place.csv"), DataFrame)
    N = 1:nrow(place_df)
    N_g = [1]  # First location is grid node
    N_c = collect(2:length(N))  # Rest are construction sites

    # Identify CEV columns by name pattern (e1, e2, ...). Any other columns
    # (e.g., hours_digging, hours_loading_swinging) are read separately below.
    place_cols = names(place_df)
    cev_cols = filter(c -> occursin(r"^e\d+$", String(c)), place_cols)
    @assert length(cev_cols) == length(E) "place.csv must have one e<i> column per CEV in ev_data.csv"

    # Create location matrix
    A = zeros(Int, length(N), length(E))
    for (i, row) in enumerate(eachrow(place_df))
        for (e, col) in enumerate(cev_cols)
            A[i, e] = row[col]
        end
    end

    # Per-site work hour requirements. Values for the grid node should be 0
    # (no work happens there); the optimizer only consumes entries for N_c.
    @assert "hours_digging" in place_cols "place.csv must include a hours_digging column"
    @assert "hours_loading_swinging" in place_cols "place.csv must include a hours_loading_swinging column"
    hours_digging = Float64.(place_df.hours_digging)
    hours_loading_swinging = Float64.(place_df.hours_loading_swinging)


    # Load travel time matrix
    tau_trv = Matrix{Float64}(CSV.read(joinpath(data_dir, "travel_time.csv"), DataFrame)[:, 2:end])

    # Load time data
    # K indexes the 96 physical intervals, e.g. [00:00, 00:15], ..., [23:45, 24:00].
    # T indexes the 97 time boundaries, e.g. 00:00, 00:15, ..., 24:00.
    time_df = CSV.read(joinpath(data_dir, "time_data.csv"), DataFrame)
    K = 1:nrow(time_df)
    T = 1:(nrow(time_df) + 1)

    # Use the new intensity_tons_emissions column if available, otherwise fall back to lambda_CO2
    if "intensity_tons_emissions" in names(time_df)
        #println("Using real CAISO CO2 intensity data from 'intensity_tons_emissions' column")
        lambda_CO2 = time_df.intensity_tons_emissions
    else
        #println("Using synthetic CO2 data from 'lambda_CO2' column")
        lambda_CO2 = time_df.lambda_CO2
    end

    lambda_whl_elec = time_df.lambda_buy

    # Load work data
    work_df = CSV.read(joinpath(data_dir, "work_flexible.csv"), DataFrame)
    time_columns = names(work_df)[3:end]  # Skip Location and EV columns - Store all column names from 3rd column onwards

    # Get the maximum location and EV indices from the data
    function get_numeric_value(val)
        if isa(val, Number)
            return Int(val)
        else
            # Try to extract number from string (e.g., "n1" -> 1)
            m = match(r"\d+", string(val))
            return m === nothing ? 1 : parse(Int, m.match)
        end
    end

    max_location = maximum(get_numeric_value(row.Location) for row in eachrow(work_df))
    max_ev = maximum(get_numeric_value(row.EV) for row in eachrow(work_df))

    # Initialize R_work with model dimensions. This avoids dimension mismatches when
    # the work file omits zero-work locations/EVs.
    @assert max_location <= length(N) "work_flexible.csv references a location outside place.csv"
    @assert max_ev <= length(E) "work_flexible.csv references an EV outside ev_data.csv"
    R_work = zeros(length(N), length(E), length(time_columns))

    for row in eachrow(work_df)
        # Skip the first row with t1, t2, etc.
        if startswith(string(row[time_columns[1]]), "t")
            continue
        end

        # Get location and EV indices
        i = get_numeric_value(row.Location)
        e = get_numeric_value(row.EV)

        for (t, col) in enumerate(time_columns)
            R_work[i, e, t] = parse(Float64, string(row[col]))
        end
    end

    # Validate data
    validate_data(
        M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
        k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
        SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch, delta_T
    )

    return M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
    k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
    SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch, delta_T, work_cap,
    lambda_demand_NC, lambda_demand_OP, carbon_price_per_ton, p_digging, p_loading_swinging, p_traveling,
    hours_digging, hours_loading_swinging
end

"""
Validate loaded data
"""
function validate_data(
    M, T, K, N, N_g, N_c, E, A, C_MCS_plug, CH_MCS, CH_CEV, DCH_MCS, DCH_MCS_plug,
    k_trv, R_work, SOE_CEV_ini, SOE_CEV_max, SOE_CEV_min, SOE_MCS_ini, SOE_MCS_max,
    SOE_MCS_min, tau_trv, lambda_whl_elec, lambda_CO2, rho_miss, rho_labor, eta_ch_dch, delta_T
)
    # Check dimensions
    @assert size(tau_trv) == (length(N), length(N)) "Travel time matrix dimensions mismatch"
    @assert length(T) == length(K) + 1 "T must have one more boundary point than interval set K"
    @assert size(R_work) == (length(N), length(E), length(K)) "Work requirements dimensions mismatch"
    @assert length(lambda_whl_elec) == length(K) "Electricity price vector must be interval-indexed"
    @assert length(lambda_CO2) == length(K) "CO2 vector must be interval-indexed"
    @assert size(A) == (length(N), length(E)) "Location matrix dimensions mismatch"

    # Check values
    @assert length(eta_ch_dch) == length(M) "eta_ch_dch length must match number of MCS"
    @assert all(0 .< eta_ch_dch .<= 1) "Efficiency must be between 0 and 1"
    @assert length(SOE_MCS_ini) == length(M) "SOE_MCS_ini length must match number of MCS"
    @assert length(SOE_MCS_max) == length(M) "SOE_MCS_max length must match number of MCS"
    @assert length(SOE_MCS_min) == length(M) "SOE_MCS_min length must match number of MCS"
    @assert length(CH_MCS) == length(M) "CH_MCS length must match number of MCS"
    @assert length(DCH_MCS) == length(M) "DCH_MCS length must match number of MCS"
    @assert length(DCH_MCS_plug) == length(M) "DCH_MCS_plug length must match number of MCS"
    @assert length(C_MCS_plug) == length(M) "C_MCS_plug length must match number of MCS"
    @assert all(SOE_MCS_min .<= SOE_MCS_ini .<= SOE_MCS_max) "Invalid MCS energy limits"
    @assert all(SOE_CEV_min .<= SOE_CEV_ini .<= SOE_CEV_max) "Invalid CEV energy limits"
    @assert all(tau_trv .>= 0) "Negative travel times not allowed"
    @assert all(R_work .>= 0) "Negative work requirements not allowed"
    @assert all(A .>= 0) "Negative location values not allowed"
    @assert all(lambda_CO2 .>= 0) "Negative CO2 prices not allowed"
    @assert all(lambda_whl_elec .>= 0) "Negative electricity prices not allowed"
    @assert rho_miss >= 0 "Negative missed work penalty not allowed"
    @assert rho_labor >= 0 "Negative labor penalty not allowed"

    @assert delta_T > 0 "Non-positive time interval not allowed"

    # Check diagonal elements
    @assert all(diag(tau_trv) .== 0) "Travel time matrix diagonal must be zero"

    # Check symmetry
    @assert all(tau_trv .== tau_trv') "Travel time matrix must be symmetric"

    # Check CEV assignments
    @assert all(sum(A, dims=1) .== 1) "Each CEV must be assigned to exactly one location"
    @assert all(A[1, :] .== 0) "No CEVs should be assigned to grid node"
end

end # module 