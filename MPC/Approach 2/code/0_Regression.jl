# #############################################################################
# 0_Regression.jl  -  module Regression
# -----------------------------------------------------------------------------
# Offline step 0 that fits the per-activity CEV power constants (digging, loading+swinging, traveling) from the recorded soil task files.
# It writes the fitted means and standard deviations into parameters.csv, which DataLoader then reads.
# This file is not part of the MILP or the MPC loop, and it runs once before them.
# The fit uses the same Bayesian model as the online estimator in 1_Common.jl, with idle pinned to zero.
# It builds the observation windows and the energy-balance equations of Section II-C of the paper, but solves them with a Bayesian fit instead of the paper's constrained NNLS.
# Five groups:
#
#   1. SETUP AND CONSTANTS
#      _HAVE_XLSX, BATTERY_CAP, MIN_DELTA_SOC, MCMC_DEFAULT, NCHAINS_DEFAULT,
#      SOIL_FILES, PRIOR_MU, PRIOR_SIGMA
#      -- the optional XLSX dependency check, the battery capacity, the SOC
#      threshold for closing a window, the sampler defaults, the list of soil
#      task files, and the fixed priors on the four activity powers.
#
#   2. TASK-FILE READING
#      _row_seconds, _read_task_file
#      -- read one Excel task file into its start time, end time, activity
#      and SOC columns, and turn each row into a duration in seconds.
#
#   3. EQUATION BUILDING
#      _equations_from_file!
#      -- walk the rows of one file and emit one equation each time the
#      cumulative SOC drop reaches the threshold, giving the activity hours
#      and the energy consumed in that window.
#
#   4. PARAMETER EXPORT
#      _write_params!
#      -- write the fitted means and standard deviations into parameters.csv,
#      updating existing rows and appending missing ones.
#
#   5. STEP 0 ENTRY POINT
#      run_regression
#      -- build the equations from every soil file, fit them with the
#      estimator from 1_Common.jl, and call the export.
#      Returns false and leaves parameters.csv untouched if the step cannot run.
# #############################################################################
module Regression

# external packages used across this file
using Printf
using Dates
using DataFrames
using CSV
using ..Common: BayesianActivityEstimator, observe!, refit!

# everything below that other files are allowed to use
export run_regression

# Records whether XLSX.jl could be loaded when this module was included.
# XLSX is the only optional package, so a missing install turns step 0 off instead of breaking the include chain.
const _HAVE_XLSX = try
    @eval import XLSX
    true
catch
    false
end

# Fixed settings: the CEV battery capacity in kWh (C_batt in Section II-A of the paper), the SOC drop in percent that closes an observation window (tau in Section II-C2), and the default NUTS draws per chain and chains.
const BATTERY_CAP   = 14.8     
const MIN_DELTA_SOC = 3.0      
const MCMC_DEFAULT  = 2000    
const NCHAINS_DEFAULT = 4      

# Names of the twelve soil task-recording Excel files, read from the data folder passed to run_regression.
# They cover the October 21 to 23 and February 2 to 3 recording days.
const SOIL_FILES = [
    "Oct_21_Tasks_1.xlsx",
    "Oct_22_Tasks_1.xlsx", "Oct_22_Tasks_2.xlsx", "Oct_22_Tasks_3.xlsx",
    "Oct_22_Tasks_4.xlsx", "Oct_22_Tasks_5.xlsx",
    "Oct_23_Tasks_1.xlsx",
    "Feb_02_Tasks_1.xlsx", "Feb_02_Tasks_2.xlsx", "Feb_02_Tasks_3.xlsx",
    "Feb_03_Tasks_1.xlsx", "Feb_03_Tasks_2.xlsx",
]

# Prior mean and prior standard deviation in kW of the four activity powers, in the order digging, loading+swinging, traveling, idling.
# A standard deviation of zero for idling makes the estimator keep idle fixed at its prior mean of 0 kW instead of sampling it.
const PRIOR_MU    = [4.79, 3.16, 4.71, 0.0]
const PRIOR_SIGMA = [0.23, 0.23, 0.54, 0.0]

# Returns the duration in seconds of one task row, computed as end time t1 minus start time t0.
# Returns 0.0 if either time is missing, if the subtraction or conversion to milliseconds fails, or if the duration is not positive.
function _row_seconds(t0, t1)
    (ismissing(t0) || ismissing(t1)) && return 0.0
    ms = try
        Dates.value(convert(Millisecond, t1 - t0))   
    catch
        return 0.0                                     
    end
    return ms > 0 ? ms / 1000 : 0.0
end

# Appends the energy-balance equations of one task file to A_rows and b_rows (Sections II-C1 and II-C2 of the paper).
# Each equation covers a run of task rows that ends once the SOC has dropped by at least MIN_DELTA_SOC, with A holding the hours in [digging, loading+swinging, traveling, idling] and b the energy consumed in kWh.
# Grading 1 counts as digging, Grading 2 and Swinging count as loading+swinging, both spellings of traveling are accepted, and any other activity name adds no time.
function _equations_from_file!(A_rows, b_rows, starts, stops, acts, socs)
    n = length(socs)
    n == 0 && return
    dur = [_row_seconds(starts[r], stops[r]) for r in 1:n]

    bstart = findfirst(!ismissing, socs)
    bstart === nothing && return
    anchor = Float64(socs[bstart])
    j = bstart + 1
    while j <= n
        if ismissing(socs[j]); j += 1; continue; end
        soc_now  = Float64(socs[j])
        cum_delta = soc_now - anchor
        if abs(cum_delta) < MIN_DELTA_SOC; j += 1; continue; end

        h = zeros(7)   
        for r in bstart:j
            a = acts[r]; ismissing(a) && continue
            s = strip(String(a)); d = dur[r]
            if     s == "Digging";    h[1] += d
            elseif s == "Grading 1";  h[2] += d
            elseif s == "Loading";    h[3] += d
            elseif s == "Swinging";   h[4] += d
            elseif s == "Grading 2";  h[5] += d
            elseif s == "Travelling" || s == "Traveling"; h[6] += d
            elseif s == "Idling";     h[7] += d
            end
        end
        h ./= 3600   
        push!(A_rows, [h[1] + h[2], h[3] + h[4] + h[5], h[6], h[7]])
        push!(b_rows, -cum_delta * BATTERY_CAP / 100)

        bstart = j + 1
        anchor = soc_now
        j += 1
    end
end

# Reads the Sheet1 sheet of one Excel task file and returns its Start time (actual), End time (actual), Activity and SoC columns, in that order.
# Returns nothing and logs a warning if the file, the sheet or any of the four columns cannot be read, so the caller can skip that file.
function _read_task_file(path)
    try
        tbl = XLSX.readtable(path, "Sheet1")
        df  = DataFrame(tbl)
        col(name) = df[!, findfirst(==(name), strip.(string.(names(df))))]
        return (col("Start time (actual)"), col("End time (actual)"),
                col("Activity"), col("SoC"))
    catch err
        @warn "STEP 0: could not read task file; skipping it." path exception = err
        return nothing
    end
end

# Writes the three fitted power means (mu) and three fitted standard deviations (sd) into the parameters.csv file at params_csv, rounded to four decimals.
# An existing row with the same key is updated in place, a missing key is appended as a new row, and the file is overwritten.
function _write_params!(params_csv, mu, sd)
    df = CSV.read(params_csv, DataFrame)
    ("Parameter" in names(df) && "Value" in names(df)) ||
        error("Regression: parameters.csv missing Parameter/Value columns -> $params_csv")
    "Unit" in names(df)        || (df.Unit = fill("", nrow(df)))
    "Description" in names(df)  || (df.Description = fill("", nrow(df)))
    df.Value = Vector{Any}(df.Value)   

    updates = ("p_digging" => mu[1], "p_loading_swinging" => mu[2], "p_traveling" => mu[3],
               "sigma_digging" => sd[1], "sigma_loading_swinging" => sd[2], "sigma_traveling" => sd[3])
    for (key, val) in updates
        idx = findfirst(==(key), strip.(string.(df.Parameter)))
        if idx === nothing
            push!(df, (key, round(val, digits = 4), "kW", "written by step-0 Julia regression"); promote = true)
        else
            df.Value[idx] = round(val, digits = 4)
        end
    end
    CSV.write(params_csv, df)
end

# Runs step 0: builds the equations from every SOIL_FILES file found in data_dir, fits them with the BayesianActivityEstimator from 1_Common.jl, and writes the fitted means and standard deviations into params_csv.
# Returns true after parameters.csv has been rewritten, and returns false with a warning, leaving parameters.csv untouched, if XLSX.jl is missing, if data_dir or params_csv does not exist, or if no equation could be built.
function run_regression(data_dir::AbstractString, params_csv::AbstractString;
                        mcmc_samples::Int = MCMC_DEFAULT,
                        nchains::Int = NCHAINS_DEFAULT)
    if !_HAVE_XLSX
        @warn "STEP 0: XLSX.jl not installed; skipping the fit (using existing parameters.csv). " *
              "Install once with: import Pkg; Pkg.add(\"XLSX\")"
        return false
    end
    if !isdir(data_dir)
        @warn "STEP 0: regression data folder not found; skipping (using existing parameters.csv)." data_dir
        return false
    end
    if !isfile(params_csv)
        @warn "STEP 0: parameters.csv not found; skipping." params_csv
        return false
    end

    println("=" ^ 78)
    println("STEP 0  Bayesian activity-power regression (pure Julia; NUTS $nchains chains x $mcmc_samples draws)")
    println("  data folder : ", data_dir)
    println("  writing     : ", params_csv)
    println("=" ^ 78)
    t0 = time()

    A_rows = Vector{Vector{Float64}}(); b_rows = Float64[]
    nfiles = 0
    for fname in SOIL_FILES
        path = joinpath(data_dir, fname)
        isfile(path) || (@warn "STEP 0: soil file missing; skipping." path; continue)
        cols = _read_task_file(path); cols === nothing && continue
        _equations_from_file!(A_rows, b_rows, cols...)
        nfiles += 1
    end
    if isempty(b_rows)
        @warn "STEP 0: no regression equations built (no readable data); keeping existing parameters.csv."
        return false
    end
    A = reduce(vcat, (reshape(r, 1, :) for r in A_rows))
    @printf("  built %d equations from %d file(s)\n", length(b_rows), nfiles)

    est = BayesianActivityEstimator(PRIOR_MU, PRIOR_SIGMA; mcmc_samples = mcmc_samples)
    for i in eachindex(b_rows)
        observe!(est, A[i, :], b_rows[i])
    end
    refit!(est; nchains = nchains)  

    _write_params!(params_csv, est.mu, est.sd)
    @printf("STEP 0 done in %.1f s; parameters.csv refreshed.\n", time() - t0)
    println("  fitted means : dig=$(round(est.mu[1],digits=3)) load=$(round(est.mu[2],digits=3)) trv=$(round(est.mu[3],digits=3)) kW")
    println("  fitted sds   : dig=$(round(est.sd[1],digits=3)) load=$(round(est.sd[2],digits=3)) trv=$(round(est.sd[3],digits=3)) kW")
    return true
end

end 