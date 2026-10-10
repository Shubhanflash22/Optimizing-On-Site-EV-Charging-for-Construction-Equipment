# #############################################################################
# 0_Regression.jl  —  module Regression   (STEP 0: offline power-model fit)
# -----------------------------------------------------------------------------
# A PURE-JULIA port of Tasks_energy_loading_swinging_bayesian.py. There is NO
# Python and NO wrapper: this reads the soil task-recording Excel files directly,
# builds the energy-balance regression equations, fits the SAME Bayesian model
# used by the online estimator (Common.activity_power_model: TruncatedNormal power
# priors + a half-normal noise, NUTS), and writes ONLY the fitted values into the
# MPC's parameters.csv:
#     p_digging, p_loading_swinging, p_traveling   = posterior MEAN per activity
#     sigma_digging, sigma_loading_swinging, ...    = posterior SD  per activity
# No plots / images / diagnostics are produced (that was the Python script's job).
#
# Model (per emitted equation i, mirroring the Python build_equations_from_tasks):
#   * walk the task rows, accumulate activity DURATIONS into a bucket until the
#     cumulative |ΔSoC| reaches MIN_DELTA_SOC (%), then emit ONE equation:
#       A_i = [dig+grading1, load+swing+grading2, travel, idle]  (hours)
#       b_i = -ΔSoC * BATTERY_CAP / 100                          (kWh consumed)
#   * fit  b_i ~ Normal(A_i . x, s),  x_a ~ TruncatedNormal(mu_a, sigma_a; ≥0),
#     s ~ HalfNormal(std(b)).  (Idle is pinned to 0 kW, matching the fleet model.)
#
# FAIL-SOFT: if XLSX.jl is not installed or the data folder is missing, this logs
# a warning and returns false so the MPC still runs against the existing
# parameters.csv. Install the one extra package with:  import Pkg; Pkg.add("XLSX")
# #############################################################################
module Regression

using Printf
using Dates
using Statistics
using DataFrames
using CSV
using ..Common: BayesianActivityEstimator, observe!, refit!

export run_regression

# Optional dependency: only XLSX is extra. Load it defensively so a missing
# package cannot break the whole include chain; step 0 then degrades gracefully.
const _HAVE_XLSX = try
    @eval import XLSX
    true
catch
    false
end

# ---- constants ported from the Python script (MATERIAL = "soil") ------------
const BATTERY_CAP   = 14.8     # kWh, full-pack capacity used to turn ΔSoC% into kWh
const MIN_DELTA_SOC = 3.0      # emit one equation per cumulative 3% SoC drop
const MCMC_DEFAULT  = 2000     # NUTS draws per chain (Python used draws=2000)
const NCHAINS_DEFAULT = 4      # NUTS chains, pooled (Python used chains=4)

# Soil task files (Python MATERIAL_FILES["soil"] = files 1..12), by basename.
const SOIL_FILES = [
    "Oct_21_Tasks_1.xlsx",
    "Oct_22_Tasks_1.xlsx", "Oct_22_Tasks_2.xlsx", "Oct_22_Tasks_3.xlsx",
    "Oct_22_Tasks_4.xlsx", "Oct_22_Tasks_5.xlsx",
    "Oct_23_Tasks_1.xlsx",
    "Feb_02_Tasks_1.xlsx", "Feb_02_Tasks_2.xlsx", "Feb_02_Tasks_3.xlsx",
    "Feb_03_Tasks_1.xlsx", "Feb_03_Tasks_2.xlsx",
]

# Soil prior on [digging, loading+swinging, traveling, idling] powers, matching
# the Python X_PRIOR_MU_SIGMA["False"]["soil"]. Idle is pinned (sigma 0 -> the
# estimator holds it at 0 kW rather than sampling a degenerate distribution).
const PRIOR_MU    = [4.79, 3.16, 4.71, 0.0]
const PRIOR_SIGMA = [0.23, 0.23, 0.54, 0.0]

# Seconds spanned by one task row (End - Start); missing/negative -> 0 so it is
# ignored by the per-activity duration sums (matches pandas skipna=True).
function _row_seconds(t0, t1)
    (ismissing(t0) || ismissing(t1)) && return 0.0
    ms = try
        Dates.value(convert(Millisecond, t1 - t0))   # DateTime-DateTime -> Millisecond
    catch
        return 0.0                                     # non-datetime cell -> ignore
    end
    return ms > 0 ? ms / 1000 : 0.0
end

# Build (A rows, b entries) for ONE task file, exactly like the Python
# build_equations_from_tasks: cumulative-|ΔSoC| bucketing, 4-activity columns.
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

        # sum hours per raw activity over the bucket rows [bstart .. j]
        h = zeros(7)   # dig, grading1, load, swing, grading2, travel, idle
        for r in bstart:j
            a = acts[r]; ismissing(a) && continue
            s = strip(String(a)); d = dur[r]
            if     s == "Digging";    h[1] += d
            elseif s == "Grading 1";  h[2] += d
            elseif s == "Loading";    h[3] += d
            elseif s == "Swinging";   h[4] += d
            elseif s == "Grading 2";  h[5] += d
            elseif s == "Travelling"; h[6] += d
            elseif s == "Idling";     h[7] += d
            end
        end
        h ./= 3600   # seconds -> hours
        # 4-activity row: [dig(+grading1), load(+swing+grading2), travel, idle]
        push!(A_rows, [h[1] + h[2], h[3] + h[4] + h[5], h[6], h[7]])
        push!(b_rows, -cum_delta * BATTERY_CAP / 100)

        bstart = j + 1
        anchor = soc_now
        j += 1
    end
end

# Read one Excel file into the four needed column vectors (Start, End, Activity,
# SoC). Returns nothing if the file/sheet is unreadable.
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

# Update-in-place (or append) the six fitted rows in parameters.csv.
function _write_params!(params_csv, mu, sd)
    df = CSV.read(params_csv, DataFrame)
    ("Parameter" in names(df) && "Value" in names(df)) ||
        error("Regression: parameters.csv missing Parameter/Value columns -> $params_csv")
    "Unit" in names(df)        || (df.Unit = fill("", nrow(df)))
    "Description" in names(df)  || (df.Description = fill("", nrow(df)))
    df.Value = Vector{Any}(df.Value)   # allow writing rounded floats uniformly

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

# -----------------------------------------------------------------------------
# STEP 0 entry point. `data_dir` holds the soil .xlsx files; `params_csv` is the
# MPC parameters file to refresh. Returns true on success, false (warned) if the
# step could not run so the caller keeps the existing parameters.csv.
# -----------------------------------------------------------------------------
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

    # ---- build the regression equations from every soil file ----
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

    # ---- fit the SAME Bayesian model as the online estimator, then export ----
    est = BayesianActivityEstimator(PRIOR_MU, PRIOR_SIGMA; mcmc_samples = mcmc_samples)
    for i in eachindex(b_rows)
        observe!(est, A[i, :], b_rows[i])
    end
    refit!(est; nchains = nchains)   # NUTS (nchains pooled); idle pinned to 0

    _write_params!(params_csv, est.mu, est.sd)
    @printf("STEP 0 done in %.1f s; parameters.csv refreshed.\n", time() - t0)
    println("  fitted means : dig=$(round(est.mu[1],digits=3)) load=$(round(est.mu[2],digits=3)) trv=$(round(est.mu[3],digits=3)) kW")
    println("  fitted sds   : dig=$(round(est.sd[1],digits=3)) load=$(round(est.sd[2],digits=3)) trv=$(round(est.sd[3],digits=3)) kW")
    return true
end

end # module Regression
