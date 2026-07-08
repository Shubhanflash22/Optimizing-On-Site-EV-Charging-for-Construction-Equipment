# #############################################################################
# Common.jl  —  module Common
# -----------------------------------------------------------------------------
# Small, dependency-light helpers shared by every other module in the pipeline:
#   * travel-time normalisation (fractional hours -> whole interval counts),
#   * the on-peak (16:00-21:00) membership test used by the demand charge,
#   * clock-label / x-tick builders for the figures and CSVs, and
#   * the STEP-plot helpers (piecewise-constant traces) so that all figures
#     match the v4_real reference style instead of smooth continuous lines.
#
# Nomenclature and behaviour deliberately mirror the reference files
# (DataLoader_v4_real.jl / MCS_OPTIMAL_v4_real.jl) so a reviewer who knows the
# reference sees the same helper names and semantics here.
# #############################################################################
module Common

using DataFrames
using Printf

export normalize_travel_steps, in_peak,
       clock_label, build_time_labels, create_fixed_2hour_xticks,
       stepify_interval_values, stepify_boundary_values,
       interval_time_dataframe, safe_get

# -----------------------------------------------------------------------------
# Travel model: convert a travel-time matrix (values already expressed in time
# INTERVALS, possibly fractional) into WHOLE interval counts. The diagonal
# (i -> i) is 0; any positive off-diagonal time becomes at least one step.
# (Same contract as `normalize_travel_steps` in MCS_OPTIMAL_v4_real.jl.)
# -----------------------------------------------------------------------------
function normalize_travel_steps(tau_trv, N)
    n = length(N)
    steps = zeros(Int, n, n)
    for i in N, j in N
        steps[i, j] = i == j ? 0 : max(1, Int(round(tau_trv[i, j])))
    end
    return steps
end

# -----------------------------------------------------------------------------
# On-peak window membership. Interval k covers [t_start+(k-1)*dt, t_start+k*dt]
# (mod 24). Returns true iff that whole interval lies inside the 16:00-21:00
# on-peak band that carries the extra on-peak demand charge.
# (Identical logic to `in_peak` in the reference main driver.)
# -----------------------------------------------------------------------------
function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

# -----------------------------------------------------------------------------
# Turn an interval index k into an "HH:MM" clock label at its START boundary,
# using the run's start hour t_start and step length delta_T.
# -----------------------------------------------------------------------------
function clock_label(t_start, delta_T, k)
    m = mod(Int(round(t_start * 60 + (k - 1) * delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

# -----------------------------------------------------------------------------
# Build the vector of BOUNDARY clock labels (one per boundary index 1..nK+1),
# so plots/CSVs show clock times that match the simulation window. Mirrors the
# `time_labels` construction in mcs_optimization_main_v4_real.jl.
# -----------------------------------------------------------------------------
function build_time_labels(t_start, delta_T, nK)
    return [begin
        clock_min = mod(Int(round(t_start * 60 + k * delta_T * 60)), 24 * 60)
        @sprintf("%02d:%02d", div(clock_min, 60), clock_min % 60)
    end for k in 0:nK]
end

# -----------------------------------------------------------------------------
# Fixed 2-hour x-ticks over a boundary-indexed axis T (= 1..nK+1). Maps every
# even hour offset from t_start onto its boundary index and labels it "HH:00".
# (Same output contract as `create_fixed_2hour_xticks` in the reference.)
# -----------------------------------------------------------------------------
function create_fixed_2hour_xticks(T, t_start::Real=0)
    Tvec = collect(T)
    n_intervals = length(Tvec) - 1
    ticks = Int[]
    labels = String[]
    span_hours = n_intervals * 0.25   # 0.25 h per interval (15-min grid)
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

# -----------------------------------------------------------------------------
# STEP helper for INTERVAL-indexed quantities. The value at interval k is held
# flat across the boundary span [k, k+1], producing the piecewise-constant
# (staircase) traces used for power/price/work figures.
# (Same as `stepify_interval_values` in MCS_OPTIMAL_v4_real.jl.)
# -----------------------------------------------------------------------------
function stepify_interval_values(K, values)
    Kvec = collect(K)
    x_step = Int[]
    y_step = eltype(values)[]
    for (idx, k) in enumerate(Kvec)
        push!(x_step, k);     push!(y_step, values[idx])
        push!(x_step, k + 1); push!(y_step, values[idx])
    end
    return x_step, y_step
end

# -----------------------------------------------------------------------------
# STEP helper for BOUNDARY-indexed states (e.g. state of energy). The value at
# boundary t is held until the next boundary; the last value is drawn at the
# final boundary without extending past it.
# (Same as `stepify_boundary_values` in the reference.)
# -----------------------------------------------------------------------------
function stepify_boundary_values(T, values)
    Tvec = collect(T)
    x_step = Int[]
    y_step = eltype(values)[]
    isempty(Tvec) && return x_step, y_step
    for idx in 1:(length(Tvec) - 1)
        push!(x_step, Tvec[idx]);     push!(y_step, values[idx])
        push!(x_step, Tvec[idx + 1]); push!(y_step, values[idx])
    end
    push!(x_step, last(Tvec)); push!(y_step, values[end])
    return x_step, y_step
end

# -----------------------------------------------------------------------------
# Base per-interval table carrying the integer interval index plus human-
# readable start/end clock labels; per-quantity reporters append columns to it.
# (Same role as `interval_time_dataframe` in the reference.)
# -----------------------------------------------------------------------------
function interval_time_dataframe(K, time_labels)
    Kvec = collect(K)
    return DataFrame(
        Time_Period = Kvec,
        Time_Start_Label = [time_labels[k] for k in Kvec],
        Time_End_Label   = [time_labels[k + 1] for k in Kvec],
    )
end

# Safe indexed accessor: v[i] if it exists, else `default` (keeps fixed-width
# logs from crashing on datasets with a different number of CEVs).
safe_get(v, i, default=NaN) = i <= length(v) ? v[i] : default

end # module Common
