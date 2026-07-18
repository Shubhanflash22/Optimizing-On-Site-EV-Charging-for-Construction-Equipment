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
using Turing
using Statistics
using Random

export normalize_travel_steps, in_peak,
       clock_label, clock_day_label, build_time_labels, build_time_labels_days,
       create_fixed_2hour_xticks, multiday_xticks,
       stepify_interval_values, stepify_boundary_values,
       interval_time_dataframe, safe_get,
       BayesianActivityEstimator, observe!, refit!

# Silence Turing's sampling progress bar at load time.
Turing.setprogress!(false)

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

# -----------------------------------------------------------------------------
# MULTI-DAY HELPERS (Receding horizon only)
# -----------------------------------------------------------------------------

# Day-tagged clock label for a within-day interval index, e.g. "D2 08:15".
clock_day_label(t_start, delta_T, day, k) = string("D", day, " ", clock_label(t_start, delta_T, k))

# Boundary clock labels for a MULTI-DAY kept horizon of `n_days` days, each with
# `nK` daytime intervals. Boundary index g (1..n_days*nK+1) is tagged with its day
# and within-day clock (e.g. "D1 08:00", ..., "D2 08:00", ...).
function build_time_labels_days(t_start, delta_T, n_days, nK)
    labels = String[]
    for g in 0:(n_days * nK)
        day = min(n_days, div(g, nK) + 1)     # boundary g belongs to this day
        wk  = g - (day - 1) * nK              # within-day boundary offset (0..nK)
        push!(labels, clock_day_label(t_start, delta_T, day, wk + 1))
    end
    return labels
end

# X-ticks for a multi-day kept horizon: one tick every `every_hours` within each
# day-block, placed on the boundary axis (1..n_days*nK+1) and labelled "D<d> HH:00".
function multiday_xticks(n_days, nK, t_start, delta_T; every_hours::Int = 4)
    step = max(1, Int(round(every_hours / delta_T)))
    ticks = Int[]; labels = String[]
    for d in 1:n_days
        for k in 1:step:nK
            push!(ticks, (d - 1) * nK + k)
            push!(labels, clock_day_label(t_start, delta_T, d, k))
        end
    end
    return ticks, labels
end

# #############################################################################
# BAYESIAN ACTIVITY-POWER ESTIMATOR  (ported from
# Avik/Tasks_energy_loading_swinging_bayesian (1).py)
# -----------------------------------------------------------------------------
# The Python reference fits the linear-in-the-powers energy model
#     b_i | x, sigma ~ Normal(A_i . x, sigma)
# with a TruncatedNormal(mu, sigma; lower=0) prior on every per-activity power x
# and a HalfNormal(sigma = std(b)) prior on the observation noise. Each row of A
# is the activity-hours spent this interval; b is the measured energy (kWh).
#
# This Julia port keeps that model EXACTLY (TruncatedNormal powers, half-normal
# noise, Normal likelihood). The only project-specific choice is the activity
# set: we use four columns [digging, loading+swinging, traveling, idling] to
# match the MCS optimiser's B = [1,2,3,4]; the Python "Mixing"/"Grading" columns
# are not part of this fleet. Idling is pinned to 0 kW with 0 std (no power is
# lost while idle), so it is treated as a deterministic zero rather than sampled
# (a TruncatedNormal with sigma = 0 is degenerate).
#
# The posterior MEAN (`mu`) is the certainty-equivalent power profile the MILP
# consumes; the posterior STD (`sd`) is the per-activity uncertainty used both
# for the convergence figure and (in Fork B) as the plant's sampling spread.
# #############################################################################

# Turing model: TruncatedNormal(prior_mu, prior_sigma; lower=0) on each of the
# four activity powers, a half-normal observation-noise std (Truncated Normal(0,
# sigma_b) with sigma_b = std(b)), and a Normal likelihood b ~ Normal(A*x, s).
Turing.@model function activity_power_model(A, b, prior_mu, prior_sigma, sigma_b)
    x1 ~ truncated(Normal(prior_mu[1], prior_sigma[1]); lower = 0.0)   # digging power
    x2 ~ truncated(Normal(prior_mu[2], prior_sigma[2]); lower = 0.0)   # loading/swinging power
    x3 ~ truncated(Normal(prior_mu[3], prior_sigma[3]); lower = 0.0)   # traveling power
    x4 ~ truncated(Normal(prior_mu[4], prior_sigma[4]); lower = 0.0)   # idling power
    x = [x1, x2, x3, x4]
    s ~ truncated(Normal(0.0, sigma_b); lower = 0.0)                   # HalfNormal(sigma_b) noise
    mu = A * x
    for j in eachindex(b)
        b[j] ~ Normal(mu[j], s)
    end
end

# Mutable estimator state carried between updates: the fixed prior, all
# observations gathered so far, and the CURRENT posterior summary (mu = the
# profile fed to the MILP; sd = its uncertainty).
mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}
    prior_sigma::Vector{Float64}
    A_obs::Matrix{Float64}
    b_obs::Vector{Float64}
    mu::Vector{Float64}
    sd::Vector{Float64}
    mcmc_samples::Int
end

# Constructor: no observations yet; posterior initialised to the prior (this is
# also the CALIBRATED-PRIOR mode, where mu/sd are used as-is without any fit).
function BayesianActivityEstimator(prior_mu, prior_sigma; mcmc_samples = 500)
    k = length(prior_mu)
    return BayesianActivityEstimator(collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     Matrix{Float64}(undef, 0, k), Float64[],
                                     collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     mcmc_samples)
end

# Record ONE telemetry observation (activity-hours row + measured energy). Only
# appends the data; inference is re-run separately in refit!.
function observe!(est::BayesianActivityEstimator, a::AbstractVector, b::Real)
    est.A_obs = vcat(est.A_obs, reshape(collect(float.(a)), 1, :))
    push!(est.b_obs, float(b))
    return est
end

# Re-run the Bayesian regression on ALL data so far and refresh mu / sd via NUTS
# (mirrors the Python fit_bayesian_activity_power -> pm.sample with target_accept
# = 0.9). Activities whose prior std is 0 (idle) are pinned to their prior mean
# with 0 uncertainty instead of sampled, avoiding a degenerate TruncatedNormal.
function refit!(est::BayesianActivityEstimator)
    isempty(est.b_obs) && return est
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0
    # Floor the prior std handed to the sampler so a pinned (sigma = 0) activity
    # does not make the model degenerate; its posterior is overwritten below.
    prior_sigma_fit = [max(s, 1e-6) for s in est.prior_sigma]
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, prior_sigma_fit, sigma_b)
    chain = sample(model, NUTS(0.9), est.mcmc_samples; progress = false)
    syms = (:x1, :x2, :x3, :x4)
    for i in 1:length(est.prior_mu)
        if est.prior_sigma[i] <= 1e-12
            est.mu[i] = est.prior_mu[i]   # pinned activity (e.g. idle): deterministic
            est.sd[i] = 0.0
        else
            col = vec(chain[syms[i]])
            est.mu[i] = mean(col)
            est.sd[i] = std(col)
        end
    end
    return est
end

end # module Common
