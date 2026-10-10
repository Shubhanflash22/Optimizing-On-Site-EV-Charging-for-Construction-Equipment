# #############################################################################
# Common.jl  -  module Common
# -----------------------------------------------------------------------------
# Shared, dependency-light utilities used by every other file in the pipeline (DataLoader, MCSModel, OneShot, Output). 
# Nothing in this file touches the MILP/MPC formulation directly -- it is infrastructure that the optimization code and the plotting/logging code both depend on. Four groups:
#
#   1. TIME / CLOCK HELPERS
#      normalize_travel_steps, in_peak, clock_label, clock_day_label,
#      build_time_labels, build_time_labels_days, multiday_xticks,
#      create_fixed_2hour_xticks
#      -- convert between interval indices (1..nK) and real clock time, flag
#      which intervals fall in the 16:00-21:00 on-peak demand-charge window,
#      and build the x-axis tick/label sets used by every figure and CSV.
#
#   2. STEP-PLOT / TABLE HELPERS
#      stepify_interval_values, stepify_boundary_values,
#      interval_time_dataframe, safe_get
#      -- turn interval- or boundary-indexed series into piecewise-constant
#      (staircase) x/y traces for plotting, and build the common per-interval
#      DataFrame skeleton (period index + start/end clock labels) that every
#      output table is built on.
#
#   3. BAYESIAN ACTIVITY-POWER ESTIMATOR
#      activity_power_model, BayesianActivityEstimator, observe!, refit!
#      -- fits the per-activity CEV power draw (digging / loading+swinging /
#      traveling / idling) from observed (activity-hours, measured energy)
#      pairs via a Turing/NUTS regression. The posterior mean/std (mu, sd)
#      it produces are the calibrated p_a power constants consumed by the
#      MILP's work-power constraint and by the stochastic "plant" below.
#      Idle is pinned deterministically (sd = 0), never sampled.
#
#   4. ACTIVITY POWER SAMPLE POOL ("plant" randomness)
#      ActivityPowerPool, draw_activity_power_pool, draw_activity_power_pool_live,
#      new_cursor, next_power!
#      -- pre-draws a shared, reproducible set of realized activity powers
#      (from the frozen Bayesian posterior, or from real recorded field data)
#      so every approach that simulates plant behavior consumes the SAME
#      underlying random draws, keeping cross-approach comparisons fair.
#      Each approach walks the shared pool with its own independent cursor.
#
#   5. RUN LOGGING (opt-in via detailed_output = true)
#      DetailedPlanLog / RealizedTupleLog        (CEV-side)
#      MCSPlanLog     / MCSRealizedLog           (MCS-side)
#      log_plan_row! / log_realized_row! / log_mcs_plan_row! / log_mcs_realized_row!
#      to_dataframe(...)
#      -- append-only Vector{NamedTuple} logs of (a) the FULL remaining-horizon
#      plan at every re-solve (what was planned but possibly overwritten
#      before ever being implemented) and (b) what was actually realized each
#      interval, for both CEVs and MCSs. Converted to a DataFrame once, at the
#      end of the run, for cheap post-hoc analysis (e.g. "did the plan change
#      from the previous resolve").
# #############################################################################

module Common

# external packages used across this file
using DataFrames   # for building output tables
using Printf       # for @sprintf in clock labels
using Turing       # for Bayesian model (activity power estimation)
using Statistics   # mean, std
using Random       # RNG for sampling

# everything below that other files are allowed to use
export DetailedPlanLog, RealizedTupleLog, log_plan_row!, log_realized_row!, to_dataframe,
       MCSPlanLog, MCSRealizedLog, log_mcs_plan_row!, log_mcs_realized_row!,
       normalize_travel_steps, in_peak,
       clock_label, clock_day_label, build_time_labels, build_time_labels_days,
       multiday_xticks, create_fixed_2hour_xticks,
       stepify_interval_values, stepify_boundary_values,
       interval_time_dataframe, safe_get,
       BayesianActivityEstimator, observe!, refit!,
       ActivityPowerPool, draw_activity_power_pool, draw_activity_power_pool_live,
       new_cursor, next_power!

Turing.setprogress!(false)  # turn off Turing's sampling progress bar/output

# Converts raw travel-time values (tau_trv) into integer interval counts (steps), for every pair of nodes i,j in N. 
# Same node (i == j) is forced to 0 steps.
# Any nonzero travel time is rounded to the nearest integer and floored at 1, so no two distinct nodes can have zero travel steps between them.
# Constraint 12b and 13b of the paper's MILP formulation.
function normalize_travel_steps(tau_trv, N)
    n = length(N)
    steps = zeros(Int, n, n)
    for i in N, j in N
        steps[i, j] = i == j ? 0 : max(1, Int(round(tau_trv[i, j])))
    end
    return steps
end

# Checks whether interval k falls fully within the on-peak window (16:00-21:00 / 4-9pm).
# Converts k into a start/stop clock time using t_start and delta_T, wraps around 24h with mod, and treats a stop of exactly 0 (midnight) as 24 (end of day) for the check.
# Objective function (4) of the paper's MILP formulation.
function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

# Converts interval index k into a "HH:MM" clock label. 
# Computes the total minutes from midnight (t_start + elapsed intervals * delta_T, both in hours, converted to minutes), wraps around a 24h clock with mod, then formats as zero-padded HH:MM.
# k=1 is the label for the start of the first interval.
function clock_label(t_start, delta_T, k)
    m = mod(Int(round(t_start * 60 + (k - 1) * delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

# Builds a "Dday HH:MM" label by prefixing clock_label's HH:MM output with the day number.
# Used for multi-day (horizon > 24h) x-axis labels.
clock_day_label(t_start, delta_T, day, k) = string("D", day, " ", clock_label(t_start, delta_T, k))

# Builds boundary-time labels for a single day's intervals 1..nK.
# Returns nK+1 labels: clock_label at k=0 (start of interval 1) through k=nK (end of interval nK), i.e. one label per interval boundary.
function build_time_labels(t_start, delta_T, nK)
    return [begin
        clock_min = mod(Int(round(t_start * 60 + k * delta_T * 60)), 24 * 60)
        @sprintf("%02d:%02d", div(clock_min, 60), clock_min % 60)
    end for k in 0:nK]
end

# Multi-day version of build_time_labels. 
# Builds "Dday HH:MM" boundary labels across n_days, each with nK intervals. 
# g is a global boundary counter from 0 to n_days*nK; it's split back into (day, wk) so each boundary gets the right day number and the right within-day clock time via clock_day_label.
function build_time_labels_days(t_start, delta_T, n_days, nK)
    labels = String[]
    for g in 0:(n_days * nK)
        day = min(n_days, div(g, nK) + 1)    
        wk  = g - (day - 1) * nK             
        push!(labels, clock_day_label(t_start, delta_T, day, wk + 1))
    end
    return labels
end

# Builds x-axis tick positions and labels for a multi-day plot, spaced every_hours apart (default 4h).
# step converts every_hours into an interval count using delta_T.
# For each day, walks through intervals 1:step:nK, records the global tick position (day offset + k), and its "Dday HH:MM" label via clock_day_label
function multiday_xticks(n_days, nK, t_start, delta_T; every_hours::Int = 4)
    step = max(1, Int(round(every_hours / delta_T)))
    ticks = Int[]; labels = String[]
    for dday in 1:n_days
        for k in 1:step:nK
            push!(ticks, (dday - 1) * nK + k)
            push!(labels, clock_day_label(t_start, delta_T, dday, k))
        end
    end
    return (ticks, labels)
end

# Builds x-axis tick positions and "HH:00" labels spaced every 2 hours, over the interval range T.
# span_hours is computed from n_intervals and delta_T, so the tick spacing works correctly regardless of interval granularity.
# For each 2-hour offset, maps it to an interval index idx by scaling hour_offset against the total span, skips it if it falls past the last interval, and labels it with the wrapped clock hour (t_start + hour_offset mod 24).
function create_fixed_2hour_xticks(T, delta_T, t_start::Real=0)
    Tvec = collect(T)
    n_intervals = length(Tvec) - 1
    ticks = Int[]
    labels = String[]
    span_hours = n_intervals * delta_T   
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

# Converts an interval-indexed series (one value per interval k in K) into a staircase (x,y) trace for plotting.
# For each interval k, emits two points, (k, value) and (k+1, value), so the plotted line stays flat across the interval instead of interpolating diagonally between interval indices.
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

# Same staircase idea as stepify_interval_values, but for a boundary-indexed series over T (like SOE, which is defined at interval boundaries rather than within intervals).
# For each pair of consecutive boundaries, emits two points holding the value flat across that segment, then appends one final point at the last boundary so the last value is still plotted.
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

# Builds a DataFrame with one row per interval k in K, giving the interval index plus its start and end clock labels, pulled from time_labels at positions k and k+1.
function interval_time_dataframe(K, time_labels)
    Kvec = collect(K)
    return DataFrame(
        Time_Period = Kvec,
        Time_Start_Label = [time_labels[k] for k in Kvec],
        Time_End_Label   = [time_labels[k + 1] for k in Kvec],
    )
end

# Returns v[i] if index i is within bounds, otherwise returns default (NaN unless overridden).
# Simple bounds-safe array lookup used when indexing might run past the end of a vector.
safe_get(v, i, default=NaN) = i <= length(v) ? v[i] : default

# Turing probabilistic model for estimating the 4 activity power constants (dig, load, travel, idle) from observed data.
# A is the design matrix (rows = observations, columns = the 4 activities), b is the observed vector (e.g. measured energy per observation).
# x1..x4 are the 4 unknown activity powers, each given a truncated (non-negative) normal prior from prior_mu/prior_sigma.
# s is the observation noise std, also truncated non-negative, with its own prior scaled by sigma_b.
# mu = A*x predicts each observation from the current activity powers, and each observed b[j] is modeled as noisy around its prediction.
Turing.@model function activity_power_model(A, b, prior_mu, prior_sigma, sigma_b)
    x1 ~ truncated(Normal(prior_mu[1], prior_sigma[1]); lower = 0.0)  
    x2 ~ truncated(Normal(prior_mu[2], prior_sigma[2]); lower = 0.0)   
    x3 ~ truncated(Normal(prior_mu[3], prior_sigma[3]); lower = 0.0)   
    x4 ~ truncated(Normal(prior_mu[4], prior_sigma[4]); lower = 0.0)   
    x = [x1, x2, x3, x4]
    s ~ truncated(Normal(0.0, sigma_b); lower = 0.0)                   
    mu = A * x
    for j in eachindex(b)
        b[j] ~ Normal(mu[j], s)
    end
end

# Holds the state needed to run and re-run the Bayesian activity_power_model as new observations arrive.
# prior_mu/prior_sigma are the fixed priors on the 4 activity powers, A_obs/b_obs accumulate observed (design row, target) pairs, mu/sd hold the latest posterior mean/std for each activity power, and mcmc_samples sets how many samples to draw per refit.
# draws holds every pooled posterior draw of the last refit, one row per draw and one column per activity (empty until refit! has run).
mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}
    prior_sigma::Vector{Float64}
    A_obs::Matrix{Float64}
    b_obs::Vector{Float64}
    mu::Vector{Float64}
    sd::Vector{Float64}
    mcmc_samples::Int
    draws::Matrix{Float64}
end

# Constructs a BayesianActivityEstimator with empty observation data (0 rows in A_obs, empty b_obs).
# Initializes mu/sd to the prior values, since no data has been observed yet to update them.
function BayesianActivityEstimator(prior_mu, prior_sigma; mcmc_samples = 500)
    k = length(prior_mu)
    return BayesianActivityEstimator(collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     Matrix{Float64}(undef, 0, k), Float64[],
                                     collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     mcmc_samples, Matrix{Float64}(undef, 0, k))
end

# Appends one new observation to the estimator: a is a row of the design matrix (activity-hours for that observation), b is the observed target value (e.g. measured energy).
# Stacks a onto A_obs and pushes b onto b_obs, growing the dataset used by the next refit!.
function observe!(est::BayesianActivityEstimator, a::AbstractVector, b::Real)
    est.A_obs = vcat(est.A_obs, reshape(collect(float.(a)), 1, :))
    push!(est.b_obs, float(b))
    return est
end

# Re-runs Bayesian inference (NUTS MCMC) on all observations collected so far and updates mu/sd in place.
# Does nothing if there are no observations yet.
# sigma_b estimates the observation noise scale from the data itself (falls back to 1.0 with a single observation), and prior_sigma_fit floors each prior std away from exactly zero so the sampler doesn't choke on a degenerate prior.
# Runs single-chain or multi-chain sampling depending on nchains, then for each of the 4 activities either keeps it pinned at its prior (if that activity's prior_sigma was effectively zero, meaning it was never meant to be estimated) or reads its posterior mean/std from the corresponding chain column.
# Every pooled draw of each column is also kept in est.draws (a pinned activity gets its prior mean in every row).
function refit!(est::BayesianActivityEstimator; nchains::Int = 1)
    isempty(est.b_obs) && return est
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0
    prior_sigma_fit = [max(s, 1e-6) for s in est.prior_sigma]
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, prior_sigma_fit, sigma_b)
    chain = nchains > 1 ?
        sample(model, NUTS(0.9), MCMCThreads(), est.mcmc_samples, nchains; progress = false) :
        sample(model, NUTS(0.9), est.mcmc_samples; progress = false)
    syms = (:x1, :x2, :x3, :x4)
    est.draws = zeros(length(vec(chain[:x1])), length(est.prior_mu))
    for i in 1:length(est.prior_mu)
        if est.prior_sigma[i] <= 1e-12
            est.mu[i] = est.prior_mu[i] 
            est.sd[i] = 0.0
            est.draws[:, i] .= est.prior_mu[i]
        else
            col = vec(chain[syms[i]])
            est.mu[i] = mean(col)
            est.sd[i] = std(col)
            est.draws[:, i] = col
        end
    end
    return est
end

# Holds a pre-drawn pool of sampled activity powers, one set per (entity, activity) pair.
# mu/sd are the underlying distribution parameters used to draw them, samples maps (entity, activity) to its vector of pre-drawn values, wrap controls whether the pool is redrawn randomly instead of walked in sequence, and rng is the random source (nothing when wrap is unused)
struct ActivityPowerPool
    mu::Vector{Float64}
    sd::Vector{Float64}
    samples::Dict{Tuple{Int,Int}, Vector{Float64}}   
    wrap::Bool                                        
    rng::Union{Nothing, AbstractRNG}                  
end

# Draws a standardized z-offset according to the requested sampling mode, used to bias sampled activity powers away from or toward the mean.
# :normal draws a plain standard normal, :near_mean shrinks its spread, :high and :low shift it into the upper or lower tail, and :spread_wide randomly picks either tail with a wide spread.
# Errors on any other symbol.
function _draw_mode_z(mode::Symbol, rng)
    if mode === :normal
        return randn(rng)                                        
    elseif mode === :near_mean
        return 0.5 * randn(rng)                                   
    elseif mode === :high
        return 2.0 + 0.5 * abs(randn(rng))                        
    elseif mode === :low
        return -2.0 - 0.5 * abs(randn(rng))                      
    elseif mode === :spread_wide
        sign = rand(rng, Bool) ? 1.0 : -1.0
        return sign * (2.0 + 0.5 * abs(randn(rng)))               
    else
        error("draw_activity_power_pool: unknown mode :$mode ",
              "(expected :normal, :near_mean, :high, :low, or :spread_wide)")
    end
end

# Builds an ActivityPowerPool by pre-drawing n_samples activity power values for every (entity, activity) combination, from a Normal(mu[a], sd[a]) shaped by the given mode, floored at 0 so no negative power is ever produced.
# wrap is set to false here, meaning the pool is meant to be walked in order (via next_power!) rather than redrawn randomly.
function draw_activity_power_pool(entities, mu, sd; n_samples::Int = 20,
                                   rng = Random.GLOBAL_RNG, mode::Symbol = :normal)
    samples = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for e in entities, a in eachindex(mu)
        samples[(e, a)] = [max(mu[a] + sd[a] * _draw_mode_z(mode, rng), 0.0) for _ in 1:n_samples]
    end
    return ActivityPowerPool(collect(float.(mu)), collect(float.(sd)), samples, false, nothing)
end

# Builds an ActivityPowerPool from real recorded data instead of a fitted distribution.
# live_values maps each activity index to its list of actually observed power values.
# For each activity, computes mu/sd empirically from those recorded values (sd = 0 if only one value exists), then for every entity draws n_samples values by resampling with replacement from the recorded values themselves (bootstrap), not from a fitted Normal.
# Errors if an activity has no recorded values at all.
function draw_activity_power_pool_live(entities, live_values::Dict{Int, Vector{Float64}};
                                        n_samples::Int = 20, rng = Random.GLOBAL_RNG)
    n_act = length(live_values)
    mu = zeros(n_act); sd = zeros(n_act)
    samples = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for a in 1:n_act
        vals = live_values[a]
        isempty(vals) && error("draw_activity_power_pool_live: no recorded values for activity $a")
        mu[a] = sum(vals) / length(vals)
        sd[a] = length(vals) > 1 ?
            sqrt(sum((v - mu[a])^2 for v in vals) / (length(vals) - 1)) : 0.0
        for e in entities
            fixed_sequence = [vals[rand(rng, 1:length(vals))] for _ in 1:n_samples]
            samples[(e, a)] = fixed_sequence
        end
    end
    return ActivityPowerPool(mu, sd, samples, false, nothing)
end

# Creates a fresh cursor dict for walking an ActivityPowerPool in order.
# One entry per (entity, activity) key in the pool's samples, each starting at index 1.
new_cursor(pool::ActivityPowerPool) = Dict{Tuple{Int,Int}, Int}(k => 1 for k in keys(pool.samples))

# Returns the next pre-drawn power value for a given (entity, activity) pair, advancing that pair's cursor.
# If the activity's sd is effectively zero, skips sampling entirely and just returns mu (deterministic case, e.g. idle).
# If wrap is set, ignores the cursor and returns a random sample from the pool instead of walking in order.
# Otherwise reads the next value at the cursor position, errors if the pool is exhausted, and advances the cursor.
function next_power!(pool::ActivityPowerPool, cursor::Dict{Tuple{Int,Int}, Int}, e::Int, a::Int)
    pool.sd[a] <= 1e-12 && return pool.mu[a]
    key = (e, a)
    vals = pool.samples[key]
    if pool.wrap
        return vals[rand(pool.rng, 1:length(vals))]
    end
    c = cursor[key]
    c > length(vals) && error("ActivityPowerPool exhausted for entity=$e, activity=$a ",
                               "(only $(length(vals)) pre-drawn samples); increase n_samples ",
                               "in draw_activity_power_pool.")
    cursor[key] = c + 1
    return vals[c]
end

# Simple append-only log of CEV plan rows, one row per (resolve step, offset step, cev) combination.
# Each row is a NamedTuple, stored generically so any fields passed via log_plan_row! are accepted.
mutable struct DetailedPlanLog
    rows::Vector{NamedTuple}
end
DetailedPlanLog() = DetailedPlanLog(NamedTuple[])

# Appends one planned row to the log: what the plan says CEV `cev` will be doing at `offset_step`, as computed during the re-solve at `resolve_step` (offset_step >= resolve_step, since this logs the full remaining-horizon plan, not just what gets implemented).
# Also stores derived clock labels for both steps and how many steps ahead offset_step is from resolve_step, plus the planned activity, power, charging flag, and SOE values for CEV and MCS at that point.
function log_plan_row!(dpl::DetailedPlanLog, d;
                        day::Int, resolve_step::Int, offset_step::Int,
                        activity_planned::AbstractString, planned_power_kW::Float64,
                        planned_charging::Bool,
                        soe_cev_planned_kWh::Float64, soe_mcs_planned_kWh::Float64,
                        cev::Int, scenario_id::Union{Missing,Int} = missing)
    push!(dpl.rows, (;
        day, resolve_step,
        resolve_clock = clock_label(d.t_start, d.delta_T, resolve_step),
        offset_step,
        offset_clock = clock_label(d.t_start, d.delta_T, offset_step),
        steps_ahead = offset_step - resolve_step,
        cev, scenario_id,
        activity_planned, planned_power_kW, planned_charging,
        soe_cev_planned_kWh, soe_mcs_planned_kWh,
    ))
end

# Converts the accumulated plan rows into a DataFrame and flags which rows changed from the previous re-solve's plan for the same (day, cev, scenario_id, offset_step).
# Sorts by that grouping first so consecutive rows for the same future step (from different resolve times) sit next to each other, walks through comparing each row's activity/power to the immediately preceding one in that group, and records a `changed_from_prior_resolve` flag (missing for the first occurrence of each group, since there's nothing prior to compare against).
# Re-sorts back into chronological (day, resolve_step, offset_step, ...) order before returning.
function to_dataframe(dpl::DetailedPlanLog)
    df = DataFrame(dpl.rows)
    isempty(df) && return df
    sort!(df, [:day, :cev, :scenario_id, :offset_step, :resolve_step])
    changed = Vector{Union{Missing,Bool}}(missing, nrow(df))
    prev_key = nothing
    prev_act = nothing
    prev_pow = nothing
    for i in 1:nrow(df)
        key = (df.day[i], df.cev[i], df.scenario_id[i], df.offset_step[i])
        if isequal(prev_key, key)
            changed[i] = (df.activity_planned[i] != prev_act) ||
                         !isapprox(df.planned_power_kW[i], prev_pow; atol = 1e-9)
        end
        prev_key = key; prev_act = df.activity_planned[i]; prev_pow = df.planned_power_kW[i]
    end
    df.changed_from_prior_resolve = changed
    sort!(df, [:day, :resolve_step, :offset_step, :cev, :scenario_id])
    return df
end

# Simple append-only log of realized (actually-implemented, not just planned) CEV activity, one row per (day, step, cev).
mutable struct RealizedTupleLog
    rows::Vector{NamedTuple}
end
RealizedTupleLog() = RealizedTupleLog(NamedTuple[])

# Appends one realized row to the log: the actual power tuple, executed activity, resulting SOE for CEV and MCS, and whether the interval hit infeasibility, at a specific (day, step) for a given cev.
# Also stores the planned activity and planned power for that same (day, step, cev) as sibling columns, so plan vs. realized can be read directly off one row with no join needed.
# planned_activity_blocked_by_min_soe flags the case where the planned activity could not start or complete because the CEV's SOE was already at its minimum floor; defaults to false.
# p_tuple is unpacked positionally into p_dig_kW, p_load_kW, p_trav_kW, p_idle_kW, assuming a fixed 4-element ordering (dig, load, travel, idle).
function log_realized_row!(rtl::RealizedTupleLog, d;
                            day::Int, step::Int, cev::Int,
                            p_tuple::AbstractVector{<:Real}, activity_executed::AbstractString,
                            activity_planned::AbstractString, planned_power_kW::Float64,
                            soe_cev_kWh::Float64, soe_mcs_kWh::Float64,
                            infeasible_flag::Bool,
                            planned_activity_blocked_by_min_soe::Bool = false)
    push!(rtl.rows, (;
        day, step, clock = clock_label(d.t_start, d.delta_T, step), cev,
        p_dig_kW = float(p_tuple[1]), p_load_kW = float(p_tuple[2]),
        p_trav_kW = float(p_tuple[3]), p_idle_kW = float(p_tuple[4]),
        activity_executed, activity_planned, planned_power_kW,
        soe_cev_kWh, soe_mcs_kWh, infeasible_flag,
        planned_activity_blocked_by_min_soe,
    ))
end

# Converts the accumulated realized rows into a DataFrame, no sorting or derived columns (unlike the plan logs).
to_dataframe(rtl::RealizedTupleLog) = DataFrame(rtl.rows)

# Same idea as DetailedPlanLog but for MCS plan rows, one row per (resolve step, offset step, mcs) combination.
mutable struct MCSPlanLog
    rows::Vector{NamedTuple}
end
MCSPlanLog() = MCSPlanLog(NamedTuple[])

# Appends one planned row to the log: what the plan says MCS `mcs` will be doing at `offset_step`, as computed during the re-solve at `resolve_step`.
# Stores derived clock labels and steps_ahead like log_plan_row!, plus the planned status, node, grid charge/discharge power, and SOE for that MCS at that point.
function log_mcs_plan_row!(mpl::MCSPlanLog, d;
                            day::Int, resolve_step::Int, offset_step::Int,
                            mcs::Int, mcs_status_planned::AbstractString,
                            mcs_node_planned::AbstractString,
                            grid_charge_kW_planned::Float64,
                            grid_discharge_kW_planned::Float64,
                            soe_mcs_planned_kWh::Float64,
                            scenario_id::Union{Missing,Int} = missing)
    push!(mpl.rows, (;
        day, resolve_step,
        resolve_clock = clock_label(d.t_start, d.delta_T, resolve_step),
        offset_step,
        offset_clock = clock_label(d.t_start, d.delta_T, offset_step),
        steps_ahead = offset_step - resolve_step,
        mcs, scenario_id,
        mcs_status_planned, mcs_node_planned,
        grid_charge_kW_planned, grid_discharge_kW_planned, soe_mcs_planned_kWh,
    ))
end

# Converts accumulated MCS plan rows into a DataFrame and flags which rows changed from the previous re-solve's plan for the same (day, mcs, scenario_id, offset_step).
# Same sort-compare-resort pattern as to_dataframe(dpl::DetailedPlanLog), but compares mcs_status_planned between consecutive rows as well as the charge/discharge power values.
function to_dataframe(mpl::MCSPlanLog)
    df = DataFrame(mpl.rows)
    isempty(df) && return df
    sort!(df, [:day, :mcs, :scenario_id, :offset_step, :resolve_step])
    changed = Vector{Union{Missing,Bool}}(missing, nrow(df))
    prev_key = nothing
    prev_status = nothing
    prev_charge = nothing
    prev_discharge = nothing
    for i in 1:nrow(df)
        key = (df.day[i], df.mcs[i], df.scenario_id[i], df.offset_step[i])
        if isequal(prev_key, key)
            changed[i] = (df.mcs_status_planned[i] != prev_status) ||
                         !isapprox(df.grid_charge_kW_planned[i], prev_charge; atol = 1e-9) ||
                         !isapprox(df.grid_discharge_kW_planned[i], prev_discharge; atol = 1e-9)
        end
        prev_key = key
        prev_status = df.mcs_status_planned[i]
        prev_charge = df.grid_charge_kW_planned[i]
        prev_discharge = df.grid_discharge_kW_planned[i]
    end
    df.changed_from_prior_resolve = changed
    sort!(df, [:day, :resolve_step, :offset_step, :mcs, :scenario_id])
    return df
end

# Same idea as RealizedTupleLog but for MCS realized rows, one row per (day, step, mcs).
mutable struct MCSRealizedLog
    rows::Vector{NamedTuple}
end
MCSRealizedLog() = MCSRealizedLog(NamedTuple[])

# Appends one realized row to the log: the actual status, node, grid charge/discharge power, and resulting SOE for MCS `mcs` at a specific (day, step)
function log_mcs_realized_row!(mrl::MCSRealizedLog, d;
                                day::Int, step::Int, mcs::Int,
                                mcs_status_realized::AbstractString,
                                mcs_node_realized::AbstractString,
                                grid_charge_kW_realized::Float64,
                                grid_discharge_kW_realized::Float64,
                                soe_mcs_kWh::Float64)
    push!(mrl.rows, (;
        day, step, clock = clock_label(d.t_start, d.delta_T, step), mcs,
        mcs_status_realized, mcs_node_realized,
        grid_charge_kW_realized, grid_discharge_kW_realized, soe_mcs_kWh,
    ))
end

# Converts accumulated realized MCS rows into a DataFrame, no sorting or derived columns, same as RealizedTupleLog's version.
to_dataframe(mrl::MCSRealizedLog) = DataFrame(mrl.rows)

end