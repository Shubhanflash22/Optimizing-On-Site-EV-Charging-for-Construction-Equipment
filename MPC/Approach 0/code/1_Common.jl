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

using DataFrames
using Printf
using Turing
using Statistics
using Random

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

Turing.setprogress!(false)

function normalize_travel_steps(tau_trv, N)
    n = length(N)
    steps = zeros(Int, n, n)
    for i in N, j in N
        steps[i, j] = i == j ? 0 : max(1, Int(round(tau_trv[i, j])))
    end
    return steps
end

function in_peak(k, delta_T, t_start)
    start    = mod(t_start + (k - 1) * delta_T, 24)
    stop     = mod(t_start + k * delta_T, 24)
    stop_eff = stop == 0 ? 24 : stop
    return start >= 16 && stop_eff <= 21
end

function clock_label(t_start, delta_T, k)
    m = mod(Int(round(t_start * 60 + (k - 1) * delta_T * 60)), 24 * 60)
    return @sprintf("%02d:%02d", div(m, 60), m % 60)
end

function build_time_labels(t_start, delta_T, nK)
    return [begin
        clock_min = mod(Int(round(t_start * 60 + k * delta_T * 60)), 24 * 60)
        @sprintf("%02d:%02d", div(clock_min, 60), clock_min % 60)
    end for k in 0:nK]
end

clock_day_label(t_start, delta_T, day, k) = string("D", day, " ", clock_label(t_start, delta_T, k))

function build_time_labels_days(t_start, delta_T, n_days, nK)
    labels = String[]
    for g in 0:(n_days * nK)
        day = min(n_days, div(g, nK) + 1)    
        wk  = g - (day - 1) * nK             
        push!(labels, clock_day_label(t_start, delta_T, day, wk + 1))
    end
    return labels
end

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

function create_fixed_2hour_xticks(T, t_start::Real=0)
    Tvec = collect(T)
    n_intervals = length(Tvec) - 1
    ticks = Int[]
    labels = String[]
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
    x_step = Int[]
    y_step = eltype(values)[]
    for (idx, k) in enumerate(Kvec)
        push!(x_step, k);     push!(y_step, values[idx])
        push!(x_step, k + 1); push!(y_step, values[idx])
    end
    return x_step, y_step
end

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

function interval_time_dataframe(K, time_labels)
    Kvec = collect(K)
    return DataFrame(
        Time_Period = Kvec,
        Time_Start_Label = [time_labels[k] for k in Kvec],
        Time_End_Label   = [time_labels[k + 1] for k in Kvec],
    )
end

safe_get(v, i, default=NaN) = i <= length(v) ? v[i] : default

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

mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}
    prior_sigma::Vector{Float64}
    A_obs::Matrix{Float64}
    b_obs::Vector{Float64}
    mu::Vector{Float64}
    sd::Vector{Float64}
    mcmc_samples::Int
end

function BayesianActivityEstimator(prior_mu, prior_sigma; mcmc_samples = 500)
    k = length(prior_mu)
    return BayesianActivityEstimator(collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     Matrix{Float64}(undef, 0, k), Float64[],
                                     collect(float.(prior_mu)), collect(float.(prior_sigma)),
                                     mcmc_samples)
end

function observe!(est::BayesianActivityEstimator, a::AbstractVector, b::Real)
    est.A_obs = vcat(est.A_obs, reshape(collect(float.(a)), 1, :))
    push!(est.b_obs, float(b))
    return est
end

function refit!(est::BayesianActivityEstimator; nchains::Int = 1)
    isempty(est.b_obs) && return est
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0
    prior_sigma_fit = [max(s, 1e-6) for s in est.prior_sigma]
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, prior_sigma_fit, sigma_b)
    chain = nchains > 1 ?
        sample(model, NUTS(0.9), MCMCThreads(), est.mcmc_samples, nchains; progress = false) :
        sample(model, NUTS(0.9), est.mcmc_samples; progress = false)
    syms = (:x1, :x2, :x3, :x4)
    for i in 1:length(est.prior_mu)
        if est.prior_sigma[i] <= 1e-12
            est.mu[i] = est.prior_mu[i] 
            est.sd[i] = 0.0
        else
            col = vec(chain[syms[i]])
            est.mu[i] = mean(col)
            est.sd[i] = std(col)
        end
    end
    return est
end

struct ActivityPowerPool
    mu::Vector{Float64}
    sd::Vector{Float64}
    samples::Dict{Tuple{Int,Int}, Vector{Float64}}   
    wrap::Bool                                        
    rng::Union{Nothing, AbstractRNG}                  
end

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

function draw_activity_power_pool(entities, mu, sd; n_samples::Int = 20,
                                   rng = Random.GLOBAL_RNG, mode::Symbol = :normal)
    samples = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for e in entities, a in eachindex(mu)
        samples[(e, a)] = [max(mu[a] + sd[a] * _draw_mode_z(mode, rng), 0.0) for _ in 1:n_samples]
    end
    return ActivityPowerPool(collect(float.(mu)), collect(float.(sd)), samples, false, nothing)
end

new_cursor(pool::ActivityPowerPool) = Dict{Tuple{Int,Int}, Int}(k => 1 for k in keys(pool.samples))

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

mutable struct DetailedPlanLog
    rows::Vector{NamedTuple}
end
DetailedPlanLog() = DetailedPlanLog(NamedTuple[])

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

mutable struct RealizedTupleLog
    rows::Vector{NamedTuple}
end
RealizedTupleLog() = RealizedTupleLog(NamedTuple[])

function log_realized_row!(rtl::RealizedTupleLog, d;
                            day::Int, step::Int, cev::Int,
                            p_tuple::AbstractVector{<:Real}, activity_executed::AbstractString,
                            soe_cev_kWh::Float64, soe_mcs_kWh::Float64,
                            infeasible_flag::Bool)
    push!(rtl.rows, (;
        day, step, clock = clock_label(d.t_start, d.delta_T, step), cev,
        p_dig_kW = float(p_tuple[1]), p_load_kW = float(p_tuple[2]),
        p_trav_kW = float(p_tuple[3]), p_idle_kW = float(p_tuple[4]),
        activity_executed, soe_cev_kWh, soe_mcs_kWh, infeasible_flag,
    ))
end

to_dataframe(rtl::RealizedTupleLog) = DataFrame(rtl.rows)

mutable struct MCSPlanLog
    rows::Vector{NamedTuple}
end
MCSPlanLog() = MCSPlanLog(NamedTuple[])

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

function to_dataframe(mpl::MCSPlanLog)
    df = DataFrame(mpl.rows)
    isempty(df) && return df
    sort!(df, [:day, :mcs, :scenario_id, :offset_step, :resolve_step])
    changed = Vector{Union{Missing,Bool}}(missing, nrow(df))
    prev_key = nothing
    prev_status = nothing
    for i in 1:nrow(df)
        key = (df.day[i], df.mcs[i], df.scenario_id[i], df.offset_step[i])
        if isequal(prev_key, key)
            changed[i] = df.mcs_status_planned[i] != prev_status
        end
        prev_key = key; prev_status = df.mcs_status_planned[i]
    end
    df.changed_from_prior_resolve = changed
    sort!(df, [:day, :resolve_step, :offset_step, :mcs, :scenario_id])
    return df
end

mutable struct MCSRealizedLog
    rows::Vector{NamedTuple}
end
MCSRealizedLog() = MCSRealizedLog(NamedTuple[])

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

to_dataframe(mrl::MCSRealizedLog) = DataFrame(mrl.rows)

end