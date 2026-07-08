# #############################################################################
# BayesianEstimator.jl  —  module BayesianEstimator
# -----------------------------------------------------------------------------
# The "learning" half of the closed loop: a Bayesian regression that turns
# measured per-interval energy into a refined estimate of each activity's power
# draw. The generative idea is linear-in-the-powers:
#
#     energy_this_interval ~= h_dig*p_dig + h_load*p_load + h_trv*p_trav + h_idle*p_idle
#
# so each observation is a pair (a, b): a = hours spent on each activity this
# interval, b = the measured energy (kWh). Regressing b on a recovers the powers
# p; doing it the Bayesian way returns a full posterior (mean + std), not just a
# point estimate. The posterior MEAN is what the MILP consumes each step; the
# posterior STD is carried for the convergence figure.
#
# This module has NO knowledge of the optimiser or the MPC loop — it only owns
# the estimator state and its update rules.
# #############################################################################
module BayesianEstimator

using Turing
using Statistics
using Random

export BayesianActivityEstimator, observe!, refit!

# Silence Turing's sampling progress bar at load time.
Turing.setprogress!(false)

# -----------------------------------------------------------------------------
# The Turing probabilistic model: priors on the four activity powers (truncated
# at 0), a HalfNormal observation-noise std, and a Normal likelihood linking the
# predicted energy (A*x) to each measured energy b. Explicit scalars x1..x4 are
# used (instead of a vector RV) because newer Turing versions retrieve named
# scalars from the chain more reliably.
# -----------------------------------------------------------------------------
Turing.@model function activity_power_model(A, b, prior_mu, prior_sigma, sigma_b)
    x1 ~ truncated(Normal(prior_mu[1], prior_sigma[1]); lower = 0.0)   # digging power
    x2 ~ truncated(Normal(prior_mu[2], prior_sigma[2]); lower = 0.0)   # loading/swinging power
    x3 ~ truncated(Normal(prior_mu[3], prior_sigma[3]); lower = 0.0)   # traveling power
    x4 ~ truncated(Normal(prior_mu[4], prior_sigma[4]); lower = 0.0)   # idling power
    x = [x1, x2, x3, x4]
    s ~ truncated(Normal(0.0, sigma_b); lower = 0.0)                   # observation-noise std
    mu = A * x
    for j in eachindex(b)
        b[j] ~ Normal(mu[j], s)
    end
end

# -----------------------------------------------------------------------------
# Mutable estimator state carried between updates: the fixed prior, all
# observations gathered so far, and the CURRENT posterior summary (mu = the
# profile fed to the MILP; sd = its uncertainty).
# -----------------------------------------------------------------------------
mutable struct BayesianActivityEstimator
    prior_mu::Vector{Float64}
    prior_sigma::Vector{Float64}
    A_obs::Matrix{Float64}
    b_obs::Vector{Float64}
    mu::Vector{Float64}
    sd::Vector{Float64}
    mcmc_samples::Int
end

# Constructor: no observations yet; posterior initialised to the prior.
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

# Re-run the Bayesian regression on ALL data so far and refresh mu / sd via NUTS.
function refit!(est::BayesianActivityEstimator)
    isempty(est.b_obs) && return est
    sigma_b = length(est.b_obs) > 1 ? max(std(est.b_obs), 1e-3) : 1.0
    model = activity_power_model(est.A_obs, est.b_obs, est.prior_mu, est.prior_sigma, sigma_b)
    chain = sample(model, NUTS(0.9), est.mcmc_samples; progress = false)
    syms = (:x1, :x2, :x3, :x4)
    for i in 1:length(est.prior_mu)
        col = vec(chain[syms[i]])
        est.mu[i] = mean(col)
        est.sd[i] = std(col)
    end
    return est
end

end # module BayesianEstimator
