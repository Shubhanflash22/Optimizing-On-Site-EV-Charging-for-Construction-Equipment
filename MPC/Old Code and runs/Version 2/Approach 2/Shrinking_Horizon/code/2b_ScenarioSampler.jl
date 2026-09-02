# #############################################################################
# ScenarioSampler.jl  —  module ScenarioSampler
# -----------------------------------------------------------------------------
# The ONLY new piece of machinery Approach 2 (Stochastic / scenario-based MPC)
# adds on top of Approach 1. Everything else — the estimator, the plant pool,
# the window MILP's physics, the closed loop's bookkeeping — is unchanged.
#
# WHAT THIS MODULE DOES
# Approach 1 (certainty-equivalent MPC) hands `build_window_model` a single
# power vector `est.mu` and plans as if it were exactly true. Approach 2 instead
# draws a small set of `n_scenarios` SAMPLE power vectors from the SAME fitted
# posterior `N(mu, sd)` and hands the whole SET to the stochastic window MILP
# (`build_window_model_stochastic` in 3_MCSModel.jl), which plans one shared
# "here-and-now" action that must be feasible under EVERY sampled scenario at
# once (non-anticipativity), while later intervals are allowed to differ per
# scenario (recourse). See `docs/Understanding_Stochastic_MPC.md` for the full
# five-level explanation and a worked numerical example.
#
# WHERE THE SAMPLES COME FROM
# Exactly the same generative model as the shared plant pool
# (`draw_activity_power_pool` in 1_Common.jl): independent draws from
# `Normal(mu[a], sd[a])`, clipped at 0. The difference is WHEN and WHY they are
# drawn:
#   - the plant pool is drawn ONCE per run, up front, and is what "reality"
#     actually turns out to be (consumed by `next_power!`, one draw per
#     occurrence);
#   - a scenario set is drawn FRESH at every re-solve step, purely as the
#     planner's internal set of "what ifs" for that one optimisation — it is
#     never consumed, never shared across steps, and the planner throws it
#     away the moment the window is solved and the first interval applied.
# These two pieces of randomness are DELIBERATELY independent (different RNG
# stream) so that a run's scenario draws can never leak into — or be confused
# with — the realized plant truth. That separation is what keeps "planning
# under uncertainty" and "what actually happened" honest and comparable to
# Approach 0 / Approach 1's own separation of planner vs plant.
#
# SCENARIO COUNT
# `n_scenarios` defaults to 5 (a small, fast, illustrative set — enough to
# capture "optimistic / a couple of average / pessimistic" without blowing up
# the MILP, which scales roughly linearly in S). It is a plain keyword
# argument everywhere it is used, so it can be changed for a run without
# touching this file: `run_mpc(d, pool; n_scenarios = 10, ...)`. Equal
# probability weights (1/S) are used throughout; nothing here assumes S = 5.
# #############################################################################
module ScenarioSampler

using Random

export sample_scenarios, equal_weights, DEFAULT_N_SCENARIOS

# Kept as one named constant (rather than sprinkling the literal `5` around)
# so the default is easy to find and change in exactly one place.
const DEFAULT_N_SCENARIOS = 5

# -----------------------------------------------------------------------------
# Draw `n_scenarios` per-activity power vectors from the current posterior
# (mu, sd) — one independent draw per scenario, same generative model as
# `draw_activity_power_pool`: Normal(mu[a], sd[a]) clipped at 0. An activity
# pinned deterministic (sd <= 0, e.g. idle) is identical across every scenario,
# exactly as it is deterministic in the plant pool.
#
# Returns a `Vector{Vector{Float64}}`: `scenarios[s][a]` = the sampled power
# (kW) of activity `a` under scenario `s`. Pass a dedicated `rng` so scenario
# draws are reproducible and never share a stream with the plant pool.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CHANGE 4 -- STRATIFIED SCENARIO SAMPLING (Issue 3)
# -----------------------------------------------------------------------------
# The i.i.d. draws above have no guarantee any of the n_scenarios lands in the
# risky tail -- 5 independent Normal draws can easily all land "comfortably
# average" on a given re-solve, missing the one branch that would have
# triggered a protective hedge (see Issue 3 in the handoff notes).
#
# When n_scenarios == 5 (the default), replace the draws with 5 FIXED, evenly
# spread bins -- extreme-low / slightly-low / near-mean / extreme-high /
# mild-high -- each still genuinely random WITHIN its bin (a fresh random
# fraction drawn every call, per activity), so consecutive re-solves get
# different concrete numbers, but a value in the tail is now structurally
# guaranteed every single re-solve instead of left to chance:
#
#   Extreme low   mu - (1+r1)*sd     r1 ~ U(0,1)
#   Slightly low  mu - r2*sd         r2 ~ U(0,1)
#   Near mean     mu + r3*sd         r3 ~ U(-0.3,0.3)
#   Extreme high  mu + (1+r4)*sd     r4 ~ U(0,1)
#   Mild high     mu + r5*sd         r5 ~ U(0,1)
#
# For any OTHER n_scenarios (this function is called with a plain keyword
# elsewhere, e.g. n_scenarios = 10), the 5-bin formula above has no natural
# generalization, so this falls back to the original i.i.d. draws for that
# call -- unchanged behaviour outside the n_scenarios = 5 default.
# -----------------------------------------------------------------------------
function sample_scenarios(mu::AbstractVector{<:Real}, sd::AbstractVector{<:Real},
                          n_scenarios::Int = DEFAULT_N_SCENARIOS; rng = Random.GLOBAL_RNG)
    n_scenarios >= 1 || error("sample_scenarios: n_scenarios must be >= 1, got $n_scenarios")
    length(mu) == length(sd) ||
        error("sample_scenarios: mu and sd must have the same length ($(length(mu)) vs $(length(sd)))")
    B = length(mu)
    scenarios = Vector{Vector{Float64}}(undef, n_scenarios)

    if n_scenarios == 5
        for s in 1:5
            scenarios[s] = Vector{Float64}(undef, B)
        end
        for a in 1:B
            if sd[a] <= 1e-12
                for s in 1:5
                    scenarios[s][a] = float(mu[a])
                end
                continue
            end
            r1 = rand(rng); r2 = rand(rng); r3 = -0.3 + 0.6 * rand(rng); r4 = rand(rng); r5 = rand(rng)
            scenarios[1][a] = max(mu[a] - (1 + r1) * sd[a], 0.0)   # extreme low
            scenarios[2][a] = max(mu[a] - r2 * sd[a],       0.0)   # slightly low
            scenarios[3][a] = max(mu[a] + r3 * sd[a],       0.0)   # near mean
            scenarios[4][a] = max(mu[a] + (1 + r4) * sd[a], 0.0)   # extreme high
            scenarios[5][a] = max(mu[a] + r5 * sd[a],       0.0)   # mild high
        end
    else
        for s in 1:n_scenarios
            scenarios[s] = [sd[a] <= 1e-12 ? float(mu[a]) : max(mu[a] + sd[a] * randn(rng), 0.0) for a in 1:B]
        end
    end

    return scenarios
end

# Equal (1/S) probability weights over `n_scenarios` scenarios. Kept as its own
# tiny function (rather than inlining `fill(1/S, S)` at every call site) so a
# future non-uniform weighting scheme (e.g. importance-weighted or reduced
# scenarios) has exactly one place to change.
equal_weights(n_scenarios::Int) = fill(1.0 / n_scenarios, n_scenarios)

end # module ScenarioSampler
