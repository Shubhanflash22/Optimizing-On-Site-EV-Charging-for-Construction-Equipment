# #############################################################################
# ScenarioSampler.jl  -  module ScenarioSampler
# -----------------------------------------------------------------------------
# Draws the sampled activity power vectors, called scenarios, that the stochastic window MILP plans against, together with their probability weights.
# Used by 4_MPCLoop.jl, which draws a fresh set at every re-solve and passes it to build_window_model_stochastic in 3_MCSModel.jl.
# The draws use their own random generator, so they never touch the plant's sample pool in 1_Common.jl.
# The mean and standard deviation passed in are the values from parameters.csv, which a run never refits.
# Three groups:
#
#   1. SCENARIO COUNT
#      DEFAULT_N_SCENARIOS
#      -- the default number of scenarios per re-solve, kept as one named constant so it is changed in one place.
#
#   2. SCENARIO SAMPLING
#      sample_scenarios
#      -- draw one power vector per scenario, using five fixed bins when the count is exactly 5 and independent normal draws for any other count.
#
#   3. SCENARIO WEIGHTS
#      equal_weights
#      -- the probability weight of each scenario, equal for every scenario.
# #############################################################################
module ScenarioSampler

# external packages used across this file
using Random

# everything below that other files are allowed to use
export sample_scenarios, equal_weights, DEFAULT_N_SCENARIOS

# The default number of scenarios drawn at each re-solve.
const DEFAULT_N_SCENARIOS = 5

# Draws n_scenarios vectors of per-activity power in kW, where scenarios[s][a] is the power of activity a under scenario s.
# With exactly 5 scenarios each one comes from a fixed bin around mu, with scenario 1 the most optimistic low draw, and with any other count the draws are independent Normal(mu, sd), floored at 0.
# An activity with sd of zero, which is idle, gets mu in every scenario.
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
            # Sets activity a's power in the five scenarios from fixed bins around mu (extreme low, slightly low, near mean, extreme high, mild high), each floored at 0.
            r1 = rand(rng); r2 = rand(rng); r3 = -0.3 + 0.6 * rand(rng); r4 = rand(rng); r5 = rand(rng)
            scenarios[1][a] = max(mu[a] - (1 + r1) * sd[a], 0.0)   
            scenarios[2][a] = max(mu[a] - r2 * sd[a],       0.0)   
            scenarios[3][a] = max(mu[a] + r3 * sd[a],       0.0)  
            scenarios[4][a] = max(mu[a] + (1 + r4) * sd[a], 0.0)   
            scenarios[5][a] = max(mu[a] + r5 * sd[a],       0.0)   
        end
    else
        for s in 1:n_scenarios
            scenarios[s] = [sd[a] <= 1e-12 ? float(mu[a]) : max(mu[a] + sd[a] * randn(rng), 0.0) for a in 1:B]
        end
    end

    return scenarios
end

# Returns the probability weight of each scenario, which is 1 / n_scenarios for every one.
equal_weights(n_scenarios::Int) = fill(1.0 / n_scenarios, n_scenarios)

end 