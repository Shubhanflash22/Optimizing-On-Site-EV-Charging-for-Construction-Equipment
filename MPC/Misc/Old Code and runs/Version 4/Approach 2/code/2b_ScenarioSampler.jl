# #############################################################################
# ScenarioSampler.jl  -  module ScenarioSampler
# -----------------------------------------------------------------------------
# Builds the fixed activity power vectors, called scenarios, that the stochastic window MILP plans against, together with their probability weights.
# The scenarios are built once at the start of a run from the saved posterior draws in posterior_draws.csv (written by step 0 in 0_Regression.jl), and 4_MPCLoop.jl passes the same scenarios to every re-solve.
# Each draw is scored by the energy the day's work would need under that draw, the draws are sorted by that score and cut into n_scenarios equal groups, and each group is averaged activity by activity.
# The work hours come from the input data (place.csv, through the loaded data d), so nothing about the day is hardcoded here.
# Five groups:
#
#   1. SCENARIO COUNT
#      DEFAULT_N_SCENARIOS
#      -- the default number of scenarios, kept as one named constant so it is changed in one place.
#
#   2. WORK HOURS
#      scenario_work_hours
#      -- the day's digging, loading+swinging and traveling hours, read from d.
#
#   3. SCENARIO CONSTRUCTION
#      scenarios_from_draws, load_scenarios
#      -- score, sort, cut and average the posterior draws into the scenarios.
#
#   4. SCENARIO EXPORT
#      write_scenarios
#      -- write the scenarios with their work energy into A2_scenarios.csv.
#
#   5. SCENARIO WEIGHTS
#      equal_weights
#      -- the probability weight of each scenario, equal for every scenario.
# #############################################################################
module ScenarioSampler

# external packages used across this file
using CSV
using DataFrames
using Statistics
using Printf

# everything below that other files are allowed to use
export load_scenarios, write_scenarios, equal_weights, DEFAULT_N_SCENARIOS

# The default number of scenarios.
const DEFAULT_N_SCENARIOS = 5

# Returns the day's work hours as (dig, load, travel).
# dig and load are the sums of hours_digging and hours_loading_swinging over all sites in place.csv.
# travel follows the pacing rule of constraints 14e/14f: a site with W = (dig + load) / delta_T work intervals needs fld(W, kappa_wt) travel intervals for each CEV assigned to it, and each travel interval lasts delta_T hours.
function scenario_work_hours(d)
    hd = d.hours_digging
    hl = d.hours_loading_swinging
    travel = 0.0
    for i in d.N_c, e in d.E
        d.A[i, e] == 1 || continue
        W = round(Int, (hd[i] + hl[i]) / d.delta_T)
        travel += fld(W, d.kappa_wt) * d.delta_T
    end
    return (sum(hd), sum(hl), travel)
end

# Builds n scenarios from the posterior draws (one row per draw, columns dig, load, travel, idle).
# hours is (dig, load, travel) in hours.
# Each draw is scored by E = dig*p_dig + load*p_load + travel*p_trv, the draws are sorted by E, cut into n groups of (almost) equal size, and each group is averaged column by column.
# Returns scenarios, where scenarios[s][a] is the power of activity a in scenario s, with scenario 1 the lowest-energy group and scenario n the highest.
function scenarios_from_draws(draws::AbstractMatrix{<:Real}, hours, n::Int)
    N = size(draws, 1)
    n >= 1 || error("scenarios_from_draws: n must be >= 1, got $n")
    N >= n || error("scenarios_from_draws: need at least $n draws, got $N")
    E = hours[1] .* draws[:, 1] .+ hours[2] .* draws[:, 2] .+ hours[3] .* draws[:, 3]
    order = sortperm(E)
    return [vec(mean(draws[order[(fld((s - 1) * N, n) + 1):fld(s * N, n)], :]; dims = 1)) for s in 1:n]
end

# Reads the saved posterior draws from draws_csv and returns the n scenarios built from them for the day described by d.
# Stops with a clear message if the file is missing or if its means do not match the means in parameters.csv (tolerance 1e-3 kW), which would mean the two files come from different regression runs.
# Prints the scenario table, so it appears at the top of the run log.
function load_scenarios(draws_csv::AbstractString, d, n::Int = DEFAULT_N_SCENARIOS)
    isfile(draws_csv) ||
        error("load_scenarios: $draws_csv not found. Run step 0 (the regression) once so it writes posterior_draws.csv.")
    df = CSV.read(draws_csv, DataFrame)
    for c in ("dig", "load", "travel", "idle")
        c in names(df) || error("load_scenarios: column '$c' missing in $draws_csv")
    end
    draws = Matrix{Float64}(df[:, ["dig", "load", "travel", "idle"]])
    for a in 1:3
        abs(mean(draws[:, a]) - d.prior_mu[a]) <= 1e-3 ||
            error("load_scenarios: posterior_draws.csv and parameters.csv disagree on activity $a (draws mean " *
                  "$(round(mean(draws[:, a]), digits = 4)) vs parameters $(d.prior_mu[a])). Rerun step 0 and copy both files.")
    end
    hours = scenario_work_hours(d)
    scenarios = scenarios_from_draws(draws, hours, n)
    println("Scenarios from ", size(draws, 1), " posterior draws (work hours dig=", round(hours[1], digits = 2),
            ", load=", round(hours[2], digits = 2), ", travel=", round(hours[3], digits = 2), "):")
    @printf("  %-4s %-8s %-8s %-8s %-8s %-12s\n", "s", "dig", "load", "travel", "idle", "work kWh")
    for (s, p) in enumerate(scenarios)
        @printf("  %-4d %-8.3f %-8.3f %-8.3f %-8.3f %-12.3f\n", s, p[1], p[2], p[3], p[4],
                hours[1] * p[1] + hours[2] * p[2] + hours[3] * p[3])
    end
    return scenarios
end

# Writes the scenarios, their equal weights and their work energy in kWh to A2_scenarios.csv in out_dir.
function write_scenarios(out_dir::AbstractString, scenarios, d)
    hours = scenario_work_hours(d)
    n = length(scenarios)
    df = DataFrame(scenario = 1:n,
                   weight = equal_weights(n),
                   dig_kW = [p[1] for p in scenarios],
                   load_kW = [p[2] for p in scenarios],
                   travel_kW = [p[3] for p in scenarios],
                   idle_kW = [p[4] for p in scenarios],
                   work_energy_kWh = [hours[1] * p[1] + hours[2] * p[2] + hours[3] * p[3] for p in scenarios])
    CSV.write(joinpath(out_dir, "A2_scenarios.csv"), df)
end

# Returns the probability weight of each scenario, which is 1 / n_scenarios for every one.
equal_weights(n_scenarios::Int) = fill(1.0 / n_scenarios, n_scenarios)

end 