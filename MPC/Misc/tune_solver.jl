# =============================================================================
# tune_solver.jl   (put in C:\Users\shubh\Desktop\MPC\Test\)
# -----------------------------------------------------------------------------
# Compares solver settings for BOTH HiGHS and Gurobi on the same Approach 2
# windows, with a short time limit, in one go.
#
# How to run (fresh Julia REPL, nothing else heavy running):
#     include(raw"C:\Users\shubh\Desktop\MPC\Test\tune_solver.jl")
#
# What happens
#   * This script starts itself twice, each time in its own Julia process:
#     once for HiGHS (3_MCSModel.jl) and once for Gurobi (mcs_model_garobi.jl).
#     Separate processes are needed because both model files define the same
#     module names.
#   * Each process builds the stochastic window starting at each interval in
#     K_STARTS from the start-of-day state (same data, same 5 scenarios) and
#     re-solves that one model once per setting, TL seconds each.
#   * Everything is combined into ONE file:  Test\solver_tuning_all.csv
#     (column "solver" says highs or gurobi). The console prints only the
#     top 5 settings per solver and window.
#
# Reading the result: lower "objective" = better plan found within the time
# limit; smaller "gap_pct" = closer to proven optimal. Differences under about
# 1% in the objective are noise.
# =============================================================================
using JuMP
using Printf
using DataFrames
using CSV

const TL        = 120.0                  # seconds per setting per window
const K_STARTS  = [1, 5]                 # window start intervals (1 = 08:00, 5 = 09:00)
const N_SCEN    = 5
const SOLVERS   = ["highs", "gurobi"]
const CODE_DIR  = raw"C:\Users\shubh\Desktop\MPC\Approach 2\code"
const MODEL_FILE = Dict("highs" => "3_MCSModel.jl", "gurobi" => "mcs_model_garobi.jl")

# The settings to compare for each solver, as (label, [attribute => value, ...]).
function settings_grid(backend)
    g = Tuple{String,Vector{Pair{String,Any}}}[]
    if backend == "highs"
        for eff in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), sym in (true)
            push!(g, ("eff=$(eff) sym=$(sym)",
                      Pair{String,Any}["mip_heuristic_effort" => eff, "mip_detect_symmetry" => sym]))
        end
    else
        for h in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), focus in (0, 1, 2, 3)
            push!(g, ("Heuristics=$(h) MIPFocus=$(focus)",
                      Pair{String,Any}["Heuristics" => h, "MIPFocus" => focus]))
        end
    end
    return g
end

# ----------------------------------------------------------------------------
# Worker: runs one solver and writes Test\_tuning_<solver>.csv
# ----------------------------------------------------------------------------
function run_worker(backend)
    input_dir = joinpath(dirname(CODE_DIR), "data", "input_data")
    d = Main.load_data(input_dir)
    scenarios = Main.load_scenarios(joinpath(input_dir, "posterior_draws.csv"), d, N_SCEN)
    weights = fill(1.0 / N_SCEN, N_SCEN)
    nK = length(collect(d.K))

    rows = NamedTuple[]
    for k0 in K_STARTS
        println("\n=== ", backend, ": window starting at interval ", k0, " ===")
        soe_mcs = copy(float.(d.SOE_MCS_ini))
        soe_cev = copy(float.(d.SOE_CEV_ini))
        node    = [first(d.N_g) for _ in d.M]
        trans   = Any[nothing for _ in d.M]
        rd      = float.(d.hours_digging)
        rl      = float.(d.hours_loading_swinging)
        hist    = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]

        # builds the model (the builder also does one solve, which a 1 s limit keeps short)
        model = Main.MCSModel.build_window_model_stochastic(d, k0:nK, soe_mcs, soe_cev, node, trans,
                                                            rd, rl, hist, 0.0, 0.0, scenarios, weights;
                                                            time_limit_sec = 1.0, silent = true)
        set_silent(model)

        for (label, attrs) in settings_grid(backend)
            for (name, val) in attrs
                set_attribute(model, name, val)
            end
            set_time_limit_sec(model, TL)
            optimize!(model)
            ok   = has_values(model)
            obj  = ok ? objective_value(model) : NaN
            bnd  = try objective_bound(model) catch; NaN end
            gap  = try 100 * relative_gap(model) catch; NaN end
            tsec = try solve_time(model) catch; NaN end
            status = string(termination_status(model))
            push!(rows, (solver = backend, k0 = k0, setting = label, status = status,
                         objective = obj, bound = bnd, gap_pct = gap, secs = tsec))
            @printf("  %-34s %-12s obj=%10.3f  bound=%10.3f  gap=%7.2f%%  %6.1fs\n",
                    label, status, obj, bnd, gap, tsec)
        end
    end
    CSV.write(joinpath(@__DIR__, "_tuning_" * backend * ".csv"), DataFrame(rows))
end

# ----------------------------------------------------------------------------
# Driver: starts one worker process per solver, then combines the results
# ----------------------------------------------------------------------------
function run_driver()
    jl   = Base.julia_cmd()
    proj = Base.active_project()
    dfs  = DataFrame[]
    for backend in SOLVERS
        println("\n", "="^72, "\nTuning ", backend, "\n", "="^72)
        script = @__FILE__
        cmd = proj === nothing ? `$jl $script $backend` :
                                 `$jl --project=$(dirname(proj)) $script $backend`
        try
            run(cmd)
        catch err
            @warn "$backend run failed; continuing with the other solver." exception = err
        end
        part = joinpath(@__DIR__, "_tuning_" * backend * ".csv")
        if isfile(part)
            push!(dfs, CSV.read(part, DataFrame))
            rm(part; force = true)
        end
    end
    isempty(dfs) && (println("No results were produced."); return)
    df = reduce(vcat, dfs)
    out = joinpath(@__DIR__, "solver_tuning_all.csv")
    CSV.write(out, df)

    println("\n", "="^72, "\nTop 5 per solver and window (lowest objective first)\n", "="^72)
    for backend in SOLVERS, k0 in K_STARTS
        sub = df[(df.solver .== backend) .& (df.k0 .== k0), :]
        isempty(sub) && continue
        sub = sort(sub, [:objective, :gap_pct])
        println(backend, ", window ", k0, ":")
        for r in eachrow(first(sub, 5))
            @printf("  %-34s obj=%10.3f  gap=%7.2f%%\n", r.setting, r.objective, r.gap_pct)
        end
    end
    println("\nAll rows written to ", out)
end

if length(ARGS) >= 1
    SCENARIO1_NO_AUTORUN = true
    MCS_MODEL_FILE = MODEL_FILE[ARGS[1]]
    include(joinpath(CODE_DIR, "6_Shrinking_Horizon_main.jl"))
    run_worker(ARGS[1])
else
    run_driver()
end
