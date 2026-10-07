# run_everything.jl  (temporary helper, delete before the public release)
#
# Runs, for each approach, the normal driver run and then its run_all_modes.jl sweep.
# Every run gets its own fresh Julia process, so the three codebases never clash.
#
# From the MPC root folder:
#   julia --project=. --threads=auto run_everything.jl
#
# Edit the three constants below to run only part of it.

using Printf

const ROOT     = dirname(@__DIR__)
# const APPROACHES = [0, 1, 2]
const APPROACHES = [2]
const RUN_NORMAL = true
const RUN_SWEEP  = true

approach_dir(a) = joinpath(ROOT, "Approach $a")
params_csv(a)   = joinpath(approach_dir(a), "data", "input_data", "parameters.csv")

# Small driver script for the normal run of one approach (written to a temp folder).
function driver_text(a)
    if a == 0
        main = joinpath(approach_dir(0), "code", "6_OneShot_main.jl")
        return "SCENARIO0_NO_AUTORUN = true\ninclude($(repr(main)))\nrun_scenario_0()\n"
    else
        main = joinpath(approach_dir(a), "code", "6_Shrinking_Horizon_main.jl")
        return "SCENARIO1_NO_AUTORUN = true\ninclude($(repr(main)))\nrun_scenario_1(run_regression = false, time_limit_sec = 1200.0)\n"
    end
end

function run_script(script, label, a)
    println("\n", "#" ^ 72, "\n# ", label, "\n", "#" ^ 72)
    t0  = time()
    cmd = Cmd(`$(Base.julia_cmd()) --project=$ROOT --threads=auto $script`; dir = approach_dir(a))
    ok  = try
        run(ignorestatus(cmd)).exitcode == 0
    catch err
        println("  could not start: ", sprint(showerror, err))
        false
    end
    return (; label, ok, secs = time() - t0)
end

results = NamedTuple[]
tmp = mktempdir()

for a in APPROACHES
    # Safety net: some runs can rewrite parameters.csv (the step 0 refit), so keep a copy and put it back.
    pfile  = params_csv(a)
    backup = isfile(pfile) ? read(pfile) : nothing

    if RUN_NORMAL
        script = joinpath(tmp, "normal_A$a.jl")
        write(script, driver_text(a))
        push!(results, run_script(script, "Approach $a : normal run", a))
    end
    if RUN_SWEEP
        script = joinpath(approach_dir(a), "code", "run_all_modes.jl")
        push!(results, run_script(script, "Approach $a : sweep (run_all_modes.jl)", a))
    end

    if backup !== nothing && read(pfile) != backup
        write(pfile, backup)
        println("\nNOTE: Approach $a parameters.csv was changed by a run; restored the original.")
    end
end

println("\n", "=" ^ 72, "\nDONE\n", "=" ^ 72)
for r in results
    @printf("%-45s %-7s %8.1f min\n", r.label, r.ok ? "OK" : "FAILED", r.secs / 60)
end
