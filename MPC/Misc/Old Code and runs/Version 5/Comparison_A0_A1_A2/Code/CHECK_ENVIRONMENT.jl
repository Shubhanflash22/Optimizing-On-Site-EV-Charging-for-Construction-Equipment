# #############################################################################
# CHECK_ENVIRONMENT.jl — pre-flight check. Run this BEFORE RUN_ALL_9_RUNS.jl.
#
# Verifies, in order, and WITHOUT running any real MILP/MCMC solve:
#   1. Julia is a reasonable version
#   2. Every required package is installed and loadable
#   3. The full real codebase (Approach 1 + Approach 2 + Comparison driver)
#      loads cleanly — this exercises every path fix from earlier, since a
#      wrong path shows up immediately as "file not found" on include()
#   4. The key input files this all depends on actually exist on disk
#   5. HiGHS can solve an actual (trivial) LP — confirms the solver, not
#      just the package, works
#   6. The Output directory is writable
#
# Prints a clear PASS/FAIL per check and a summary at the end. Exits with a
# non-zero code if anything failed, so you can also use it in a script:
#   julia --project=. CHECK_ENVIRONMENT.jl && julia --project=. RUN_ALL_9_RUNS.jl
#
# Place this file in the SAME folder as 7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl
# and RUN_ALL_9_RUNS.jl (Comparison_A0_A1_A2/Code/).
# #############################################################################

const _THIS_DIR = @__DIR__
const RESULTS = Dict{String, Bool}()
const DETAILS = Dict{String, String}()

function _check(f::Function, name::String)
    print(rpad("[$(name)]", 40))
    try
        detail = f()
        RESULTS[name] = true
        DETAILS[name] = detail === nothing ? "" : string(detail)
        println("PASS  ", DETAILS[name])
    catch e
        RESULTS[name] = false
        DETAILS[name] = sprint(showerror, e)
        println("FAIL")
        println("        -> ", DETAILS[name])
    end
end

println("="^78)
println("ENVIRONMENT PRE-FLIGHT CHECK")
println("Julia version : ", VERSION)
println("Running from  : ", _THIS_DIR)
println("="^78)

# -----------------------------------------------------------------------------
# 1. Julia version sanity (warn only, not a hard fail — adjust the bound if
#    you know the exact version you developed against)
# -----------------------------------------------------------------------------
_check("Julia version >= 1.9") do
    VERSION >= v"1.9" || error("Julia $(VERSION) is older than 1.9 — upgrade via juliaup")
    "OK ($(VERSION))"
end

# -----------------------------------------------------------------------------
# 2. Each required package individually — so a missing package gives a clean
#    one-line answer instead of a stack trace buried inside a later include()
# -----------------------------------------------------------------------------
const REQUIRED_PACKAGES = ["JuMP", "HiGHS", "DataFrames", "CSV", "Turing", "Plots", "Random", "Printf", "Dates"]

for pkg in REQUIRED_PACKAGES
    _check("Package: $(pkg)") do
        Base.require(Main, Symbol(pkg))
        "loadable"
    end
end

missing_pkgs = [p for p in REQUIRED_PACKAGES if !get(RESULTS, "Package: $(p)", false)]
if !isempty(missing_pkgs)
    println()
    println("Missing packages detected. Install them with:")
    println("  julia --project=. -e 'using Pkg; Pkg.add([", join(["\"$(p)\"" for p in missing_pkgs], ", "), "])'")
    println("Then re-run this check before continuing.")
    println()
end

# -----------------------------------------------------------------------------
# 3. Load the REAL codebase (both Approach modules + Comparison driver) via
#    the same Sweep file RUN_ALL_9_RUNS.jl uses. NO_AUTORUN guards prevent
#    any actual solve from firing — this only defines functions/modules and
#    resolves every path constant. If a path fix from earlier is wrong, this
#    is where it will surface, as a clear "file not found" on a specific
#    joinpath(...).
# -----------------------------------------------------------------------------
if isempty(missing_pkgs)
    _check("Load full codebase (Approach 1 + 2 + Comparison driver)") do
        # The Sweep file itself declares `const COMPARISON_NO_AUTORUN = true`
        # before including the base driver, so that one doesn't need a guard
        # here. But the Sweep file ALSO has its own bottom-of-file autorun,
        # guarded separately by `SWEEP_NO_AUTORUN` — without this, just
        # loading the codebase for a check would kick off a real 5-mode solve.
        global SWEEP_NO_AUTORUN = true
        include(joinpath(_THIS_DIR, "7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl"))
        "all modules loaded, all include() paths resolved, no solve triggered"
    end
else
    println(rpad("[Load full codebase]", 40), "SKIPPED (fix missing packages first)")
end

# -----------------------------------------------------------------------------
# 4. Key input files actually exist on disk, using the SAME path constants
#    the real run will use (only meaningful if step 3 passed)
# -----------------------------------------------------------------------------
if get(RESULTS, "Load full codebase (Approach 1 + 2 + Comparison driver)", false)
    _check("Comparison Input/ CSVs exist") do
        required_csvs = ["ev_data.csv", "mcs_data.csv", "parameters.csv", "place.csv",
                          "time_data.csv", "travel_time.csv", "work_flexible.csv"]
        missing = [f for f in required_csvs if !isfile(joinpath(Main._COMPARISON_INPUT, f))]
        isempty(missing) || error("missing from $(Main._COMPARISON_INPUT): $(join(missing, ", "))")
        "all 7 present in $(Main._COMPARISON_INPUT)"
    end

    _check("Approach 1 code directory resolves") do
        isdir(Main.A1ShrinkingApp._DIR) || error("not found: $(Main.A1ShrinkingApp._DIR)")
        "$(Main.A1ShrinkingApp._DIR)"
    end

    _check("Approach 2 code directory resolves") do
        isdir(Main.A2ShrinkingApp._DIR) || error("not found: $(Main.A2ShrinkingApp._DIR)")
        "$(Main.A2ShrinkingApp._DIR)"
    end

    _check("Bayesian Regression folder resolves (not used this run, but checked)") do
        isdir(Main._DEFAULT_REGRESSION_DATA_DIR) || error("not found: $(Main._DEFAULT_REGRESSION_DATA_DIR)")
        "$(Main._DEFAULT_REGRESSION_DATA_DIR)"
    end
end

# -----------------------------------------------------------------------------
# 5. HiGHS actually solves — a tiny trivial LP, nothing to do with your model.
#    Confirms the solver binary/package works end-to-end on this machine.
#    NOTE: `using` must happen at top level (not inside the do-block) because
#    @variable/@objective/@constraint are macros — they get expanded when
#    Julia parses this code, before any runtime line inside the block (like a
#    dynamic Base.require) has run. A real top-level `using` is what makes
#    JuMP.@variable resolvable at all.
# -----------------------------------------------------------------------------
if get(RESULTS, "Package: JuMP", false) && get(RESULTS, "Package: HiGHS", false)
    using JuMP
    using HiGHS
end

if get(RESULTS, "Package: JuMP", false) && get(RESULTS, "Package: HiGHS", false)
    _check("HiGHS solves a trivial LP") do
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x >= 0)
        @objective(m, Min, x)
        @constraint(m, x >= 3)
        optimize!(m)
        status = termination_status(m)
        String(Symbol(status)) == "OPTIMAL" || error("unexpected status: $(status)")
        "solved, x=$(value(x)) as expected"
    end
end

# -----------------------------------------------------------------------------
# 6. Output directory is writable
# -----------------------------------------------------------------------------
_check("Output_AllRuns directory is writable") do
    out_dir = normpath(joinpath(_THIS_DIR, "..", "Output_AllRuns"))
    mkpath(out_dir)
    testfile = joinpath(out_dir, ".write_test")
    open(testfile, "w") do io
        write(io, "ok")
    end
    rm(testfile)
    "$(out_dir)"
end

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
println()
println("="^78)
n_pass = count(values(RESULTS))
n_total = length(RESULTS)
println("SUMMARY: $(n_pass)/$(n_total) checks passed")
if n_pass < n_total
    println()
    println("FAILED CHECKS:")
    for (name, ok) in RESULTS
        ok || println("  - $(name): $(DETAILS[name])")
    end
    println()
    println("Fix these before running RUN_ALL_9_RUNS.jl.")
    println("="^78)
    exit(1)
else
    println("All checks passed. Safe to proceed to RUN_ALL_9_RUNS.jl.")
    println("="^78)
end
