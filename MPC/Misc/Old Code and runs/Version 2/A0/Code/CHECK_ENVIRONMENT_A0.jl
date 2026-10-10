# #############################################################################
# CHECK_ENVIRONMENT_A0.jl — pre-flight check. Run this BEFORE
# RUN_A0_ONLY_25_RUNS.jl. Adapted from the original Comparison_A0_A1_A2's
# CHECK_ENVIRONMENT.jl, trimmed to only what this A0-only folder needs
# (Approach 1's codebase + this folder's own A0App.jl -- no Approach 2, no
# Plots/plotting, since this folder never draws a figure).
#
#   julia --project=. CHECK_ENVIRONMENT_A0.jl && julia --project=. RUN_A0_ONLY_25_RUNS.jl
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
println("A0-ONLY ENVIRONMENT PRE-FLIGHT CHECK")
println("Julia version : ", VERSION)
println("Running from  : ", _THIS_DIR)
println("="^78)

_check("Julia version >= 1.9") do
    VERSION >= v"1.9" || error("Julia $(VERSION) is older than 1.9 — upgrade via juliaup")
    "OK ($(VERSION))"
end

# Turing is required only because 1_Common.jl unconditionally `using Turing`
# (it defines the Bayesian estimator A1S/A2S use; A0 itself never calls it,
# but the file still needs to load) -- same requirement your existing
# codebase already has.
const REQUIRED_PACKAGES = ["JuMP", "HiGHS", "DataFrames", "CSV", "Turing", "Random", "Printf", "Dates"]

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

if isempty(missing_pkgs)
    _check("Load A0App (Approach 1 codebase + detailed-A0 logic)") do
        include(joinpath(_THIS_DIR, "A0App.jl"))
        "all modules loaded, all include() paths resolved, no solve triggered"
    end
else
    println(rpad("[Load A0App]", 40), "SKIPPED (fix missing packages first)")
end

if get(RESULTS, "Load A0App (Approach 1 codebase + detailed-A0 logic)", false)
    _check("Input/ CSVs exist") do
        input_dir = normpath(joinpath(_THIS_DIR, "..", "Input"))
        required_csvs = ["ev_data.csv", "mcs_data.csv", "parameters.csv", "place.csv",
                          "time_data.csv", "travel_time.csv", "work_flexible.csv", "live_powers.csv"]
        missing = [f for f in required_csvs if !isfile(joinpath(input_dir, f))]
        isempty(missing) || error("missing from $(input_dir): $(join(missing, ", "))")
        "all 8 present in $(input_dir)"
    end

    _check("Input data loads via A0App.DataLoader") do
        input_dir = normpath(joinpath(_THIS_DIR, "..", "Input"))
        d = Main.A0App.DataLoader.load_data(:input; input_dir = input_dir)
        "loaded: $(length(d.E)) CEV(s), $(length(d.M)) MCS unit(s), $(length(collect(d.K))) intervals/day"
    end
end

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

_check("Output_A0_Only directory is writable") do
    out_dir = normpath(joinpath(_THIS_DIR, "..", "Output_A0_Only"))
    mkpath(out_dir)
    testfile = joinpath(out_dir, ".write_test")
    open(testfile, "w") do io
        write(io, "ok")
    end
    rm(testfile)
    "$(out_dir)"
end

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
    println("Fix these before running RUN_A0_ONLY_25_RUNS.jl.")
    println("="^78)
    exit(1)
else
    println("All checks passed. Safe to proceed to RUN_A0_ONLY_25_RUNS.jl.")
    println("="^78)
end
