# #############################################################################
# Comparison_main.jl  —  TOP-LEVEL 3-WAY + 3-SUBSET COMPARISON DRIVER
# -----------------------------------------------------------------------------
# Runs, side by side, from the SAME input data and the SAME shared power pool:
#   * Approach 0    (A0)  — one-shot plan, executed open-loop, no replanning
#                    (taken from ONE canonical codebase's run_one_shot by
#                    default — see `approach0_source` below; verified
#                    byte-identical across both candidate sources, so either
#                    one is an equally valid choice — see the note above
#                    `module A1RecedingApp`)
#   * Approach 1b   (A1R) — Approach 1, Receding Horizon closed-loop MPC
#                    (n_days = n_days_receding)
#   * Approach 2b   (A2R) — Approach 2, Receding Horizon, STOCHASTIC
#                    scenario-based closed-loop MPC (same n_days_receding)
#
# All three are solved EXACTLY ONCE per run_comparison() call and then sliced
# into 4 output folders — the full 3-way comparison plus the 3 requested
# subsets — so nothing is re-solved per comparison:
#
#   Output/A0_A1R/         A0 vs A1R
#   Output/A0_A2R/         A0 vs A2R
#   Output/A0_A1R_A2R/     A0 vs A1R vs A2R
#   Output/A1R_A2R/        A1R vs A2R
#
# Each subfolder gets the FULL merged artefact set (see 8_ComparisonOutput.jl
# module header): 01_total_grid_power_profile.png/.csv … 09_mcs_<m>_power_
# profile.png/.csv, 07_mcs_optimization_summary.png,
# 07_approach_timeline_comparison.png, 08_kpi_metrics_summary.png,
# 08_cost_kpi_metrics.csv, a <keys...>.html KPI table, and (new)
# 10_diagnostic_dispatch_trace.csv / 11_diagnostic_capacity_summary.csv --
# see 8_ComparisonOutput.jl's write_diagnostic_dispatch_trace for what these
# contain. No changes needed here: it's called automatically from inside
# write_comparison_outputs, using data already returned by run_mpc /
# run_one_shot -- Approach 1/2's own solver code (4_MPCLoop.jl etc.) is
# untouched.
#
# MULTI-DAY (n_days_receding) — WHO ACTUALLY HONORS IT:
#   A1R / A2R (Receding Horizon)      -- yes, natively. Their own MPCLoop.jl
#                                         has a day-loop with real state
#                                         carried from one day into the next.
#   A0 (one-shot, source = receding)  -- yes. run_one_shot re-solves once per
#                                         KEPT day, each time starting from
#                                         the REAL carried-over state. Default
#                                         is now approach0_source =
#                                         :a1_receding so A0 is genuinely
#                                         multi-day too, comparable to A1R/A2R.
# THE CODE IS CALLED IN PLACE — NOTHING IS COPIED. This driver include()s both
# source codebases directly from their own folders (paths resolved relative
# to THIS file, not hardcoded, so the whole Comparison_A0_A1_A2 folder is
# portable as long as it stays a sibling of Approach 1 / Approach 2 under the
# same MPC root — see _MPC_ROOT below). Edit either codebase in place and the
# next run picks the change up automatically.
#
# HOW THE TWO CODEBASES ARE KEPT SEPARATE (WITHOUT DUPLICATING Common.jl)
# Both codebases define modules with the SAME names (Common, DataLoader,
# MCSModel, MPCLoop, Output, plus ScenarioSampler for Approach 2), so each is
# include()-d inside its own wrapper module below to avoid one silently
# overwriting another. Common.jl is the ONE exception: only A1RecedingApp
# includes its own copy; A2RecedingApp ALIASES it. This is what makes it
# possible to build exactly ONE `ActivityPowerPool` object and hand the SAME
# object to all three runs — they are otherwise distinct Julia modules, so
# without this alias the pool's type from one app would be rejected by the
# other's `run_mpc`/`run_one_shot` (a different nominal type), forcing
# separately-built pools and quietly voiding the shared-plant comparison.
#
# This aliasing choice was VERIFIED against the actual codebase (checksummed,
# not assumed) before this driver was written:
#   - Approach 1/Receding_Horizon/code/1_Common.jl
#       ==  Approach 2/Receding_Horizon/code/1_Common.jl    (byte-identical)
# So Approach 1's copy is aliased into A2RecedingApp below — every app gets
# everything it needs, nothing is lost.
# The same checksum pass also confirmed 2_DataLoader.jl matches between
# Approach 1 / Approach 2 (so dA1R/dA2R are loaded through literally the same
# DataLoader code), and that run_one_shot's function BODY is byte-identical
# between A1R and A2R, consistent with A0 being interchangeable across both.
#
# WHAT GETS WRITTEN — see the folder list above; each subfolder additionally
# gets its own run_log.txt is NOT written per-subfolder (there's one shared
# Output/run_log.txt for the whole run — see _with_console_log below).
# #############################################################################

using Printf
using Random
using Dates

const _CODE_DIR = @__DIR__

# -----------------------------------------------------------------------------
# STATUS LOGGING — timestamped progress lines so a long run (each of A0/A1R/A2R
# can take up to time_limit_sec) shows visible life instead of going silent.
# _status prints "[HH:MM:SS] msg". _timed_status wraps a stage: prints a
# "... starting" line, runs f(), then prints "... done in Xs" (or "FAILED
# after Xs" if f() throws, then rethrows so the run still stops on error).
# -----------------------------------------------------------------------------
_status(msg::AbstractString) = println("[$(Dates.format(now(), "HH:MM:SS"))] ", msg)

function _timed_status(f, label::AbstractString)
    _status("-> starting: $(label)")
    t0 = time()
    local result
    try
        result = f()
    catch e
        _status("-> FAILED: $(label) (after $(round(time() - t0, digits=1))s)")
        rethrow(e)
    end
    _status("-> done: $(label) ($(round(time() - t0, digits=1))s)")
    return result
end

# -----------------------------------------------------------------------------
# DEFAULT PATHS — resolved relative to THIS file, not a hardcoded machine
# path (see RUN_ALL.jl / mainA2.jl for the same convention). _CODE_DIR is
# .../Comparison_A0_A1_A2/Code, so one level up is Comparison_A0_A1_A2 and two
# levels up is the MPC root that Approach 1 / Approach 2 also live under.
# -----------------------------------------------------------------------------
const _ROOT     = normpath(joinpath(_CODE_DIR, ".."))
# --- TEMP-RUN PATCH (see CHANGES.md) -----------------------------------------
# This copy of Comparison_A0_A1_A2 lives in Downloads, NOT as a sibling of
# Approach 1 / Approach 2 under the Desktop MPC root, so the normal "two
# levels up" resolution (`normpath(joinpath(_ROOT, ".."))`) would look for
# Approach 1/2 inside Downloads and fail. Hardcode the real MPC root instead
# -- Approach 1 and Approach 2 still live here, unmoved, on the Desktop.
const _MPC_ROOT = raw"C:\Users\shubh\Desktop\MPC"
# ------------------------------------------------------------------------------
const _A1_ROOT  = joinpath(_MPC_ROOT, "Approach 1")
const _A2_ROOT  = joinpath(_MPC_ROOT, "Approach 2")

const _A1R_CODE  = joinpath(_A1_ROOT, "Receding_Horizon",  "code")
const _A2R_CODE  = joinpath(_A2_ROOT, "Receding_Horizon",  "code")

const _A1R_INPUT = joinpath(_A1_ROOT, "Receding_Horizon",  "data", "input_data")
const _A2R_INPUT = joinpath(_A2_ROOT, "Receding_Horizon",  "data", "input_data")

const _COMPARISON_INPUT = joinpath(_ROOT, "Input")
const _COMPARISON_OUT   = joinpath(_ROOT, "Output")
const _DEFAULT_REGRESSION_DATA_DIR = raw"C:\Users\shubh\Desktop\Bayesian Regression"

# The 6 input files confirmed byte-identical across BOTH input_data
# folders (checksummed on the actual codebase before writing this driver —
# see the note above). parameters.csv is deliberately excluded here: it is
# (re)built ONCE by the step-0 regression below and shared by all three runs.
const _SHARED_INPUT_FILES = ["time_data.csv", "travel_time.csv", "work_flexible.csv",
                              "ev_data.csv", "mcs_data.csv", "place.csv"]

# =============================================================================
# NAMESPACED APP WRAPPERS — see the big header comment above for why each is
# its own module and why only A1RecedingApp includes 1_Common.jl.
# =============================================================================
module A1RecedingApp
    # TEMP-RUN PATCH (see CHANGES.md): was normpath(joinpath(@__DIR__, "..", "..",
    # "Approach 1", "Receding_Horizon", "code")) -- that resolves relative to
    # wherever THIS file physically sits, which is now Downloads, not Desktop\MPC.
    # Hardcoded to the real (unmoved) Desktop location of Approach 1/2 instead.
    const _DIR = raw"C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\code"
    include(joinpath(_DIR, "1_Common.jl"))
    include(joinpath(_DIR, "0_Regression.jl"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

module A2RecedingApp
    import ..A1RecedingApp
    const Common = A1RecedingApp.Common
    # TEMP-RUN PATCH (see CHANGES.md): hardcoded for the same reason as A1RecedingApp above.
    const _DIR = raw"C:\Users\shubh\Desktop\MPC\Approach 2\Receding_Horizon\code"
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "2b_ScenarioSampler.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

include(joinpath(_CODE_DIR, "8_ComparisonOutput.jl"))
using .ComparisonOutput: Approach, write_comparison_outputs

# -----------------------------------------------------------------------------
# CONSOLE LOG CAPTURE — mirrors each source codebase's own _with_console_log,
# so `Output/run_log.txt` contains everything printed during the whole run
# (regression fit, both MPC loops solving live, all 4 comparison writes).
# -----------------------------------------------------------------------------
function _with_console_log(f, out_dir)
    mkpath(out_dir)
    log_path = joinpath(out_dir, "run_log.txt")
    open(log_path, "w") do logfile
        pipe = Pipe()
        orig_stdout = stdout
        orig_stderr = stderr
        tee_task = @async begin
            while !eof(pipe)
                data = readavailable(pipe)
                write(orig_stdout, data)
                write(logfile, data)
                flush(orig_stdout)
                flush(logfile)
            end
        end
        try
            redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    f()
                end
            end
        finally
            close(pipe.in)
            wait(tee_task)
        end
    end
end

# =============================================================================
# STEP: build Comparison/Input from ONE canonical source folder (either of
# the two — confirmed identical on the actual codebase, see
# _SHARED_INPUT_FILES note above) plus the step-0 regression's freshly
# (re)built parameters.csv.
# =============================================================================
function build_comparison_input(; input_dir::AbstractString = _COMPARISON_INPUT,
                                  csv_source_dir::AbstractString = _A1R_INPUT,
                                  # TEMP-RUN PATCH (see CHANGES.md): regression turned off by
                                  # default -- this run reuses the pre-staged, hand-edited
                                  # parameters.csv shipped in Input/ instead of refitting it.
                                  run_regression::Bool = false,
                                  regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                                  regression_samples::Int = 2000,
                                  regression_chains::Int = 4)
    mkpath(input_dir)
    for f in _SHARED_INPUT_FILES
        dst = joinpath(input_dir, f)
        # TEMP-RUN PATCH (see CHANGES.md): if the file is already staged in
        # input_dir (this run ships a fully pre-populated, hand-edited Input/
        # folder), keep it as-is rather than overwriting it from csv_source_dir.
        # This is what stops the stress-test edits to ev_data.csv (and the
        # other shared CSVs) from being silently clobbered on every run.
        if isfile(dst)
            continue
        end
        src = joinpath(csv_source_dir, f)
        isfile(src) || error("build_comparison_input: expected shared input file missing -> $src")
        cp(src, dst; force = true)
    end
    params_path = joinpath(input_dir, "parameters.csv")
    ok = false
    if run_regression
        ok = A1RecedingApp.Regression.run_regression(regression_data_dir, params_path;
                                                      mcmc_samples = regression_samples,
                                                      nchains = regression_chains)
    end
    if !ok && !isfile(params_path)
        @warn "build_comparison_input: regression did not produce parameters.csv; " *
              "falling back to a copy from $csv_source_dir"
        cp(joinpath(csv_source_dir, "parameters.csv"), params_path; force = true)
    end
    return input_dir
end

# =============================================================================
# THE 4 COMPARISONS — (output subfolder name, approach keys in the order
# they should appear as columns). Keys map to the `all_apps` Dict built in
# run_comparison(). Pass a different `combos` vector to run_comparison() to
# write a different subset without touching this default list.
# =============================================================================
const _ALL_COMBOS = [
    ("A0_A1R",     ["A0", "A1R"]),
    ("A0_A2R",     ["A0", "A2R"]),
    ("A0_A1R_A2R", ["A0", "A1R", "A2R"]),
    ("A1R_A2R",    ["A1R", "A2R"]),
]

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
function run_comparison(; input_dir::AbstractString = _COMPARISON_INPUT,
                          out_dir::AbstractString = _COMPARISON_OUT,
                          csv_source_dir::AbstractString = _A1R_INPUT,
                          # TEMP-RUN PATCH (see CHANGES.md): regression turned off by
                          # default -- reuses the pre-staged, hand-edited parameters.csv.
                          run_regression::Bool = false,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          # Which codebase's run_one_shot is "Approach 0". Verified
                          # byte-identical between A1R and A2R on the actual codebase,
                          # so either is an equally valid choice; :a1_receding is the
                          # default only to match the original comparison scripts.
                          approach0_source::Symbol = :a1_receding,   # :a1_receding / :a2_receding
                          # PLANT MODE for Approach 0's open-loop replay:
                          #   :sampled  the one-shot plan drifts under the shared pool
                          #   :mean     realized power pinned to mu, so realized == planned
                          #             and Approach 0's KPIs are the MILP's own optimum
                          # Both closed loops (A1R/A2R) are ALWAYS :sampled, so a :mean
                          # Approach 0 makes the reported gaps mix plan drift with the
                          # value of re-planning rather than isolating the latter.
                          approach0_plant::Symbol = :sampled,
                          # time_limit_sec::Float64 = Inf,
                          time_limit_sec::Float64 = 600.0, #<------------------------------------------ COMMENT THIS LINE FOR THE FULL RUN (Inf) ------------------------------------>
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          mcmc_samples::Int = 500,
                          H::Int = 16,
                          n_days_receding::Int = 1,
                          pool_n_samples::Union{Nothing, Int} = nothing,
                          # APPROACH 2: scenarios sampled from the posterior at every
                          # re-solve, for the Receding stochastic closed loop (A2R).
                          n_scenarios::Int = A2RecedingApp.ScenarioSampler.DEFAULT_N_SCENARIOS,
                          # Which comparisons to write -- defaults to the full set of
                          # 4 (3-way + all 3 requested subsets). Pass a subset of
                          # _ALL_COMBOS to skip some, e.g. during a quick test run.
                          combos = _ALL_COMBOS,
                          seed::Int = 1)
    approach0_source in (:a1_receding, :a2_receding) ||
        error("run_comparison: approach0_source must be :a1_receding or :a2_receding")
    approach0_plant in (:sampled, :mean) ||
        error("run_comparison: approach0_plant must be :sampled or :mean")

    build_comparison_input(; input_dir, csv_source_dir, run_regression,
                            regression_data_dir, regression_samples, regression_chains)

    return _with_console_log(out_dir) do
        _run_t0 = time()
        println("="^78)
        println("3-WAY COMPARISON — Approach 0 vs A1-Receding vs A2-Receding")
        println("A1 Receding  code: $(_A1R_CODE)")
        println("A2 Receding  code: $(_A2R_CODE)")
        println("Input             : $(abspath(input_dir))")
        println("Output            : $(abspath(out_dir))")
        println("Comparisons       : ", join(first.(combos), ", "))
        _status("Run started")
        println("="^78)

        # ---- load data separately with each app's OWN DataLoader, from the
        # SAME Comparison/Input folder ----
        dA1R, dA2R = _timed_status("loading input data (A1R + A2R DataLoaders)") do
            (A1RecedingApp.DataLoader.load_data(:input;  input_dir = input_dir),
             A2RecedingApp.DataLoader.load_data(:input;  input_dir = input_dir))
        end

        # ---- ONE shared ActivityPowerPool, built ONCE, passed as the literal
        # SAME object into all three runs below (see the module header note
        # for why this is possible across two distinct codebases).
        #
        # SIZING. next_power! ERRORS (it does not wrap) once a cursor walks past
        # the end of the pre-drawn samples, so the pool must cover the LONGEST
        # run, not the reported horizon. Both Receding runs simulate
        # n_days_receding + 1 days -- they always add a BUFFER day, which is
        # simulated in full and only then dropped from the reported outputs --
        # so their cursors keep drawing through that extra day. Sizing on
        # nK_day alone would leave roughly half the margin a Receding run can
        # need, and the failure would land mid-run after the MILPs had already
        # been solving for a while. Size on the buffer-inclusive length
        # instead; unconsumed samples cost nothing but a little memory.
        nK_day = length(collect(dA1R.K))
        n_samples = pool_n_samples === nothing ?
            nK_day * (n_days_receding + 1) + 5 : pool_n_samples
        pool = _timed_status("building shared power pool ($(n_samples) samples/entity-activity)") do
            # mode = :normal -> unbiased draws (unchanged); see 1_Common.jl's
            # "DRAW MODE" doc for the 4 sensitivity-sweep modes -- swept by
            # 7_Comparison_main_RecedingOnlyVersion_Sweep.jl, which reuses this
            # same driver rather than duplicating it.
            A1RecedingApp.Common.draw_activity_power_pool(dA1R.E, dA1R.prior_mu, dA1R.prior_sigma;
                                                           n_samples = n_samples,
                                                           rng = MersenneTwister(seed),
                                                           mode = :normal)
        end
        println("\nShared power pool: n_samples=$(n_samples) per (entity, activity) ",
                "($(nK_day) intervals/day x $(n_days_receding + 1) days incl. the Receding ",
                "buffer day, + 5); mu=", round.(pool.mu, digits = 2),
                " kW; sd=", round.(pool.sd, digits = 2), " kW")

        # ---- APPROACH 0 (one-shot, no replanning) ----
        println("\n--- Approach 0 (one-shot, plant = :$(approach0_plant), source = :$(approach0_source)) ---")
        res0 = _timed_status("Approach 0 solve (source = :$(approach0_source))") do
            if approach0_source == :a1_receding
                A1RecedingApp.MPCLoop.run_one_shot(dA1R, pool; plant = approach0_plant,
                                                   time_limit_sec, multi_activity,
                                                   require_site_visit, single_visit_per_site,
                                                   n_days = n_days_receding, seed)
            else # :a2_receding
                A2RecedingApp.MPCLoop.run_one_shot(dA2R, pool; plant = approach0_plant,
                                                   time_limit_sec, multi_activity,
                                                   require_site_visit, single_visit_per_site,
                                                   n_days = n_days_receding, seed)
            end
        end

        # ---- APPROACH 1b: Receding Horizon closed-loop MPC ----
        println("\n--- Approach 1 - Receding (n_days = $(n_days_receding)) ---")
        resA1R = _timed_status("Approach 1 - Receding solve (n_days = $(n_days_receding))") do
            A1RecedingApp.MPCLoop.run_mpc(dA1R, pool; time_limit_sec, multi_activity,
                                          require_site_visit, single_visit_per_site,
                                          mcmc_samples, n_days = n_days_receding, seed)
        end

        # ---- APPROACH 2b: Receding Horizon, stochastic scenario-based closed-loop MPC ----
        println("\n--- Approach 2 - Receding (stochastic, $(n_scenarios) scenarios, n_days = $(n_days_receding)) ---")
        resA2R = _timed_status("Approach 2 - Receding solve ($(n_scenarios) scenarios, n_days = $(n_days_receding))") do
            A2RecedingApp.MPCLoop.run_mpc(dA2R, pool; time_limit_sec, multi_activity,
                                          require_site_visit, single_visit_per_site,
                                          mcmc_samples, n_scenarios, n_days = n_days_receding, seed)
        end

        # ---- the three Approach identities, keyed exactly as _ALL_COMBOS expects ----
        all_apps = Dict(
            "A0"  => Approach("A0",  "Approach 0 (one-shot, :$(approach0_plant))", res0,   :gray40),
            "A1R" => Approach("A1R", "Approach 1 - Receding",                      resA1R, :firebrick),
            "A2R" => Approach("A2R", "Approach 2 - Receding (stochastic)",         resA2R, :darkorange),
        )

        # ---- write every requested comparison — same 3 solved results,
        # sliced into as many output folders as `combos` lists ----
        println("\n--- Writing $(length(combos)) comparison(s) ---")
        for (folder, keys) in combos
            apps = [all_apps[k] for k in keys]
            sub_out = joinpath(out_dir, folder)
            println("  $(folder): ", join([a.label for a in apps], " vs "), "  ->  $(sub_out)")
            _timed_status("writing $(folder)") do
                Base.invokelatest(write_comparison_outputs, apps, sub_out)
            end
        end

        # ---- headline KPI printout (all three, regardless of which combos ran) ----
        println("\n" * "="^78)
        println("KPI SUMMARY (all three)")
        @printf("%-28s %14s %14s %14s\n", "Metric", "A0", "A1-Reced", "A2-Reced")
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Grid energy (kWh)", res0.total_energy, resA1R.total_energy, resA2R.total_energy)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Energy cost (USD)", res0.total_cost, resA1R.total_cost, resA2R.total_cost)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "CO2 (kg)", res0.total_co2, resA1R.total_co2, resA2R.total_co2)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "NCD peak (kW)", res0.nc_peak, resA1R.nc_peak, resA2R.nc_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "OPD peak (kW)", res0.op_peak, resA1R.op_peak, resA2R.op_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Missed work (h)", res0.missed, resA1R.missed, resA2R.missed)
        println("="^78)
        println("\nResults written to: $(abspath(out_dir))")
        for (folder, keys) in combos
            println("  $(folder)/  (", join(keys, " vs "), ")")
        end
        println("  run_log.txt  (this console log, shared across all $(length(combos)) comparisons)")
        _status("Run finished — total elapsed $(round(time() - _run_t0, digits=1))s")

        return (; res0, resA1R, resA2R, all_apps, dA1R, dA2R, pool)
    end
end

# Auto-run unless a harness defines COMPARISON_NO_AUTORUN = true first.
if !(@isdefined(COMPARISON_NO_AUTORUN) && COMPARISON_NO_AUTORUN)
    run_comparison(n_days_receding = 5, approach0_source = :a1_receding)
end
