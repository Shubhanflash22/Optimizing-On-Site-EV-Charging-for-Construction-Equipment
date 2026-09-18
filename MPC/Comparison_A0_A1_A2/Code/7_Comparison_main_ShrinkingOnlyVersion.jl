# #############################################################################
# Comparison_main.jl  —  TOP-LEVEL 3-WAY + 3-SUBSET COMPARISON DRIVER
# -----------------------------------------------------------------------------
# Runs, side by side, from the SAME input data and the SAME shared power pool:
#   * Approach 0    (A0)  — one-shot plan, executed open-loop, no replanning.
#                    Its own independent codebase now (../Approach 0/) --
#                    no longer borrowed from Approach 1 or Approach 2.
#   * Approach 1b   (A1S) — Approach 1, Shrinking Horizon closed-loop MPC
#                    (n_day_run = n_day_run)
#   * Approach 2b   (A2S) — Approach 2, Shrinking Horizon, STOCHASTIC
#                    scenario-based closed-loop MPC (same n_day_run)
#
# All three are solved EXACTLY ONCE per run_comparison() call and then sliced
# into 4 output folders — the full 3-way comparison plus the 3 requested
# subsets — so nothing is re-solved per comparison:
#
#   Output/A0_A1S/         A0 vs A1S
#   Output/A0_A2S/         A0 vs A2S
#   Output/A0_A1S_A2S/     A0 vs A1S vs A2S
#   Output/A1S_A2S/        A1S vs A2S
#
# Each subfolder gets the FULL merged artefact set (see 8_ComparisonOutput.jl
# module header): 01_total_grid_power_profile.png/.csv … 09_mcs_<m>_power_
# profile.png/.csv, 07_mcs_optimization_summary.png,
# 07_approach_timeline_comparison.png, 08_kpi_metrics_summary.png,
# 08_cost_kpi_metrics.csv, a <keys...>.html KPI table, and
# 10_diagnostic_dispatch_trace.csv / 11_diagnostic_capacity_summary.csv --
# see 8_ComparisonOutput.jl's write_diagnostic_dispatch_trace for what these
# contain. No changes needed here: it's called automatically from inside
# write_comparison_outputs, using data already returned by run_mpc /
# run_one_shot -- Approach 0/1/2's own solver code (4_OneShot.jl /
# 4_MPCLoop.jl etc.) is untouched. ALL cross-approach images and comparison
# CSVs live here, in this Comparison folder, and only here -- Approach 0/1/2's
# own `write_outputs` only ever draws that one approach's own single-run
# figures.
#
# MULTI-DAY (n_day_run) — WHO ACTUALLY HONORS IT:
#   A1S / A2S (Shrinking Horizon)  -- yes, natively. Their own MPCLoop.jl has
#                                     an internal day loop with real state
#                                     carried from one day into the next.
#   A0 (one-shot)                  -- yes. run_one_shot re-solves once per
#                                     kept day, each time starting from the
#                                     REAL carried-over state (see
#                                     ../Approach 0/docs/README.md §5 for what
#                                     carries over and what resets daily).
#
# THE CODE IS CALLED IN PLACE — NOTHING IS COPIED. This driver include()s all
# three source codebases directly from their own folders (paths resolved
# relative to THIS file, not hardcoded, so the whole Comparison_A0_A1_A2
# folder is portable as long as it stays a sibling of Approach 0 / Approach 1
# / Approach 2 under the same MPC root — see _MPC_ROOT below). Edit any
# codebase in place and the next run picks the change up automatically.
#
# HOW THE THREE CODEBASES ARE KEPT SEPARATE (WITHOUT DUPLICATING Common.jl)
# All three codebases define modules with the SAME names (Common, DataLoader,
# MCSModel, plus MPCLoop/Output for A1/A2, ScenarioSampler for A2), so each is
# include()-d inside its own wrapper module below to avoid one silently
# overwriting another. Common.jl is the ONE exception: only A1ShrinkingApp
# includes its own copy; A0App and A2ShrinkingApp both ALIAS it. This is what
# makes it possible to build exactly ONE `ActivityPowerPool` object and hand
# the SAME object to all three runs — they are otherwise distinct Julia
# modules, so without this alias the pool's type from one app would be
# rejected by another app's `run_mpc`/`run_one_shot` (a different nominal
# type), forcing separately-built pools and quietly voiding the shared-plant
# comparison.
#
# WHAT GETS WRITTEN — see the folder list above; each subfolder additionally
# gets its own run_log.txt is NOT written per-subfolder (there's one shared
# Output/run_log.txt for the whole run — see _with_console_log below).
# #############################################################################

using Printf
using Random
using Dates
using CSV

const _CODE_DIR = @__DIR__

# -----------------------------------------------------------------------------
# STATUS LOGGING — timestamped progress lines so a long run (each of A0/A1S/A2S
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
# _MPC_ROOT is the folder that contains Approach 0 / Approach 1 / Approach 2 /
# Comparison_A0_A1_A2 as siblings -- two levels up from this file
# (.../Comparison_A0_A1_A2/Code). Resolved relative to THIS file so the whole
# Comparison_A0_A1_A2 folder is portable as long as it stays a sibling of the
# three Approach folders, wherever that is.
const _MPC_ROOT = normpath(joinpath(_ROOT, ".."))
const _A1_ROOT  = joinpath(_MPC_ROOT, "Approach 1")
const _A2_ROOT  = joinpath(_MPC_ROOT, "Approach 2")
const _A0_ROOT  = joinpath(_MPC_ROOT, "Approach 0")

const _A1S_CODE  = joinpath(_A1_ROOT, "code")
const _A2S_CODE  = joinpath(_A2_ROOT, "code")
const _A0_CODE   = joinpath(_A0_ROOT, "code")

const _A1S_INPUT = joinpath(_A1_ROOT, "data", "input_data")
const _A2S_INPUT = joinpath(_A2_ROOT, "data", "input_data")

const _COMPARISON_INPUT = joinpath(_ROOT, "Input")
const _COMPARISON_OUT   = joinpath(_ROOT, "Output")
const _DEFAULT_REGRESSION_DATA_DIR = raw"C:\Users\shubh\Desktop\Bayesian Regression"

# The 6 input files confirmed byte-identical across BOTH input_data
# folders (checksummed on the actual codebase before writing this driver —
# see the note above). parameters.csv is deliberately excluded here: it is
# (re)built ONCE by the step-0 regression below and shared by all three runs.
const _SHARED_INPUT_FILES = ["time_data.csv", "travel_time.csv", "work_flexible.csv",
                              "ev_data.csv", "mcs_data.csv", "place.csv", "live_powers.csv"]

# =============================================================================
# NAMESPACED APP WRAPPERS — see the big header comment above for why each is
# its own module and why only A1ShrinkingApp includes 1_Common.jl. A0App and
# A2ShrinkingApp both ALIAS A1ShrinkingApp's Common instead of including their
# own copy: `ActivityPowerPool` is a nominal struct type declared inside
# `module Common`, so two separately-include()-d copies of 1_Common.jl would
# produce two DIFFERENT types, and a pool built by one app's
# draw_activity_power_pool would be rejected by another app's
# run_mpc/run_one_shot. Aliasing is what makes it possible to build exactly
# ONE ActivityPowerPool object and hand the literal SAME object to all three
# solves. (DataLoader's `d` is a plain NamedTuple, not a nominal struct, so it
# has no such restriction -- each app is free to keep its own copy of that.)
# =============================================================================
module A1ShrinkingApp
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Approach 1", "code"))
    include(joinpath(_DIR, "1_Common.jl"))
    include(joinpath(_DIR, "0_Regression.jl"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

module A0App
    # Approach 0 is its own independent codebase now (../../Approach 0/code) --
    # no longer borrowed from Approach 1 or Approach 2's run_one_shot, so there
    # is no more approach0_source switch to get wrong. Common is ALIASED (see
    # this section's header note), not included from Approach 0's own copy.
    import ..A1ShrinkingApp
    const Common = A1ShrinkingApp.Common
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Approach 0", "code"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_OneShot.jl"))
end

module A2ShrinkingApp
    import ..A1ShrinkingApp
    const Common = A1ShrinkingApp.Common
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Approach 2", "code"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "2b_ScenarioSampler.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

include(joinpath(_CODE_DIR, "8_ComparisonOutput.jl"))
using .ComparisonOutput: Approach, write_comparison_outputs

# =============================================================================
# DETAILED OUTPUT WRITER (opt-in via `detailed_output = true`)
# -----------------------------------------------------------------------------
# Writes the 4 CSVs (<prefix>_plan_full, <prefix>_realized_tuple,
# <prefix>_MCS_plan_full, <prefix>_MCS_realized_tuple) for ONE approach into a
# COMPLETELY SEPARATE tree, Detailed_Output/ -- a sibling to `out_dir`, never
# nested inside it or inside any comparison subfolder. Called once per
# approach (A0, A1S, A2S) right after all three solves finish.
# =============================================================================
function _write_detailed_output(res, approach_prefix::AbstractString, out_dir::AbstractString)
    detailed_root = joinpath(dirname(out_dir), "Detailed_" * basename(out_dir))
    mkpath(detailed_root)
    if res.detailed_plan_df !== nothing
        CSV.write(joinpath(detailed_root, "$(approach_prefix)_plan_full.csv"), res.detailed_plan_df)
    end
    if res.detailed_realized_df !== nothing
        CSV.write(joinpath(detailed_root, "$(approach_prefix)_realized_tuple.csv"), res.detailed_realized_df)
    end
    if res.detailed_mcs_plan_df !== nothing
        CSV.write(joinpath(detailed_root, "$(approach_prefix)_MCS_plan_full.csv"), res.detailed_mcs_plan_df)
    end
    if res.detailed_mcs_realized_df !== nothing
        CSV.write(joinpath(detailed_root, "$(approach_prefix)_MCS_realized_tuple.csv"), res.detailed_mcs_realized_df)
    end
    return detailed_root
end

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
                                  csv_source_dir::AbstractString = _A1S_INPUT,
                                  # TEMP-RUN PATCH: regression turned off by
                                  # default -- this run reuses the pre-staged, hand-edited
                                  # parameters.csv shipped in Input/ instead of refitting it.
                                  run_regression::Bool = false,
                                  regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                                  regression_samples::Int = 2000,
                                  regression_chains::Int = 4)
    mkpath(input_dir)
    for f in _SHARED_INPUT_FILES
        dst = joinpath(input_dir, f)
        # TEMP-RUN PATCH: if the file is already staged in
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
        ok = A1ShrinkingApp.Regression.run_regression(regression_data_dir, params_path;
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
    ("A0_A1S",     ["A0", "A1S"]),
    ("A0_A2S",     ["A0", "A2S"]),
    ("A0_A1S_A2S", ["A0", "A1S", "A2S"]),
    ("A1S_A2S",    ["A1S", "A2S"]),
]

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
function run_comparison(; input_dir::AbstractString = _COMPARISON_INPUT,
                          out_dir::AbstractString = _COMPARISON_OUT,
                          csv_source_dir::AbstractString = _A1S_INPUT,
                          # TEMP-RUN PATCH: regression turned off by
                          # default -- reuses the pre-staged, hand-edited parameters.csv.
                          run_regression::Bool = false,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          # PLANT MODE for Approach 0's open-loop replay:
                          #   :sampled  the one-shot plan drifts under the shared pool
                          #   :mean     realized power pinned to mu, so realized == planned
                          #             and Approach 0's KPIs are the MILP's own optimum
                          # Both closed loops (A1S/A2S) are ALWAYS :sampled, so a :mean
                          # Approach 0 makes the reported gaps mix plan drift with the
                          # value of re-planning rather than isolating the latter.
                          approach0_plant::Symbol = :sampled,
                          # time_limit_sec::Float64 = Inf,
                          time_limit_sec::Float64 = 600.0, #<------------------------------------------ COMMENT THIS LINE FOR THE FULL RUN (Inf) ------------------------------------>
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          mcmc_samples::Int = 500,
                          # SHRINKING-HORIZON knobs, passed straight through to A1S/A2S's
                          # run_mpc: `shrinking` toggles whether the window actually shrinks
                          # through the day (true) or stays a fixed-length lookahead (false);
                          # `H` is the fixed-length lookahead size when shrinking = false, and
                          # is otherwise unused. See 4_MPCLoop.jl for the full mechanics.
                          shrinking::Bool = true,
                          H::Int = 16,
                          n_day_run::Int = 1,
                          pool_n_samples::Union{Nothing, Int} = nothing,
                          # APPROACH 2: scenarios sampled from the posterior at every
                          # re-solve, for the stochastic closed loop (A2S).
                          n_scenarios::Int = A2ShrinkingApp.ScenarioSampler.DEFAULT_N_SCENARIOS,
                          # Which comparisons to write -- defaults to the full set of
                          # 4 (3-way + all 3 requested subsets). Pass a subset of
                          # _ALL_COMBOS to skip some, e.g. during a quick test run.
                          combos = _ALL_COMBOS,
                          # PLANT MODE: :normal (default) is the unbiased Bayesian draw;
                          # :high/:low/:near_mean/:spread_wide bias it (1_Common.jl's "DRAW
                          # MODE" doc); :live_data draws instead from real recorded values in
                          # Input/live_powers.csv (per-CEV independent, without-replacement
                          # draws). Same `mode` name as run_comparison_sweep's `modes` tuple
                          # in this same file's Sweep sibling.
                          mode::Symbol = :normal,
                          # DETAILED OUTPUT (opt-in): when true, every 15-min
                          # plan/realized decision is written to CSV for all
                          # three approaches (CEV + MCS), into a
                          # Detailed_Output/ tree parallel to `out_dir` -- see
                          # _write_detailed_output below. Off by default since
                          # it costs extra memory/time per solve.
                          detailed_output::Bool = false,
                          seed::Int = 1)
    approach0_plant in (:sampled, :mean) ||
        error("run_comparison: approach0_plant must be :sampled or :mean")

    build_comparison_input(; input_dir, csv_source_dir, run_regression,
                            regression_data_dir, regression_samples, regression_chains)

    return _with_console_log(out_dir) do
        _run_t0 = time()
        println("="^78)
        println("3-WAY COMPARISON — Approach 0 vs A1-Shrinking vs A2-Shrinking")
        println("A1 Shrinking code: $(_A1S_CODE)")
        println("A2 Shrinking code: $(_A2S_CODE)")
        println("Input             : $(abspath(input_dir))")
        println("Output            : $(abspath(out_dir))")
        println("Comparisons       : ", join(first.(combos), ", "))
        _status("Run started")
        println("="^78)

        # ---- load data separately with each app's OWN DataLoader, from the
        # SAME Comparison/Input folder ----
        dA0, dA1S, dA2S = _timed_status("loading input data (A0 + A1S + A2S DataLoaders)") do
            (A0App.DataLoader.load_data(:input;    input_dir = input_dir),
             A1ShrinkingApp.DataLoader.load_data(:input;  input_dir = input_dir),
             A2ShrinkingApp.DataLoader.load_data(:input;  input_dir = input_dir))
        end

        # ---- ONE shared ActivityPowerPool, built ONCE, passed as the literal
        # SAME object into all three runs below (see the module header note
        # for why this is possible across two distinct codebases).
        #
        # SIZING. next_power! ERRORS (it does not wrap) once a cursor walks past
        # the end of the pre-drawn samples, so the pool must cover the LONGEST
        # run: n_day_run days' worth of intervals, plus a small safety margin.
        nK_day = length(collect(dA1S.K))
        n_samples = pool_n_samples === nothing ?
            nK_day * n_day_run + 5 : pool_n_samples
        pool = _timed_status("building shared power pool ($(n_samples) samples/entity-activity, :$(mode))") do
            if mode == :live_data
                A1ShrinkingApp.Common.draw_activity_power_pool_live(
                    dA1S.E, A1ShrinkingApp.DataLoader.load_live_powers(input_dir);
                    rng = MersenneTwister(seed))
            else
                # mode = :normal -> unbiased draws (unchanged); see 1_Common.jl's
                # "DRAW MODE" doc for the 4 sensitivity-sweep modes -- swept by
                # 7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl, which reuses this
                # same driver rather than duplicating it.
                A1ShrinkingApp.Common.draw_activity_power_pool(dA1S.E, dA1S.prior_mu, dA1S.prior_sigma;
                                                               n_samples = n_samples,
                                                               rng = MersenneTwister(seed),
                                                               mode = mode)
            end
        end
        println("\nShared power pool: n_samples=$(n_samples) per (entity, activity) ",
                "($(nK_day) intervals/day x $(n_day_run) day(s), + 5); mu=",
                round.(pool.mu, digits = 2), " kW; sd=", round.(pool.sd, digits = 2), " kW")

        # ---- APPROACH 0 (one-shot, no replanning) ----
        println("\n--- Approach 0 (one-shot, plant = :$(approach0_plant)) ---")
        res0 = _timed_status("Approach 0 solve") do
            A0App.OneShot.run_one_shot(dA0, pool; plant = approach0_plant,
                                       time_limit_sec, multi_activity,
                                       require_site_visit, single_visit_per_site,
                                       n_day_run, seed, detailed_output)
        end

        # ---- APPROACH 1b: Shrinking Horizon closed-loop MPC ----
        println("\n--- Approach 1 - Shrinking (shrinking=$(shrinking), H=$(H), n_day_run = $(n_day_run)) ---")
        resA1S = _timed_status("Approach 1 - Shrinking solve (n_day_run = $(n_day_run))") do
            A1ShrinkingApp.MPCLoop.run_mpc(dA1S, pool; shrinking, H, time_limit_sec, multi_activity,
                                          require_site_visit, single_visit_per_site,
                                          mcmc_samples, plant = :sampled, n_day_run, seed, detailed_output)
        end

        # ---- APPROACH 2b: Shrinking Horizon, stochastic scenario-based closed-loop MPC ----
        println("\n--- Approach 2 - Shrinking (stochastic, $(n_scenarios) scenarios, n_day_run = $(n_day_run)) ---")
        resA2S = _timed_status("Approach 2 - Shrinking solve ($(n_scenarios) scenarios, n_day_run = $(n_day_run))") do
            A2ShrinkingApp.MPCLoop.run_mpc(dA2S, pool; shrinking, H, time_limit_sec, multi_activity,
                                          require_site_visit, single_visit_per_site,
                                          mcmc_samples, plant = :sampled, n_scenarios, n_day_run, seed, detailed_output)
        end

        # ---- the three Approach identities, keyed exactly as _ALL_COMBOS expects ----
        all_apps = Dict(
            "A0"  => Approach("A0",  "Approach 0 (one-shot, :$(approach0_plant))", res0,   :gray40),
            "A1S" => Approach("A1S", "Approach 1 - Shrinking",                     resA1S, :firebrick),
            "A2S" => Approach("A2S", "Approach 2 - Shrinking (stochastic)",        resA2S, :darkorange),
        )

        # ---- DETAILED OUTPUT (opt-in): separate tree, all three approaches ----
        if detailed_output
            d0 = _write_detailed_output(res0,   "A0",  out_dir)
            _write_detailed_output(resA1S, "A1S", out_dir)
            _write_detailed_output(resA2S, "A2S", out_dir)
            println("\nDetailed output -> $(d0)  (A0_*.csv, A1S_*.csv, A2S_*.csv, incl. *_MCS_*.csv)")
        end

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
        @printf("%-28s %14s %14s %14s\n", "Metric", "A0", "A1-Shrink", "A2-Shrink")
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Grid energy (kWh)", res0.total_energy, resA1S.total_energy, resA2S.total_energy)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Energy cost (USD)", res0.total_cost, resA1S.total_cost, resA2S.total_cost)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "CO2 (kg)", res0.total_co2, resA1S.total_co2, resA2S.total_co2)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "NCD peak (kW)", res0.nc_peak, resA1S.nc_peak, resA2S.nc_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "OPD peak (kW)", res0.op_peak, resA1S.op_peak, resA2S.op_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Missed work (h)", res0.missed, resA1S.missed, resA2S.missed)
        println("="^78)
        println("\nResults written to: $(abspath(out_dir))")
        for (folder, keys) in combos
            println("  $(folder)/  (", join(keys, " vs "), ")")
        end
        println("  run_log.txt  (this console log, shared across all $(length(combos)) comparisons)")
        _status("Run finished — total elapsed $(round(time() - _run_t0, digits=1))s")

        return (; res0, resA1S, resA2S, all_apps, dA0, dA1S, dA2S, pool)
    end
end

# Auto-run unless a harness defines COMPARISON_NO_AUTORUN = true first.
if !(@isdefined(COMPARISON_NO_AUTORUN) && COMPARISON_NO_AUTORUN)
    run_comparison(n_day_run = 5)
end
