# #############################################################################
# Comparison_main.jl  —  TOP-LEVEL 3-WAY COMPARISON DRIVER   [APPROACH 2 TREE]
# -----------------------------------------------------------------------------
# Runs, side by side, from the SAME input data and the SAME shared power pool:
#   * Approach 0   — one-shot plan, executed open-loop, no replanning, DETERMINISTIC
#                    (taken from the Shrinking Horizon codebase's run_one_shot
#                    by default — see `approach0_source` below)
#   * Approach 2a  — Shrinking Horizon, STOCHASTIC scenario-based closed-loop MPC
#   * Approach 2b  — Receding Horizon, STOCHASTIC scenario-based closed-loop MPC
#                    (run with n_days = 1 so its single reported day is directly
#                    comparable to the Shrinking Horizon's single-day scope)
#
# NOTE: this is the Approach 2 (stochastic) tree, so `run_mpc` in both codebases
# below is the scenario-based controller (see each codebase's 4_MPCLoop.jl /
# 2b_ScenarioSampler.jl and docs/Understanding_Stochastic_MPC.md). `run_one_shot`
# (Approach 0) stays certainty-equivalent on purpose, as the fixed baseline.
#
# THE CODE IS CALLED IN PLACE — NOTHING IS COPIED.
# This driver `include()`s the two source codebases directly from THIS Approach 2
# tree's own Shrinking_Horizon/code and Receding_Horizon/code (resolved relative
# to this file's own location, so the whole "Approach 2" folder is self-contained
# and portable — no machine-specific path to edit). Edit either codebase there
# and the next run picks it up automatically — there is no copy of either
# codebase living under Comparison/.
#
# HOW THE TWO CODEBASES ARE KEPT SEPARATE (WITHOUT DUPLICATING Common.jl)
# Both codebases define modules with the SAME names (Common, DataLoader,
# MCSModel, MPCLoop, Output), so each is `include`-d inside its own wrapper
# module (RecedingApp / ShrinkingApp) below to avoid one silently overwriting
# the other. The ONE exception is Common.jl: ShrinkingApp does NOT include its
# own copy — it re-uses RecedingApp's Common module directly (verified
# byte-identical apart from 3 extra multi-day-only helper functions Shrinking
# never calls; see the note above `module ShrinkingApp`). This is what makes
# it possible to build exactly ONE `ActivityPowerPool` object and hand the
# SAME object to both codebases' MPC loops — they are otherwise two distinct
# Julia modules, so without this alias the pool's type from one app would be
# rejected by the other app's `run_mpc`/`run_one_shot` (a different nominal
# type), forcing two separately-built pools and quietly voiding the shared-
# plant comparison. With the alias, all three runs draw from the literal same
# pre-generated sample sequence.
#
# WHAT GETS WRITTEN — everything lands directly in `out_dir` (no
# per-approach subfolders, no native per-codebase report set):
#   Output/01_total_grid_power_profile.png/.csv  … 09_mcs_<m>_power_profile.*
#   Output/08_kpi_metrics_summary.png
#   Output/08_cost_kpi_metrics.csv
#   Output/approach0_vs_shrinking_vs_receding.html
#   Output/run_log.txt   everything printed during the whole run
# #############################################################################

using Printf
using Random

const _CODE_DIR = @__DIR__

# -----------------------------------------------------------------------------
# DEFAULT PATHS
# -----------------------------------------------------------------------------
# _CODE_DIR = .../Approach 2/Comparison/Code, so two levels up is the Approach 2 root.
const _APPROACH2_ROOT    = normpath(joinpath(_CODE_DIR, "..", ".."))
const _RECEDING_CODE_DIR  = joinpath(_APPROACH2_ROOT, "Receding_Horizon",  "code")
const _SHRINKING_CODE_DIR = joinpath(_APPROACH2_ROOT, "Shrinking_Horizon", "code")
const _RECEDING_INPUT    = joinpath(_APPROACH2_ROOT, "Receding_Horizon",  "data", "input_data")
const _SHRINKING_INPUT   = joinpath(_APPROACH2_ROOT, "Shrinking_Horizon", "data", "input_data")
const _COMPARISON_INPUT  = joinpath(_APPROACH2_ROOT, "Comparison", "Input")
const _COMPARISON_OUT    = joinpath(_APPROACH2_ROOT, "Comparison", "Output")
const _DEFAULT_REGRESSION_DATA_DIR = raw"C:\Users\shubh\Desktop\Bayesian Regression"

# The 7 input files that end up in Comparison/Input. All but parameters.csv
# are confirmed identical between the two source input_data folders; those 6
# are copied verbatim from `csv_source_dir`. parameters.csv is (re)built by
# the step-0 regression (see build_comparison_input below).
const _SHARED_INPUT_FILES = ["time_data.csv", "travel_time.csv", "work_flexible.csv",
                              "ev_data.csv", "mcs_data.csv", "place.csv", "live_powers.csv"]

# =============================================================================
# NAMESPACED APP WRAPPERS — include the two codebases IN PLACE, from their own
# folders, each inside its own module so their identically-named submodules
# (Common, DataLoader, MCSModel, MPCLoop, Output) don't collide.
# =============================================================================
module RecedingApp
    # Resolved relative to THIS file (Comparison/Code/7_Comparison_main.jl), not a
    # hardcoded machine path, so the Approach 2 folder works wherever it's placed.
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Receding_Horizon", "code"))
    include(joinpath(_DIR, "1_Common.jl"))
    include(joinpath(_DIR, "0_Regression.jl"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "2b_ScenarioSampler.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

# ShrinkingApp deliberately does NOT include its own 1_Common.jl. Instead it
# reuses RecedingApp.Common (aliased below) so that BOTH codebases' MPCLoop
# and MCSModel modules resolve `..Common` to the exact same Julia module —
# meaning `ActivityPowerPool`, `draw_activity_power_pool`, `new_cursor` and
# `next_power!` are the SAME type/functions for both apps, and one pool
# object built from either can be passed to both apps' run_mpc/run_one_shot.
# (Verified: Shrinking_Horizon/code/1_Common.jl is byte-identical to
# Receding_Horizon/code/1_Common.jl except for 3 extra multi-day-only helper
# functions -- clock_day_label, build_time_labels_days, multiday_xticks --
# that only Receding's own MPCLoop/Output use; Shrinking never calls them.)
module ShrinkingApp
    import ..RecedingApp
    const Common = RecedingApp.Common
    # Resolved relative to THIS file, not a hardcoded machine path (see RecedingApp).
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Shrinking_Horizon", "code"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "2b_ScenarioSampler.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))
end

include(joinpath(_CODE_DIR, "8_ComparisonOutput.jl"))
using .ComparisonOutput: Approach, write_comparison_outputs

# -----------------------------------------------------------------------------
# CONSOLE LOG CAPTURE — mirrors both sibling drivers' own _with_console_log,
# so `Output/run_log.txt` contains everything printed during the whole run
# (regression fit, both MPC loops solving live, all writers).
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
# STEP: build Comparison/Input from the two source input folders (see the
# _SHARED_INPUT_FILES note above for exactly what's copied vs regenerated).
# =============================================================================
function build_comparison_input(; input_dir::AbstractString = _COMPARISON_INPUT,
                                  csv_source_dir::AbstractString = _RECEDING_INPUT,
                                  run_regression::Bool = true,
                                  regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                                  regression_samples::Int = 2000,
                                  regression_chains::Int = 4)
    mkpath(input_dir)
    for f in _SHARED_INPUT_FILES
        src = joinpath(csv_source_dir, f)
        isfile(src) || error("build_comparison_input: expected shared input file missing -> $src")
        cp(src, joinpath(input_dir, f); force = true)
    end
    params_path = joinpath(input_dir, "parameters.csv")
    ok = false
    if run_regression
        ok = RecedingApp.Regression.run_regression(regression_data_dir, params_path;
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
# MAIN ENTRY POINT
# =============================================================================
function run_comparison(; input_dir::AbstractString = _COMPARISON_INPUT,
                          out_dir::AbstractString = _COMPARISON_OUT,
                          csv_source_dir::AbstractString = _RECEDING_INPUT,
                          run_regression::Bool = true,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          approach0_source::Symbol = :shrinking,   # :shrinking or :receding
                          # PLANT MODE for Approach 0's open-loop replay:
                          #   :sampled  the one-shot plan drifts under the shared pool
                          #   :mean     realized power pinned to mu, so realized == planned
                          #             and Approach 0's KPIs are the MILP's own optimum
                          # Both closed loops (1a and 1b) are ALWAYS :sampled, so a :mean
                          # Approach 0 makes the reported gaps mix plan drift with the
                          # value of re-planning rather than isolating the latter.
                          approach0_plant::Symbol = :sampled,
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          mcmc_samples::Int = 500,
                          H::Int = 16,
                          n_days_receding::Int = 1,
                          pool_n_samples::Union{Nothing, Int} = nothing,
                          mode::Symbol = :normal,
                          # APPROACH 2: scenarios sampled from the posterior at every
                          # re-solve, for BOTH the Shrinking and Receding stochastic
                          # closed loops below.
                          n_scenarios::Int = RecedingApp.ScenarioSampler.DEFAULT_N_SCENARIOS,
                          seed::Int = 1)
    approach0_source in (:shrinking, :receding) ||
        error("run_comparison: approach0_source must be :shrinking or :receding")
    approach0_plant in (:sampled, :mean) ||
        error("run_comparison: approach0_plant must be :sampled or :mean")

    build_comparison_input(; input_dir, csv_source_dir, run_regression,
                            regression_data_dir, regression_samples, regression_chains)

    return _with_console_log(out_dir) do
        println("="^78)
        println("3-WAY COMPARISON — Approach 0 (deterministic) vs Shrinking (stochastic) vs Receding (stochastic)")
        println("Scenarios/re-solve: $(n_scenarios)")
        println("Receding code : $(_RECEDING_CODE_DIR)")
        println("Shrinking code: $(_SHRINKING_CODE_DIR)")
        println("Input         : $(abspath(input_dir))")
        println("Output        : $(abspath(out_dir))")
        println("="^78)

        # ---- load data separately with each app's OWN DataLoader, from the
        # SAME Comparison/Input folder ----
        dS = ShrinkingApp.DataLoader.load_data(:input; input_dir = input_dir)
        dR = RecedingApp.DataLoader.load_data(:input;  input_dir = input_dir)

        # ---- ONE shared ActivityPowerPool, built ONCE, passed as the literal
        # SAME object into all three runs below (see the ShrinkingApp module
        # note above for why this is possible across the two codebases).
        #
        # SIZING. next_power! ERRORS (it does not wrap) once a cursor walks past
        # the end of the pre-drawn samples, so the pool must cover the LONGEST
        # run, not the reported horizon. The Receding run simulates n_days + 1
        # days -- it always adds a BUFFER day, which is simulated in full and
        # only then dropped from the reported outputs -- so its cursor keeps
        # drawing through that extra day. Sizing on nK_day alone would leave
        # roughly half the margin the Receding run can need, and the failure
        # would land mid-run after the MILPs had already been solving for a
        # while. Size on the buffer-inclusive length instead; unconsumed
        # samples cost nothing but a little memory. ----
        nK_day = length(collect(dS.K))
        n_samples = pool_n_samples === nothing ?
            nK_day * (n_days_receding + 1) + 5 : pool_n_samples
        pool = if mode == :live_data
            RecedingApp.Common.draw_activity_power_pool_live(
                dS.E, RecedingApp.DataLoader.load_live_powers(input_dir);
                rng = MersenneTwister(seed))
        else
            # mode = :normal -> unbiased draws (unchanged); see 1_Common.jl's
            # "DRAW MODE" doc for the 4 sensitivity-sweep modes.
            RecedingApp.Common.draw_activity_power_pool(dS.E, dS.prior_mu, dS.prior_sigma;
                                                        n_samples = n_samples,
                                                        rng = MersenneTwister(seed),
                                                        mode = mode)
        end
        println("\nShared power pool: n_samples=$(n_samples) per (entity, activity) ",
                "($(nK_day) intervals/day x $(n_days_receding + 1) days incl. the Receding ",
                "buffer day, + 5); mu=", round.(pool.mu, digits = 2),
                " kW; sd=", round.(pool.sd, digits = 2), " kW")

        # ---- APPROACH 0 (one-shot, no replanning) ----
        println("\n--- Approach 0 (one-shot, plant = :$(approach0_plant)) ---")
        res0 = if approach0_source == :shrinking
            ShrinkingApp.MPCLoop.run_one_shot(dS, pool; plant = approach0_plant,
                                              time_limit_sec, multi_activity,
                                              require_site_visit, single_visit_per_site, seed)
        else
            RecedingApp.MPCLoop.run_one_shot(dR, pool; plant = approach0_plant,
                                             time_limit_sec, multi_activity,
                                             require_site_visit, single_visit_per_site,
                                             n_days = n_days_receding, seed)
        end

        # ---- APPROACH 2a: Shrinking Horizon, stochastic scenario-based closed-loop MPC ----
        println("\n--- Approach 2a (Shrinking Horizon, stochastic, $(n_scenarios) scenarios) ---")
        resS = ShrinkingApp.MPCLoop.run_mpc(dS, pool; shrinking = true, H, time_limit_sec,
                                            multi_activity, require_site_visit, single_visit_per_site,
                                            mcmc_samples, n_scenarios, seed)

        # ---- APPROACH 2b: Receding Horizon, stochastic scenario-based closed-loop MPC (n_days = 1) ----
        println("\n--- Approach 2b (Receding Horizon, stochastic, $(n_scenarios) scenarios) ---")
        resR = RecedingApp.MPCLoop.run_mpc(dR, pool; time_limit_sec, multi_activity,
                                           require_site_visit, single_visit_per_site,
                                           mcmc_samples, n_scenarios, n_days = n_days_receding, seed)

        # ---- the merged 3-way comparison — written straight into out_dir ----
        println("\n--- Writing 3-way comparison outputs ---")
        apps = [
            Approach("approach0",  "Approach 0 (one-shot, :$(approach0_plant))", res0, :gray40),
            Approach("shrinking",  "Approach 2 — Shrinking (stochastic)", resS, :steelblue),
            Approach("receding",   "Approach 2 — Receding (stochastic)",  resR, :firebrick),
        ]
        Base.invokelatest(write_comparison_outputs, apps, out_dir)

        # ---- headline KPI printout ----
        println("\n" * "="^78)
        println("KPI SUMMARY")
        @printf("%-28s %14s %14s %14s\n", "Metric", "Approach0", "Shrinking", "Receding")
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Grid energy (kWh)", res0.total_energy, resS.total_energy, resR.total_energy)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Energy cost (USD)", res0.total_cost, resS.total_cost, resR.total_cost)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "CO2 (kg)", res0.total_co2, resS.total_co2, resR.total_co2)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "NCD peak (kW)", res0.nc_peak, resS.nc_peak, resR.nc_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "OPD peak (kW)", res0.op_peak, resS.op_peak, resR.op_peak)
        @printf("%-28s %14.2f %14.2f %14.2f\n", "Missed work (h)", res0.missed, resS.missed, resR.missed)
        println("="^78)
        println("\nResults written to: $(abspath(out_dir))")
        println("  01_total_grid_power_profile.png/.csv … 09_mcs_<m>_power_profile.png/.csv")
        println("  08_kpi_metrics_summary.png")
        println("  08_cost_kpi_metrics.csv")
        println("  approach0_vs_shrinking_vs_receding.html")
        println("  run_log.txt  (this console log)")

        return (; res0, resS, resR, dS, dR, pool)
    end
end

# Auto-run unless a harness defines COMPARISON_NO_AUTORUN = true first.
if !(@isdefined(COMPARISON_NO_AUTORUN) && COMPARISON_NO_AUTORUN)
    run_comparison()
end
