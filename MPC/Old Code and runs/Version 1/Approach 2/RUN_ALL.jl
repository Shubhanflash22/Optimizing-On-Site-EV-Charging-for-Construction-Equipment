# #############################################################################
#  RUN_ALL.jl  —  ONE-CLICK MASTER RUNNER   [APPROACH 2: STOCHASTIC MPC]
#  Place anywhere; paths below resolve relative to this file's own location
#  (see ROOT), so the whole "Approach 2" folder is portable as a unit.
#  Run with:  julia RUN_ALL.jl
#             (or just include() it from the Julia REPL)
# -----------------------------------------------------------------------------
#  Runs, in this order:
#     1. Shrinking Horizon  — :input        (stochastic scenario-based MPC)
#     2. Shrinking Horizon  — :synthetic    (stochastic scenario-based MPC)
#     3. Shrinking Horizon  — SOE sweep
#     4. Receding  Horizon  — :input        (stochastic scenario-based MPC)
#     5. Receding  Horizon  — :synthetic    (stochastic scenario-based MPC)
#     6. Receding  Horizon  — SOE sweep
#     7. Comparison         — 3-way (Approach 0 deterministic vs Shrinking vs
#                              Receding, both stochastic)
#
#  Approach 0 (the one-shot baseline) stays certainty-equivalent throughout, on
#  purpose — see docs/Understanding_Stochastic_MPC.md. Only Approach 1's old
#  role is now filled by the scenario-based Approach 2 controller.
#
#  EACH STAGE RUNS IN ITS OWN JULIA SUBPROCESS. This is deliberate and not
#  optional: the two codebases both define modules called Common, DataLoader,
#  MCSModel, MPCLoop and Output. Including them into one session would have the
#  second silently redefine the first's modules, so results would depend on load
#  order. Separate processes give each stage a clean namespace. It also means a
#  stage that crashes cannot take the rest of the batch down with it.
# #############################################################################

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║   ############   EDIT HERE — PLANT MODE SWITCH   ############             ║
# ║                                                                           ║
# ║   This is the ONLY line to change to switch between the two modes.        ║
# ║   It applies to every stage below, including both sweeps and the          ║
# ║   3-way comparison.                                                       ║
# ║                                                                           ║
# ║     :sampled   Approach 0's fixed plan is replayed against the STOCHASTIC ║
# ║                plant — realized power is drawn from the shared random     ║
# ║                pool, so the day drifts away from the plan with no         ║
# ║                feedback to correct it. Approach 2 faces the SAME draws,   ║
# ║                so the reported gap isolates the value of re-planning.     ║
# ║                >>> USE THIS FOR NORMAL / HEADLINE RUNS. <<<               ║
# ║                                                                           ║
# ║     :mean      Approach 0's plan is replayed against the DETERMINISTIC    ║
# ║                plant — realized power is pinned to the same mean mu the   ║
# ║                MILP planned on, and each interval realizes its single     ║
# ║                planned activity in full. Realized == planned EXACTLY, so  ║
# ║                Approach 0's KPIs ARE the MILP's own optimum. Approach 2   ║
# ║                is still stochastic, so the reported gap then MIXES plan   ║
# ║                drift with the value of hedging + re-planning.             ║
# ║                >>> USE THIS TO GET THE CLEAN OPTIMUM REFERENCE. <<<       ║
# ║                                                                           ║
# ║   To separate the two effects, run the batch BOTH ways and difference     ║
# ║   the two Approach 0 numbers: that difference is the cost of plan drift   ║
# ║   alone. A :mean run consumes no samples from the shared pool, so it can  ║
# ║   never perturb a :sampled run.                                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const PLANT_MODE = :sampled          # <<<<<<  CHANGE THIS LINE: :sampled  or  :mean


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║   ############   EDIT HERE — WHICH STAGES TO RUN   ############           ║
# ║   Comment out (prepend #) any line to skip that stage. The sweeps are     ║
# ║   by far the slowest (10 full closed-loop runs each) — skip them for a    ║
# ║   quick pass.                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# const STAGES = [
#     :shrinking_input,
#     :shrinking_synthetic,
#     :shrinking_sweep,
#     :receding_input,
#     :receding_synthetic,
#     :receding_sweep,
#     :comparison,
# ]
const STAGES = [
    :shrinking_input,
    :shrinking_sweep,
    :receding_input,
    :receding_sweep,
    :comparison,
]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║   ############   EDIT HERE — SHARED RUN SETTINGS   ############           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const SEED            = 1        # RNG seed for the stochastic plant
const TIME_LIMIT_SEC  = Inf      # HiGHS seconds per window solve; Inf = solve to the MIP gap
const MCMC_SAMPLES    = 500      # NUTS draws (only used by the dormant online-refit path)
const H_SHRINKING     = 16       # fixed-H lookahead length; ONLY used if SHRINKING_MODE=false
const SHRINKING_MODE  = true     # true = shrinking window; false = fixed H (EXPERIMENTAL,
                                 #   terminal rules drop out — see the Shrinking docs §8.7)
const N_DAYS_RECEDING = 1        # reported days the Receding runs KEEP. A BUFFER day is
                                 #   ALWAYS simulated on top and then dropped, so the loop
                                 #   actually runs N_DAYS_RECEDING + 1 days. Set 1 for
                                 #   "1 reported day + 1 buffer" (and for a like-for-like
                                 #   comparison against the single-day Shrinking runs);
                                 #   set 2 for two reported days + buffer.
const RUN_REGRESSION  = true     # re-fit parameters.csv from the task .xlsx files (step 0).
                                 #   Set false to reuse whatever parameters.csv already holds
                                 #   — much faster, and required if the .xlsx folder is absent.
                                 #   NOTE: :synthetic never runs step 0 regardless.
const APPROACH0_SOURCE = :shrinking   # comparison only: whose one-shot solver is "Approach 0"
                                      #   (:shrinking or :receding)
const N_SCENARIOS      = 5           # APPROACH 2: scenarios sampled from the posterior at
                                      #   every re-solve (see 2b_ScenarioSampler.jl). Applies
                                      #   to every stage below, including both sweeps and the
                                      #   comparison.

# ---- PATHS. Change only if you move folders. ---------------------------------
# Resolved relative to THIS file, not a hardcoded machine path, so the folder
# is portable. @__DIR__ is Approach 2's own root since RUN_ALL.jl lives there.
const ROOT            = @__DIR__
const SHRINKING_CODE  = joinpath(ROOT, "Shrinking_Horizon", "code")
const RECEDING_CODE   = joinpath(ROOT, "Receding_Horizon",  "code")
const COMPARISON_CODE = joinpath(ROOT, "Comparison", "Code")
const REGRESSION_DATA = raw"C:\Users\shubh\Desktop\Bayesian Regression"
const BATCH_LOG_DIR   = joinpath(ROOT, "batch_logs")
# Each stage is executed by writing a tiny throwaway .jl into BATCH_LOG_DIR and
# running it in a subprocess. Keep it only when you need to inspect exactly what
# a stage was asked to do; the default is to delete it once the stage finishes so
# batch_logs holds nothing but the .log files.
const KEEP_STAGE_SCRIPTS = false


# #############################################################################
#  Below this line is machinery — no need to edit.
# #############################################################################

using Printf, Dates

const JULIA_EXE = joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia")

_q(p) = replace(p, "\\" => "\\\\")     # escape a Windows path for embedding in Julia source

# NOTE ON :synthetic. DataLoader.load_data(:synthetic) returns build_default_data()
# -- the scenario HARDCODED in 2_DataLoader.jl. It reads no files. The CSVs under
# data\synthetic_data\ are a human-readable MIRROR of those same hardcoded values
# (handy for inspecting the scenario without opening the source), not an input:
# nothing in either codebase reads that folder. Only :input takes an input_dir.

# Each stage is (id, human label, the Julia source that stage's subprocess runs).
function _stage_source(id::Symbol)
    plant = repr(PLANT_MODE)
    common = """
        time_limit_sec = $(TIME_LIMIT_SEC),
        mcmc_samples   = $(MCMC_SAMPLES),
        approach0_plant = $(plant),
        n_scenarios = $(N_SCENARIOS),
        seed = $(SEED),
    """
    if id === :shrinking_input
        return """
        SCENARIO1_NO_AUTORUN = true
        include(raw"$(_q(joinpath(SHRINKING_CODE, "6_Shrinking_Horizon_main.jl")))")
        run_scenario_1(; mode = :input, shrinking = $(SHRINKING_MODE), H = $(H_SHRINKING),
                         run_regression = $(RUN_REGRESSION),
                         regression_data_dir = raw"$(_q(REGRESSION_DATA))",
                         $common)
        """
    elseif id === :shrinking_synthetic
        return """
        SCENARIO1_NO_AUTORUN = true
        include(raw"$(_q(joinpath(SHRINKING_CODE, "6_Shrinking_Horizon_main.jl")))")
        run_scenario_1(; mode = :synthetic, shrinking = $(SHRINKING_MODE), H = $(H_SHRINKING),
                         run_regression = false, $common)
        """
    elseif id === :shrinking_sweep
        # The sweep sets SCENARIO1_NO_AUTORUN itself and includes its own main.
        # MASTER_A0_PLANT / MASTER_N_SCENARIOS must be defined BEFORE the include
        # so the sweep's own constants pick them up instead of their defaults.
        return """
        MASTER_A0_PLANT = $(plant)
        MASTER_N_SCENARIOS = $(N_SCENARIOS)
        include(raw"$(_q(joinpath(SHRINKING_CODE, "run_soe_sweep.jl")))")
        """
    elseif id === :receding_input
        return """
        SCENARIO1_NO_AUTORUN = true
        include(raw"$(_q(joinpath(RECEDING_CODE, "6_Receding_Horizon_main.jl")))")
        run_scenario_1(; mode = :input, n_days = $(N_DAYS_RECEDING),
                         run_regression = $(RUN_REGRESSION),
                         regression_data_dir = raw"$(_q(REGRESSION_DATA))",
                         $common)
        """
    elseif id === :receding_synthetic
        return """
        SCENARIO1_NO_AUTORUN = true
        include(raw"$(_q(joinpath(RECEDING_CODE, "6_Receding_Horizon_main.jl")))")
        run_scenario_1(; mode = :synthetic, n_days = $(N_DAYS_RECEDING),
                         run_regression = false, $common)
        """
    elseif id === :receding_sweep
        return """
        MASTER_A0_PLANT = $(plant)
        MASTER_N_SCENARIOS = $(N_SCENARIOS)
        MASTER_N_DAYS_RECEDING = $(N_DAYS_RECEDING)
        include(raw"$(_q(joinpath(RECEDING_CODE, "run_soe_sweep.jl")))")
        """
    elseif id === :comparison
        # The comparison driver always runs :input (it builds Comparison/Input
        # from the shared CSVs) and always keeps 1 Receding day so the Receding
        # run is directly comparable to the single-day Shrinking one.
        return """
        COMPARISON_NO_AUTORUN = true
        include(raw"$(_q(joinpath(COMPARISON_CODE, "7_Comparison_main.jl")))")
        run_comparison(; approach0_source = $(repr(APPROACH0_SOURCE)),
                         approach0_plant  = $(plant),
                         run_regression   = $(RUN_REGRESSION),
                         regression_data_dir = raw"$(_q(REGRESSION_DATA))",
                         time_limit_sec = $(TIME_LIMIT_SEC),
                         mcmc_samples   = $(MCMC_SAMPLES),
                         H = $(H_SHRINKING),
                         n_days_receding = 1,
                         n_scenarios = $(N_SCENARIOS),
                         seed = $(SEED))
        """
    else
        error("RUN_ALL: unknown stage :$id")
    end
end

const _LABELS = Dict(
    :shrinking_input     => "1/7  Shrinking Horizon  —  :input",
    :shrinking_synthetic => "2/7  Shrinking Horizon  —  :synthetic",
    :shrinking_sweep     => "3/7  Shrinking Horizon  —  SOE sweep",
    :receding_input      => "4/7  Receding  Horizon  —  :input",
    :receding_synthetic  => "5/7  Receding  Horizon  —  :synthetic",
    :receding_sweep      => "6/7  Receding  Horizon  —  SOE sweep",
    :comparison          => "7/7  Comparison         —  3-way",
)

# Run ONE stage in its own Julia process. Output is streamed to this console AND
# captured to batch_logs/<stage>.log. A non-zero exit is recorded and the batch
# continues, so one failing stage does not cost you the whole run.
function _run_stage(id::Symbol)
    mkpath(BATCH_LOG_DIR)
    label   = _LABELS[id]
    scratch = joinpath(BATCH_LOG_DIR, "_stage_$(id).jl")
    logfile = joinpath(BATCH_LOG_DIR, "$(id).log")
    write(scratch, _stage_source(id))

    println("\n" * "="^78)
    println(label)
    println("  plant mode : :$(PLANT_MODE)")
    println("  started    : ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("  log        : ", logfile)
    println("="^78)
    flush(stdout)

    t0 = time()
    ok = true
    open(logfile, "w") do lf
        try
            # `pipeline(..., stdout=..., stderr=...)` would swallow the console
            # stream, so tee manually: read the process output and write it to
            # both places as it arrives.
            proc_out = Pipe()
            p = run(pipeline(`$(JULIA_EXE) --color=no $(scratch)`,
                             stdout = proc_out, stderr = proc_out), wait = false)
            close(proc_out.in)
            while !eof(proc_out)
                chunk = readavailable(proc_out)
                write(stdout, chunk); flush(stdout)
                write(lf, chunk);     flush(lf)
            end
            wait(p)
            ok = success(p)
        catch err
            ok = false
            msg = "STAGE FAILED: $(sprint(showerror, err))\n"
            print(msg); write(lf, msg)
        end
    end
    dt = time() - t0
    # Remove the throwaway stage script so batch_logs stays clean (see
    # KEEP_STAGE_SCRIPTS above). Never let a cleanup failure mask the result.
    if !KEEP_STAGE_SCRIPTS
        try
            isfile(scratch) && rm(scratch; force = true)
        catch err
            @warn "RUN_ALL: could not remove stage script" scratch exception = err
        end
    end
    @printf("\n---- %s : %s in %.1f s (%.1f min)\n", label, ok ? "OK" : "FAILED", dt, dt / 60)
    flush(stdout)
    return (; id, ok, seconds = dt)
end

function run_all()
    println("#"^78)
    println("#  MASTER BATCH RUN")
    println("#  root       : ", ROOT)
    println("#  julia      : ", JULIA_EXE)
    println("#  PLANT MODE : :", PLANT_MODE,
            PLANT_MODE === :mean ?
              "   (Approach 0 DETERMINISTIC — its KPIs are the MILP optimum)" :
              "   (Approach 0 stochastic — gap isolates hedging + re-planning value)")
    println("#  scenarios  : ", N_SCENARIOS, " (Approach 2's scenario-based MPC, every stage)")
    println("#  stages     : ", join(string.(STAGES), ", "))
    println("#  started    : ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("#"^78)

    isdir(ROOT) || error("RUN_ALL: ROOT does not exist -> $ROOT")
    results = [_run_stage(id) for id in STAGES]

    println("\n" * "#"^78)
    println("#  BATCH SUMMARY")
    println("#"^78)
    @printf("%-40s %-8s %10s\n", "Stage", "Status", "Minutes")
    for r in results
        @printf("%-40s %-8s %10.1f\n", _LABELS[r.id], r.ok ? "OK" : "FAILED", r.seconds / 60)
    end
    total = sum(r.seconds for r in results; init = 0.0)
    @printf("%-40s %-8s %10.1f\n", "TOTAL", "", total / 60)
    nfail = count(!r.ok for r in results)
    nfail > 0 && println("\n  $(nfail) stage(s) FAILED — see batch_logs/<stage>.log for the full output.")
    println("\n  Per-stage logs : ", BATCH_LOG_DIR)
    println("  Results        : <Shrinking|Receding>_Horizon/output/<input|synthetic>/,")
    println("                   <Shrinking|Receding>_Horizon/output/input_testing/summary.html,")
    println("                   Comparison/Output/")
    return results
end

# Auto-run unless a harness defines RUN_ALL_NO_AUTORUN = true first.
if !(@isdefined(RUN_ALL_NO_AUTORUN) && RUN_ALL_NO_AUTORUN)
    run_all()
end
