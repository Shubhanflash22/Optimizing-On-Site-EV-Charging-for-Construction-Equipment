# #############################################################################
# RUN_ALL_25_RUNS_GHOST.jl — single smoke-test run before the real 25-run sweep.
#
# Supersedes RUN_ALL_9_RUNS.jl (High/Low/NearMean) and RUN_ALL_5_RUNS.jl
# (GroundTruth/Normal) with a single script covering all 5 modes:
# GroundTruth (:live_data), High, Low, NearMean, Normal.
#
#   3-seed block  @ 1200s, 1 day  x 5 modes x seeds {7,18,55}  -> 15 runs
#   5-day block   @ 1200s, seed 7 x 5 modes                    -> 5 runs
#   1-hour block  @ 3600s, seed 7 x 5 modes                    -> 5 runs
# = 25 runs total.
#
# NOTE: the 1-hour block uses a HARD 3600s cap, not the "inf" (1e9s) the
# earlier scripts used for their third block -- this is a deliberate change
# per the run spec, not a typo.
#
# "GroundTruth" = :live_data (LIVE_DATA_MODE): plant power drawn from real
# recorded values in Input/live_powers.csv. High/Low/NearMean/Normal are the
# Bayesian Normal(mu,sd) draw-mode sensitivity sweep, unchanged.
#
# Output goes into Output_AllRuns/ with its OWN manifest (manifest_25runs.csv)
# so it never collides with any manifest left over from the two scripts this
# replaces -- but see the run guide for why you should delete the old run
# folders before starting this one anyway (label numbers below start fresh
# at 01, which WILL collide by name with the old scripts' 01_.. through
# 15_.. folders if they're still on disk).
#
# Usage (from this file's directory, or anywhere — it's location-independent):
#   julia --project=. RUN_ALL_25_RUNS.jl
# #############################################################################

using Dates
using Printf

const _THIS_DIR = @__DIR__
global SWEEP_NO_AUTORUN = true
include(joinpath(_THIS_DIR, "7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl"))

# -----------------------------------------------------------------------------
# Results root — same Output_AllRuns/ folder convention as before.
# -----------------------------------------------------------------------------
const RESULTS_ROOT = normpath(joinpath(_THIS_DIR, "..", "Output_AllRuns_GHOST"))
mkpath(RESULTS_ROOT)
const MANIFEST_PATH = joinpath(RESULTS_ROOT, "manifest_ghost.csv")

# -----------------------------------------------------------------------------
# The 25 solves. Each entry runs ONE mode via `modes = (mode,)` so its output
# nests as <out_dir>/<mode>/... with nothing else sharing that folder.
# -----------------------------------------------------------------------------
const RUNS = [
    (label = "01_GroundTruth_seed7_1200s",  mode = :live_data, seed = 7,  n_day_run = 1, time_limit_sec = 30.0),
]

# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------
function _init_manifest()
    if !isfile(MANIFEST_PATH)
        open(MANIFEST_PATH, "w") do io
            println(io, "label,mode,seed,n_day_run,time_limit_sec,status,elapsed_sec,started_at,finished_at,out_dir,error")
        end
    end
end

function _log_manifest(row)
    open(MANIFEST_PATH, "a") do io
        esc(x) = replace(string(x), "," => ";")
        println(io, join([
            row.label, row.mode, row.seed, row.n_day_run, row.time_limit_sec,
            row.status, @sprintf("%.1f", row.elapsed_sec),
            row.started_at, row.finished_at, row.out_dir, esc(row.error),
        ], ","))
    end
end

# -----------------------------------------------------------------------------
# Main loop — one run_comparison_sweep call per entry, wrapped in try/catch so
# one failure doesn't kill the batch, manifest updated after every run so
# progress survives a kill mid-batch.
# -----------------------------------------------------------------------------
function main()
    _init_manifest()

    println("="^78)
    println("GHOST RUN -- single smoke test (01_GroundTruth_seed7_1200s, 30s cap)")
    println("Results root : $(RESULTS_ROOT)")
    println("Manifest     : $(MANIFEST_PATH)")
    println("Total runs   : $(length(RUNS))")
    println("="^78)

    batch_t0 = time()

    for (i, r) in enumerate(RUNS)
        out_dir = joinpath(RESULTS_ROOT, r.label)
        started_at = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        println("\n" * "#"^78)
        println("# RUN $(i)/$(length(RUNS)): $(r.label)")
        println("#   mode=$(r.mode)  seed=$(r.seed)  n_day_run=$(r.n_day_run)  time_limit_sec=$(r.time_limit_sec)")
        println("#   out_dir=$(out_dir)")
        println("#"^78)

        t0 = time()
        status = "OK"
        errmsg = ""
        try
            run_comparison_sweep(;
                out_dir = out_dir,
                modes = (r.mode,),
                seed = r.seed,
                n_day_run = r.n_day_run,
                time_limit_sec = r.time_limit_sec,
            )
        catch e
            status = "FAILED"
            errmsg = sprint(showerror, e)
            @error "Run $(r.label) failed" exception=(e, catch_backtrace())
        end
        elapsed = time() - t0
        finished_at = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

        _log_manifest((; label = r.label, mode = r.mode, seed = r.seed,
                         n_day_run = r.n_day_run, time_limit_sec = r.time_limit_sec,
                         status, elapsed_sec = elapsed, started_at, finished_at,
                         out_dir, error = errmsg))

        @printf("\n>>> RUN %d/%d (%s) finished: %s  [%.1f min]\n",
                i, length(RUNS), r.label, status, elapsed / 60)
    end

    total_elapsed = time() - batch_t0
    println("\n" * "="^78)
    @printf("ALL 25 RUNS COMPLETE — total wall time: %.1f min (%.2f hr)\n",
            total_elapsed / 60, total_elapsed / 3600)
    println("Manifest: $(MANIFEST_PATH)")
    println("="^78)
end

main()
