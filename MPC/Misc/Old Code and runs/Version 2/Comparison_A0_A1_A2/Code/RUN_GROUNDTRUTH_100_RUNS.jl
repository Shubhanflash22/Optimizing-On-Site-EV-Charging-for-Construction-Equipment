# #############################################################################
# RUN_GROUNDTRUTH_100_RUNS.jl — 100-run sweep, GroundTruth (:live_data) ONLY.
#
# Unlike RUN_ALL_25_RUNS.jl (which sweeps 5 modes across 3 duration/cap tiers),
# this script holds mode and time_limit_sec fixed and varies ONLY the seed --
# 100 distinct seeds, all at 1200s / 1-day, all :live_data. Purpose: a much
# larger sample of the real-recorded-power draw's run-to-run variance for
# GroundTruth specifically (e.g. to characterize the infeasibility rate, cost
# spread, etc. across many independent realizations, not just 3 seeds).
#
# Output folder names are DELIBERATELY DIFFERENT from RUN_ALL_25_RUNS.jl's
# Output_AllRuns / Detailed_Output_AllRuns, so this sweep's results can never
# collide with or be confused for the 25-run sweep's -- same parent directory
# (Comparison_A0_A1_A2/), different folder names:
#   Output_GroundTruth_100Runs/            (comparison output, per run)
#   Detailed_Output_GroundTruth_100Runs/   (plan-vs-realized CSVs, per run)
# The detailed-output sibling-folder naming is derived automatically from
# whatever RESULTS_ROOT is named below (see the generalized fix in
# 7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl's _write_detailed_output) --
# rename RESULTS_ROOT and the detailed tree renames itself to match, no other
# edit needed.
#
# Usage (from this file's directory, or anywhere — it's location-independent):
#   julia --project=. RUN_GROUNDTRUTH_100_RUNS.jl
# #############################################################################

using Dates
using Printf

const _THIS_DIR = @__DIR__
global SWEEP_NO_AUTORUN = true
include(joinpath(_THIS_DIR, "7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl"))

# -----------------------------------------------------------------------------
# Results root — renamed, sibling to (not overlapping with) Output_AllRuns/.
# -----------------------------------------------------------------------------
const RESULTS_ROOT = normpath(joinpath(_THIS_DIR, "..", "Output_GroundTruth_100Runs"))
mkpath(RESULTS_ROOT)
const MANIFEST_PATH = joinpath(RESULTS_ROOT, "manifest_100runs_groundtruth.csv")

# -----------------------------------------------------------------------------
# The 100 solves: GroundTruth (:live_data) only, 1200s cap, 1-day runs,
# seeds 1..100. Each entry runs ONE mode via `modes = (mode,)` so its output
# nests as <out_dir>/<mode>/... with nothing else sharing that folder.
# -----------------------------------------------------------------------------
const RUNS = [
    (label = @sprintf("%03d_GroundTruth_seed%d_1200s", i, i),
     mode = :live_data, seed = i, n_day_run = 1, time_limit_sec = 1200.0,
     detailed_output = true)
    for i in 1:100
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
    println("100-RUN GROUNDTRUTH-ONLY SWEEP (seeds 1..100, all @ 1200s / 1-day)")
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
                detailed_output = r.detailed_output,
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
    @printf("ALL 100 RUNS COMPLETE — total wall time: %.1f min (%.2f hr)\n",
            total_elapsed / 60, total_elapsed / 3600)
    println("Manifest: $(MANIFEST_PATH)")
    println("="^78)
end

main()
