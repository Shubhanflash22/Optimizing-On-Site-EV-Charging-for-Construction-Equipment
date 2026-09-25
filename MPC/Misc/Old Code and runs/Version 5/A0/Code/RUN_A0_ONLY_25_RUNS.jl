# #############################################################################
# RUN_A0_ONLY_25_RUNS.jl — Approach 0 ONLY, for the same 25 cases as the
# original RUN_ALL_25_RUNS.jl (Comparison_A0_A1_A2/Code), with full detailed
# output (CEV + MCS plan/realized) and an A0-only KPI summary table per run.
#
#   3-seed block  @ 1200s, 1 day  x 5 modes x seeds {7,18,55}  -> 15 runs
#   5-day block   @ 1200s, seed 7 x 5 modes                    -> 5 runs
#   1-hour block  @ 3600s, seed 7 x 5 modes                    -> 5 runs
# = 25 runs total.  Same labels, modes, seeds, n_day_run, time_limit_sec as
# the original 25-run sweep (kept EXACTLY the same, per request) -- only A0
# is solved this time (A1S/A2S are never built).
#
# Usage (from this file's own folder):
#   julia --project=. RUN_A0_ONLY_25_RUNS.jl
#
# Folder layout expected (this is exactly what the A0/ zip already contains):
#   A0/
#     Code/   <- this file, A0App.jl, run_one_shot_detailed.jl, kpi_tables.jl,
#                outputs_a0.jl, src/ (copied Approach-1 source, unmodified)
#     Input/  <- the 8 shared input CSVs (the same ones the real 25-run sweep
#                used -- copied from Comparison_A0_A1_A2/Input)
#
# Output goes into A0/Output_A0_Only/<run_label>/ :
#   A0_CEV_plan_full.csv      full-day plan, every 15-min step, per CEV
#   A0_CEV_realized.csv       what was actually realized, every 15-min step, per CEV
#   A0_MCS_plan_full.csv      full-day plan, every 15-min step, for the MCS
#   A0_MCS_realized.csv       what the MCS actually did, every 15-min step
#   A0_solve_log.csv          one row per day: solver status/objective/gap/time
#   A0_day_snapshots.csv      one row per day: cumulative backlog/shortfall
#   A0_kpi_summary.csv        the A0-only full KPI table (whole run)
#   A0_kpi_summary_by_day.csv (5-day runs only) same table, Day1..Day5 + Overall
# No images, no extra plot CSVs are written.
# #############################################################################

using Dates
using Printf
using Random
using CSV
using DataFrames

const _THIS_DIR = @__DIR__
include(joinpath(_THIS_DIR, "A0App.jl"))
using .A0App

const _A0_ROOT      = normpath(joinpath(_THIS_DIR, ".."))
const _INPUT_DIR    = joinpath(_A0_ROOT, "Input")
const _RESULTS_ROOT = joinpath(_A0_ROOT, "Output_A0_Only")
mkpath(_RESULTS_ROOT)
const _MANIFEST_PATH = joinpath(_RESULTS_ROOT, "manifest_A0_25runs.csv")

# -----------------------------------------------------------------------------
# The 25 solves -- IDENTICAL configuration to the original RUN_ALL_25_RUNS.jl
# (labels, modes, seeds, n_day_run, time_limit_sec all kept exactly the same,
# per request). approach0_plant is fixed at :sampled, matching the original
# comparison driver's default for Approach 0's open-loop replay.
# -----------------------------------------------------------------------------
const RUNS = [
    # ---- 3-seed block @ 1200s, 1 day (15 runs) ----
    (label = "01_GroundTruth_seed7_1200s",  mode = :live_data, seed = 7,  n_day_run = 1, time_limit_sec = 1200.0),
    (label = "02_GroundTruth_seed18_1200s", mode = :live_data, seed = 18, n_day_run = 1, time_limit_sec = 1200.0),
    (label = "03_GroundTruth_seed55_1200s", mode = :live_data, seed = 55, n_day_run = 1, time_limit_sec = 1200.0),

    (label = "04_High_seed7_1200s",         mode = :high,      seed = 7,  n_day_run = 1, time_limit_sec = 1200.0),
    (label = "05_High_seed18_1200s",        mode = :high,      seed = 18, n_day_run = 1, time_limit_sec = 1200.0),
    (label = "06_High_seed55_1200s",        mode = :high,      seed = 55, n_day_run = 1, time_limit_sec = 1200.0),

    (label = "07_Low_seed7_1200s",          mode = :low,       seed = 7,  n_day_run = 1, time_limit_sec = 1200.0),
    (label = "08_Low_seed18_1200s",         mode = :low,       seed = 18, n_day_run = 1, time_limit_sec = 1200.0),
    (label = "09_Low_seed55_1200s",         mode = :low,       seed = 55, n_day_run = 1, time_limit_sec = 1200.0),

    (label = "10_NearMean_seed7_1200s",     mode = :near_mean, seed = 7,  n_day_run = 1, time_limit_sec = 1200.0),
    (label = "11_NearMean_seed18_1200s",    mode = :near_mean, seed = 18, n_day_run = 1, time_limit_sec = 1200.0),
    (label = "12_NearMean_seed55_1200s",    mode = :near_mean, seed = 55, n_day_run = 1, time_limit_sec = 1200.0),

    (label = "13_Normal_seed7_1200s",       mode = :normal,    seed = 7,  n_day_run = 1, time_limit_sec = 1200.0),
    (label = "14_Normal_seed18_1200s",      mode = :normal,    seed = 18, n_day_run = 1, time_limit_sec = 1200.0),
    (label = "15_Normal_seed55_1200s",      mode = :normal,    seed = 55, n_day_run = 1, time_limit_sec = 1200.0),

    # ---- 5-day block, seed 7, 1200s (5 runs) ----
    (label = "16_GroundTruth_seed7_5days",  mode = :live_data, seed = 7, n_day_run = 5, time_limit_sec = 1200.0),
    (label = "17_High_seed7_5days",         mode = :high,      seed = 7, n_day_run = 5, time_limit_sec = 1200.0),
    (label = "18_Low_seed7_5days",          mode = :low,       seed = 7, n_day_run = 5, time_limit_sec = 1200.0),
    (label = "19_NearMean_seed7_5days",     mode = :near_mean, seed = 7, n_day_run = 5, time_limit_sec = 1200.0),
    (label = "20_Normal_seed7_5days",       mode = :normal,    seed = 7, n_day_run = 5, time_limit_sec = 1200.0),

    # ---- 1-hour (3600s) block, seed 7, 1 day (5 runs) ----
    (label = "21_GroundTruth_seed7_3600s",  mode = :live_data, seed = 7, n_day_run = 1, time_limit_sec = 3600.0),
    (label = "22_High_seed7_3600s",         mode = :high,      seed = 7, n_day_run = 1, time_limit_sec = 3600.0),
    (label = "23_Low_seed7_3600s",          mode = :low,       seed = 7, n_day_run = 1, time_limit_sec = 3600.0),
    (label = "24_NearMean_seed7_3600s",     mode = :near_mean, seed = 7, n_day_run = 1, time_limit_sec = 3600.0),
    (label = "25_Normal_seed7_3600s",       mode = :normal,    seed = 7, n_day_run = 1, time_limit_sec = 3600.0),
]

# -----------------------------------------------------------------------------
# Manifest helpers -- same pattern as the original RUN_ALL_25_RUNS.jl.
# -----------------------------------------------------------------------------
function _init_manifest()
    if !isfile(_MANIFEST_PATH)
        open(_MANIFEST_PATH, "w") do io
            println(io, "label,mode,seed,n_day_run,time_limit_sec,status,elapsed_sec,started_at,finished_at,out_dir,error")
        end
    end
end

function _log_manifest(row)
    open(_MANIFEST_PATH, "a") do io
        esc(x) = replace(string(x), "," => ";")
        println(io, join([
            row.label, row.mode, row.seed, row.n_day_run, row.time_limit_sec,
            row.status, @sprintf("%.1f", row.elapsed_sec),
            row.started_at, row.finished_at, row.out_dir, esc(row.error),
        ], ","))
    end
end

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
function main()
    _init_manifest()

    println("="^78)
    println("A0-ONLY 25-RUN SWEEP (GroundTruth / High / Low / NearMean / Normal)")
    println("Input        : $(abspath(_INPUT_DIR))")
    println("Results root : $(_RESULTS_ROOT)")
    println("Manifest     : $(_MANIFEST_PATH)")
    println("Total runs   : $(length(RUNS))")
    println("="^78)

    # ---- load input data ONCE, shared across every run -- only the plant
    # pool changes between runs, never the underlying input data ----
    dA1S = A0App.DataLoader.load_data(:input; input_dir = _INPUT_DIR)
    nK_day = length(collect(dA1S.K))

    batch_t0 = time()

    for (i, r) in enumerate(RUNS)
        out_dir = joinpath(_RESULTS_ROOT, r.label)
        mkpath(out_dir)
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
            n_samples = nK_day * r.n_day_run + 5

            pool = if r.mode == :live_data
                A0App.Common.draw_activity_power_pool_live(
                    dA1S.E, A0App.DataLoader.load_live_powers(_INPUT_DIR);
                    rng = MersenneTwister(r.seed))
            else
                A0App.Common.draw_activity_power_pool(dA1S.E, dA1S.prior_mu, dA1S.prior_sigma;
                                                       n_samples = n_samples,
                                                       rng = MersenneTwister(r.seed),
                                                       mode = r.mode)
            end

            resd = A0App.run_one_shot_detailed(dA1S, pool; plant = :sampled,
                                                time_limit_sec = r.time_limit_sec,
                                                n_day_run = r.n_day_run, seed = r.seed)

            A0App.write_a0_detailed_outputs(resd, out_dir)
            A0App.write_a0_kpi_outputs(resd, dA1S, out_dir)
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
    @printf("ALL 25 A0-ONLY RUNS COMPLETE — total wall time: %.1f min (%.2f hr)\n",
            total_elapsed / 60, total_elapsed / 3600)
    println("Manifest: $(_MANIFEST_PATH)")
    println("="^78)
end

main()
