# #############################################################################
# OneShot_main.jl  -  driver script, not a module
# -----------------------------------------------------------------------------
# The single entry point for running Approach 0 end to end: wires together
# Common, DataLoader, MCSModel, OneShot, and Output (in that dependency
# order), exposes one configurable function (run_scenario_0) that runs a full
# scenario from raw input files to printed KPIs and written CSVs, and then
# auto-runs itself with default settings the moment this file is included.
# Three groups:
#
#   1. MODULE WIRING
#      the six `include` calls plus the five `using` lines right after them
#      -- loads the other five files as modules, in the order each one
#      depends on the last, and imports only the specific functions this
#      driver actually calls from each (load_data/load_live_powers from
#      DataLoader, the two pool constructors from Common, run_one_shot from
#      OneShot, and the three output functions from Output).
#
#   2. SCENARIO ENTRY POINT
#      run_scenario_0
#      -- resolves the input directory (falling back to a couple of common
#      alternate locations if the given one doesn't exist), loads the
#      problem data, builds the shared stochastic ActivityPowerPool (from
#      real recorded field data when mode = :live_data, otherwise sampled
#      around the calibrated/prior mean with the requested spread), runs
#      Approach 0 via run_one_shot, prints the KPI summary, and -- when
#      detailed_output is true -- writes every CSV via Output.jl and prints
#      where they landed.
#
#   3. AUTO-RUN
#      the `if !(@isdefined(SCENARIO0_NO_AUTORUN) ...)` guard at the bottom
#      -- runs run_scenario_0() with every default the instant this file is
#      included, unless the includer has already defined
#      SCENARIO0_NO_AUTORUN = true beforehand (letting, for example, a
#      comparison script across Approaches 0/1/2 include this file for its
#      module definitions without triggering an unwanted extra run).
# #############################################################################

# external packages used across this file
using Printf
using Random

const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_MCSModel.jl"))
include(joinpath(_CODE_DIR, "4_OneShot.jl"))
include(joinpath(_CODE_DIR, "5_Output.jl"))

using .DataLoader: load_data, load_live_powers
using .Common: draw_activity_power_pool, draw_activity_power_pool_live
using .OneShot: run_one_shot
using .Output: write_detailed_output, write_kpi_summary, print_kpis

# Runs Approach 0 end to end for one scenario: resolves the input directory, loads the problem data, builds the shared stochastic pool, runs the one-shot solve-and-simulate loop, prints the KPIs, and writes the CSVs when detailed_output is true.
function run_scenario_0(; mode::Symbol = :normal,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          plant::Symbol = :sampled,
                          n_day_run::Int = 1,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          detailed_output::Bool = true,
                          seed::Int = 1)
    # Falls back to a couple of common alternate locations if the given input_dir doesn't exist, so the script still finds the data when run from a different working directory.
    if !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    # Loads the problem data and makes sure the output directory exists before anything gets written to it.
    d = load_data(input_dir)

    mkpath(out_dir)

    # Builds the shared ActivityPowerPool the whole run will draw from.
    # n_samples is sized generously for the number of intervals actually needed across the whole run (nK_day * n_day_run), with a small +5 buffer, since next_power! errors if the pool ever runs out for a given (entity, activity) pair.
    # mode = :live_data builds the pool from real recorded field data (live_powers.csv); any other mode value is passed straight through as the sampling shape to draw_activity_power_pool (:normal, :near_mean, :high, :low, or :spread_wide), sampled around the prior mean/sigma from parameters.csv.
    nK_day = length(collect(d.K))
    n_samples = nK_day * n_day_run + 5
    pool = if mode == :live_data
        live_values = load_live_powers(input_dir)
        draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
    else
        draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                 n_samples = n_samples, rng = MersenneTwister(seed), mode = mode)
    end

    # Runs Approach 0 itself: one whole-day MILP solve per day, executed open-loop against the pool.
    res = run_one_shot(d, pool; time_limit_sec = time_limit_sec,
                       multi_activity = multi_activity,
                       plant = plant, n_day_run = n_day_run, seed = seed,
                       detailed_output = detailed_output)

    # Always prints the KPI summary; only writes CSVs (the four detailed logs plus the KPI/solve/interval tables) when detailed_output was requested.
    print_kpis(res)
    if detailed_output
        write_detailed_output(res, out_dir)
        write_kpi_summary(res, d, out_dir; day_snapshots_df = res.day_snapshots_df, solve_log = res.solve_log)
        println("\nResults written to: $(abspath(out_dir))")
        println("  A0_plan_full.csv, A0_realized_tuple.csv               (CEV, per 15-min step)")
        println("  A0_MCS_plan_full.csv, A0_MCS_realized_tuple.csv       (MCS, per 15-min step)")
        println("  A0_kpi_summary.csv                                    (whole-run KPI table)")
        println("  A0_solve_log.csv                                      (per-day solver status, objective, MIP gap)")
        println("  A0_interval_log.csv                                   (plain per-interval log)")
        n_day_run > 1 && println("  A0_kpi_summary_by_day.csv                             (Day1..Day$(n_day_run) + Overall)")
    else
        println("\n(detailed_output = false — no CSVs written; pass detailed_output = true to get them)")
    end
    return res
end

# Runs a default scenario automatically the instant this file is included, unless the includer set SCENARIO0_NO_AUTORUN = true beforehand (for example, a script that includes this file only to reuse its function/module definitions without triggering an extra run).
if !(@isdefined(SCENARIO0_NO_AUTORUN) && SCENARIO0_NO_AUTORUN)
    run_scenario_0()
end
