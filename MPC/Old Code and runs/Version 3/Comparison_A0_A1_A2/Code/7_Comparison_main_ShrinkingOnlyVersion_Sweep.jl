# #############################################################################
# 7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl — DRAW-MODE SENSITIVITY SWEEP
# -----------------------------------------------------------------------------
# Runs the SAME 3-way comparison as 7_Comparison_main_ShrinkingOnlyVersion.jl
# (A0 one-shot vs A1-Shrinking vs A2-Shrinking, from the same shared power
# pool) FIVE TIMES OVER — once per plant `mode` supported by
# `draw_activity_power_pool` in 1_Common.jl (see that function's "CHANGE 6"
# header for the full detail):
#
#   :normal       — the original, unbiased draw: z ~ N(0,1), can land anywhere
#   :near_mean    — plant draws clustered within ~0.5 sigma of the mean
#   :high         — plant draws >= 2 sigma ABOVE the mean, every single draw
#   :low          — plant draws >= 2 sigma BELOW the mean, every single draw
#   :spread_wide  — plant draws >= 2 sigma from the mean, sign random each draw
#
# Each mode gets its OWN freshly-built ActivityPowerPool (same seed, same
# frozen mu/sigma, same n_samples sizing as the base driver — ONLY `mode`
# differs) and its own output subfolder:
#
#   Output/normal/A0_A1S/           Output/near_mean/A0_A1S/        … etc
#   Output/normal/A0_A2S/           Output/near_mean/A0_A2S/
#   Output/normal/A0_A1S_A2S/       Output/near_mean/A0_A1S_A2S/
#   Output/normal/A1S_A2S/          Output/near_mean/A1S_A2S/
#   Output/high/…    Output/low/…    Output/spread_wide/…
#
# i.e. exactly the 4 comparison folders the base driver writes, nested one
# level deeper under the mode name. A cross-mode KPI table is also printed at
# the end and written to Output/mode_sweep_kpi_summary.csv.
#
# FIXED AT n_day_run = 1 (1-day runs) FOR ALL FIVE MODES — override via the
# `n_day_run` keyword to `run_comparison_sweep` if a longer horizon is needed;
# nothing else about the sweep depends on that value.
#
# NOTHING about the two codebases, Approach 0's source, the data loading, or
# `write_comparison_outputs` is duplicated or reimplemented here — this file
# `include()`s 7_Comparison_main_ShrinkingOnlyVersion.jl (with its own
# auto-run disabled) to get the SAME `A1ShrinkingApp` / `A2ShrinkingApp`
# namespaced wrappers, the SAME `build_comparison_input`, the SAME
# `_ALL_COMBOS`, and the SAME `write_comparison_outputs`, then adds ONLY the
# mode loop around the same three solves that driver already performs once.
# Edit either codebase (or the base comparison driver) in place and the next
# sweep run picks the change up automatically, exactly like the base driver.
# #############################################################################

# Load the base driver WITHOUT letting it auto-run its own single-mode
# comparison — this file supplies its own entry point (`run_comparison_sweep`)
# at the bottom instead.
const COMPARISON_NO_AUTORUN = true
include(joinpath(@__DIR__, "7_Comparison_main_ShrinkingOnlyVersion.jl"))

using Printf
using Random

# The 5 draw modes swept by default (see header above). Pass a different
# `modes` tuple/vector to `run_comparison_sweep` to run a subset, e.g. during
# a quick test.
const _DEFAULT_SWEEP_MODES = (:normal, :near_mean, :high, :low, :spread_wide)

# =============================================================================
# MAIN ENTRY POINT — same knobs as `run_comparison` (see the base driver for
# the full docstring on each), plus `modes` to control which draw modes run.
# `n_day_run` defaults to 1 (1-day runs) here, for fast, repeated checks
# across all five plant modes.
# =============================================================================
function run_comparison_sweep(; input_dir::AbstractString = _COMPARISON_INPUT,
                                     out_dir::AbstractString = _COMPARISON_OUT,
                                     csv_source_dir::AbstractString = _A1S_INPUT,
                                     run_regression::Bool = false,
                                     regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                                     regression_samples::Int = 2000,
                                     regression_chains::Int = 4,
                                     approach0_source::Symbol = :a1_shrinking,   # :a1_shrinking / :a2_shrinking
                                     approach0_plant::Symbol = :sampled,         # :sampled / :mean
                                    time_limit_sec::Float64 = 1200.0,
                                     multi_activity::Bool = false,
                                     require_site_visit::Bool = false,
                                     single_visit_per_site::Bool = false,
                                     mcmc_samples::Int = 500,
                                     shrinking::Bool = true,
                                     H::Int = 16,
                                     n_day_run::Int = 1,                        # 1-day runs, per request
                                     pool_n_samples::Union{Nothing, Int} = nothing,
                                     n_scenarios::Int = A2ShrinkingApp.ScenarioSampler.DEFAULT_N_SCENARIOS,
                                     combos = _ALL_COMBOS,
                                     modes = _DEFAULT_SWEEP_MODES,
                                     seed::Int = 1)
    approach0_source in (:a1_shrinking, :a2_shrinking) ||
        error("run_comparison_sweep: approach0_source must be :a1_shrinking or :a2_shrinking")
    approach0_plant in (:sampled, :mean) ||
        error("run_comparison_sweep: approach0_plant must be :sampled or :mean")
    isempty(modes) && error("run_comparison_sweep: `modes` must list at least one mode")

    build_comparison_input(; input_dir, csv_source_dir, run_regression,
                            regression_data_dir, regression_samples, regression_chains)

    return _with_console_log(out_dir) do
        _sweep_t0 = time()
        println("="^78)
        println("DRAW-MODE SWEEP — 3-WAY COMPARISON x $(length(modes)) plant modes")
        println("Modes             : ", join(modes, ", "))
        println("A1 Shrinking code: $(_A1S_CODE)")
        println("A2 Shrinking code: $(_A2S_CODE)")
        println("Input             : $(abspath(input_dir))")
        println("Output            : $(abspath(out_dir))")
        println("Comparisons       : ", join(first.(combos), ", "), "  (written per mode)")
        println("n_day_run         : $(n_day_run)  (1-day runs)")
        _status("Sweep started")
        println("="^78)

        # ---- load data ONCE, shared across every mode -- only the plant pool
        # changes between modes, never the underlying input data or (mu, sd) ----
        dA1S, dA2S = _timed_status("loading input data (A1S + A2S DataLoaders)") do
            (A1ShrinkingApp.DataLoader.load_data(:input;  input_dir = input_dir),
             A2ShrinkingApp.DataLoader.load_data(:input;  input_dir = input_dir))
        end

        # Same sizing rule as the base driver: n_day_run days' worth of
        # intervals, plus a small safety margin.
        nK_day = length(collect(dA1S.K))
        n_samples = pool_n_samples === nothing ?
            nK_day * n_day_run + 5 : pool_n_samples

        mode_summaries = NamedTuple[]

        for mode in modes
            println("\n" * "#"^78)
            println("# MODE: :$(mode)")
            println("#"^78)

            # ---- ONE shared pool per mode, built ONCE and passed as the
            # literal SAME object into all three runs for THAT mode ----
            pool = _timed_status("building :$(mode) power pool ($(n_samples) samples/entity-activity)") do
                A1ShrinkingApp.Common.draw_activity_power_pool(dA1S.E, dA1S.prior_mu, dA1S.prior_sigma;
                                                               n_samples = n_samples,
                                                               rng = MersenneTwister(seed),
                                                               mode = mode)
            end
            println("[:$(mode)] pool: n_samples=$(n_samples) per (entity, activity); mu=",
                    round.(pool.mu, digits = 2), " kW; sd=", round.(pool.sd, digits = 2), " kW")

            # ---- APPROACH 0 (one-shot, no replanning) ----
            res0 = _timed_status("[:$(mode)] Approach 0 solve (source = :$(approach0_source))") do
                if approach0_source == :a1_shrinking
                    A1ShrinkingApp.MPCLoop.run_one_shot(dA1S, pool; plant = approach0_plant,
                                                       time_limit_sec, multi_activity,
                                                       require_site_visit, single_visit_per_site,
                                                       n_day_run, seed)
                else # :a2_shrinking
                    A2ShrinkingApp.MPCLoop.run_one_shot(dA2S, pool; plant = approach0_plant,
                                                       time_limit_sec, multi_activity,
                                                       require_site_visit, single_visit_per_site,
                                                       n_day_run, seed)
                end
            end

            # ---- APPROACH 1b: Shrinking Horizon closed-loop MPC ----
            resA1S = _timed_status("[:$(mode)] Approach 1 - Shrinking solve (n_day_run = $(n_day_run))") do
                A1ShrinkingApp.MPCLoop.run_mpc(dA1S, pool; shrinking, H, time_limit_sec, multi_activity,
                                              require_site_visit, single_visit_per_site,
                                              mcmc_samples, plant = :sampled, n_day_run, seed)
            end

            # ---- APPROACH 2b: Shrinking Horizon, stochastic scenario-based closed-loop MPC ----
            resA2S = _timed_status("[:$(mode)] Approach 2 - Shrinking solve ($(n_scenarios) scenarios, n_day_run = $(n_day_run))") do
                A2ShrinkingApp.MPCLoop.run_mpc(dA2S, pool; shrinking, H, time_limit_sec, multi_activity,
                                              require_site_visit, single_visit_per_site,
                                              mcmc_samples, plant = :sampled, n_scenarios, n_day_run, seed)
            end

            all_apps = Dict(
                "A0"  => Approach("A0",  "Approach 0 (one-shot, :$(approach0_plant))", res0,   :gray40),
                "A1S" => Approach("A1S", "Approach 1 - Shrinking",                     resA1S, :firebrick),
                "A2S" => Approach("A2S", "Approach 2 - Shrinking (stochastic)",        resA2S, :darkorange),
            )

            # ---- write every requested comparison for THIS mode, nested
            # under Output/<mode>/ ----
            mode_out = joinpath(out_dir, String(mode))
            println("\n--- [:$(mode)] writing $(length(combos)) comparison(s) to $(mode_out) ---")
            for (folder, keys) in combos
                apps = [all_apps[k] for k in keys]
                sub_out = joinpath(mode_out, folder)
                println("  $(folder): ", join([a.label for a in apps], " vs "), "  ->  $(sub_out)")
                _timed_status("[:$(mode)] writing $(folder)") do
                    Base.invokelatest(write_comparison_outputs, apps, sub_out)
                end
            end

            push!(mode_summaries, (; mode, res0, resA1S, resA2S))
        end

        # ---- cross-mode KPI summary — how the SAME three approaches moved
        # under each of the five plant modes, all in one place ----
        println("\n" * "="^78)
        println("CROSS-MODE KPI SUMMARY")
        for row in mode_summaries
            header = "--[ :$(row.mode) ]"
            println("\n", header, "-"^max(4, 78 - length(header)))
            @printf("%-28s %14s %14s %14s\n", "Metric", "A0", "A1-Shrink", "A2-Shrink")
            @printf("%-28s %14.2f %14.2f %14.2f\n", "Grid energy (kWh)",        row.res0.total_energy,   row.resA1S.total_energy,   row.resA2S.total_energy)
            @printf("%-28s %14.2f %14.2f %14.2f\n", "Energy cost (USD)",        row.res0.total_cost,     row.resA1S.total_cost,     row.resA2S.total_cost)
            @printf("%-28s %14.2f %14.2f %14.2f\n", "CO2 (kg)",                 row.res0.total_co2,      row.resA1S.total_co2,      row.resA2S.total_co2)
            @printf("%-28s %14.2f %14.2f %14.2f\n", "NCD peak (kW)",            row.res0.nc_peak,        row.resA1S.nc_peak,        row.resA2S.nc_peak)
            @printf("%-28s %14.2f %14.2f %14.2f\n", "OPD peak (kW)",            row.res0.op_peak,        row.resA1S.op_peak,        row.resA2S.op_peak)
            @printf("%-28s %14.2f %14.2f %14.2f\n", "Missed work (h)",          row.res0.missed,         row.resA1S.missed,         row.resA2S.missed)
            @printf("%-28s %14.3f %14.3f %14.3f\n", "Terminal shortfall (kWh)", row.res0.shortfall_kWh,  row.resA1S.shortfall_kWh,  row.resA2S.shortfall_kWh)
        end
        println("="^78)

        # ---- same table, flattened to CSV for easy plotting/diffing across modes ----
        summary_csv = joinpath(out_dir, "mode_sweep_kpi_summary.csv")
        open(summary_csv, "w") do io
            println(io, "mode,approach,total_energy_kWh,total_cost_USD,total_co2_kg,nc_peak_kW,op_peak_kW,missed_h,shortfall_kWh")
            for row in mode_summaries
                for (label, res) in (("A0", row.res0), ("A1S", row.resA1S), ("A2S", row.resA2S))
                    println(io, join(Any[String(row.mode), label, res.total_energy, res.total_cost,
                                          res.total_co2, res.nc_peak, res.op_peak, res.missed,
                                          res.shortfall_kWh], ","))
                end
            end
        end
        println("\nCross-mode KPI summary written to: $(summary_csv)")

        println("\nResults written to: $(abspath(out_dir))")
        for mode in modes
            println("  $(mode)/")
            for (folder, keys) in combos
                println("    $(folder)/  (", join(keys, " vs "), ")")
            end
        end
        println("  mode_sweep_kpi_summary.csv")
        println("  run_log.txt  (this console log, shared across all $(length(modes)) modes x $(length(combos)) comparisons)")
        _status("Sweep finished — total elapsed $(round(time() - _sweep_t0, digits=1))s")

        return (; mode_summaries, dA1S, dA2S)
    end
end

# Auto-run unless a harness defines SWEEP_NO_AUTORUN = true first.
if !(@isdefined(SWEEP_NO_AUTORUN) && SWEEP_NO_AUTORUN)
    run_comparison_sweep(n_day_run = 1, approach0_source = :a1_shrinking)
end
