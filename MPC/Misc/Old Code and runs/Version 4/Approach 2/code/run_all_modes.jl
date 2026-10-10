# #############################################################################
# run_all_modes.jl  -  standalone sweep script, not part of the pipeline
# -----------------------------------------------------------------------------
# Runs Approach 2 once for every mode (6 options) on ONE input dataset -- 6
# runs total -- saving each run's CSVs into its own folder named
# input_<mode>_sampled. Every run uses important_only = true, so only the 8
# key files are written for each (no figures, no replan grids, no activity
# comparison) -- this keeps a 6-run sweep fast, since Approach 2 already
# solves ~96 windows per day, each with one copy of the window per scenario,
# and rendering ~30 figures on top of that per run would dominate the
# sweep's runtime.
#
# PLACE THIS FILE IN THE code/ FOLDER, next to 6_Shrinking_Horizon_main.jl.
# All paths below are then derived automatically from @__DIR__ (this file's
# own location), the same trick 6_Shrinking_Horizon_main.jl itself uses --
# nothing here is hardcoded to a specific machine, username, or drive letter,
# so the whole "Approach 2" folder can be moved or copied anywhere and this
# still works unmodified, as long as the code/, data/input_data/, and
# output/ folders stay siblings of each other (matching the README's
# documented layout).
#
# Sets SCENARIO1_NO_AUTORUN itself, so including 6_Shrinking_Horizon_main.jl
# here does not also trigger an extra default run on top of the 6 in the
# sweep.
#
# From the Julia REPL:
#   include("run_all_modes.jl")
# #############################################################################

using Printf

const CODE_DIR  = @__DIR__
const ROOT_DIR  = dirname(CODE_DIR)
const INPUT_DIR = joinpath(ROOT_DIR, "data", "input_data")
const OUT_ROOT  = joinpath(ROOT_DIR, "output", "sweep_output")

SCENARIO1_NO_AUTORUN = true
include(joinpath(CODE_DIR, "6_Shrinking_Horizon_main.jl"))

const MODES = [:normal, :near_mean, :high, :low, :spread_wide, :live_data]
const PLANTS = [:sampled] # :mean can be added if desired, but it is not a real plant and will produce a lot of missed hours

results = NamedTuple[]

for mode in MODES
    label = "input_$(mode)_sampled"
    out_dir = joinpath(OUT_ROOT, label)
    println("\n", "=" ^ 60)
    println("Running: ", label)
    println("=" ^ 60)
    try
        res = run_scenario_1(mode = mode, input_dir = INPUT_DIR, out_dir = out_dir,
                              n_day_run = 1, n_scenarios = DEFAULT_N_SCENARIOS, important_only = true,
                              time_limit_sec = 1200.0,
                              run_regression = false)
                              # run_regression = (mode == first(MODES)))
        d = res.d
        # Realized total cost, computed the same way 5_Output.jl's _cost_components
        # does: this is what ACTUALLY happened (mode-dependent), not what any one
        # window's plan assumed. Unlike Approach 0, there is no single planning
        # objective to report here -- Approach 2 solves one window per interval
        # (about 96 a day), each pricing only its own remaining horizon, so no
        # single res.solve_log.objective value stands for "the day's plan".
        # mean_gap_pct below is the average MIP gap across every window that
        # solved (NaN/held windows excluded), as a rough signal of solver health
        # across the whole run -- it is not comparable to Approach 0's single
        # per-day gap.
        energy_cost = res.total_cost
        carbon_cost = (d.carbon_price_per_ton / 1000.0) * res.total_co2
        ncd_cost    = d.lambda_demand_NC * res.nc_peak
        opd_cost    = d.lambda_demand_OP * res.op_peak
        missed_cost = d.rho_miss * res.missed
        realized_total = energy_cost + carbon_cost + ncd_cost + opd_cost +
                          missed_cost + res.labour_cost + res.shortfall_penalty_cost
        gaps = filter(!isnan, res.solve_log.gap_percent)
        mean_gap_pct = isempty(gaps) ? NaN : sum(gaps) / length(gaps)
        push!(results, (; label, status = "OK",
                         realized_cost = realized_total,
                         mean_gap_pct = mean_gap_pct,
                         missed_hours = res.missed,
                         n_infeasible = res.n_infeasible,
                         n_capped = res.n_capped))
    catch err
        println("  FAILED: ", sprint(showerror, err))
        push!(results, (; label, status = "FAILED", realized_cost = NaN,
                         mean_gap_pct = NaN, missed_hours = NaN,
                         n_infeasible = -1, n_capped = -1))
    end
end

println("\n", "=" ^ 78)
println("SWEEP SUMMARY  (", length(results), " runs, output under ", OUT_ROOT, ")")
println("=" ^ 78)
@printf("%-24s %-8s %-14s %-12s %-12s %-14s %s\n",
        "Run", "Status", "Realized Cost", "Mean Gap %", "Missed (h)", "n_infeasible", "n_capped")
for r in results
    @printf("%-24s %-8s %-14.4g %-12.3g %-12.4g %-14s %s\n",
            r.label, r.status, r.realized_cost, r.mean_gap_pct, r.missed_hours,
            r.n_infeasible, r.n_capped)
end