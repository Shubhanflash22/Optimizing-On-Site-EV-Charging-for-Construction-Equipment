# #############################################################################
# run_all_modes.jl  -  standalone sweep script, not part of the pipeline
# -----------------------------------------------------------------------------
# Runs Approach 0 once for every combination of mode (6 options) and plant
# (2 options) on ONE input dataset -- 12 runs total -- saving each run's CSVs
# into its own folder named input_<mode>_<plant>.
#
# PLACE THIS FILE IN THE code/ FOLDER, next to 6_OneShot_main.jl. All paths
# below are then derived automatically from @__DIR__ (this file's own
# location), the same trick 6_OneShot_main.jl itself uses -- nothing here is
# hardcoded to a specific machine, username, or drive letter, so the whole
# "Approach 0" folder can be moved or copied anywhere and this still works
# unmodified, as long as the code/, data/input_data/, and output/ folders
# stay siblings of each other (matching the README's documented layout).
#
# Sets SCENARIO0_NO_AUTORUN itself, so including 6_OneShot_main.jl here does
# not also trigger an extra default run on top of the 12 in the sweep.
#
# From the Julia REPL:
#   include("run_all_modes.jl")
# #############################################################################

using Printf

const CODE_DIR  = @__DIR__
const ROOT_DIR  = dirname(CODE_DIR)
const INPUT_DIR = joinpath(ROOT_DIR, "data", "input_data")
const OUT_ROOT  = joinpath(ROOT_DIR, "output", "sweep_output")

SCENARIO0_NO_AUTORUN = true
include(joinpath(CODE_DIR, "6_OneShot_main.jl"))

const MODES  = [:normal, :near_mean, :high, :low, :spread_wide, :live_data]
const PLANTS = [:sampled] # :mean can be added if desired, but it is not a real plant and will produce a lot of missed hours

results = NamedTuple[]

for mode in MODES, plant in PLANTS
    label = "input_$(mode)_$(plant)"
    out_dir = joinpath(OUT_ROOT, label)
    println("\n", "=" ^ 60)
    println("Running: ", label)
    println("=" ^ 60)
    try
        res = run_scenario_0(mode = mode, input_dir = INPUT_DIR, out_dir = out_dir,
                              plant = plant, n_day_run = 1, detailed_output = true)
        d = res.d
        # Realized total cost, computed the same way 5_Output.jl's _cost_components
        # does: this is what ACTUALLY happened (mode-dependent), not what the
        # planner assumed up front (res.solve_log.objective is planning-time only,
        # and is the SAME for every mode by design, since planning always uses
        # d.prior_mu regardless of mode -- it is not a bug that it doesn't vary).
        energy_cost = res.total_cost
        carbon_cost = (d.carbon_price_per_ton / 1000.0) * res.total_co2
        ncd_cost    = d.lambda_demand_NC * res.nc_peak
        opd_cost    = d.lambda_demand_OP * res.op_peak
        missed_cost = d.rho_miss * res.missed
        realized_total = energy_cost + carbon_cost + ncd_cost + opd_cost +
                          missed_cost + res.labour_cost + res.shortfall_penalty_cost
        push!(results, (; label, status = "OK",
                         planned_cost = res.solve_log.objective[1],
                         realized_cost = realized_total,
                         gap_pct = res.solve_log.gap_percent[1],
                         missed_hours = res.missed,
                         n_capped = res.n_capped))
    catch err
        println("  FAILED: ", sprint(showerror, err))
        push!(results, (; label, status = "FAILED", planned_cost = NaN,
                         realized_cost = NaN, gap_pct = NaN, missed_hours = NaN,
                         n_capped = -1))
    end
end

println("\n", "=" ^ 78)
println("SWEEP SUMMARY  (", length(results), " runs, output under ", OUT_ROOT, ")")
println("=" ^ 78)
@printf("%-28s %-8s %-14s %-14s %-10s %-12s %s\n",
        "Run", "Status", "Planned Cost", "Realized Cost", "Gap %", "Missed (h)", "n_capped")
for r in results
    @printf("%-28s %-8s %-14.4g %-14.4g %-10.3g %-12.4g %s\n",
            r.label, r.status, r.planned_cost, r.realized_cost, r.gap_pct, r.missed_hours, r.n_capped)
end
println("\nNote: Planned Cost is the same across every mode by design (planning always")
println("uses parameters.csv, regardless of mode). Realized Cost is where the mode's")
println("effect actually shows up -- that's the number worth comparing across rows.")
