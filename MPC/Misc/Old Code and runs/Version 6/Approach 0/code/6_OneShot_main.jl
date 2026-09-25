# #############################################################################
# 6_OneShot_main.jl  —  TOP-LEVEL DRIVER for Approach 0 (thin)
# -----------------------------------------------------------------------------
# WHAT APPROACH 0 IS
# The "commit once" baseline: at 08:00, solve ONE whole-day MILP over the full
# 24 h horizon (no shrinking, no replanning), then execute that fixed plan
# open-loop for the rest of the day against the (possibly stochastic) plant.
# Contrast with Approach 1 (this repo's shrinking-horizon MPC) and Approach 2
# (its stochastic/scenario-based sibling), which both re-solve every 15
# minutes. Approach 0 exists to give those two something to be measured
# against: the gap between Approach 0 and Approach 1/2 is the value of
# re-planning.
#
# This file is only the ORCHESTRATOR. The work lives in focused modules, named
# in include / dependency order:
#   1_Common.jl    shared helpers (travel steps, clock labels, the detailed-
#                  output log structs for CEV and MCS)
#   2_DataLoader.jl load :synthetic / :input data (full 24 h horizon)
#   3_MCSModel.jl  the single 24 h window MILP (identical to Approach 1/2's)
#   4_OneShot.jl   the one-shot executor (the whole of Approach 0)
#   5_Output.jl    CSV outputs: detailed plan/realized (CEV + MCS) + KPI tables
# #############################################################################

using Printf
using Random

# ---- include the modules in dependency order (Common first) ----
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

# =============================================================================
# ENTRY POINT
# =============================================================================
# Load the chosen dataset, run Approach 0, print the KPI summary, and write
# the detailed CSVs + KPI table(s) to output/<mode>/.
function run_scenario_0(; mode::Symbol = :normal,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          # SOLVER TIME LIMIT: seconds the solver may spend on the single
                          # whole-day window MILP each day. Defaults to no limit (solve to
                          # the MIP gap); pass a finite value to shorten it.
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          # PLANT MODE for the fixed plan's execution:
                          #   :sampled  realized power = the next unused draw from the shared
                          #             pool, so the fixed plan drifts with no feedback to
                          #             correct it. >>> Use this for normal / headline runs. <<<
                          #   :mean     realized power pinned to the same mu the MILP planned
                          #             on, so realized == planned EXACTLY and the KPIs ARE the
                          #             whole-day MILP's own optimum (deterministic reference).
                          plant::Symbol = :sampled,
                          n_day_run::Int = 1,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          detailed_output::Bool = true,
                          seed::Int = 1)
    # Resolve the input folder (mirrors Approach 1/2's own fallback).
    if mode != :synthetic && !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    load_mode = mode in (:input, :synthetic) ? mode : :input
    d = load_data(load_mode; input_dir = input_dir)

    mkpath(out_dir)

    # ---- power-sample pool: sized for the WHOLE multi-day run, not one day ----
    nK_day = length(collect(d.K))
    n_samples = nK_day * n_day_run + 5
    pool = if mode == :live_data
        live_values = load_live_powers(input_dir)
        draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
    else
        draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                 n_samples = n_samples, rng = MersenneTwister(seed), mode = mode)
    end

    res = run_one_shot(d, pool; time_limit_sec = time_limit_sec,
                       multi_activity = multi_activity, require_site_visit = require_site_visit,
                       single_visit_per_site = single_visit_per_site,
                       plant = plant, n_day_run = n_day_run, seed = seed,
                       detailed_output = detailed_output)

    print_kpis(res)
    if detailed_output
        write_detailed_output(res, out_dir)
        write_kpi_summary(res, d, out_dir; day_snapshots_df = res.day_snapshots_df, solve_log = res.solve_log)
        println("\nResults written to: $(abspath(out_dir))")
        println("  A0_plan_full.csv, A0_realized_tuple.csv               (CEV, per 15-min step)")
        println("  A0_MCS_plan_full.csv, A0_MCS_realized_tuple.csv       (MCS, per 15-min step)")
        println("  A0_kpi_summary.csv                                    (whole-run KPI table)")
        n_day_run > 1 && println("  A0_kpi_summary_by_day.csv                             (Day1..Day$(n_day_run) + Overall)")
    else
        println("\n(detailed_output = false — no CSVs written; pass detailed_output = true to get them)")
    end
    return res
end

# Auto-run unless a harness defines SCENARIO0_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO0_NO_AUTORUN) && SCENARIO0_NO_AUTORUN)
    run_scenario_0()
end
