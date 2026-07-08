# #############################################################################
# Receding_Horizon_main.jl  —  TOP-LEVEL DRIVER (thin)
# -----------------------------------------------------------------------------
# WHAT THIS PROGRAM DOES (the 30-second version)
# We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
# fleet of electric excavators (Construction EVs, "CEVs"). This is the MULTI-DAY,
# CROSS-DAY RECEDING horizon: it simulates `n_days` reported days plus one dropped
# BUFFER day, and every 15 minutes it (1) OPTIMISES a MILP over the rest of today
# plus `lookahead_days` future daytime blocks, and (2) LEARNS the activity powers
# from measured energy. Only the first interval is applied; nights recharge the
# MCS to full while the CEV battery and unfinished work carry over. The buffer
# day is dropped so the CEV energy-neutral wrap-up never lands on a reported day.
#
# This file is only the ORCHESTRATOR. The work lives in focused modules:
#   Common.jl            shared helpers (travel steps, clock labels, step plots)
#   DataLoader.jl        load :synthetic / :input data; infer day_end_hour
#   BayesianEstimator.jl the online activity-power learner
#   MCSModel.jl          the CROSS-DAY window MILP + deterministic overnight charge
#   MPCLoop.jl           the multi-day closed loop (optimise + learn + apply)
#   Plotting.jl          the v4_real-style STEP figures (+ CSVs)
#   Reporting.jl         KPI / cost CSVs, worker schedule, per-day replan grids
# #############################################################################

using Printf

const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "Common.jl"))
include(joinpath(_CODE_DIR, "DataLoader.jl"))
include(joinpath(_CODE_DIR, "BayesianEstimator.jl"))
include(joinpath(_CODE_DIR, "MCSModel.jl"))
include(joinpath(_CODE_DIR, "MPCLoop.jl"))
include(joinpath(_CODE_DIR, "Plotting.jl"))
include(joinpath(_CODE_DIR, "Reporting.jl"))

using .DataLoader: load_data
using .MPCLoop: run_mpc
using .Plotting: write_trajectory_figures
using .Reporting: write_reports

# =============================================================================
# ENTRY POINT
# =============================================================================
# Load the chosen dataset, run the multi-day cross-day receding loop, print the
# KPI summary (kept days only), and write the full figure + report set.
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          time_limit_sec::Float64 = 60.0,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          refit_every::Int = 8, mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          soft_prec::Bool = false, soft_pace::Bool = false,
                          soft_term::Bool = false, term_tol::Float64 = 0.1,
                          n_days::Union{Nothing, Int} = nothing,   # days to KEEP in the results
                          lookahead_days::Int = 1,                 # cross-day window depth
                          seed::Int = 1)
    if mode == :input && !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    d = load_data(mode; input_dir = input_dir)

    res = run_mpc(d; time_limit_sec = time_limit_sec, multi_activity = multi_activity,
                     require_site_visit = require_site_visit, single_visit_per_site = single_visit_per_site,
                     refit_every = refit_every, mcmc_samples = mcmc_samples,
                     soft_prec = soft_prec, soft_pace = soft_pace,
                     soft_term = soft_term, term_tol = term_tol,
                     n_days = n_days, lookahead_days = lookahead_days, seed = seed)

    _print_kpis(res)

    write_trajectory_figures(res, out_dir)
    write_reports(res, out_dir)

    println("\nResults written to: $(abspath(out_dir))")
    println("  Figures (v4_real style, kept days): 01..07 + mcs_<m>_power_profile + 11_power_estimate_convergence")
    println("  Reports: 08 cost/emissions, 09 KPI, 10 mip-convergence, closed_loop_trajectory,")
    println("           overnight_mcs_charge_day*.csv, worker_schedule, replan_grids/day*/*.csv+*.html")
    return res.log
end

function _print_kpis(res)
    d = res.d
    println("\n==== Scenario 1 RECEDING-horizon KPIs (kept days 1..$(res.n_days_keep); buffer day dropped) ====")
    @printf("Total grid energy   : %.2f kWh\n", res.total_energy)
    @printf("Total energy cost   : \$%.2f\n", res.total_cost)
    res.total_co2 > 1e-9 && @printf("Total CO2 emissions : %.2f kg\n", res.total_co2)
    @printf("NC peak demand      : %.2f kW\n", res.nc_peak)
    @printf("On-peak demand      : %.2f kW\n", res.op_peak)
    @printf("Missed work (hours) : %.2f\n", res.missed)
    @printf("Labour (towing)     : \$%.2f  (%.2f h in transit @ \$%.2f/h)\n",
            res.labour_cost, res.transit_intervals * d.delta_T, d.rho_labor)
    @printf("Overnight recharge  : %.2f kWh grid  ->  \$%.2f (cheapest hours, kept days)\n",
            res.overnight_energy, res.overnight_cost)
end

# Auto-run unless a harness defines SCENARIO1_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
