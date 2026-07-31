# #############################################################################
# Receding_Horizon_main.jl  —  TOP-LEVEL DRIVER (thin)
# -----------------------------------------------------------------------------
# WHAT THIS PROGRAM DOES (the 30-second version)
# We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
# fleet of electric excavators (Construction EVs, "CEVs"). This is a MULTI-DAY
# RECEDING horizon with a FIXED-LENGTH rolling window: it simulates `n_days`
# reported days plus one dropped BUFFER day, and every 15 minutes it (1) OPTIMISES
# a MILP over EXACTLY the next 24h (nothing more, nothing less - always the same
# length, whatever time of day it is) using a power model that is fitted ONCE and
# held FIXED, and (2) APPLIES only the first interval to the stochastic plant
# (FORK B: realized power is drawn from the fixed posterior; the model is never
# re-fitted online). Every window's own 24h-out terminal point requires the MCS/CEV
# batteries to be back at their initial energy level, so recovery is guaranteed on
# a rolling basis rather than only on the last day. The MCS/CEV state, and the
# shared applied-activity history, carry over continuously - nothing is reset at a
# day boundary. Work quotas beyond the given data (the buffer day, and beyond)
# repeat the LAST defined day's quota. The buffer day is dropped from reported
# KPIs so the wrap-up never lands on a reported day.
#
# This file is only the ORCHESTRATOR (step 6). The work lives in focused modules,
# named in include / dependency order:
#   1_Common.jl            shared helpers (travel steps, clock/multi-day labels,
#                          step plots) PLUS the Bayesian activity-power estimator
#                          PLUS the Bayesian activity-power estimator
#   0_Regression.jl        STEP 0 (pure Julia; needs Common): reads the soil .xlsx
#                          task files, fits the Bayesian power model, and refreshes
#                          parameters.csv (mu + per-activity sigma) BEFORE the MPC.
#                          Runs by default in :input mode; skip via run_regression=
#                          false. Fail-soft if XLSX.jl / the data folder is absent.
#   2_DataLoader.jl        load :synthetic / :input data; multi-day quotas + day_end
#   3_MCSModel.jl          the fixed-length 24h rolling window MILP (overnight
#                          recharge is just part of that window's own terminal target)
#   4_MPCLoop.jl           the multi-day closed loop (optimise + fixed-model plant + apply)
#   5_Output.jl            ALL on-disk artefacts: v4_real-style STEP figures (+ CSVs)
#                          PLUS KPI/cost CSVs, worker schedule, per-day replan grids
# #############################################################################

using Printf

# ---- include the modules in dependency order (Common first) ----
const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "0_Regression.jl"))
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_MCSModel.jl"))
include(joinpath(_CODE_DIR, "4_MPCLoop.jl"))
include(joinpath(_CODE_DIR, "5_Output.jl"))

using .DataLoader: load_data
using .MPCLoop: run_mpc
using .Output: write_outputs

# Default folder holding the soil task-recording .xlsx files (step-0 regression).
const _DEFAULT_REGRESSION_DATA_DIR = raw"C:\Users\shubh\Desktop\Bayesian Regression"

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
                          mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          n_days::Union{Nothing, Int} = nothing,   # days to KEEP in the results
                          run_regression::Bool = true,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          seed::Int = 1)
    # Resolve the input folder (with a couple of legacy fallbacks).
    if mode == :input && !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end
    
     # ---- STEP 0: (re)fit the Bayesian power model and refresh parameters.csv ----
    # Pure-Julia regression over the soil .xlsx files (no Python). Only meaningful
    # in :input mode (synthetic builds its powers in code). Runs by default; pass
    # run_regression=false to reuse the last-fitted parameters.csv.
    if mode == :input && run_regression
        Regression.run_regression(regression_data_dir, joinpath(input_dir, "parameters.csv");
                                  mcmc_samples = regression_samples, nchains = regression_chains)
    end

    d = load_data(mode; input_dir = input_dir)

    res = run_mpc(d; time_limit_sec = time_limit_sec, multi_activity = multi_activity,
                     require_site_visit = require_site_visit, single_visit_per_site = single_visit_per_site,
                     mcmc_samples = mcmc_samples,
                     n_days = n_days, seed = seed)

    _print_kpis(res)

    write_outputs(res, out_dir)

    println("\nResults written to: $(abspath(out_dir))")
    println("  Figures (v4_real style, kept days): 01..07 + mcs_<m>_power_profile + 11_power_estimate_convergence")
    println("  Reports: 08 cost/emissions, 09 KPI, 10 mip-convergence, closed_loop_trajectory,")
    println("           worker_schedule, replan_grids/day*/*.csv+*.html")
    return res.log
end

# Human-readable KPI block (kept days only).
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
end

# Auto-run unless a harness defines SCENARIO1_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
