# #############################################################################
# Shrinking_Horizon_main.jl  —  TOP-LEVEL DRIVER (thin)
# -----------------------------------------------------------------------------
# WHAT THIS PROGRAM DOES (the 30-second version)
# We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
# fleet of electric excavators (Construction EVs, "CEVs"). Over a work day the
# MCS drives around and tops the excavators up so none runs flat, while paying
# the least for electricity (time-of-use price + demand charges + carbon) and
# getting all the digging/loading work done. We do NOT know each activity's exact
# power draw, so we fit a Bayesian power model ONCE and then every 15 minutes we
# (1) OPTIMISE a MILP over the remaining day using that fixed model and (2) APPLY
# only the first interval to the stochastic plant — classic MPC applied one
# interval at a time (SHRINKING horizon).
#
# This file is only the ORCHESTRATOR (step 6). The work lives in focused modules,
# named in include / dependency order:
#   1_Common.jl            shared helpers (travel steps, clock labels, step plots)
#                          PLUS the Bayesian activity-power estimator
#   0_Regression.jl        STEP 0 (pure Julia; needs Common): reads the soil .xlsx
#                          task files, fits the Bayesian power model, and refreshes
#                          parameters.csv (mu + per-activity sigma) BEFORE the MPC.
#                          Runs by default in :input mode; skip via run_regression=
#                          false. Fail-soft if XLSX.jl / the data folder is absent.
#   2_DataLoader.jl        load :synthetic / :input data (full 24 h horizon)
#   3_MCSModel.jl          the single 24 h window MILP (Eq. 1-13)
#   4_MPCLoop.jl           the closed loop (optimise + fixed-model plant + apply)
#   5_Output.jl            ALL on-disk artefacts: v4_real-style STEP figures (+ CSVs)
#                          PLUS KPI/cost CSVs, worker schedule, replan grids
# #############################################################################

using Printf

# ---- include the modules in dependency order (Common first) ----
const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "0_Regression.jl"))   # after Common (uses ..Common)
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
# Load the chosen dataset, run the shrinking-horizon closed loop, print the KPI
# summary, and write the full figure + report set to output/<mode>/.
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,
                          # SOLVER TIME LIMIT (control point #1, the top-level knob). Seconds the
                          # solver may spend on EACH 15-min window MILP. This flows through to
                          # run_mpc -> build_window_model -> set_time_limit_sec (3_MCSModel.jl).
                          #   e.g.  run_scenario_1(mode = :input, time_limit_sec = Inf)
                          # Pass Inf to REMOVE the limit and solve every window to the MIP gap.
                          time_limit_sec::Float64 = 60.0,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
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

    res = run_mpc(d; shrinking = shrinking, H = H, time_limit_sec = time_limit_sec,
                     multi_activity = multi_activity, require_site_visit = require_site_visit,
                     single_visit_per_site = single_visit_per_site,
                     mcmc_samples = mcmc_samples, seed = seed)

    _print_kpis(res)

    write_outputs(res, out_dir)

    println("\nResults written to: $(abspath(out_dir))")
    println("  Figures (v4_real style): 01..09 (09 = per-MCS power profiles)")
    println("  Reports: 08 KPI, replan_grids/*.csv+*.html,")
    println("           plan_vs_actual.html + plan_vs_actual_costs.png  (08:00 plan vs realised, financial)")
    println("           plan_vs_actual_activity.png, plan_vs_actual_side_by_side.html, plan_vs_actual_by_entity.html  (ACTIVITY)")
    return res.log
end

# Human-readable KPI block (mirrors the previous summary).
function _print_kpis(res)
    d = res.d
    println("\n==== Scenario 1 closed-loop KPIs (full 24 h horizon: 08:00 -> 08:00 next day) ====")
    @printf("Total grid energy   : %.2f kWh\n", res.total_energy)
    @printf("Total energy cost   : \$%.2f\n", res.total_cost)
    res.total_co2 > 1e-9 && @printf("Total CO2 emissions : %.2f kg\n", res.total_co2)
    @printf("NC peak demand      : %.2f kW\n", res.nc_peak)
    @printf("On-peak demand      : %.2f kW\n", res.op_peak)
    @printf("Missed work (hours) : %.2f\n", res.missed)
    @printf("Labour (towing)     : \$%.2f  (%.2f h in transit @ \$%.2f/h)\n",
            res.labour_cost, res.transit_intervals * d.delta_T, d.rho_labor)
    @printf("CEV SOE at horizon  : %s kWh (target %s)\n",
            string(round.(res.soe_cev_end, digits = 2)), string(round.(d.SOE_CEV_ini, digits = 2)))
    @printf("MCS SOE at horizon  : %s kWh (target %s)\n",
            string(round.(res.soe_mcs_end, digits = 2)), string(round.(d.SOE_MCS_ini, digits = 2)))
end

# Auto-run unless a harness defines SCENARIO1_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
