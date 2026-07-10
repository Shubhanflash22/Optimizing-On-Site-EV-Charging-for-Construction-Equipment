# #############################################################################
# Shrinking_Horizon_main.jl  —  TOP-LEVEL DRIVER (thin)
# -----------------------------------------------------------------------------
# WHAT THIS PROGRAM DOES (the 30-second version)
# We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
# fleet of electric excavators (Construction EVs, "CEVs"). Over a work day the
# MCS drives around and tops the excavators up so none runs flat, while paying
# the least for electricity (time-of-use price + demand charges + carbon) and
# getting all the digging/loading work done. We do NOT know each activity's exact
# power draw, so every 15 minutes we (1) OPTIMISE a MILP over the remaining day
# and (2) LEARN the powers from measured energy — classic MPC with an online-
# learned model, applied one interval at a time (SHRINKING horizon).
#
# This file is only the ORCHESTRATOR (step 7). The work lives in focused modules,
# named in include / dependency order:
#   1_Common.jl            shared helpers (travel steps, clock labels, step plots)
#   2_DataLoader.jl        load :synthetic / :input data; infer day_end_hour
#   3_BayesianEstimator.jl the online activity-power learner
#   4_MCSModel.jl          the window MILP + deterministic overnight charge
#   5_MPCLoop.jl           the closed loop (optimise + learn + apply)
#   6_Output.jl            ALL on-disk artefacts: v4_real-style STEP figures (+ CSVs)
#                          PLUS KPI/cost CSVs, worker schedule, replan grids
# #############################################################################

using Printf

# ---- include the modules in dependency order (Common first) ----
const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_BayesianEstimator.jl"))
include(joinpath(_CODE_DIR, "4_MCSModel.jl"))
include(joinpath(_CODE_DIR, "5_MPCLoop.jl"))
include(joinpath(_CODE_DIR, "6_Output.jl"))

using .DataLoader: load_data
using .MPCLoop: run_mpc
using .Output: write_outputs

# =============================================================================
# ENTRY POINT
# =============================================================================
# Load the chosen dataset, run the shrinking-horizon closed loop, print the KPI
# summary, and write the full figure + report set to output/<mode>/.
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,
                          time_limit_sec::Float64 = 60.0,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          refit_every::Int = 8, mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          soft_prec::Bool = false, soft_pace::Bool = false,
                          soft_term::Bool = false, term_tol::Float64 = 0.1,
                          seed::Int = 1)
    # Resolve the input folder (with a couple of legacy fallbacks).
    if mode == :input && !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    d = load_data(mode; input_dir = input_dir)

    res = run_mpc(d; shrinking = shrinking, H = H, time_limit_sec = time_limit_sec,
                     multi_activity = multi_activity, require_site_visit = require_site_visit,
                     single_visit_per_site = single_visit_per_site,
                     refit_every = refit_every, mcmc_samples = mcmc_samples,
                     soft_prec = soft_prec, soft_pace = soft_pace,
                     soft_term = soft_term, term_tol = term_tol, seed = seed)

    _print_kpis(res)

    write_outputs(res, out_dir)

    println("\nResults written to: $(abspath(out_dir))")
    println("  Figures (v4_real style): 01..07 + mcs_<m>_power_profile + 11_power_estimate_convergence")
    println("  Reports: 08 cost/emissions, 09 KPI, 10 mip-convergence, closed_loop_trajectory,")
    println("           overnight_mcs_charge, worker_schedule, replan_grids/*.csv+*.html")
    return res.log
end

# Human-readable KPI block (mirrors the previous summary).
function _print_kpis(res)
    d = res.d
    println("\n==== Scenario 1 closed-loop KPIs (daytime horizon 08:00-$(Int(round(d.day_end_hour))):00) ====")
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
    @printf("Overnight recharge  : %.2f kWh grid  ->  \$%.2f (cheapest hours)\n",
            res.overnight_energy, res.overnight_cost)
end

# Auto-run unless a harness defines SCENARIO1_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
