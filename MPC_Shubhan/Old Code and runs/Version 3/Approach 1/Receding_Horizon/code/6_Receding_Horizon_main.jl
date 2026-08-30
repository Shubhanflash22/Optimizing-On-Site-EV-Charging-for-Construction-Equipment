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
using Random

# ---- include the modules in dependency order (Common first) ----
const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "0_Regression.jl"))
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_MCSModel.jl"))
include(joinpath(_CODE_DIR, "4_MPCLoop.jl"))
include(joinpath(_CODE_DIR, "5_Output.jl"))

using .DataLoader: load_data
using .Common: draw_activity_power_pool
using .MPCLoop: run_mpc, run_one_shot
using .Output: write_outputs, write_approach_comparison

# Default folder holding the soil task-recording .xlsx files (step-0 regression).
const _DEFAULT_REGRESSION_DATA_DIR = raw"C:\Users\shubh\Desktop\Bayesian Regression"

# -----------------------------------------------------------------------------
# CONSOLE LOG CAPTURE: mirror everything printed (println/@printf to stdout,
# @warn to stderr) to a run_log.txt file under out_dir, in ADDITION to the
# terminal -- nothing currently printed changes, it's just also saved.
# -----------------------------------------------------------------------------
function _with_console_log(f, out_dir)
    mkpath(out_dir)
    log_path = joinpath(out_dir, "run_log.txt")

    open(log_path, "w") do logfile
        pipe = Pipe()
        orig_stdout = stdout
        orig_stderr = stderr

        # Asynchronously read from the pipe and write to both console and file
        tee_task = @async begin
            while !eof(pipe)
                data = readavailable(pipe)
                write(orig_stdout, data)
                write(logfile, data)
                flush(orig_stdout)
                flush(logfile)
            end
        end

        try
            redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    f()
                end
            end
        finally
            # Close the writing end of the pipe and wait for the tee task to finish
            close(pipe.in)
            wait(tee_task)
        end
    end
end

# =============================================================================
# ENTRY POINT
# =============================================================================
# Load the chosen dataset, run the multi-day cross-day receding loop, print the
# KPI summary (kept days only), and write the full figure + report set.
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          # APPROACH 0 PLANT MODE. Approach 0 solves each kept day's
                          # 24 h window once at that day's 08:00 and replays it
                          # open-loop under ONE plant:
                          #   :sampled  realized power = the next unused draw from the
                          #             shared pool, so each day's fixed plan drifts
                          #             with no feedback until the next day's solve
                          #             picks up the real state (stochastic baseline).
                          #   :mean     realized power pinned to the same mu the MILP
                          #             planned on, single planned activity per
                          #             interval, so realized == planned EXACTLY within
                          #             each day and the KPIs are the per-day MILPs' own
                          #             optima chained through the real carry-over
                          #             (deterministic baseline; consumes no pool
                          #             samples, needs no seed).
                          # Exactly one runs per call -- pick the baseline you want to
                          # compare Approach 1 against. Approach 1 is always :sampled.
                          approach0_plant::Symbol = :sampled,
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

    return _with_console_log(out_dir) do
        # Resolve how many days Approach 1 will actually keep (mirrors the same
        # resolution inside run_mpc/run_one_shot) so the SHARED pool is sized
        # generously enough: Approach 1 also runs one dropped BUFFER day, so it
        # needs one more day's worth of occurrences than the kept-day count.
        n_days_keep_resolved = n_days === nothing ? d.n_days : max(1, n_days)

        # ---- SHARED power-sample pool (Common.jl): generated ONCE, from the
        # frozen d.prior_mu/d.prior_sigma, so Approach 0 and Approach 1 draw
        # their realized plant power from the exact same underlying samples.
        # n_samples scales with the number of days in play (20/day/activity is
        # the single-day rule of thumb; the buffer day means Approach 1 needs
        # up to n_days_keep+1 days' worth).
        # mode = :normal -> unbiased draws (unchanged); see 1_Common.jl's
        # "DRAW MODE" doc for the 4 sensitivity-sweep modes.
        pool = draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                        n_samples = 20 * (n_days_keep_resolved + 1),
                                        rng = MersenneTwister(seed),
                                        mode = :normal)

        # ---- APPROACH 0: one-shot PER DAY, executed open-loop for that day ----
        # approach0_plant picks WHICH baseline this is:
        #   :mean     -> realized == planned within each day; the KPIs ARE the per-day
        #                MILP optima (deterministic reference, no sampling anywhere).
        #   :sampled  -> the same per-day plans drifting under the stochastic pool,
        #                with no intra-day feedback.
        approach0_plant in (:sampled, :mean) ||
            error("run_scenario_1: approach0_plant must be :sampled or :mean, got :$approach0_plant")
        res0 = run_one_shot(d, pool; plant = approach0_plant, time_limit_sec = time_limit_sec,
                            multi_activity = multi_activity, require_site_visit = require_site_visit,
                            single_visit_per_site = single_visit_per_site,
                            n_days = n_days, seed = seed)

        # ---- APPROACH 1: closed-loop MPC against the stochastic pool ----
        res = run_mpc(d, pool; plant = :sampled, time_limit_sec = time_limit_sec, multi_activity = multi_activity,
                         require_site_visit = require_site_visit, single_visit_per_site = single_visit_per_site,
                         mcmc_samples = mcmc_samples,
                         n_days = n_days, seed = seed)

        _print_kpis(res)
        _print_approach_summary(res0, res)

        write_outputs(res, out_dir)
        write_approach_comparison(res0, res, out_dir)

        println("\nResults written to: $(abspath(out_dir))")
        println("  Figures (v4_real style, kept days): 01..07 + mcs_<m>_power_profile + 11_power_estimate_convergence")
        println("  Reports: 08 cost/emissions, 09 KPI, 10 mip-convergence, closed_loop_trajectory,")
        println("           worker_schedule, replan_grids/day*/*.csv+*.html")
        println("           approach0_vs_approach1.html  (Approach 0 one-shot/day vs Approach 1 closed-loop, totals + per-day;")
        println("                                         column labelled by A0's plant mode, plus run diagnostics)")
        println("  Console log: run_log.txt")
        res.log
    end
end

# Human-readable KPI block (kept days only).
# Two-run console summary. The label adapts to which baseline Approach 0 was run as,
# because that decides what the gap actually measures.
function _print_approach_summary(res0, res1)
    # TOTAL operating cost (energy + carbon + demand + missed + labour), not just the
    # energy term that res.total_cost holds -- same decomposition the HTML report uses.
    tot(r) = Output._cost_components(r).total
    a0_lbl = res0.plant === :mean ?
        "A0 one-shot/day, mean plant (MILP optima)" :
        "A0 one-shot/day, sampled plant (open loop)"
    gap_lbl = res0.plant === :mean ?
        "  A1 - A0 (drift + re-planning, net)      " :
        "  A1 - A0 (value of intra-day re-planning)"
    println("\n==== Approach comparison (both fully realised, kept days) ====")
    @printf("%s : \$%.2f  (%.2f h transit, %.2f h missed)\n",
            a0_lbl, tot(res0), res0.transit_intervals * res0.d.delta_T, res0.missed)
    @printf("A1 closed loop (re-planning)              : \$%.2f  (%.2f h transit, %.2f h missed)\n",
            tot(res1), res1.transit_intervals * res1.d.delta_T, res1.missed)
    @printf("%s: %+.2f  (negative = A1 cheaper)\n", gap_lbl, tot(res1) - tot(res0))
    res0.plant === :mean && println("  NOTE: A0 is DETERMINISTIC here, so this gap mixes plan drift with the",
                                    "\n        value of re-planning. Re-run with approach0_plant = :sampled to",
                                    "\n        separate the two.")
    nclamp = res0.n_capped + res1.n_capped
    nclamp > 0 && @printf("  NOTE: %d interval(s) had work capped by available CEV energy -- the\n           shortfall is reflected honestly in rem_dig/rem_load, not hidden.\n", nclamp)
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
end

# Auto-run unless a harness defines SCENARIO1_NO_AUTORUN = true first.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end
