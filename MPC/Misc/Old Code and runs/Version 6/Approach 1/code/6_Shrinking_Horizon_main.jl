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
#                          PLUS the Bayesian activity-power estimator, PLUS the
#                          detailed-output log structs (CEV and MCS)
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
#
# This is Approach 1 ON ITS OWN — it no longer runs or reports on Approach 0.
# For an Approach 0 vs Approach 1 vs Approach 2 comparison, use
# ../Comparison_A0_A1_A2/Code/ instead, which solves all three and produces the
# comparison figures/CSVs. That's also where all image/plot output now lives
# for a cross-approach comparison; this file's own `write_outputs` only ever
# draws Approach 1's own single-run figures.
# #############################################################################

using Printf
using Random

# ---- include the modules in dependency order (Common first) ----
const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "0_Regression.jl"))   # after Common (uses ..Common)
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_MCSModel.jl"))
include(joinpath(_CODE_DIR, "4_MPCLoop.jl"))
include(joinpath(_CODE_DIR, "5_Output.jl"))

using .DataLoader: load_data, load_live_powers
using .Common: draw_activity_power_pool, draw_activity_power_pool_live
using .MPCLoop: run_mpc
using .Output: write_outputs, write_detailed_output

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
# Load the chosen dataset, run the shrinking-horizon closed loop, print the KPI
# summary, and write the full figure + report set to output/<mode>/.
#
# NOTE: `dataset` (:input / :synthetic — WHICH data to load) and `mode`
# (:normal / :high / :low / :near_mean / :live_data — HOW the simulated
# plant's realized power is drawn) are two separate, independent choices.
# They used to share the name `mode` for both, which is not valid Julia (a
# function cannot declare the same keyword argument name twice) — `dataset`
# is the fix, kept distinct from `mode` so it matches the convention the
# Comparison_A0_A1_A2 drivers already use (`mode`/`modes` = draw mode only).
function run_scenario_1(; dataset::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,
                          # SOLVER TIME LIMIT (control point #1, the top-level knob). Seconds the
                          # solver may spend on EACH window MILP (every 15-min window). This flows
                          # through to run_mpc -> build_window_model -> set_time_limit_sec
                          # (3_MCSModel.jl). Defaults to NO LIMIT (solve to the MIP gap); pass a
                          # finite value to shorten it, e.g. run_scenario_1(dataset = :input, time_limit_sec = 60.0)
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(dataset)),
                          run_regression::Bool = true,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          # PLANT MODE: how the simulated plant's realized power is drawn.
                          # :normal (default, unchanged) -> unbiased Bayesian draws from
                          # Normal(mu,sd); :high/:low/:near_mean/:spread_wide -> the same
                          # Bayesian pool biased per 1_Common.jl's "DRAW MODE" doc;
                          # :live_data -> draws instead from real recorded values in
                          # data/input_data/live_powers.csv (see DataLoader.load_live_powers
                          # / Common.draw_activity_power_pool_live). One unified `mode` name
                          # is used everywhere in this codebase for this choice -- see also
                          # the Comparison_A0_A1_A2 drivers' `mode`/`modes` arguments.
                          mode::Symbol = :normal,
                          detailed_output::Bool = false,
                          seed::Int = 1)
    # Resolve the input folder (with a couple of legacy fallbacks).
    if dataset == :input && !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    # ---- STEP 0: (re)fit the Bayesian power model and refresh parameters.csv ----
    # Pure-Julia regression over the soil .xlsx files (no Python). Only meaningful
    # in :input mode (synthetic builds its powers in code). Runs by default; pass
    # run_regression=false to reuse the last-fitted parameters.csv.
    if dataset == :input && run_regression
        Regression.run_regression(regression_data_dir, joinpath(input_dir, "parameters.csv");
                                  mcmc_samples = regression_samples, nchains = regression_chains)
    end

    d = load_data(dataset; input_dir = input_dir)

    return _with_console_log(out_dir) do
        # ---- power-sample pool (Common.jl): generated ONCE, from the frozen
        # d.prior_mu/d.prior_sigma. mode = :normal -> unbiased draws (the
        # original behaviour, unchanged). See draw_activity_power_pool's
        # "DRAW MODE" doc in 1_Common.jl for the other 4 sensitivity-sweep
        # modes (:near_mean/:high/:low/:spread_wide).
        pool = if mode == :live_data
            live_values = load_live_powers(input_dir)
            draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
        else
            draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                     n_samples = 20, rng = MersenneTwister(seed),
                                     mode = mode)
        end

        # ---- APPROACH 1: closed-loop MPC against the stochastic pool ----
        res = run_mpc(d, pool; shrinking = shrinking, H = H, time_limit_sec = time_limit_sec,
                         multi_activity = multi_activity, require_site_visit = require_site_visit,
                         single_visit_per_site = single_visit_per_site,
                         mcmc_samples = mcmc_samples, plant = :sampled, seed = seed,
                         detailed_output = detailed_output)

        _print_kpis(res)

        write_outputs(res, out_dir)
        detailed_output && write_detailed_output(res, out_dir)

        println("\nResults written to: $(abspath(out_dir))")
        println("  Figures (v4_real style): 01..09 (09 = per-MCS power profiles)")
        println("  Reports: 08 KPI, replan_grids/*.csv+*.html,")
        println("           plan_vs_actual.html + plan_vs_actual_costs.png  (08:00 plan vs realised, financial)")
        println("           plan_vs_actual_activity.png, plan_vs_actual_side_by_side.html, plan_vs_actual_by_entity.html  (ACTIVITY)")
        detailed_output && println("           plan_full.csv, realized_tuple.csv, MCS_plan_full.csv, MCS_realized_tuple.csv  (per 15-min step)")
        println("  Console log: run_log.txt")
        println("\nFor an Approach 0 vs 1 vs 2 comparison, run ../Comparison_A0_A1_A2/Code/ instead.")
        res.log
    end
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
