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
using .MPCLoop: run_mpc, run_one_shot
using .Output: write_outputs, write_approach_comparison

# Default folder holding the soil task-recording .xlsx files (step-0 regression).
const _DEFAULT_REGRESSION_DATA_DIR = normpath(joinpath(_CODE_DIR, "..", "..", "..", "..", "Bayesian Regression"))

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
function run_scenario_1(; mode::Symbol = :synthetic,
                          input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          shrinking::Bool = true, H::Int = 16,
                          # SOLVER TIME LIMIT (control point #1, the top-level knob). Seconds the
                          # solver may spend on EACH window MILP (every 15-min window for
                          # Approach 1, the single whole-day window for Approach 0). This flows
                          # through to run_mpc / run_one_shot -> build_window_model ->
                          # set_time_limit_sec (3_MCSModel.jl). Defaults to NO LIMIT (solve to the
                          # MIP gap); pass a finite value to shorten it, e.g.
                          #   run_scenario_1(mode = :input, time_limit_sec = 60.0)
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          require_site_visit::Bool = false,
                          single_visit_per_site::Bool = false,
                          # APPROACH 0 PLANT MODE. Approach 0 solves the whole-day MILP
                          # once at 08:00 and replays it open-loop under ONE plant:
                          #   :sampled  realized power = the next unused draw from the
                          #             shared pool, so the fixed plan drifts with no
                          #             feedback to correct it (stochastic baseline).
                          #   :mean     realized power pinned to the same mu the MILP
                          #             planned on, single planned activity per interval,
                          #             so realized == planned EXACTLY and the KPIs are
                          #             the whole-day MILP's own optimum (deterministic
                          #             baseline; consumes no pool samples, needs no seed).
                          # Exactly one runs per call -- pick the baseline you want to
                          # compare Approach 1 against.
                          approach0_plant::Symbol = :sampled,
                          mcmc_samples::Int = 500,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
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
        # ---- SHARED power-sample pool (Common.jl): generated ONCE, from the
        # frozen d.prior_mu/d.prior_sigma, so Approach 0 and Approach 1 draw
        # their realized plant power from the exact same underlying samples.
        # mode = :normal -> unbiased draws (the original behaviour, unchanged).
        # See draw_activity_power_pool's "DRAW MODE" doc in 1_Common.jl for the
        # other 4 sensitivity-sweep modes (:near_mean/:high/:low/:spread_wide);
        # this single-run driver always uses :normal.
        pool = if mode == :live_data
            live_values = load_live_powers(input_dir)
            draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
        else
            draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                     n_samples = 20, rng = MersenneTwister(seed),
                                     mode = mode)
        end

        # ---- APPROACH 0: one-shot 8:00 plan, replayed open-loop under ONE plant ----
        # approach0_plant picks WHICH baseline this is:
        #   :mean     -> realized == planned exactly; the KPIs ARE the whole-day MILP
        #                optimum (deterministic reference, no sampling anywhere).
        #   :sampled  -> the same fixed plan drifting under the stochastic pool, with
        #                no feedback (what re-planning is being credited against).
        approach0_plant in (:sampled, :mean) ||
            error("run_scenario_1: approach0_plant must be :sampled or :mean, got :$approach0_plant")
        res0 = run_one_shot(d, pool; time_limit_sec = time_limit_sec,
                            multi_activity = multi_activity, require_site_visit = require_site_visit,
                            single_visit_per_site = single_visit_per_site,
                            plant = approach0_plant, seed = seed)

        # ---- APPROACH 1: closed-loop MPC against the stochastic pool ----
        res = run_mpc(d, pool; shrinking = shrinking, H = H, time_limit_sec = time_limit_sec,
                         multi_activity = multi_activity, require_site_visit = require_site_visit,
                         single_visit_per_site = single_visit_per_site,
                         mcmc_samples = mcmc_samples, plant = :sampled, seed = seed)

        _print_kpis(res)
        _print_approach_summary(res0, res)

        write_outputs(res, out_dir)
        write_approach_comparison(res0, res, out_dir)

        println("\nResults written to: $(abspath(out_dir))")
        println("  Figures (v4_real style): 01..09 (09 = per-MCS power profiles)")
        println("  Reports: 08 KPI, replan_grids/*.csv+*.html,")
        println("           plan_vs_actual.html + plan_vs_actual_costs.png  (08:00 plan vs realised, financial)")
        println("           plan_vs_actual_activity.png, plan_vs_actual_side_by_side.html, plan_vs_actual_by_entity.html  (ACTIVITY)")
        println("           approach0_vs_approach1.html  (Approach 0 one-shot vs Approach 1 closed-loop, both fully")
        println("                                         realised; the A0 column is labelled by its plant mode, and a")
        println("                                         run-diagnostics table reports infeasible windows + SOE clamps)")
        println("  Console log: run_log.txt")
        res.log
    end
end

# Three-way console summary: the deterministic MILP optimum, the same plan drifting
# under the stochastic plant with no feedback, and the closed loop correcting it.
# Splits the headline A1-vs-A0 gap into "cost of drift" and "value of re-planning".
function _print_approach_summary(res0, res1)
    # TOTAL operating cost (energy + carbon + demand + missed + labour), not just the
    # energy term that res.total_cost holds -- same decomposition the HTML report uses.
    tot(r) = Output._cost_components(r).total
    a0_lbl = res0.plant === :mean ?
        "A0 one-shot, mean plant (MILP optimum) " :
        "A0 one-shot, sampled plant (open loop) "
    gap_lbl = res0.plant === :mean ?
        "  A1 - A0 (drift + re-planning, net)   " :
        "  A1 - A0 (value of re-planning)       "
    println("\n==== Approach comparison (both fully realised) ====")
    @printf("%s : \$%.2f  (%.2f h transit, %.2f h missed)\n",
            a0_lbl, tot(res0), res0.transit_intervals * res0.d.delta_T, res0.missed)
    @printf("A1 closed loop (re-planning)            : \$%.2f  (%.2f h transit, %.2f h missed)\n",
            tot(res1), res1.transit_intervals * res1.d.delta_T, res1.missed)
    @printf("%s: %+.2f  (negative = A1 cheaper)\n", gap_lbl, tot(res1) - tot(res0))
    res0.plant === :mean && println("  NOTE: A0 is DETERMINISTIC here, so this gap mixes plan drift with the",
                                    "\n        value of re-planning. Re-run with approach0_plant = :sampled to",
                                    "\n        separate the two.")
    nclamp = res0.n_capped + res1.n_capped
    nclamp > 0 && @printf("  NOTE: %d interval(s) had work capped by available CEV energy -- the\n           shortfall is reflected honestly in rem_dig/rem_load, not hidden.\n", nclamp)
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
