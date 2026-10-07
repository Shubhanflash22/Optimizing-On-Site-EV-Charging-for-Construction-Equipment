# #############################################################################
# 6_Shrinking_Horizon_main.jl  -  main script of Approach 1
# -----------------------------------------------------------------------------
# Runs the whole Approach 1 pipeline: the optional power regression, data loading, the closed-loop shrinking-horizon MPC, the KPI printout and the output files.
# Including this file runs run_scenario_1 with its defaults, unless SCENARIO1_NO_AUTORUN is set to true first.
# Four groups:
#
#   1. SETUP
#      _CODE_DIR, _DEFAULT_REGRESSION_DATA_DIR
#      -- the file includes in dependency order, the imports, and the default folder of the regression task files.
#
#   2. CONSOLE LOG
#      _with_console_log
#      -- run a function while copying everything printed on the screen into a log file.
#
#   3. SCENARIO RUNNER
#      run_scenario_1
#      -- run the regression, load the data, build the plant's power pool, run the MPC, and write the results.
#
#   4. KPI PRINTOUT
#      _print_kpis
#      -- print the headline KPIs of one run.
# #############################################################################
# external packages used across this file
using Printf
using Random

const _CODE_DIR = @__DIR__
include(joinpath(_CODE_DIR, "1_Common.jl"))
include(joinpath(_CODE_DIR, "0_Regression.jl"))    
include(joinpath(_CODE_DIR, "2_DataLoader.jl"))
include(joinpath(_CODE_DIR, "3_MCSModel.jl"))
include(joinpath(_CODE_DIR, "4_MPCLoop.jl"))
include(joinpath(_CODE_DIR, "5_Output.jl"))

using .DataLoader: load_data, load_live_powers
using .Common: draw_activity_power_pool, draw_activity_power_pool_live
using .MPCLoop: run_mpc
using .Output: write_outputs, write_important_outputs, write_detailed_output

const _DEFAULT_REGRESSION_DATA_DIR = joinpath(dirname(dirname(_CODE_DIR)), "Bayesian Regression")

# Runs f() while copying everything printed to the screen, including warnings, into A1_run_log.txt in out_dir.
# A background task reads from a pipe that stdout and stderr are both redirected into, writing each chunk to both the real console and the log file as it arrives.
function _with_console_log(f, out_dir)
    mkpath(out_dir)
    log_path = joinpath(out_dir, "A1_run_log.txt")

    open(log_path, "w") do logfile
        pipe = Pipe()
        orig_stdout = stdout
        orig_stderr = stderr

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
            close(pipe.in)
            wait(tee_task)
        end
    end
end

# Runs Approach 1 end to end and returns the result of run_mpc.
# It first refits the activity powers into parameters.csv when run_regression is true, then loads the input data from input_dir, and builds the pool of sampled activity powers that the plant draws from.
# mode sets how those powers are drawn (normal, high, low, near_mean, or live_data to resample the recorded values in live_powers.csv), and the pool is sized for n_day_run days so a multi-day run never exhausts its samples.
# It then runs the closed-loop MPC over n_day_run days, with time_limit_sec limiting each window solve, prints the KPIs, and writes the results into out_dir.
# detailed_output also writes the four detailed plan and realized logs, and important_only writes only the eight most important files (write_important_outputs), forcing detailed_output on regardless of its own setting.
function run_scenario_1(; input_dir::AbstractString = joinpath(dirname(_CODE_DIR), "data", "input_data"),
                          time_limit_sec::Float64 = Inf,
                          multi_activity::Bool = false,
                          mcmc_samples::Int = 500,
                          n_day_run::Int = 1,
                          mode::Symbol = :normal,
                          out_dir::String = joinpath(dirname(_CODE_DIR), "output", String(mode)),
                          run_regression::Bool = false,
                          regression_data_dir::AbstractString = _DEFAULT_REGRESSION_DATA_DIR,
                          regression_samples::Int = 2000,
                          regression_chains::Int = 4,
                          detailed_output::Bool = false,
                          important_only::Bool = false,
                          seed::Int = 1)
    if !isdir(input_dir)
        for alt in (joinpath(_CODE_DIR, "input_data"), joinpath(dirname(_CODE_DIR), "input_data"))
            isdir(alt) && (input_dir = alt; break)
        end
    end

    if run_regression
        Regression.run_regression(regression_data_dir, joinpath(input_dir, "parameters.csv");
                                  mcmc_samples = regression_samples, nchains = regression_chains)
    end

    d = load_data(input_dir)
    detailed_output = detailed_output || important_only

    return _with_console_log(out_dir) do
        pool = if mode == :live_data
            live_values = load_live_powers(input_dir)
            draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
        else
            draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                     n_samples = length(collect(d.K)) * n_day_run + 5, rng = MersenneTwister(seed),
                                     mode = mode)
        end

        res = run_mpc(d, pool; time_limit_sec = time_limit_sec,
                         multi_activity = multi_activity,
                         mcmc_samples = mcmc_samples, plant = :sampled, seed = seed,
                         n_day_run = n_day_run, detailed_output = detailed_output)

        _print_kpis(res)

        if important_only
            write_important_outputs(res, out_dir)
        else
            write_outputs(res, out_dir)
            detailed_output && write_detailed_output(res, out_dir)
        end

        println("\nResults written to: $(abspath(out_dir))")
        if important_only
            println("  Important mode: A1_interval_log.csv, A1_solve_log.csv, A1_kpi_summary.csv,")
            println("                  A1_plan_vs_actual.html (+ day1.. for n_day_run > 1),")
            println("                  A1_plan_full.csv, A1_realized_tuple.csv, A1_MCS_plan_full.csv, A1_MCS_realized_tuple.csv")
        else
            println("  Figures: A1_01..A1_09 (09 = per-MCS power profiles)")
            println("  Reports: A1_kpi_summary.csv(+_by_day), A1_replan_grids/*.csv+*.html,")
            println("           A1_plan_vs_actual.html + A1_plan_vs_actual_costs.png  (Overall, plus day1.. for n_day_run > 1)")
            println("           day<N>/A1_plan_vs_actual_activity.png, A1_plan_vs_actual_side_by_side.html, A1_plan_vs_actual_by_entity.html  (ACTIVITY, per day)")
            detailed_output && println("           A1_plan_full.csv, A1_realized_tuple.csv, A1_MCS_plan_full.csv, A1_MCS_realized_tuple.csv  (per 15-min step)")
        end
        println("  Console log: A1_run_log.txt")
        res
    end
end

# Prints a short summary of one run's headline KPIs: grid energy, energy cost, CO2 (skipped if effectively zero), the two demand peaks, missed work, MCS transit time and labour cost, and the end-of-run SOE of every CEV and MCS against its target.
function _print_kpis(res)
    d = res.d
    println("\n==== Scenario 1 closed-loop KPIs — $(res.n_day_run) day(s), $(res.nK) intervals ====")
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
    @printf("Terminal SOE shortfall penalty : \$%.2f\n", res.shortfall_penalty_cost)
end

# Runs the scenario with its default arguments when the file is included, unless SCENARIO1_NO_AUTORUN is defined as true.
if !(@isdefined(SCENARIO1_NO_AUTORUN) && SCENARIO1_NO_AUTORUN)
    run_scenario_1()
end