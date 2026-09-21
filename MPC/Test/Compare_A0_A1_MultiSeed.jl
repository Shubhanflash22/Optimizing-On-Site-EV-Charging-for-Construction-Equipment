# #############################################################################
# Compare_A0_A1_MultiMode.jl  —  A0 vs A1 across 5 modes x 20 seeds each
# -----------------------------------------------------------------------------
# Lives in:      C:\Users\shubh\Desktop\MPC\Test\Compare_A0_A1_MultiMode.jl
# Reads code from (siblings of Test, under MPC):
#                C:\Users\shubh\Desktop\MPC\Approach 0\code\
#                C:\Users\shubh\Desktop\MPC\Approach 1\code\
# Writes output to:
#                C:\Users\shubh\Desktop\MPC\Test\Output\<mode>\seed_<seed>\A0\
#                C:\Users\shubh\Desktop\MPC\Test\Output\<mode>\seed_<seed>\A1\
#                C:\Users\shubh\Desktop\MPC\Test\Output\all_modes_summary.xlsx
#                C:\Users\shubh\Desktop\MPC\Test\Output\all_modes_summary.csv
#
# Runs all 5 modes x 20 seeds = 100 total (A0, A1) pairs. Full detailed output
# (plan_full.csv, realized_tuple.csv, MCS_plan_full.csv, MCS_realized_tuple.csv,
# KPI CSVs) is written for every single one, in its own subfolder -- nothing
# overwrites anything else across modes or seeds.
#
# The Excel summary is REWRITTEN after every single run (not just at the end),
# so even if this gets interrupted partway through, whatever completed so far
# is already saved in a valid, complete .xlsx file -- nothing is lost.
#
# EDIT THESE TO CHANGE THE SCAN:
CONFIG_MODES           = [:low]
CONFIG_N_SEEDS         = 20
CONFIG_MASTER_SEED     = 55          # same seeds get reused across all 5 modes
CONFIG_TIME_LIMIT_SEC  = 1200.0
# #############################################################################

using Random
using Printf
using XLSX

const _TEST_CODE_DIR = @__DIR__
const _MPC_ROOT       = normpath(joinpath(_TEST_CODE_DIR, ".."))
const _A0_CODE        = joinpath(_MPC_ROOT, "Approach 0", "code")
const _A1_CODE        = joinpath(_MPC_ROOT, "Approach 1", "code")
const _A0_DATA_INPUT  = joinpath(_MPC_ROOT, "Approach 0", "data", "input_data")
const _A1_DATA_INPUT  = joinpath(_MPC_ROOT, "Approach 1", "data", "input_data")
const _OUT_DIR        = normpath(joinpath(_TEST_CODE_DIR, "Output"))
mkpath(_OUT_DIR)

isdir(_A0_CODE) || error("Approach 0 code not found at: $(_A0_CODE) -- check the folder layout.")
isdir(_A1_CODE) || error("Approach 1 code not found at: $(_A1_CODE) -- check the folder layout.")

# -----------------------------------------------------------------------------
# Same 20 seeds reused for every mode, so results are directly comparable
# mode-to-mode (seed 886732 means the same underlying random draws in
# :low as in :normal, just interpreted through each mode's own bias).
# -----------------------------------------------------------------------------
master_rng = MersenneTwister(CONFIG_MASTER_SEED)
const SEEDS = unique(rand(master_rng, 1:1_000_000, CONFIG_N_SEEDS * 2))[1:CONFIG_N_SEEDS]

println("="^78)
println("A0 vs A1 MULTI-MODE SCAN")
println("Modes   : $(CONFIG_MODES)")
println("Seeds   : $(SEEDS)")
println("Total runs: $(length(CONFIG_MODES) * length(SEEDS))")
println("A0 code : $(_A0_CODE)")
println("A1 code : $(_A1_CODE)")
println("Output  : $(_OUT_DIR)")
println("="^78)

# -----------------------------------------------------------------------------
# Load each approach ONCE, into its own module namespace.
# -----------------------------------------------------------------------------
module A0App
    const CODE_DIR = Main._A0_CODE
    include(joinpath(CODE_DIR, "1_Common.jl"))
    include(joinpath(CODE_DIR, "2_DataLoader.jl"))
    include(joinpath(CODE_DIR, "3_MCSModel.jl"))
    include(joinpath(CODE_DIR, "4_OneShot.jl"))
    include(joinpath(CODE_DIR, "5_Output.jl"))
end

module A1App
    const CODE_DIR = Main._A1_CODE
    include(joinpath(CODE_DIR, "1_Common.jl"))
    include(joinpath(CODE_DIR, "2_DataLoader.jl"))
    include(joinpath(CODE_DIR, "3_MCSModel.jl"))
    include(joinpath(CODE_DIR, "4_MPCLoop.jl"))
    include(joinpath(CODE_DIR, "5_Output.jl"))
end

# -----------------------------------------------------------------------------
# Load data ONCE -- mode/seed-independent.
# -----------------------------------------------------------------------------
println("\nLoading data for both approaches (once)...")
d0 = A0App.DataLoader.load_data(:input; input_dir = _A0_DATA_INPUT)
d1 = A1App.DataLoader.load_data(:input; input_dir = _A1_DATA_INPUT)
println("  done.")

function _build_pool(App, d, data_input_dir, mode, seed)
    if mode == :live_data
        live_values = App.DataLoader.load_live_powers(data_input_dir)
        return App.Common.draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(seed))
    else
        return App.Common.draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                                     n_samples = 20, rng = MersenneTwister(seed),
                                                     mode = mode)
    end
end

# -----------------------------------------------------------------------------
# Write (or overwrite) the Excel summary from whatever is in `results` so
# far. Called after every single run, so the file is always complete and
# valid even if the script is interrupted mid-scan.
# -----------------------------------------------------------------------------
function _write_excel(results, xlsx_path)
    XLSX.openxlsx(xlsx_path, mode="w") do xf
        # Sheet 1: every result, all modes, all seeds.
        sheet1 = xf[1]
        XLSX.rename!(sheet1, "All Results")
        headers = ["mode","seed","a0_nc_peak","a1_nc_peak","ncd_gap","a0_cost","a1_cost",
                   "cost_gap","a0_missed","a1_missed","a0_energy","a1_energy"]
        for (c, h) in enumerate(headers)
            sheet1[1, c] = h
        end
        for (i, r) in enumerate(results)
            row = i + 1
            sheet1[row, 1] = String(r.mode)
            sheet1[row, 2] = r.seed
            sheet1[row, 3] = r.a0_nc_peak
            sheet1[row, 4] = r.a1_nc_peak
            sheet1[row, 5] = r.ncd_gap
            sheet1[row, 6] = r.a0_cost
            sheet1[row, 7] = r.a1_cost
            sheet1[row, 8] = r.cost_gap
            sheet1[row, 9] = r.a0_missed
            sheet1[row, 10] = r.a1_missed
            sheet1[row, 11] = r.a0_energy
            sheet1[row, 12] = r.a1_energy
        end

        # Sheet 2: ONLY the cases where A0's NCD < A1's NCD (A1 higher) --
        # the specific anomaly being hunted for, across every mode/seed.
        flagged = filter(r -> r.ncd_gap > 1e-6, results)
        sheet2 = XLSX.addsheet!(xf, "A0 NCD less than A1")
        for (c, h) in enumerate(headers)
            sheet2[1, c] = h
        end
        for (i, r) in enumerate(sort(flagged, by = x -> -x.ncd_gap))
            row = i + 1
            sheet2[row, 1] = String(r.mode)
            sheet2[row, 2] = r.seed
            sheet2[row, 3] = r.a0_nc_peak
            sheet2[row, 4] = r.a1_nc_peak
            sheet2[row, 5] = r.ncd_gap
            sheet2[row, 6] = r.a0_cost
            sheet2[row, 7] = r.a1_cost
            sheet2[row, 8] = r.cost_gap
            sheet2[row, 9] = r.a0_missed
            sheet2[row, 10] = r.a1_missed
            sheet2[row, 11] = r.a0_energy
            sheet2[row, 12] = r.a1_energy
        end

        # Sheet 3: quick per-mode counts, so you can see at a glance which
        # mode(s) actually produce the anomaly and how often.
        sheet3 = XLSX.addsheet!(xf, "Per-Mode Summary")
        sheet3[1,1]="mode"; sheet3[1,2]="n_runs"; sheet3[1,3]="n_a1_higher"; sheet3[1,4]="pct_a1_higher"
        row = 2
        for m in CONFIG_MODES
            mrows = filter(r -> r.mode == m, results)
            n = length(mrows)
            n_hi = count(r -> r.ncd_gap > 1e-6, mrows)
            sheet3[row,1] = String(m)
            sheet3[row,2] = n
            sheet3[row,3] = n_hi
            sheet3[row,4] = n > 0 ? round(100*n_hi/n, digits=1) : 0.0
            row += 1
        end
    end
end

# -----------------------------------------------------------------------------
# Main loop: every mode x every seed, full detailed output each time.
# -----------------------------------------------------------------------------
results = NamedTuple[]
csv_path  = joinpath(_OUT_DIR, "all_modes_summary.csv")
xlsx_path = joinpath(_OUT_DIR, "all_modes_summary.xlsx")

total_runs = length(CONFIG_MODES) * length(SEEDS)
run_i = 0

for mode in CONFIG_MODES
    mode_dir = joinpath(_OUT_DIR, String(mode))
    for seed in SEEDS
        global run_i += 1
        println("\n" * "-"^78)
        @printf("[%d/%d] mode=%s seed=%d\n", run_i, total_runs, String(mode), seed)
        println("-"^78)

        pool0 = _build_pool(A0App, d0, _A0_DATA_INPUT, mode, seed)
        pool1 = _build_pool(A1App, d1, _A1_DATA_INPUT, mode, seed)

        t0 = time()
        res0 = Base.invokelatest(A0App.OneShot.run_one_shot, d0, pool0;
            time_limit_sec = CONFIG_TIME_LIMIT_SEC, seed = seed, detailed_output = true)
        el0 = time() - t0

        t1 = time()
        res1 = Base.invokelatest(A1App.MPCLoop.run_mpc, d1, pool1;
            time_limit_sec = CONFIG_TIME_LIMIT_SEC, seed = seed, plant = :sampled, detailed_output = true)
        el1 = time() - t1

        seed_dir = joinpath(mode_dir, "seed_$(seed)")
        out0 = joinpath(seed_dir, "A0")
        out1 = joinpath(seed_dir, "A1")
        A0App.Output.write_kpi_summary(res0, d0, out0)
        A0App.Output.write_detailed_output(res0, out0)
        A1App.Output.write_outputs(res1, out1)
        A1App.Output.write_detailed_output(res1, out1)

        ncd_gap = res1.nc_peak - res0.nc_peak
        cost_gap = res1.total_cost - res0.total_cost

        @printf("  A0: nc_peak=%.3f cost=%.2f (%.1fs)   A1: nc_peak=%.3f cost=%.2f (%.1fs)   NCD gap=%.3f\n",
                res0.nc_peak, res0.total_cost, el0, res1.nc_peak, res1.total_cost, el1, ncd_gap)
        if ncd_gap > 1e-6
            println("  *** A1's NCD peak is HIGHER than A0's (mode=$(mode), seed=$(seed)) ***")
        end

        push!(results, (; mode, seed, a0_nc_peak = res0.nc_peak, a1_nc_peak = res1.nc_peak, ncd_gap,
                         a0_cost = res0.total_cost, a1_cost = res1.total_cost, cost_gap,
                         a0_missed = res0.missed, a1_missed = res1.missed,
                         a0_energy = res0.total_energy, a1_energy = res1.total_energy))

        # Write BOTH the CSV and the Excel after every single run -- full
        # resilience against an interruption at any point.
        open(csv_path, "w") do io
            println(io, "mode,seed,a0_nc_peak,a1_nc_peak,ncd_gap,a0_cost,a1_cost,cost_gap,a0_missed,a1_missed,a0_energy,a1_energy")
            for r in results
                println(io, join([String(r.mode), r.seed, r.a0_nc_peak, r.a1_nc_peak, r.ncd_gap,
                                   r.a0_cost, r.a1_cost, r.cost_gap, r.a0_missed, r.a1_missed,
                                   r.a0_energy, r.a1_energy], ","))
            end
        end
        _write_excel(results, xlsx_path)
    end
end

# -----------------------------------------------------------------------------
# Final console summary.
# -----------------------------------------------------------------------------
n_higher = count(r -> r.ncd_gap > 1e-6, results)
println("\n" * "="^78)
@printf("SUMMARY: %d of %d total runs show A1's NCD peak higher than A0's\n", n_higher, length(results))
for m in CONFIG_MODES
    mrows = filter(r -> r.mode == m, results)
    n_hi = count(r -> r.ncd_gap > 1e-6, mrows)
    @printf("  %-12s : %d of %d\n", String(m), n_hi, length(mrows))
end
println("="^78)
println("  CSV  : $(csv_path)")
println("  XLSX : $(xlsx_path)")
println("\nOpen the XLSX file -- sheet 'A0 NCD less than A1' has every flagged case, sorted worst first.")
