# #############################################################################
# run_soe_level_comparisons.jl
# -----------------------------------------------------------------------------
# Runs the REAL 5-way comparison harness (7_Comparison_main.jl -- shared pool,
# a single Approach-0 realization per point, all 5 methods on equal footing)
# at several different starting CEV battery charges, one full "All5" output
# folder per level. This is deliberately different from run_soe_sweep.jl:
# that script re-solves Approach 0 SEPARATELY for each MPC variant it's
# compared against (so A0's own random draw differs slightly between the
# "vs A1" and "vs A2" tables at the very same starting charge). This script
# gives every one of the 5 approaches the SAME shared pool draw at each SOE
# level, so the 5-way table at each level is a genuinely apples-to-apples
# comparison, not just 4 separate 2-way ones stitched together.
#
# HOW TO RUN: open this file in Julia (same way you'd run
# 7_Comparison_main.jl itself) from Approach 1's Comparison\Code folder, or
# adjust MAIN_SCRIPT below to point at wherever 7_Comparison_main.jl and
# 8_ComparisonOutput.jl actually live on this machine.
#
# WHAT YOU GET: under OUT_ROOT below, one subfolder per SOE level
# ("SOE_02.96", "SOE_06.91", ...), each containing the full "All5" output set
# (01..09 figures/CSVs, 08_cost_kpi_metrics.csv, the KPI .html table, etc.)
# for A0 vs A1S vs A2S vs A1R vs A2R at that one starting charge -- exactly
# what write_comparison_outputs already writes for the "All5" combo, just
# repeated once per SOE level into its own folder.
# #############################################################################

# ---- EDIT THESE THREE PATHS FOR YOUR MACHINE ----
using Printf   # defensive -- 7_Comparison_main.jl also brings this in via its own `using Printf`, but this doesn't rely on include order
const MAIN_SCRIPT = joinpath(@__DIR__, "7_Comparison_main.jl")   # this file's folder, or set explicitly
const OUT_ROOT     = raw"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Test"
const COMPARISON_INPUT_DIR = raw"C:\Users\shubh\Desktop\MPC\Approach 1\Comparison\Input"

# ---- SOE levels to run ----
# Deliberately reusing the most informative points from the earlier 10-point
# run_soe_sweep.jl sweep, rather than blind even-spacing: 2.96 = the floor,
# 4.71 = a mid-range point, 6.47 = a point near the middle, 6.91 = the "cliff" where 3/4 configs posted their worst result, 8.22 = the
# threshold where missed work first hits 0 for most configs, 9.54 = just
# past that threshold, 14.80 = full charge. Edit this array freely -- any
# value in [2.96, 14.80] is valid (that's the battery's real min/max).
const SOE_LEVELS = [2.96, 4.71, 6.47, 6.91, 8.22, 9.54, 14.80]

# ---- Only the 5-way table per level (skip the other 10 combos -- they'd
# just be redundant subsets of the same 5 solves, repeated 5x for nothing) ----
const SOE_LEVEL_COMBOS = [("All5", ["A0", "A1S", "A2S", "A1R", "A2R"])]

# Prevent 7_Comparison_main.jl's own bottom-of-file auto-run (which would
# fire once with default kwargs the moment it's included) -- we call
# run_comparison() ourselves, in the loop below, with the kwargs we want.
const COMPARISON_NO_AUTORUN = true
include(MAIN_SCRIPT)

println("="^78)
println("SOE-LEVEL SWEEP -- 5-way comparison, $(length(SOE_LEVELS)) starting-charge levels")
println("Output root: ", OUT_ROOT)
println("="^78)

for soe in SOE_LEVELS
    tag = @sprintf("SOE_%05.2f", soe)   # e.g. "SOE_02.96", "SOE_14.80"
    level_out = joinpath(OUT_ROOT, tag)
    println("\n" * "#"^60)
    println("####  ", tag, "  (starting charge = ", soe, " kWh)")
    println("#"^60)

    kwargs = (; combos = SOE_LEVEL_COMBOS,
                soe_cev_ini_override = soe,
                out_dir = level_out,
                n_days_receding = 1,
                approach0_source = :a1_receding,
                seed = 1)
    kwargs = COMPARISON_INPUT_DIR === nothing ? kwargs : merge(kwargs, (; input_dir = COMPARISON_INPUT_DIR))

    run_comparison(; kwargs...)
end

println("\n" * "="^78)
println("DONE. ", length(SOE_LEVELS), " SOE levels written under: ", OUT_ROOT)
for soe in SOE_LEVELS
    println("  ", @sprintf("SOE_%05.2f", soe), "/All5/  (08_cost_kpi_metrics.csv has the headline numbers)")
end
println("="^78)
