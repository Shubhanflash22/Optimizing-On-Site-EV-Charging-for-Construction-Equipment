# #############################################################################
# Compare_A0_A1.jl  —  ONE-CLICK A0 vs A1 comparison
# -----------------------------------------------------------------------------
# Lives in:      C:\Users\shubh\Desktop\MPC\Test\Code\Compare_A0_A1.jl
# Reads code from (siblings of Test, under MPC):
#                C:\Users\shubh\Desktop\MPC\Approach 0\code\
#                C:\Users\shubh\Desktop\MPC\Approach 1\code\
# Writes output to:
#                C:\Users\shubh\Desktop\MPC\Test\Output\
#
# Both approaches are wrapped in their own Julia module (A0App / A1App) so
# their identically-named files/functions never collide with each other.
# Both are run against MATCHED pools -- one built per module (since
# ActivityPowerPool is a separate type in each module), using the same seed
# and mode so their content is identical despite being different Julia
# types. Any difference in outcome reflects a real difference in strategy,
# not different randomness -- the same principle already verified for the
# 3-way Comparison_A0_A1_A2 driver.
#
# EDIT THESE TWO LINES to change what gets compared:
CONFIG_MODE            = :low
CONFIG_SEED            = 123456
CONFIG_TIME_LIMIT_SEC  = 1200.0
CONFIG_DETAILED_OUTPUT = true
# #############################################################################

using Random
using Printf

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

println("="^78)
println("A0 vs A1 COMPARISON  (mode=$(CONFIG_MODE), seed=$(CONFIG_SEED))")
println("A0 code : $(_A0_CODE)")
println("A1 code : $(_A1_CODE)")
println("Output  : $(_OUT_DIR)")
println("="^78)

# -----------------------------------------------------------------------------
# Load each approach into its OWN module namespace, so both codebases' many
# identically-named files/functions (1_Common.jl, run-loop internals, etc.)
# never clash with each other in this single Julia session.
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
# Load each approach's own data (from ITS OWN input_data folder -- should be
# identical files, but loaded through each approach's own DataLoader so the
# resulting `d` struct is the correct type for that approach's own functions).
# -----------------------------------------------------------------------------
println("\n[1/4] Loading data for both approaches...")
d0 = A0App.DataLoader.load_data(:input; input_dir = _A0_DATA_INPUT)
d1 = A1App.DataLoader.load_data(:input; input_dir = _A1_DATA_INPUT)
println("      done.")

# -----------------------------------------------------------------------------
# TWO pools, same content, different (module-specific) types.
# -----------------------------------------------------------------------------
# ActivityPowerPool is a separate `struct` inside EACH module's own copy of
# Common.jl -- even though the code is identical, Julia treats a struct
# defined in two different modules as two DIFFERENT types (no structural
# typing). A single pool object built from one module's type cannot be
# passed to the other module's function, which requires its own type.
#
# The fix: build the pool once per module, using the SAME seed and mode.
# Since both modules run the identical deterministic algorithm on the same
# input data (same parameters.csv, same seed), the two pools end up with
# IDENTICAL content -- just wrapped in two different, correctly-typed
# objects, one per approach. This is the same "reconstructed, not literally
# shared" synchronization already verified for the 3-way comparison.
# -----------------------------------------------------------------------------
println("\n[2/4] Building matched power pools (mode=$(CONFIG_MODE), seed=$(CONFIG_SEED))...")

function _build_pool(App, d, data_input_dir)
    if CONFIG_MODE == :live_data
        live_values = App.DataLoader.load_live_powers(data_input_dir)
        return App.Common.draw_activity_power_pool_live(d.E, live_values; rng = MersenneTwister(CONFIG_SEED))
    else
        return App.Common.draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                                     n_samples = 20, rng = MersenneTwister(CONFIG_SEED),
                                                     mode = CONFIG_MODE)
    end
end

pool0 = _build_pool(A0App, d0, _A0_DATA_INPUT)
pool1 = _build_pool(A1App, d1, _A1_DATA_INPUT)
println("      pool0 (A0 type): mu=$(round.(pool0.mu, digits=2)) kW, sd=$(round.(pool0.sd, digits=2)) kW")
println("      pool1 (A1 type): mu=$(round.(pool1.mu, digits=2)) kW, sd=$(round.(pool1.sd, digits=2)) kW")
println("      (should match exactly -- same seed, same deterministic draw)")

# -----------------------------------------------------------------------------
# Run both approaches, each against its OWN correctly-typed (but
# content-identical) pool. Base.invokelatest guards against the Julia 1.12
# "world age" issue that can arise when a module defined earlier in the same
# top-level script is called immediately after.
# -----------------------------------------------------------------------------
println("\n[3/4] Running Approach 0 (one-shot)...")
t0 = time()
res0 = Base.invokelatest(A0App.OneShot.run_one_shot, d0, pool0;
    time_limit_sec = CONFIG_TIME_LIMIT_SEC, seed = CONFIG_SEED,
    detailed_output = CONFIG_DETAILED_OUTPUT)
println(@sprintf("      done in %.1f s", time() - t0))

println("\n      Running Approach 1 (closed-loop MPC)...")
t1 = time()
res1 = Base.invokelatest(A1App.MPCLoop.run_mpc, d1, pool1;
    time_limit_sec = CONFIG_TIME_LIMIT_SEC, seed = CONFIG_SEED, plant = :sampled,
    detailed_output = CONFIG_DETAILED_OUTPUT)
println(@sprintf("      done in %.1f s", time() - t1))

# -----------------------------------------------------------------------------
# Write each approach's own normal + detailed outputs into Test/Output/A0
# and Test/Output/A1 respectively.
# -----------------------------------------------------------------------------
out0 = joinpath(_OUT_DIR, "A0")
out1 = joinpath(_OUT_DIR, "A1")
A0App.Output.write_kpi_summary(res0, d0, out0)
A1App.Output.write_outputs(res1, out1)
if CONFIG_DETAILED_OUTPUT
    A0App.Output.write_detailed_output(res0, out0)
    A1App.Output.write_detailed_output(res1, out1)
end

# -----------------------------------------------------------------------------
# Console KPI comparison table.
# -----------------------------------------------------------------------------
println("\n[4/4] Writing comparison report...")
rows = [
    ("Total grid energy (kWh)",   res0.total_energy,   res1.total_energy),
    ("Total cost (\$)",            res0.total_cost,      res1.total_cost),
    ("CO2 emissions (kg)",        res0.total_co2,      res1.total_co2),
    ("NC demand peak (kW)",       res0.nc_peak,        res1.nc_peak),
    ("OP demand peak (kW)",       res0.op_peak,        res1.op_peak),
    ("Missed work (h)",           res0.missed,         res1.missed),
    ("Travel labour (\$)",         res0.labour_cost,    res1.labour_cost),
]

println("\n" * "="^78)
@printf("%-28s %15s %15s\n", "Metric", "A0 (one-shot)", "A1 (MPC)")
println("-"^78)
for (label, v0, v1) in rows
    @printf("%-28s %15.3f %15.3f\n", label, v0, v1)
end
println("="^78)

# -----------------------------------------------------------------------------
# Self-contained HTML report (no external CDN -- works fully offline).
# -----------------------------------------------------------------------------
function _html_row(label, v0, v1)
    diff = v1 - v0
    pct = abs(v0) > 1e-9 ? (diff / v0) * 100 : 0.0
    color = diff > 1e-6 ? "#c0392b" : (diff < -1e-6 ? "#27ae60" : "#555")
    return """
    <tr>
      <td>$(label)</td>
      <td style="text-align:right">$(round(v0, digits=3))</td>
      <td style="text-align:right">$(round(v1, digits=3))</td>
      <td style="text-align:right; color:$(color)">$(round(diff, digits=3)) ($(round(pct, digits=1))%)</td>
    </tr>"""
end

html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>A0 vs A1 Comparison — mode=$(CONFIG_MODE), seed=$(CONFIG_SEED)</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 40px; background: #fafafa; color: #222; }
  h1 { font-size: 20px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 24px; }
  table { border-collapse: collapse; width: 100%; max-width: 720px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 10px 14px; border-bottom: 1px solid #eee; font-size: 14px; }
  th { background: #2F5496; color: white; text-align: left; }
  tr:hover { background: #f5f7fa; }
</style>
</head>
<body>
<h1>Approach 0 (one-shot) vs Approach 1 (closed-loop MPC)</h1>
<div class="meta">
  Mode: <b>$(CONFIG_MODE)</b> &nbsp;|&nbsp; Seed: <b>$(CONFIG_SEED)</b> &nbsp;|&nbsp;
  Time limit: $(CONFIG_TIME_LIMIT_SEC)s/window &nbsp;|&nbsp;
  Pool: mu=$(round.(pool1.mu, digits=2)) kW, sd=$(round.(pool1.sd, digits=2)) kW (matched pools -- same seed/mode for both approaches)
</div>
<table>
  <tr><th>Metric</th><th style="text-align:right">A0 (one-shot)</th><th style="text-align:right">A1 (MPC)</th><th style="text-align:right">A1 − A0</th></tr>
  $(join([_html_row(r...) for r in rows], "\n"))
</table>
<p class="meta">Generated by Compare_A0_A1.jl. Detailed per-interval CSVs (if enabled) are in Output/A0/ and Output/A1/.</p>
</body>
</html>
"""

html_path = joinpath(_OUT_DIR, "comparison_report.html")
open(html_path, "w") do io
    write(io, html)
end

println("\nDone.")
println("  A0 outputs      : $(out0)")
println("  A1 outputs      : $(out1)")
println("  HTML comparison : $(html_path)")
println("\nOpen the HTML file directly in your browser to view the comparison.")
