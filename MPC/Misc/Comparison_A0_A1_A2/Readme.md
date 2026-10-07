# Comparison driver — Approach 0 vs Approach 1 vs Approach 2

Runs all three approaches side by side, from the same input data and the same
shared power pool, and writes every cross-approach figure/CSV. This is the
**only** place cross-approach comparison output is produced — Approach 0/1/2
each write only their own single-run figures from their own folders.

## Folder layout

```
Comparison_A0_A1_A2/
  Code/
    7_Comparison_main_ShrinkingOnlyVersion.jl        base driver (run this) — `run_comparison`
    7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl   sweeps 5 draw modes — `run_comparison_sweep`
    8_ComparisonOutput.jl                             the N-way (2-3 approaches) figures/reports module
  Input/                       built automatically on first run
  Output/                      comparison folders land here (see below)
```

This must sit as a **sibling** of `Approach 0/`, `Approach 1/`, and
`Approach 2/`. The driver resolves all three source codebases relative to
its own location (two levels up) — nothing here is a hardcoded path except
the Bayesian Regression data folder, so the whole tree is portable as a unit.

**Nothing is copied.** The driver `include()`s your three codebases in place
(`../Approach 0/code`, `../Approach 1/code`, `../Approach 2/code`). Edit any
of them later and the next run picks the change up automatically.

## How to run

```julia
include(joinpath(@__DIR__, "7_Comparison_main_ShrinkingOnlyVersion.jl"))
```

This auto-runs `run_comparison(n_day_run = 5)`, which:

1. Builds `Comparison_A0_A1_A2/Input/` — copies the 7 shared CSVs from
   Approach 1's `input_data` folder if they aren't already staged there
   (a pre-staged, hand-edited `Input/` is kept as-is, never overwritten), then
   optionally runs the step-0 Bayesian regression to (re)write
   `parameters.csv` (off by default — `run_regression = false` reuses the
   pre-staged file).
2. Loads that one `Input/` dataset three times — once through each app's own
   `DataLoader` (needed so each app's `run_mpc`/`run_one_shot` sees data
   built with its own module's types).
3. Builds **one** shared power-sample pool, sized for the longest run.
4. **Solves, once each**, all three approaches from that one pool: Approach 0
   (one-shot), Approach 1 – Shrinking, Approach 2 – Shrinking (stochastic).
5. Slices those three solved results into **4 output folders** under
   `Comparison_A0_A1_A2/Output/` — nothing is re-solved per comparison:

   | Folder | Compares |
   |---|---|
   | `A0_A1S/` | A0 vs A1-Shrinking |
   | `A0_A2S/` | A0 vs A2-Shrinking |
   | `A0_A1S_A2S/` | A0 vs A1-Shrinking vs A2-Shrinking |
   | `A1S_A2S/` | A1-Shrinking vs A2-Shrinking |

   Each folder gets the full artefact set: `01_total_grid_power_profile.png/.csv`
   … `09_mcs_<m>_power_profile.png/.csv`, `07_mcs_optimization_summary.png`,
   `07_approach_timeline_comparison.png` (one column per approach in that
   folder), `08_kpi_metrics_summary.png`, `08_cost_kpi_metrics.csv`, a
   `<key1>_vs_<key2>[...].html` KPI table, and
   `10_diagnostic_dispatch_trace.csv` / `11_diagnostic_capacity_summary.csv`.

   `Output/run_log.txt` captures the console output from the whole run.

To customise instead of using the auto-run:

```julia
COMPARISON_NO_AUTORUN = true
include(joinpath(@__DIR__, "7_Comparison_main_ShrinkingOnlyVersion.jl"))
run_comparison(time_limit_sec = 60.0, seed = 2)
```

| keyword | default | meaning |
|---|---|---|
| `input_dir` | `Comparison_A0_A1_A2/Input` | where the merged input dataset is built |
| `out_dir` | `Comparison_A0_A1_A2/Output` | where the 4 comparison folders are written |
| `csv_source_dir` | Approach 1's `input_data` | which folder the shared CSVs are copied from if not already staged |
| `run_regression` | `false` | re-fit `parameters.csv` from the Bayesian Regression `.xlsx` files |
| `approach0_plant` | `:sampled` | `:sampled` (drifts under the shared pool, isolates the value of re-planning) or `:mean` (pinned to the mean, Approach 0's KPIs are the MILP's own optimum) |
| `time_limit_sec` | `Inf` | HiGHS per-window solve time limit |
| `H` | `16` | Shrinking Horizon lookahead length (only used if `shrinking = false`) |
| `n_scenarios` | Approach 2's `DEFAULT_N_SCENARIOS` (5) | scenarios sampled per re-solve for A2 |
| `mcmc_samples` | `500` | regression MCMC draws |
| `n_day_run` | (caller-supplied; the auto-run uses `5`) | how many days each approach runs |
| `seed` | `1` | RNG seed for the shared plant pool and the scenario sampler |
| `combos` | all 4 listed above (`_ALL_COMBOS`) | pass a subset to skip some comparisons |

`7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl`'s `run_comparison_sweep`
takes the same keywords (minus `n_day_run`, fixed at `1` there) plus `modes`
(default: all 5 draw-mode sensitivity sweeps — `:normal`, `:near_mean`,
`:high`, `:low`, `:spread_wide`) and nests the same 4 folders one level
deeper, under `Output/<mode>/`. It also writes a cross-mode
`mode_sweep_kpi_summary.csv`.

## Why three codebases can share one power pool

All three codebases define modules with the same names (`Common`,
`DataLoader`, `MCSModel`, plus `MPCLoop`/`Output` for Approach 1/2,
`ScenarioSampler` for Approach 2), so each is wrapped in its own namespaced
module (`A0App`, `A1ShrinkingApp`, `A2ShrinkingApp`) to stop one from
silently overwriting another. `1_Common.jl` is the one exception: only
`A1ShrinkingApp` includes its own copy; `A0App` and `A2ShrinkingApp` both
**alias** it (`const Common = A1ShrinkingApp.Common`), so there's exactly one
`ActivityPowerPool` type across all three, and the same pool object can be
handed to all three solves. Without this alias, `ActivityPowerPool` — a
nominal struct type declared inside `module Common` — would be a different
type per app, and a pool built by one app would be rejected by another app's
`run_mpc`/`run_one_shot`.

`2_DataLoader.jl` does not need this treatment: it returns a plain
`NamedTuple`, which is structurally typed in Julia (identical field names and
types are the same type, regardless of which module produced it), so each
app is free to keep its own independent copy.

## Output module generalization

`8_ComparisonOutput.jl` is a generalized module: every figure/table function
takes `apps::Vector{Approach}` and only ever loops over it or reads
`length(apps)` — nothing is hardcoded to exactly 2 or 3 approaches, so the
same code path serves every 2-way and 3-way subset. The KPI HTML table's Δ
column is added only when comparing exactly 2 approaches (with 3 it's
ambiguous which pair to difference, so it's left out — the CSV and bar chart
still give every approach's numbers side-by-side).
