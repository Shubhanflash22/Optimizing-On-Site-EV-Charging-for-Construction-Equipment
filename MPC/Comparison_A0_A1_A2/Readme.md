# Comparison driver — A0 vs A1-Shrinking vs A2-Shrinking vs A1-Receding vs A2-Receding

> **Recent changes (Sept 2026):** `R_work` removed and replaced with the paper's
> Eq. 8b (activity indicators, including idle, are hard-zeroed outside working
> hours via a new `is_working` mask) — required infrastructure for if/when idle
> power (`p_idling`, currently still 0) is ever made nonzero. Also added an
> opt-in **LIVE_DATA_MODE** that draws the simulated plant's realized power from
> real recorded values (`live_powers.csv`) instead of the Bayesian pool — see
> `docs/constraints_code_vs_model.txt` for details.

## Folder layout

Place these two files at
`C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Code`:

```
Comparison_A0_A1_A2/
  Code/
    7_Comparison_main.jl      top-level driver (run this)
    8_ComparisonOutput.jl     the generalized N-way figures/reports module
  Input/                      built automatically on first run
  Output/                     all 11 comparisons land here (see below)
```

This must sit as a **sibling** of `Approach 1\` and `Approach 2\` under the
same `MPC\` root — the driver resolves all four source codebases relative to
its own location (two levels up), so nothing here is a hardcoded path except
the Bayesian Regression data folder.

**Nothing is copied.** The driver calls your four existing codebases in
place:

```
C:\Users\shubh\Desktop\MPC\Approach 1\Shrinking_Horizon\code
C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\code
C:\Users\shubh\Desktop\MPC\Approach 2\Shrinking_Horizon\code
C:\Users\shubh\Desktop\MPC\Approach 2\Receding_Horizon\code
```

Edit any of them later and the next run picks the change up automatically.

## How to run

```julia
include(raw"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Code\7_Comparison_main.jl")
```

This auto-runs `run_comparison()` with all defaults, which:

1. Builds `Comparison_A0_A1_A2\Input\` — copies the 6 confirmed-identical
   CSVs (`time_data.csv`, `travel_time.csv`, `work_flexible.csv`,
   `ev_data.csv`, `mcs_data.csv`, `place.csv`) from one canonical
   `input_data` folder (Approach 1 Shrinking by default — all four are
   checksummed-identical, so it doesn't matter which), then runs the step-0
   Bayesian regression to (re)write `parameters.csv` into that same folder.
2. Loads that one `Input\` dataset four times — once through each codebase's
   own `DataLoader` (needed so each app's `run_mpc`/`run_one_shot` sees data
   built with its own types; see the driver's header comment for why this is
   safe and cheap).
3. Builds **one** shared power-sample pool, sized for the longest run
   (a Receding run).
4. **Actually solves, once each**, all five approaches from that one pool:
   Approach 0 (one-shot MILP), Approach 1 – Shrinking, Approach 2 – Shrinking
   (stochastic), Approach 1 – Receding, Approach 2 – Receding (stochastic).
5. Slices those five solved results into **11 output folders** under
   `Comparison_A0_A1_A2\Output\` — nothing is re-solved per comparison:

   | Folder | Compares |
   |---|---|
   | `All5/` | A0 vs A1-Shrink vs A2-Shrink vs A1-Reced vs A2-Reced |
   | `A0_A1S/` | A0 vs A1-Shrinking |
   | `A0_A2S/` | A0 vs A2-Shrinking |
   | `A0_A1R/` | A0 vs A1-Receding |
   | `A0_A2R/` | A0 vs A2-Receding |
   | `A0_A1S_A2S/` | A0 vs A1-Shrinking vs A2-Shrinking |
   | `A0_A1R_A2R/` | A0 vs A1-Receding vs A2-Receding |
   | `A1S_A2S/` | A1-Shrinking vs A2-Shrinking |
   | `A1R_A2R/` | A1-Receding vs A2-Receding |
   | `A1S_A1R/` | A1-Shrinking vs A1-Receding |
   | `A2S_A2R/` | A2-Shrinking vs A2-Receding |

   Each folder gets the full artefact set: `01_total_grid_power_profile.png/.csv`
   … `09_mcs_<m>_power_profile.png/.csv`, `07_mcs_optimization_summary.png`,
   `07_approach_timeline_comparison.png` (one column per approach in that
   folder), `08_kpi_metrics_summary.png`, `08_cost_kpi_metrics.csv`, and a
   `<key1>_vs_<key2>[...].html` KPI table (a Δ column is added only for the
   six 2-way folders, where the comparison is unambiguous).

   `Output\run_log.txt` captures the console output from the whole run
   (shared across all 11 comparisons, since they're all written from the
   same five solved results in one pass).

To customise instead of using the auto-run:

```julia
COMPARISON_NO_AUTORUN = true
include(raw"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Code\7_Comparison_main.jl")
run_comparison(time_limit_sec = 60.0, seed = 2)
```

| keyword | default | meaning |
|---|---|---|
| `input_dir` | `Comparison_A0_A1_A2\Input` | where the merged 7-file input dataset is built |
| `out_dir` | `Comparison_A0_A1_A2\Output` | where all 11 comparison folders are written |
| `csv_source_dir` | Approach 1 Shrinking's `input_data` | which folder the 6 shared CSVs are copied from (any of the four works — confirmed identical) |
| `run_regression` | `true` | re-fit `parameters.csv` from the Bayesian Regression `.xlsx` files |
| `regression_data_dir` | `C:\Users\shubh\Desktop\Bayesian Regression` | where the regression reads its raw data from |
| `approach0_source` | `:a1_shrinking` | which codebase's `run_one_shot` produces Approach 0 — `:a1_shrinking` / `:a1_receding` / `:a2_shrinking` / `:a2_receding` (verified byte-identical within each horizon-type pair, so any choice is equally valid) |
| `approach0_plant` | `:sampled` | `:sampled` (drifts under the shared pool, isolates the value of re-planning) or `:mean` (pinned to the mean, Approach 0's KPIs are the MILP's own optimum) |
| `time_limit_sec` | `Inf` | HiGHS per-window solve time limit |
| `H` | `16` | Shrinking Horizon lookahead length (only used if `SHRINKING_MODE = false` inside the Shrinking codebase itself) |
| `n_days_receding` | `1` | reported days the Receding runs keep (a buffer day is always simulated on top and dropped) |
| `n_scenarios` | Approach 2's `DEFAULT_N_SCENARIOS` (5) | scenarios sampled per re-solve for both A2-Shrinking and A2-Receding |
| `mcmc_samples` | `500` | regression MCMC draws |
| `seed` | `1` | RNG seed for the shared plant pool and both scenario samplers |
| `combos` | all 11 listed above (`_ALL_COMBOS`) | pass a subset to skip some comparisons, e.g. for a quick test run |

## Why four codebases can share one power pool

All four codebases define modules with the same names (`Common`,
`DataLoader`, `MCSModel`, `MPCLoop`, `Output`, plus `ScenarioSampler` for the
two Approach 2 folders), so each is wrapped in its own namespaced module
(`A1RecedingApp`, `A1ShrinkingApp`, `A2RecedingApp`, `A2ShrinkingApp`) to stop
one from silently overwriting another. `1_Common.jl` is the one exception:
only `A1RecedingApp` includes its own copy; the other three alias it, so
there's exactly one `ActivityPowerPool` type and the same pool object can be
handed to all five runs.

This was **checksummed against your actual codebase**, not assumed:

- Approach 1/Shrinking's `1_Common.jl` == Approach 2/Shrinking's `1_Common.jl` (byte-identical)
- Approach 1/Receding's `1_Common.jl` == Approach 2/Receding's `1_Common.jl` (byte-identical)
- the Receding variant == the Shrinking variant + 3 extra multi-day-only
  helper functions that Shrinking's own code never calls, so aliasing the
  Receding (superset) variant into all four apps loses nothing.
- `2_DataLoader.jl` also matches within each horizon type across Approach 1
  and Approach 2.
- `run_one_shot`'s function body is byte-identical within each horizon-type
  pair (A1-Shrinking == A2-Shrinking, A1-Receding == A2-Receding), consistent
  with Approach 0 being interchangeable across all four codebases.
- The 6 shared input CSVs are byte-identical across all four `input_data`
  folders.

## Output module generalization

`8_ComparisonOutput.jl` here is a generalized rewrite of your existing 3-way
`ComparisonOutput` module: every figure/table function takes
`apps::Vector{Approach}` and only ever loops over it or reads
`length(apps)` — nothing is hardcoded to exactly 3 approaches anymore, so the
same code path serves the full 5-way run and every 2-way/3-way subset.
Concretely: the timeline grid is `layout = (3, length(apps))` instead of a
fixed `(3, 3)`, plot titles and the HTML `<h2>`/filename are built from
`join([a.label for a in apps], " vs ")` instead of a hardcoded string, and
the KPI HTML table's Δ column is added only when comparing exactly 2
approaches (with 3+ it's ambiguous which pair to difference, so it's left
out — the CSV and bar chart still give you every approach's numbers
side-by-side).

## Change 3 (this session)

This is the true 5-way harness (A0, A1S, A1R, A2S, A2R together), so it gets Change 3 too:
`8_ComparisonOutput.jl`'s `_cost_components` now adds a `shortfall_cost` line to `TOTAL cost` for
every approach shown, sourced from each of the four solver codebases' own `4_MPCLoop.jl`
(`_terminal_soe_shortfall`). This addresses Issue 1 (Approach 0's realized day-end shortfall wasn't
previously reflected anywhere in reported cost) uniformly across all five approaches — a closed-loop
approach can still end a particular run short of `SOE_CEV_ini` despite re-solving throughout the
day, and this now charges that honestly too, separate from `missed_cost` (which only reflects work
capped live against the physical `SOE_CEV_min` floor). See the four source codebases' own
`docs/README.md` for the full rationale (Issues 1–3, Changes 2–5) and this session's
`CHANGES_SUMMARY.md` at the repo root for a complete file-by-file list.
