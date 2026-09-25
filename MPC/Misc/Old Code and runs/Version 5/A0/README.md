# A0-only detailed run (25 cases)

This is a **self-contained** folder: it does not need `Approach 1/`, `Approach 2/`,
or `Comparison_A0_A1_A2/` to sit next to it. It copies in everything it needs.

Nothing in Approach 1's original source (`Code/src/`) was modified. All new logic
lives in the 5 files listed below.

## Folder layout

```
A0/
  Code/
    src/                          <- Approach 1's own source, copied verbatim, untouched
      1_Common.jl
      2_DataLoader.jl
      3_MCSModel.jl
      4_MPCLoop.jl
    A0App.jl                      <- wraps src/ into a namespaced module (same pattern
                                      the original Comparison driver uses for A1ShrinkingApp)
    run_one_shot_detailed.jl      <- NEW: Approach 0's one-shot solve, with full
                                      per-15-min plan + realized logging (CEV and MCS)
    kpi_tables.jl                 <- NEW: the A0-only KPI summary table builder
    outputs_a0.jl                 <- NEW: CSV writers
    RUN_A0_ONLY_25_RUNS.jl        <- the main script you run
    CHECK_ENVIRONMENT_A0.jl       <- optional pre-flight check, run this first
  Input/                          <- the same 8 CSVs the real 25-run sweep used
                                      (copied from Comparison_A0_A1_A2/Input, the
                                      hand-edited/stress-test versions, NOT Approach 1's
                                      own input_data/ defaults)
  README.md                       <- this file
  PORTING_GUIDE.md                <- how to add this same logging to your own
                                      Comparison_A0_A1_A2 codebase instead of using
                                      this standalone copy
```

## How to run

Since you'll be running from `C:\Users\shubh\Downloads\A0`, just open that folder
in VS Code / Julia as you normally do, `cd` into `Code`, and run:

```
julia --project=. CHECK_ENVIRONMENT_A0.jl   # optional, quick sanity check first
julia --project=. RUN_A0_ONLY_25_RUNS.jl    # the real 25-run sweep
```

No `Project.toml` was included (none existed in the codebase you gave me) — this
assumes the packages you already use for the rest of the project (JuMP, HiGHS,
DataFrames, CSV, Turing, etc.) are already available in whatever environment you
normally launch Julia/VS Code with, exactly as you described.

Each of the 25 runs uses the **same label, mode, seed, n_day_run and
time_limit_sec** as the original `RUN_ALL_25_RUNS.jl` (1200s for the first 20,
3600s for the last 5) — nothing shortened, per your request. Only Approach 0
is solved; A1S/A2S are never built, so each run is much faster than the original
comparison sweep.

## Output

Everything lands in `A0/Output_A0_Only/<run_label>/`:

| File | Contents |
|---|---|
| `A0_CEV_plan_full.csv` | The **whole day's plan**, one row per (day, 15-min step, CEV): activity planned, planned power, whether it's charging, planned SOE (CEV and MCS), plus that step's electricity price and CO2 factor. Same column layout as A1/A2's `A1_plan_full.csv` (with `resolve_step` always 1, since A0 never replans) plus the 2 new price/CO2 columns. |
| `A0_CEV_realized.csv` | What actually happened, one row per (day, 15-min step, CEV): realized dig/load/travel/idle kW, activity executed, realized SOE, plus price/CO2. Same layout as `A1_realized_tuple.csv` + price/CO2. |
| `A0_MCS_plan_full.csv` | The MCS's own planned trajectory, one row per (day, 15-min step): status (Charging (grid) / Serving CEV / Traveling / Idle), which node it's parked at (or "Transit"), planned grid charge/discharge power, planned SOE, plus price/CO2. |
| `A0_MCS_realized.csv` | What the MCS actually did, same columns as above but realized. |
| `A0_solve_log.csv` | One row per day: solver status, objective value, MIP gap %, solve time. |
| `A0_day_snapshots.csv` | One row per day: cumulative missed-work hours and terminal SOE shortfall **as of that day's end** (see note below). |
| `A0_kpi_summary.csv` | The full KPI table (same 17 metrics as the original `08_cost_kpi_metrics.csv`), one column, for the **whole run**. |
| `A0_kpi_summary_by_day.csv` | **Only for the 5-day-block runs.** Same 17 metrics, one column per day (`Day1`..`Day5`) plus an `Overall` column. |

No images and no extra plot CSVs are written.

### Note on the per-day KPI table

`Missed_Work_hour` and `Terminal_SOE_Shortfall_kWh` are **cumulative-to-date**
figures in the per-day table, not each day's own isolated contribution. That's
because the underlying model (by design — see `4_MPCLoop.jl`'s "CHANGE 5" notes)
carries unfinished work and any SOE shortfall forward from one day into the next
rather than resetting it, so "as of end of day X" is the only number that's
physically meaningful. Cost, energy, CO2 and the two demand peaks, by contrast,
**are** each day's own number (not cumulative).

## Explainability data (the "why" behind each 15-min decision)

Every plan/realized row carries that interval's electricity price
(`price_USD_per_kWh`) and CO2 factor (`co2_factor`) alongside the decision — the
two signals that most directly drive the MILP's charging/discharging/timing
choices. Together with `A0_solve_log.csv` (objective value and MIP gap per day)
this should let you reconstruct, for any 15-minute slot, both what A0 decided
and the price/carbon/SOE context that pushed the optimizer toward it.
