# Comparison driver — Approach 0 vs Shrinking Horizon vs Receding Horizon

## Folder layout

Place just these two files at
`C:\Users\shubh\Desktop\MPC\Approach 1\Comparison\Code`:

```
Comparison/Code/
  7_Comparison_main.jl      top-level driver (run this)
  8_ComparisonOutput.jl     the merged 3-way figures/reports module
  README.md                 this file
```

**Nothing is copied.** The driver calls your two existing codebases in place:

```
C:\Users\shubh\Desktop\MPC\Approach 1\Shrinking_Horizon\code
C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\code
```

If you edit either codebase later, the next run picks the change up
automatically — there is no stale copy anywhere under `Comparison/`.

## How to run

In Julia, from anywhere:

```julia
include(raw"C:\Users\shubh\Desktop\MPC\Approach 1\Comparison\Code\7_Comparison_main.jl")
```

This auto-runs `run_comparison()` with all defaults, which:

1. Builds `Comparison/Input/` — copies the 6 confirmed-identical CSVs
   (`time_data.csv`, `travel_time.csv`, `work_flexible.csv`, `ev_data.csv`,
   `mcs_data.csv`, `place.csv`) from the Receding Horizon input folder, then
   runs the step-0 Bayesian regression to (re)write `parameters.csv` into
   that same folder — so `Comparison/Input/` ends up with exactly the 7
   files a normal run needs, sourced from your two existing folders.
2. Loads that one `Comparison/Input/` dataset twice — once through each
   codebase's own `DataLoader` (see "why twice" below).
3. Builds **one** shared power-sample pool.
4. **Actually solves**, in this order, all from that one pool: Approach 0
   (one-shot MILP, solved once), Approach 1 in Shrinking-Horizon mode (MILP
   re-solved every 15-min interval), Approach 1 in Receding-Horizon mode
   (`n_days = 1`, MILP re-solved every interval).
5. Writes the merged 3-way figures/reports straight into
   `Comparison/Output/`.

To customise instead of using the auto-run:

```julia
COMPARISON_NO_AUTORUN = true
include(raw"C:\Users\shubh\Desktop\MPC\Approach 1\Comparison\Code\7_Comparison_main.jl")
run_comparison(time_limit_sec = 60.0, seed = 2)
```

| keyword | default | meaning |
|---|---|---|
| `input_dir` | `Comparison/Input` | where the merged input CSVs live |
| `out_dir` | `Comparison/Output` | where everything gets written |
| `csv_source_dir` | Receding Horizon's `input_data` | source for the 6 shared CSVs |
| `run_regression` | `true` | refit `parameters.csv` via step-0 |
| `approach0_source` | `:shrinking` | which codebase's one-shot solver is "Approach 0" |
| `n_days_receding` | `1` | days the Receding Horizon run keeps (1 = apples-to-apples with Shrinking) |
| `pool_n_samples` | `nK_day · (n_days_receding + 1) + 5` | pre-drawn samples per (entity, activity) pair in the shared pool. Sized on the **buffer-inclusive** Receding length, not one day — see "Pool sizing" below |
| `time_limit_sec`, `multi_activity`, `require_site_visit`, `single_visit_per_site`, `mcmc_samples`, `H`, `seed` | same as your existing mains | forwarded as-is |

## Why `DataLoader` runs "twice"

`RecedingApp.DataLoader` and `ShrinkingApp.DataLoader` are your two actual,
different `DataLoader.jl` files (Receding's understands multi-day quotas,
Shrinking's doesn't) — they are genuinely two different pieces of code, both
pointed at the same `Comparison/Input/` folder, both producing a `d`
NamedTuple with the same values (since the files they read are identical).
This is not wasted work: it's the only way to get each codebase's own,
unmodified data-loading logic to run against the shared input.

## One shared power pool — not two

You asked for a single `ActivityPowerPool`, not two statistically-equivalent
ones. That required solving a real problem: `RecedingApp` and `ShrinkingApp`
are two separate Julia module namespaces, and each codebase's `MPCLoop`
declares `run_mpc(d, pool::ActivityPowerPool; ...)` against its **own**
`Common.ActivityPowerPool` type. Julia dispatches on nominal types, so a pool
built by one app would normally be *rejected* by the other app's `run_mpc` —
even if the two `ActivityPowerPool` structs are field-for-field identical,
they'd be two different types.

The fix, in `7_Comparison_main.jl`: `ShrinkingApp` does **not** include its
own `1_Common.jl`. Instead it does

```julia
module ShrinkingApp
    import ..RecedingApp
    const Common = RecedingApp.Common
    ...
end
```

so `ShrinkingApp.Common` *is* `RecedingApp.Common` — the literal same module.
This is safe because the two `1_Common.jl` files were checked and are
byte-identical except for three extra multi-day-only helper functions
(`clock_day_label`, `build_time_labels_days`, `multiday_xticks`) that only
Receding's own `MPCLoop`/`Output` call — Shrinking's code never references
them, so it doesn't need its own copy. Everything Shrinking's `MCSModel` /
`MPCLoop` / `Output` actually import from `..Common`
(`ActivityPowerPool`, `draw_activity_power_pool`, `new_cursor`,
`next_power!`, `clock_label`, `in_peak`, `stepify_*`, `BayesianActivityEstimator`,
etc.) is present, unchanged, in Receding's `Common.jl`.

With that alias in place, the driver builds **one** pool —

```julia
pool = RecedingApp.Common.draw_activity_power_pool(dS.E, dS.prior_mu, dS.prior_sigma;
                                                    n_samples = n_samples, rng = MersenneTwister(seed))
```

### Pool sizing — why it multiplies by `n_days_receding + 1`

`next_power!` **errors** when a cursor runs past the end of the pre-drawn samples; it does not
wrap around. So the pool has to cover the **longest** run, not the reported horizon.

The catch is the Receding run's buffer day. `n_days_receding = 1` does not mean one day is
simulated — the Receding loop always simulates `n_days + 1` days and drops the last one from the
reported outputs. The dropped day is still fully simulated, so its cursor keeps consuming draws
right through it. Sizing the pool on `nK_day` alone would give the Receding run roughly half the
margin it can need, and the failure would surface mid-run, after the MILPs had already spent a
long time solving.

Hence `nK_day · (n_days_receding + 1) + 5`. Unconsumed samples cost nothing but a little memory,
so over-provisioning is the right trade. Pass `pool_n_samples` explicitly to override.

(Idle never consumes a draw — its `sd` is 0, so `next_power!` returns `mu` without touching the
cursor — which is why a run's actual consumption is well under its interval count. The sizing
above is deliberately worst-case rather than typical.)

— and passes that same Julia object into all three calls
(`ShrinkingApp.MPCLoop.run_one_shot`/`run_mpc` and
`RecedingApp.MPCLoop.run_mpc`). Each call still makes its own `new_cursor`
(as the source codebases already do internally, on purpose, so each
approach's own realized-activity sequence determines its own draw order) but
every cursor walks the exact same pre-generated sample sequence stored once,
in one place, on one object.

## Why `approach0_source` exists

Both codebases ship their own "Approach 0" one-shot solver, and they are
*not* the same MILP (Receding's targets a rolling 24 h recharge; Shrinking's
is a single fixed day). A 3-way comparison needs exactly one baseline, so
this driver picks ONE of the two one-shot solvers to be "Approach 0" — the
Shrinking Horizon's, by default (simpler, buffer-free). Pass
`approach0_source = :receding` to use the other one instead.

## Output layout

Everything lands directly in `Comparison/Output/` — no per-approach
subfolders, and neither codebase's own native `Output.write_outputs` /
`write_approach_comparison` gets called at all (only the merged 3-way
artefacts below are produced):

```
Comparison/Output/
  01_total_grid_power_profile.png/.csv
  02_work_profiles_by_site.png/.csv
  03_mcs_state_of_energy.png/.csv
  04_cev_state_of_energy.png/.csv
  05_electricity_prices_emissions.png/.csv   (drawn once — shared input, identical for all 3)
  06_mcs_location_trajectory.png/.csv
  07_mcs_optimization_summary.png
  07_approach_timeline_comparison.png
  08_kpi_metrics_summary.png
  08_cost_kpi_metrics.csv                    (one column per approach)
  09_mcs_<m>_power_profile.png/.csv
  approach0_vs_shrinking_vs_receding.html    (3-way analogue of approach0_vs_approach1.html)
  run_log.txt                                everything printed during the run
```

Every figure above overlays all 3 approaches (one colour per approach:
Approach 0 = gray, Shrinking = steel blue, Receding = firebrick red).

`07_mcs_optimization_summary.png` is the original 4x2 overview grid (price/
CO2, grid power, MCS SOE, work-by-site, CEV SOE, location, plus a text
summary panel). `07_approach_timeline_comparison.png` is a separate, newer
3x3 grid — one column per approach, rows = MCS power / CEV state of energy /
CEV work power — extending each source codebase's own 2-approach 3x2
Approach0-vs-Approach1 timeline figure to all three approaches side by side.

(The `plan_vs_actual*` family from the previous version of this driver has
been removed per your request — those compared one approach's own forecast
against its own outcome, which doesn't extend cleanly to 3 approaches.)