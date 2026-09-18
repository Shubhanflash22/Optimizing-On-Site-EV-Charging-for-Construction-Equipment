# Approach 0 — One-Shot Baseline

## 1. What this is

Approach 0 is the "commit once" baseline every other approach in this project is
measured against. At 08:00 it solves **one** MILP over the **entire 24-hour
day** — full lookahead, nothing shrinking, nothing re-solved — and then
executes that fixed plan **open-loop** for the rest of the day, whatever the
plant actually realizes. There is no feedback: if reality drifts from the
plan, Approach 0 does not notice or correct for it until the next day's 08:00
re-solve.

Contrast with:
- **Approach 1** (`../Approach 1/Shrinking_Horizon`) — re-solves a shrinking
  window every 15 minutes, closed-loop, deterministic power model.
- **Approach 2** (`../Approach 2/Shrinking_Horizon`) — the same closed-loop
  idea, but scenario-based/stochastic at every re-solve.

The gap between Approach 0 and Approach 1/2 on identical inputs is, by
construction, **the value of re-planning** — that's the entire reason this
baseline exists.

## 2. Why Approach 0 now has its own folder

Previously, Approach 0 was not its own thing — the `Comparison_A0_A1_A2`
driver reached into whichever of Approach 1's or Approach 2's own codebase
happened to be selected (`approach0_source = :a1_shrinking` or
`:a2_shrinking`) and called that codebase's `run_one_shot` function. Both
copies were required to be kept byte-identical by hand for this to be safe.

That's gone. Approach 0 is now this folder, on its own, with its own copy of
the shared model-building code (`1_Common.jl`, `2_DataLoader.jl`,
`3_MCSModel.jl` — the physics and the MILP itself are unchanged and identical
to Approach 1/2's, since the underlying vehicle/MCS model doesn't depend on
which control approach is driving it) and its own executor (`4_OneShot.jl`).
The Comparison driver now calls `A0App.OneShot.run_one_shot(...)` directly
(aliasing `Common` from `A1ShrinkingApp` so the shared power pool still works
across all three apps — see `Comparison_A0_A1_A2/Readme.md`); there is no
more `approach0_source` switch to get wrong.

Approach 1 and Approach 2 have also been made fully independent: their own
`run_one_shot` copies, `run_soe_sweep.jl` scripts, and the "Approach 0 vs
Approach 1/2" comparison figures they used to draw internally have all been
removed. Each of the three approaches is now a clean, standalone codebase;
the only place a cross-approach comparison is produced is this folder. This
is the canonical, independent home for Approach 0.

## 3. Folder layout

```
Approach 0/
  code/
    1_Common.jl        shared helpers + the detailed-output log structs
                        (DetailedPlanLog/RealizedTupleLog for the CEV(s),
                        MCSPlanLog/MCSRealizedLog for the MCS)
    2_DataLoader.jl     loads :synthetic / :input data (identical to A1/A2's)
    3_MCSModel.jl       the single 24h window MILP (identical to A1/A2's)
    4_OneShot.jl        module OneShot — the whole of Approach 0
    5_Output.jl         module Output — CSV writers + KPI table builder
    6_OneShot_main.jl   standalone entry point (run_scenario_0)
  data/
    input_data/         the 8 real-data CSVs (same as Approach 1's)
    synthetic_data/      human-readable mirror of the hardcoded :synthetic
                         scenario (not read by any code — see 2_DataLoader.jl)
  docs/
    README.md           this file
```

## 4. Running it standalone

```julia
julia --project=.
include("code/6_OneShot_main.jl")
```

or, without auto-run:

```julia
SCENARIO0_NO_AUTORUN = true
include("code/6_OneShot_main.jl")
res = run_scenario_0(; mode = :normal, n_day_run = 1, plant = :sampled)
```

Key `run_scenario_0` arguments:

| Argument | Meaning |
|---|---|
| `mode` | `:normal` (default, unbiased draws), `:high`/`:low`/`:near_mean` (biased sensitivity sweeps), `:live_data` (draw from recorded `live_powers.csv`), or `:synthetic` (built-in hardcoded scenario, no CSVs read) |
| `plant` | `:sampled` (stochastic — the normal/headline case) or `:mean` (deterministic — realized == planned exactly, KPIs are the MILP's own optimum) |
| `n_day_run` | how many days to run back to back (see §5 below for what carries over and what doesn't) |
| `detailed_output` | `true` (default) writes the 4 detailed CSVs + KPI table(s); `false` just prints the console KPI summary |
| `time_limit_sec` | solver seconds per day's MILP; `Inf` (default) solves to the MIP gap |

Output lands in `output/<mode>/`:

| File | Contents |
|---|---|
| `A0_plan_full.csv` | The whole day's plan, one row per (day, 15-min step, CEV): activity, planned power, whether charging, planned SOE, `changed_from_prior_resolve` (always blank/missing for Approach 0 — see §5). |
| `A0_realized_tuple.csv` | What actually happened, one row per (day, 15-min step, CEV). |
| `A0_MCS_plan_full.csv` / `A0_MCS_realized_tuple.csv` | The MCS's own planned/realized status, node, charge/discharge power, SOE — the same idea, for the MCS instead of the CEV(s). |
| `A0_kpi_summary.csv` | The full 17-metric KPI table for the whole run (one column). Same metrics/formulas as `Comparison_A0_A1_A2/Code/8_ComparisonOutput.jl`'s `cost_components`. |
| `A0_kpi_summary_by_day.csv` | Only written when `n_day_run > 1`: the same table, one column per day plus an `Overall` column. |

## 5. Multi-day runs: what carries over, and what resets each day

Three kinds of state exist across a multi-day Approach 0 run:

1. **Physical state** (battery SOE, MCS location) — carries over day to day.
   This is real: whatever charge or position the CEV/MCS actually ends the
   day with is where tomorrow starts.
2. **Work backlog** (`rem_dig`/`rem_load`) — carries over day to day. Each new
   day's fresh requirement is *added on top of* whatever's still outstanding
   from before (see the `CHANGE 5` comment in `4_OneShot.jl`). This is also
   real — if the CEV falls behind, it should stay behind.
3. **Scheduling-rule memory** (`hist`, the applied-activity history) —
   **reset to empty at the start of every day.**

That third one needs explaining, because getting it wrong produces a subtle,
counter-intuitive bug. `hist` feeds three constraints inside
`3_MCSModel.jl`'s `build_window_model`:
- the **precedence rule** (cumulative loading ≤ 2× cumulative digging),
- the **rest rule** (no more than 4 consecutive work-intervals),
- the **travel-pacing rule** (1 travel required per 4 work-intervals).

If `hist` is allowed to accumulate across the whole multi-day run (as it
originally did), these three rules stop being "one clean rule per day" and
become a **running tally with a memory that never clears**. Concretely: if a
day's own work requirement is, say, 18 work-intervals, and the pacing rule
wants 1 travel per 4 (18 ÷ 4 = 4.5), there's no way to satisfy that exactly
every single day — some days need 4 travels, some need 5, and which day needs
which depends on the *entire history of the run so far*, not just that day.
That forces irregular travel/rest timing onto specific days, which fragments
the CEV's charging needs across the afternoon instead of leaving one clean
gap for the MCS to make its own trip to the grid — so the MCS ends up cramming
the same daily energy into a shorter and shorter overnight window, and the
demand-charge peak (`NCD_Peak_kW`) climbs day after day even though every
day requires **identical** work. Nothing is "getting worse" — it's a pure
side-effect of a counter that should have reset nightly but didn't.

**The fix, one line, at the top of the day loop in `run_one_shot`:**
```julia
hist = [Vector{Tuple{Int, Vector{Float64}}}() for _ in d.E]
```
This clears only the three rules' memory. It does **not** touch battery SOE,
MCS location, or the `rem_dig`/`rem_load` backlog — those are separate
variables and still carry over exactly as they should. With this in place,
every day is once again an independently well-posed problem, and
`NCD_Peak_kW` should come out flat (or very close to it) across a multi-day
run instead of climbing.

This fix is Approach-0-specific and lives only in this folder's
`4_OneShot.jl` — Approach 1/2's own `run_mpc` (in their respective
`4_MPCLoop.jl`) still carries `hist` across days by original design, since
their closed-loop re-solving every 15 minutes is a different situation (a
day boundary is not a natural "reset point" the same way it is for a
once-a-day optimizer). If you want the identical reset applied to Approach
1/2's multi-day closed-loop runs too, that's a separate, explicit change to
make there.

## 6. Why `changed_from_prior_resolve` is always blank

`A0_plan_full.csv` uses the exact same `DetailedPlanLog`/`log_plan_row!`
structure Approach 1/2 use, which includes a `changed_from_prior_resolve`
column (did this step's plan change from the immediately preceding resolve
for the same step?). Since Approach 0 only ever resolves **once** per day —
there is no "preceding resolve" to compare against — this column is always
blank/missing for every row. That's expected, not a bug: it's a direct,
visible confirmation that Approach 0 genuinely never replans.

## 7. Detailed-output CSV schema note

`A0_plan_full.csv`/`A0_realized_tuple.csv` (CEV) and
`A0_MCS_plan_full.csv`/`A0_MCS_realized_tuple.csv` (MCS) use the identical
column layout Approach 1/2 write for their own `run_mpc`, so a run from any
of the three approaches can be loaded and compared column-for-column. The
only structural difference is `resolve_step`, which is always `1` here
(Approach 0 has exactly one resolve per day) versus 1..nKd for Approach 1/2's
per-interval replanning.
