# Approach 2 — Closed-Loop Shrinking-Horizon Scenario-Based MPC

## 1. What this is

Approach 2 is a **closed-loop, shrinking-horizon, scenario-based stochastic MPC**, not a one-shot plan. At every 15-minute interval `k0` it:

1. Draws `n_scenarios` fresh **power scenarios** (default 5) and solves the **window MILP from `k0` to the end of the day** against all of them at once, using the real battery and position state at that moment — not the state the previous plan assumed. The decisions of interval `k0` itself are forced to be the same in every scenario, while later intervals may differ per scenario.
2. Applies **only that first interval**, `k0`, to the simulated plant.
3. Advances the real state and moves to `k0 + 1`. The window shrinks by one interval each step until only the last interval of the day needs solving, then resets to the full day at the start of the next day.

This re-plans 96 times a day, so it reacts to whatever the plant actually did, unlike Approach 0's single whole-day plan executed open-loop. Unlike Approach 1, the planning side is **not** certainty-equivalent: each window is solved against several sampled activity powers (scenarios drawn around the fitted means from `parameters.csv`, with the fitted standard deviations as their spread), so the action chosen for interval `k0` has to be feasible in every scenario. The simulated plant still draws its own stochastic power from the shared sample pool, and that draw is none of the scenarios. Running it with `plant = :mean` pins the plant to the fitted means; Section 6 describes what to expect from that check. Running it with `plant = :sampled` (the default) is the apples-to-apples comparison against Approach 0 and Approach 1.

The underlying optimization is the same mixed-integer program as Approach 0 and Approach 1, repeated once per scenario, from:

> A. Ghosh, A. Taşcıkaraoğlu, et al., *"Power Estimation and Optimal Work–Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations,"* arXiv:2608.18494.

Objective function (4) and constraints (5a)–(14f) in that paper are implemented in `3_MCSModel.jl`, once per scenario, with the objective taken as the probability-weighted average over the scenarios. Section 7 lists every place the code adds to, relaxes, or otherwise departs from the paper, and why.

## 2. Paper symbols ↔ code names

| Paper symbol | Code name | Where |
|---|---|---|
| Objective (4) | the `@objective` block | `3_MCSModel.jl` |
| P^ch,tot / P^dch,tot | `P_ch_tot` / `P_dch_tot` | `3_MCSModel.jl` |
| P^MCS→CEV | `P_MCS_CEV` | `3_MCSModel.jl` |
| P^work | `P_work` | `3_MCSModel.jl` |
| SOE^MCS / SOE^CEV | `SOE_MCS` / `SOE_CEV` | `3_MCSModel.jl` |
| u_i,e,t,a | `u` | `3_MCSModel.jl` |
| μ_i,e,t / ρ_m,i,e,t | `mu` / `rho` | `3_MCSModel.jl` |
| z_m,i,t (presence) | `z` | `3_MCSModel.jl` |
| x_m,i,j,t / y_m,i,j,t (travel) | `x` / `y_trv` | `3_MCSModel.jl` |
| β^arr / β^dep | `beta_arr` / `beta_dep` | `3_MCSModel.jl` |
| CH^MCS_m / DCH^MCS_m | `d.CH_MCS` / `d.DCH_MCS` | `2_DataLoader.jl` |
| CH^CEV_e | `d.CH_CEV` | `2_DataLoader.jl` |
| A_i,e (assignment) | `d.A` | `2_DataLoader.jl` |
| τ^trv_i,j | `d.tau_trv` | `2_DataLoader.jl` |
| p_a (activity power) | `d.p_digging` / `d.p_loading_swinging` / `d.p_traveling` / `d.p_idling`, `d.prior_mu` | `2_DataLoader.jl` |
| ρ^miss / ρ^travel | `d.rho_miss` / `d.rho_labor` | `2_DataLoader.jl` |
| λ^NC / λ^OP | `d.lambda_demand_NC` / `d.lambda_demand_OP` | `2_DataLoader.jl` |
| λ^em | `d.carbon_price_per_ton` (÷1000 for $/kg) | `2_DataLoader.jl` |

In Approach 2 every variable in the table above carries one extra trailing index, the power scenario `s`, so the paper's `z_m,i,t` is `z[m, i, k, s]` in the code. `P_peak_NC` and `P_peak_OP` are one per scenario. Section 7 lists the extra constraints that tie the scenarios together.

`d.prior_mu` (and `d.prior_sigma`) are normally the `p_digging` / `p_loading_swinging` / `p_traveling` rows of `parameters.csv` as-is, but can instead come from `0_Regression.jl`'s offline Bayesian fit over the field task-recording files — see that file and Section 4's note on `parameters.csv`.

## 3. Folder layout

```
project_root/
├── code/
│   ├── 0_Regression.jl    optional offline step 0: Bayesian fit of the
│   │                      digging/loading+swinging/traveling activity powers
│   │                      from the soil task-recording Excel files, writing
│   │                      the result into parameters.csv before the run
│   ├── 1_Common.jl        shared helpers: time/clock utilities, the Bayesian
│   │                      activity-power estimator, the stochastic sample
│   │                      pool ("the plant"), and the run-logging structs
│   ├── 2_DataLoader.jl    reads the input CSVs into one named tuple `d`
│   ├── 2b_ScenarioSampler.jl   draws the sampled activity power vectors
│   │                      (the power scenarios) and their weights for each
│   │                      window
│   ├── 3_MCSModel.jl      builds and solves the MILP for one window, copied
│   │                      once per scenario (the paper's objective (4) and
│   │                      constraints (5)–(14), plus the constraints that
│   │                      tie the scenarios together)
│   ├── 4_MPCLoop.jl       Approach 2 itself: draws fresh scenarios and
│   │                      re-solves the window MILP every interval from the
│   │                      real state (shrinking horizon), applies only that
│   │                      interval to the plant
│   ├── 5_Output.jl        turns a run's results into figures, CSVs, and HTML
│   │                      reports
│   └── 6_Shrinking_Horizon_main.jl   the driver script — run this one
├── data/
│   └── input_data/        the 8 input CSVs (see Section 4)
└── output/
    └── <mode>/            files written by a run (see Section 5); <mode> is
                            named after the run's `out_dir` argument by default
```

Files 0–5, including 2b, are Julia modules (`module Regression`, `module Common`, etc.) and are `include`d by file 6 in that dependency order (2b comes before 3 and 4). File 6 is a plain script, not a module — it wires everything together and auto-runs a default scenario the moment it's included.

## 4. Input file structure

All 8 files live in `data/input_data/`. Every value below is taken directly from the sample dataset checked in — it reproduces the paper's **Scenario 1** (1 CEV, 1 MCS, one grid node, one construction site). This loader, and every file it reads, is identical to Approach 0's and Approach 1's — the approaches differ only in how the plan is built and used, not in what data they read.

### `parameters.csv`
One row per scalar model parameter: `Parameter, Value, Unit, Description`. The loader (`2_DataLoader.jl`) reads each one by name, so row order doesn't matter, but the `Parameter` names must match exactly.

| Parameter | Sample value | Meaning |
|---|---|---|
| `rho_miss` | 2000 | Missed-work penalty, $/hour |
| `delta_T` | 0.25 | Interval length, hours |
| `p_digging` / `p_loading_swinging` / `p_traveling` | 4.7967 / 3.1502 / 4.7131 | Mean activity power, kW — rewritten by `0_Regression.jl` if `run_regression = true` |
| `lambda_demand_NC` / `lambda_demand_OP` | 20.12 / 20.58 | Demand charge rates, $/kW |
| `carbon_price_per_ton` | 50 | $/ton CO2 (matches the paper's $0.05/kg) |
| `rho_labor` | 20 | MCS towing labour cost, $/hour |
| `p_idling` | 0 | Idle activity power, kW (see Section 7) |
| `scale` | 2 | Loading-vs-digging precedence ratio (constraint 14c) |
| `t_limit_rest` | 1 | Mandatory-rest window, hours (constraint 14d) |
| `prior_sigma_frac` | 0.2 | Fallback prior std, as a fraction of the mean, used only if a `sigma_*` row below is absent |
| `sigma_digging` / `sigma_loading_swinging` / `sigma_traveling` | 0.5495 / 0.3999 / 0.6517 | Prior std per activity, kW — also rewritten by `0_Regression.jl`, and the spread the power scenarios are drawn with |
| `obs_noise_std` | 0.05 | Simulation-only telemetry noise, kWh |
| `co2_unit_scale` | 1 | Unit conversion applied to `intensity_tons_emissions` below |

`kappa_wt` (sample value 4) is the travel-pacing ratio: at most one travel interval per `kappa_wt` productive work intervals (constraints 14e and 14f). It is optional and defaults to 4 if the row is absent.

### `ev_data.csv`
One row per CEV. First column is the CEV's ID (`e1`, `e2`, ...) — read by position, not by its header name (which may read `Unnamed: 0` if exported from pandas).

| Column | Meaning |
|---|---|
| `SOE_min` / `SOE_max` / `SOE_ini` | Battery bounds and starting/target level, kWh |
| `ch_rate` | Charging acceptance rate, kW (paper's CH^CEV) |
| `eta_ch_dch_cev` | Charging efficiency |
| `work_cap` | Present in the sample file but **not read by the loader** |

### `mcs_data.csv`
One row per MCS (`m1`, `m2`, ...), same by-position ID convention.

| Column | Meaning |
|---|---|
| `SOE_min` / `SOE_max` / `SOE_ini` | Battery bounds, kWh |
| `CH_MCS` / `DCH_MCS` | Grid-charging / CEV-discharging power capacity, kW |
| `C_MCS_plug` | Number of outlet plugs |
| `DCH_MCS_plug` | Per-plug discharge limit, kW |
| `eta_ch_dch_mcs` | Charging/discharging efficiency |

### `place.csv`
One row per **node**. `site` is the node ID; one column per CEV ID (here, `e1`) holds `1` if that CEV is assigned to that node, `0` otherwise. A node with no CEV assigned becomes a grid node; a node with a CEV assigned becomes a construction site — this split is derived automatically, not stated explicitly.

| Column | Meaning |
|---|---|
| `hours_digging` / `hours_loading_swinging` | Required productive work at that site, hours/day |

In the sample data: `i1` (no CEV assigned → grid node), `i2` (`e1`=1 → construction site, needs 3h digging + 1.5h loading+swinging/day).

### `time_data.csv`
One row per interval (96 rows for a 24-hour, 15-minute-interval day). First column is a clock label (`8:15:00`, `8:30:00`, ...) — this is the **end** of that interval, not the start; the loader subtracts one `delta_T` from the first label to get the horizon's start time.

| Column | Meaning |
|---|---|
| `lambda_buy` | Electricity price for that interval, $/kWh |
| `intensity_tons_emissions` | Grid carbon intensity for that interval |
| `lambda_CO2` | Present in the sample file but **not read by the loader** (a different, unused carbon-price series) |

### `travel_time.csv`
A node-by-node matrix: row/column labels are node IDs, cell values are travel time between them in **intervals** (not hours). Matching against `place.csv`'s node IDs is case-insensitive.

### `work_flexible.csv`
One row per (node, CEV) pair, with one column per clock time across the full 24-hour day. A nonzero value at a given time means that CEV is on shift at that node during that interval; `0` means off-shift. This is where the 8am–12pm / 2pm–5pm working hours actually come from.

### `live_powers.csv`
Optional — only used when a run is started with `mode = :live_data`. Two columns, `activity` and `power_kW`: one row per real recorded measurement. Must contain at least one row for each of `p_digging`, `p_loading_swinging`, `p_traveling`, and `p_idling`, or the loader errors.

## 5. Output file structure

Written to `output/<out_dir>/` (wherever `out_dir` points). Every file name starts with `A2_`, matching the `A0_` and `A1_` convention of the other approaches. For a run of several days (`n_day_run > 1`), the plan-vs-realized comparisons are split into an **Overall** version at the top level and one version per day under `day<N>/`; a single-day run writes everything straight into `out_dir`, with no `day1/` subfolder.

**Written on every run, both modes:**

| File | Contents |
|---|---|
| `A2_interval_log.csv` | The plain per-interval log, all days in one table: price, CO2 intensity, grid power, work power, one SOE column per MCS and per CEV, one node column per MCS, and the planning-power mean/std for the 4 activities. |
| `A2_solve_log.csv` | One row per window solve — about 96 per day, since Approach 2 re-solves every interval, each solve covering all the scenarios at once: solver status, objective value, MIP gap %, and solve time. Held (infeasible) intervals appear with no objective or gap. |
| `A2_kpi_summary.csv` | One column of whole-run KPIs — cost broken into its 6 objective-function components plus the terminal shortfall penalty, grid energy, CO2, demand peaks, missed work, MCS transit hours, the number of infeasible windows, the loop wall time, and the total solve time. |
| `A2_kpi_summary_by_day.csv` | Only for `n_day_run > 1`. Same KPI rows, one column per day plus an Overall column. Missed work and the shortfall terms are genuine per-day deltas, not running totals. |
| `A2_plan_vs_actual.html` | The Overall plan-vs-realized cost comparison (sum of every day's first plan, maxed for the two demand peaks, vs. the whole run's realized values). For a single-day run this is just that day's comparison. |
| `day<N>/A2_plan_vs_actual.html` | Only for `n_day_run > 1`. That day's own first-plan-vs-realized comparison, with a per-interval grid power table. |
| `A2_run_log.txt` | Everything printed to the console during the run, including warnings. |

In Approach 2 each window holds one plan per power scenario. The plan-vs-actual files, the replan grids and the planned columns of the realized logs use the plan of scenario `min(3, n_scenarios)`, which is the near-average scenario when there are 5. The detailed plan logs below keep every scenario.

**Written only in normal mode (`write_outputs`, `important_only = false`):**

| File | Contents |
|---|---|
| `A2_01`–`A2_07`, `A2_09_mcs_<m>` `.png`/`.csv` | Trajectory figures over the whole run: total grid power, work by site, MCS SOE, CEV SOE, price/emissions, MCS location, the combined summary, and one power profile per MCS. |
| `A2_replan_grids/` (or `.../day<N>/` for multi-day) | One grid per planned quantity — grid power, every MCS's SOE and status, every CEV's SOE and activity — with one row per re-solve and one column per interval. |
| `A2_plan_vs_actual_costs.png` | The bar chart matching each `A2_plan_vs_actual.html`. |
| `day<N>/A2_plan_vs_actual_activity.png`, `A2_plan_vs_actual_side_by_side.html`, `A2_plan_vs_actual_by_entity.html` | That day's planned-vs-executed activity label of every CEV and MCS, as a heatmap and two HTML tables. There is no Overall version of this — an activity label has no meaningful sum or average across days. |

**Written only when `detailed_output = true`** (forced on automatically when `important_only = true`):

| File | Written by | Contents |
|---|---|---|
| `A2_plan_full.csv` | `DetailedPlanLog` | Every re-solve's plan for every CEV and every interval of its window: planned activity, planned work power, charging flag, planned SOE for both the CEV and the MCS serving it. One block of rows per scenario, identified by the `scenario_id` column, so with 5 scenarios it has 5 times as many rows as Approach 1's. |
| `A2_realized_tuple.csv` | `RealizedTupleLog` | What actually happened each interval for each CEV: the realized power split across dig/load/travel/idle, the executed activity label, planned activity and power (from scenario `min(3, n_scenarios)`) side by side with the realized ones, whether the SOE floor blocked the planned activity, and realized SOE. |
| `A2_MCS_plan_full.csv` | `MCSPlanLog` | Every re-solve's plan for every MCS and every interval of its window: planned status, planned node, planned grid charge/discharge power, planned SOE. One block of rows per scenario, identified by the `scenario_id` column. |
| `A2_MCS_realized_tuple.csv` | `MCSRealizedLog` | What actually happened each interval for each MCS: realized status, node (or "Transit"), realized charge/discharge power, realized SOE. |

**Important mode (`important_only = true`, via `write_important_outputs`)** writes only the files in the first table above plus the four detailed logs — no figures, no replan grids, no activity comparison, no bar charts. It exists for sweeps over many modes where rendering ~30 figures per run would dominate the runtime.

## 6. Running it standalone

**First-time setup (VS Code):**
1. Install Julia from [julialang.org/downloads](https://julialang.org/downloads).
2. In VS Code, install the **Julia** extension (Extensions panel, search "Julia").
3. `File > Open Folder...` and select the folder containing `code/` and `data/` as siblings.
4. `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → **"Julia: Start REPL"**.
5. In the REPL, one time only:
   ```julia
   using Pkg
   Pkg.add(["CSV", "DataFrames", "JuMP", "HiGHS", "Turing", "XLSX"])
   ```
   `LinearAlgebra`, `Printf`, `Random`, and `Statistics` are built into Julia already. `XLSX` is only needed if `0_Regression.jl`'s `run_regression = true` path runs — without it, step 0 is skipped with a warning and `parameters.csv` is used as-is.

**Running:**
```julia
include("code/6_Shrinking_Horizon_main.jl")   # auto-runs once with every default

# Re-run with different settings, without restarting Julia:
res = run_scenario_1(
    mode              = :normal,      # :normal / :near_mean / :high / :low / :spread_wide
                                       # (sampling shape for the plant), or :live_data to
                                       # draw from live_powers.csv instead
    input_dir         = "data/input_data",
    time_limit_sec    = Inf,          # solver time limit PER WINDOW SOLVE (there are ~96
                                       # of these a day, each holding one copy of the
                                       # window per power scenario); set this for large
                                       # multi-MCS cases (see Section 7)
    n_scenarios       = 5,            # power scenarios planned against at every re-solve
    multi_activity    = false,        # split an interval between its scheduled activity
                                       # and idle instead of giving it the whole interval
    n_day_run         = 1,            # number of days to simulate back-to-back
    out_dir           = "output/normal",
    run_regression    = false,        # set true to refit parameters.csv from the field data first
                                       # (needs XLSX.jl; see above)
    detailed_output   = false,        # also write the four detailed per-interval CSVs
    important_only    = false,        # true writes only the 8 key files, no figures —
                                       # forces detailed_output on regardless of its setting
    seed              = 1,
)
```
Clicking VS Code's "Run" button on `6_Shrinking_Horizon_main.jl` does the same thing as the `include(...)` line above.

`run_scenario_1` is only the function's name, kept from Approach 1. It has nothing to do with the sampled power scenarios or with the paper's Scenario 1.

To skip the auto-run when only reusing the function/module definitions elsewhere (e.g. a sweep script), define `SCENARIO1_NO_AUTORUN = true` before including the file — see `run_all_modes.jl`.

**Validation check:** run with `plant = :mean` (pass it straight to `run_mpc` if you call that directly, since `run_scenario_1` always uses `:sampled`). Approach 1's exact-zero check does not carry over: the plan here is built against several sampled powers, not the mean, so the plant's mean power is not one of the powers the plan assumed. What to expect instead is that `Terminal_Shortfall_Penalty_USD` in `A2_kpi_summary.csv` is 0 or very close to it, since the plant then sits inside the range the scenarios cover, and that `Infeasible_windows` is 0. A clearly nonzero shortfall, or any infeasible window, is worth investigating. Unlike Approach 0, there is no single `objective` value to check this against — Approach 2 solves ~96 separate windows a day, each pricing only its own remaining horizon, so no one of them equals the whole day's realized cost.

## 7. Additional changes compared to the paper

Everything below was found by comparing the code line by line against arXiv:2608.18494's equations, and confirmed with Avik. None of these are bugs — they are documented, deliberate departures. Items 1–6 are in `3_MCSModel.jl` and apply to every scenario's copy of the window. Items 9–12 have no counterpart in the paper or in Approaches 0 and 1.

**In the MILP itself (`3_MCSModel.jl`):**

1. **Idling is an explicit subactivity** with its own tracked power, added on top of the paper's formulation (the paper only identifies it as one of the 4 power-estimation subactivities in Section II, not as an optimization variable). While on shift, a CEV performs exactly one of the four activities, and can only charge while idle.
2. **A small tie-breaking term** (weight 1e-6, negligible against real costs) nudges the solver toward charging CEVs earlier in the window when multiple schedules are otherwise equally good, averaged over the scenarios with the same weights as the cost terms.
3. **Terminal CEV SOE uses `>=`** where the paper's (10b) is an equality. Equivalent whenever `SOE_CEV_ini == SOE_CEV_max`, which holds in the sample data (Table IX).
4. **Constraint (13d)** is implemented as an arrival/departure balance that does not force the MCS to end the day at the node it started from — a deliberate relaxation. In practice the terminal MCS SOE condition and the cost of extra travel already bring it back to the grid node to recharge overnight.
5. **Rolling-window adaptations**, needed because `build_window_model_stochastic` builds one window at a time rather than the whole horizon at once — this is the mechanism Approach 2 leans on hardest, since it calls `build_window_model_stochastic` once per interval rather than once per day: carried-over MCS transit state across window boundaries, a starting-position condition at each window's first interval, remaining work (not total work) on the right-hand side of (14b), and cumulative history feeding (14c)–(14f). The terminal SOE conditions (10a)/(10b) only apply once a window reaches the end of the day, so a shrinking window only "sees" them on its last few intervals, same as it would for Approach 0's single full-day window. The MCS position handed to the next window uses only decisions up to the current interval: an MCS parked in interval `k0` is reported as parked even if the plan has it leaving in `k0 + 1`, so that departure is decided by the next re-solve.
6. **Solver settings:** 8 threads in parallel mode, symmetry detection off, and a 1% relative MIP gap (rather than solving to proven optimality) as a deliberate speed trade-off. Approaches 0 and 1 run single-threaded, so solve times are not directly comparable. The achieved gap is recorded for every window solve in `A2_solve_log.csv`.

**In the simulation and KPI layer (`4_MPCLoop.jl`, `5_Output.jl`), shared logic with Approach 0's `apply_and_simulate!`:**

7. **A terminal-shortfall penalty** is added to the total cost — not part of objective (4). A stochastic run can still end the day with a CEV below its target SOE despite each window's plan promising otherwise, even with constant re-planning; the shortfall is priced as if it were missed work hours, at the same `rho_miss` rate. This is zero whenever `plant = :mean`.
8. **The plant is stochastic** (`plant = :sampled`, the default): realized CEV activity power is drawn from a shared sample pool rather than fixed at the plan's mean. If realized work would drain a CEV below its SOE floor, the work is capped and the leftover time recorded as idle at zero cost (not the idle activity's own power, regardless of what `p_idling` is set to) — this reflects that the CEV physically cannot keep drawing power once its battery reaches its minimum SOE. Any energy an already-full CEV can't accept is refunded back to the MCS that supplied it, converted through both the CEV and the MCS charging efficiencies (constraints 9b and 9a) — this path is a safety net in Approach 2 as well, since each window re-plans from the real SOE and so cannot itself plan a charge that overflows the battery; it only matters if a future change makes that possible.

**In the scenario layer (`2b_ScenarioSampler.jl`, `3_MCSModel.jl`, `4_MPCLoop.jl`), not in the paper:**

9. **Power scenarios.** Each window is copied once per scenario, and the copies differ only in the activity power used for each CEV's work (`P_work`, constraint 8d). The scenarios are drawn fresh at every re-solve around the fitted means, with the fitted standard deviations as their spread; idle has no spread, so it is the same in every scenario. With exactly 5 scenarios, each one comes from its own fixed band around the mean (extreme low, slightly low, near mean, extreme high, mild high, from about 2 standard deviations below to 2 above) with a random position inside the band. With any other count they are independent normal draws, so results for different counts are not directly comparable. All values are floored at zero. The scenarios have equal weights, and the objective is their weighted average of objective (4), including a separate demand peak per scenario. Because the five bands put about 40% of the weight on the 1 to 2 standard deviation bands, against about 27% of the probability a normal puts there, this average is a deliberately cautious one, not an expectation under the normal.
10. **Ties between the scenarios (non-anticipativity).** At the window's first interval, every binary decision (`u`, `mu`, `rho`, `z`, `x`, `y_trv`, `beta_arr`, `beta_dep`) and the power flows `P_ch_MCS`, `P_dch_MCS`, `P_MCS_CEV`, `P_ch_tot` and `P_dch_tot` are forced equal in every scenario, so that one action is chosen now. Later intervals are free to differ. `P_work`, `SOE_CEV` and the missed-work slack are deliberately not tied, since they are the uncertain consequence of the shared action.
11. **Which scenario is recorded.** The plan grids, the plan-vs-actual files and the `planned_power_kW` columns come from scenario `min(3, n_scenarios)`, the near-average one when there are 5. Only the shared first interval is identical across scenarios; every later column of a recorded plan is that one scenario's own future.
12. **Hard constraints in every scenario.** The SOE bounds and terminal conditions hold in every scenario, so the most pessimistic scenario can make a window infeasible. The loop then holds the plant's state for that interval and counts it in `Infeasible_windows`. This is more likely than in Approach 1, whose single plan uses the mean power.
