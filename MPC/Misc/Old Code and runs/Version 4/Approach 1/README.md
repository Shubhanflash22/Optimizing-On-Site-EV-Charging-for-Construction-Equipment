# Approach 1 — Closed-Loop Shrinking-Horizon MPC

## 1. What this is

Approach 1 is a **closed-loop, shrinking-horizon MPC**, not a one-shot plan. At every 15-minute interval `k0` it:

1. Solves the **window MILP from `k0` to the end of the day**, using the real battery and position state at that moment — not the state the previous plan assumed.
2. Applies **only that first interval**, `k0`, to the simulated plant.
3. Advances the real state and moves to `k0 + 1`. The window shrinks by one interval each step until only the last interval of the day needs solving, then resets to the full day at the start of the next day.

This re-plans 96 times a day, so it reacts to whatever the plant actually did, unlike Approach 0's single whole-day plan executed open-loop. The planning side is still **certainty-equivalent**: each window is solved as if the activity powers were known exactly (the fitted means from `parameters.csv`), and only the simulated plant draws stochastic power around them. Running it with `plant = :mean` pins the plant to those same means, so realized equals planned and the terminal shortfall penalty comes out at exactly 0 — use this to sanity-check the code. Running it with `plant = :sampled` (the default) is the apples-to-apples comparison against Approach 0 and Approach 2.

The underlying optimization is the same mixed-integer program as Approach 0, from:

> A. Ghosh, A. Taşcıkaraoğlu, et al., *"Power Estimation and Optimal Work–Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations,"* arXiv:2608.18494.

Objective function (4) and constraints (5a)–(14f) in that paper are implemented in `3_MCSModel.jl`, shared unchanged with Approach 0. Section 7 lists every place the code adds to, relaxes, or otherwise departs from the paper, and why.

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
│   ├── 3_MCSModel.jl      builds and solves the MILP for one window (the
│   │                      paper's objective (4) and constraints (5)–(14)),
│   │                      shared unchanged with Approach 0
│   ├── 4_MPCLoop.jl       Approach 1 itself: re-solves the window MILP every
│   │                      interval from the real state (shrinking horizon),
│   │                      applies only that interval to the plant
│   ├── 5_Output.jl        turns a run's results into figures, CSVs, and HTML
│   │                      reports
│   └── 6_Shrinking_Horizon_main.jl   the driver script — run this one
├── data/
│   └── input_data/        the 8 input CSVs (see Section 4)
└── output/
    └── <mode>/            files written by a run (see Section 5); <mode> is
                            named after the run's `out_dir` argument by default
```

Files 0–5 are Julia modules (`module Regression`, `module Common`, etc.) and are `include`d by file 6 in that dependency order. File 6 is a plain script, not a module — it wires everything together and auto-runs a default scenario the moment it's included.

## 4. Input file structure

All 8 files live in `data/input_data/`. Every value below is taken directly from the sample dataset checked in — it reproduces the paper's **Scenario 1** (1 CEV, 1 MCS, one grid node, one construction site). This loader, and every file it reads, is identical to Approach 0's — the two approaches differ only in how the plan is used, not in what data they read.

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
| `sigma_digging` / `sigma_loading_swinging` / `sigma_traveling` | 0.5495 / 0.3999 / 0.6517 | Prior std per activity, kW — also rewritten by `0_Regression.jl` |
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

Written to `output/<out_dir>/` (wherever `out_dir` points). Every file name starts with `A1_`, matching Approach 0's `A0_` convention. For a run of several days (`n_day_run > 1`), the plan-vs-realized comparisons are split into an **Overall** version at the top level and one version per day under `day<N>/`; a single-day run writes everything straight into `out_dir`, with no `day1/` subfolder.

**Written on every run, both modes:**

| File | Contents |
|---|---|
| `A1_interval_log.csv` | The plain per-interval log, all days in one table: price, CO2 intensity, grid power, work power, one SOE column per MCS and per CEV, one node column per MCS, and the planning-power mean/std for the 4 activities. |
| `A1_solve_log.csv` | One row per window solve — about 96 per day, since Approach 1 re-solves every interval: solver status, objective value, MIP gap %, and solve time. Held (infeasible) intervals appear with no objective or gap. |
| `A1_kpi_summary.csv` | One column of whole-run KPIs — cost broken into its 6 objective-function components plus the terminal shortfall penalty, grid energy, CO2, demand peaks, missed work, MCS transit hours, the number of infeasible windows, the loop wall time, and the total solve time. |
| `A1_kpi_summary_by_day.csv` | Only for `n_day_run > 1`. Same KPI rows, one column per day plus an Overall column. Missed work and the shortfall terms are genuine per-day deltas, not running totals. |
| `A1_plan_vs_actual.html` | The Overall plan-vs-realized cost comparison (sum of every day's first plan, maxed for the two demand peaks, vs. the whole run's realized values). For a single-day run this is just that day's comparison. |
| `day<N>/A1_plan_vs_actual.html` | Only for `n_day_run > 1`. That day's own first-plan-vs-realized comparison, with a per-interval grid power table. |
| `A1_run_log.txt` | Everything printed to the console during the run, including warnings. |

**Written only in normal mode (`write_outputs`, `important_only = false`):**

| File | Contents |
|---|---|
| `A1_01`–`A1_07`, `A1_09_mcs_<m>` `.png`/`.csv` | Trajectory figures over the whole run: total grid power, work by site, MCS SOE, CEV SOE, price/emissions, MCS location, the combined summary, and one power profile per MCS. |
| `A1_replan_grids/` (or `.../day<N>/` for multi-day) | One grid per planned quantity — grid power, every MCS's SOE and status, every CEV's SOE and activity — with one row per re-solve and one column per interval. |
| `A1_plan_vs_actual_costs.png` | The bar chart matching each `A1_plan_vs_actual.html`. |
| `day<N>/A1_plan_vs_actual_activity.png`, `A1_plan_vs_actual_side_by_side.html`, `A1_plan_vs_actual_by_entity.html` | That day's planned-vs-executed activity label of every CEV and MCS, as a heatmap and two HTML tables. There is no Overall version of this — an activity label has no meaningful sum or average across days. |

**Written only when `detailed_output = true`** (forced on automatically when `important_only = true`):

| File | Written by | Contents |
|---|---|---|
| `A1_plan_full.csv` | `DetailedPlanLog` | Every re-solve's plan for every CEV and every interval of its window: planned activity, planned work power, charging flag, planned SOE for both the CEV and the MCS serving it. |
| `A1_realized_tuple.csv` | `RealizedTupleLog` | What actually happened each interval for each CEV: the realized power split across dig/load/travel/idle, the executed activity label, planned activity and power side by side with the realized ones, whether the SOE floor blocked the planned activity, and realized SOE. |
| `A1_MCS_plan_full.csv` | `MCSPlanLog` | Every re-solve's plan for every MCS and every interval of its window: planned status, planned node, planned grid charge/discharge power, planned SOE. |
| `A1_MCS_realized_tuple.csv` | `MCSRealizedLog` | What actually happened each interval for each MCS: realized status, node (or "Transit"), realized charge/discharge power, realized SOE. |

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
                                       # of these a day); set this for large multi-MCS
                                       # scenarios (see Section 7)
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

To skip the auto-run when only reusing the function/module definitions elsewhere (e.g. a sweep script), define `SCENARIO1_NO_AUTORUN = true` before including the file — see `run_all_modes.jl`.

**Validation check:** run with `plant = :mean` (pass it straight to `run_mpc` if you call that directly, since `run_scenario_1` always uses `:sampled`). `Terminal_Shortfall_Penalty_USD` in `A1_kpi_summary.csv` should come out at exactly 0, because the plant then draws exactly the powers the plan assumed, so no CEV ever ends below its planned target. Unlike Approach 0, there is no single `objective` value to check this against — Approach 1 solves ~96 separate windows a day, each pricing only its own remaining horizon, so no one of them equals the whole day's realized cost.

## 7. Additional changes compared to the paper

Everything below was found by comparing the code line by line against arXiv:2608.18494's equations, and confirmed with Avik. None of these are bugs — they are documented, deliberate departures. Items 1–6 are in `3_MCSModel.jl`, shared unchanged with Approach 0.

**In the MILP itself (`3_MCSModel.jl`):**

1. **Idling is an explicit subactivity** with its own tracked power, added on top of the paper's formulation (the paper only identifies it as one of the 4 power-estimation subactivities in Section II, not as an optimization variable). While on shift, a CEV performs exactly one of the four activities, and can only charge while idle.
2. **A small tie-breaking term** (weight 1e-6, negligible against real costs) nudges the solver toward charging CEVs earlier in the window when multiple schedules are otherwise equally good.
3. **Terminal CEV SOE uses `>=`** where the paper's (10b) is an equality. Equivalent whenever `SOE_CEV_ini == SOE_CEV_max`, which holds in the sample data (Table IX).
4. **Constraint (13d)** is implemented as an arrival/departure balance that does not force the MCS to end the day at the node it started from — a deliberate relaxation. In practice the terminal MCS SOE condition and the cost of extra travel already bring it back to the grid node to recharge overnight.
5. **Rolling-window adaptations**, needed because `build_window_model` builds one window at a time rather than the whole horizon at once — this is the mechanism Approach 1 leans on hardest, since it calls `build_window_model` once per interval rather than once per day: carried-over MCS transit state across window boundaries, a starting-position condition at each window's first interval, remaining work (not total work) on the right-hand side of (14b), and cumulative history feeding (14c)–(14f). The terminal SOE conditions (10a)/(10b) only apply once a window reaches the end of the day, so a shrinking window only "sees" them on its last few intervals, same as it would for Approach 0's single full-day window.
6. **Solver settings:** single-threaded, symmetry detection off, and a 0.1% relative MIP gap (rather than solving to proven optimality) as a deliberate speed trade-off. The achieved gap is recorded for every window solve in `A1_solve_log.csv`.

**In the simulation and KPI layer (`4_MPCLoop.jl`, `5_Output.jl`), shared logic with Approach 0's `apply_and_simulate!`:**

7. **A terminal-shortfall penalty** is added to the total cost — not part of objective (4). A stochastic run can still end the day with a CEV below its target SOE despite each window's plan promising otherwise, even with constant re-planning; the shortfall is priced as if it were missed work hours, at the same `rho_miss` rate. This is zero whenever `plant = :mean`.
8. **The plant is stochastic** (`plant = :sampled`, the default): realized CEV activity power is drawn from a shared sample pool rather than fixed at the plan's mean. If realized work would drain a CEV below its SOE floor, the work is capped and the leftover time recorded as idle at zero cost (not the idle activity's own power, regardless of what `p_idling` is set to) — this reflects that the CEV physically cannot keep drawing power once its battery reaches its minimum SOE. Any energy an already-full CEV can't accept is refunded back to the MCS that supplied it, converted through both the CEV and the MCS charging efficiencies (constraints 9b and 9a) — this path is a safety net in Approach 1 specifically, since each window re-plans from the real SOE and so cannot itself plan a charge that overflows the battery; it only matters if a future change makes that possible.
