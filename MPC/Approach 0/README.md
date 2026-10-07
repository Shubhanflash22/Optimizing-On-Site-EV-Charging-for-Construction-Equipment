# Approach 0 — One-Shot MPC Baseline

## 1. What this is

Approach 0 is a **one-shot baseline**, not a true rolling-horizon MPC. For each day it:

1. Solves the **entire day's MILP once**, at the start of the day, using the full 24-hour interval set.
2. Executes that fixed plan **open-loop** against a (usually stochastic) simulated plant, interval by interval, with **no replanning** during the day.

This is deliberately the simplest possible strategy: commit to a full-day plan and hope reality matches it. It's this codebase's reproduction of Avik's original single-shot model — same MILP, solved once — with a stochastic plant executed on top of it. Running it with `plant = :mean` reproduces Avik's model most exactly, since nothing is stochastic and realized equals planned; running it with `plant = :sampled` (the default) is the apples-to-apples baseline Approaches 1 and 2 are compared against.

The underlying optimization is the mixed-integer program from:

> A. Ghosh, A. Taşcıkaraoğlu, et al., *"Power Estimation and Optimal Work–Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations,"* arXiv:2608.18494.

Objective function (4) and constraints (5a)–(14f) in that paper are implemented in `3_MCSModel.jl`. Section 7 lists every place the code adds to, relaxes, or otherwise departs from the paper, and why.

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

## 3. Folder layout

```
project_root/
├── code/
│   ├── 1_Common.jl        shared helpers: time/clock utilities, the Bayesian
│   │                      activity-power estimator, the stochastic sample
│   │                      pool ("the plant"), and the run-logging structs
│   ├── 2_DataLoader.jl    reads the input CSVs into one named tuple `d`
│   ├── 3_MCSModel.jl      builds and solves the MILP for one window (the
│   │                      paper's objective (4) and constraints (5)–(14))
│   ├── 4_OneShot.jl       Approach 0 itself: one whole-day solve per day,
│   │                      executed open-loop against the plant
│   ├── 5_Output.jl        turns a run's results into CSVs and a console
│   │                      summary
│   └── 6_OneShot_main.jl  the driver script — run this one
├── data/
│   └── input_data/        the 8 input CSVs (see Section 4)
└── output/
    └── <mode>/            CSVs written by a run (see Section 5); <mode> is
                            named after the run's `mode` argument by default
```

Files 1–5 are Julia modules (`module Common`, `module DataLoader`, etc.) and are `include`d by file 6 in that dependency order. File 6 is a plain script, not a module — it wires everything together and auto-runs a default scenario the moment it's included.

## 4. Input file structure

All 8 files live in `data/input_data/`. Every value below is taken directly from the sample dataset checked in — it reproduces the paper's **Scenario 1** (1 CEV, 1 MCS, one grid node, one construction site).

### `parameters.csv`
One row per scalar model parameter: `Parameter, Value, Unit, Description`. The loader (`2_DataLoader.jl`) reads each one by name, so row order doesn't matter, but the `Parameter` names must match exactly.

| Parameter | Sample value | Meaning |
|---|---|---|
| `rho_miss` | 2000 | Missed-work penalty, $/hour |
| `delta_T` | 0.25 | Interval length, hours |
| `p_digging` / `p_loading_swinging` / `p_traveling` | 4.7967 / 3.1502 / 4.7131 | Prior mean activity power, kW |
| `lambda_demand_NC` / `lambda_demand_OP` | 20.12 / 20.58 | Demand charge rates, $/kW |
| `carbon_price_per_ton` | 50 | $/ton CO2 (matches the paper's $0.05/kg) |
| `rho_labor` | 20 | MCS towing labour cost, $/hour |
| `p_idling` | 0 | Idle activity power, kW (see Section 7) |
| `scale` | 2 | Loading-vs-digging precedence ratio (constraint 14c) |
| `t_limit_rest` | 1 | Mandatory-rest window, hours (constraint 14d) |
| `prior_sigma_frac` | 0.2 | Fallback prior std, as a fraction of the mean, used only if a `sigma_*` row below is absent |
| `sigma_digging` / `sigma_loading_swinging` / `sigma_traveling` | 0.5495 / 0.3999 / 0.6517 | Prior std per activity, kW |
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

Written to `output/<mode>/` (or wherever `out_dir` points), only when a run is called with `detailed_output = true` (the default).

| File | Written by | Contents |
|---|---|---|
| `A0_plan_full.csv` | `DetailedPlanLog` | The full day's plan for every CEV, every interval, logged right after the solve: planned activity, planned work power, charging flag, planned SOE for both the CEV and the MCS serving it. `resolve_step` is always 1, since Approach 0 never re-solves. |
| `A0_realized_tuple.csv` | `RealizedTupleLog` | What actually happened each interval for each CEV: the realized power split across dig/load/travel/idle, the executed activity label, planned activity and power side by side with the realized ones, whether the SOE floor blocked the planned activity, and realized SOE. |
| `A0_MCS_plan_full.csv` | `MCSPlanLog` | The full day's plan for every MCS, every interval: planned status, planned node, planned grid charge/discharge power, planned SOE. |
| `A0_MCS_realized_tuple.csv` | `MCSRealizedLog` | What actually happened each interval for each MCS: realized status, node (or "Transit"), realized charge/discharge power, realized SOE. |
| `A0_kpi_summary.csv` | `Output.jl` | One column of whole-run KPIs — 17 rows, from total cost broken into its 6 objective-function components plus the terminal shortfall penalty, down to grid energy, CO2, demand peaks, missed work, MCS transit hours, and total solve time. |
| `A0_solve_log.csv` | `Output.jl` | One row per day: solver status, objective value, MIP gap %, and solve time. Written on every run, single-day included — check `gap_percent` here after any run. |
| `A0_interval_log.csv` | `Output.jl` | The plain per-interval log: price, CO2 intensity, grid power, work power, one SOE column per CEV and per MCS, one node column per MCS, and the planning-power mean/std for the 4 activities. Written on every run. |
| `A0_kpi_summary_by_day.csv` | `Output.jl` | Only written for multi-day runs. Same 17 KPI rows, one column per day plus an Overall column. Missed work and the shortfall terms are genuine per-day deltas (that day's own change), not running totals. |

## 6. Running it standalone

**First-time setup (VS Code):**
1. Install Julia from [julialang.org/downloads](https://julialang.org/downloads).
2. In VS Code, install the **Julia** extension (Extensions panel, search "Julia").
3. `File > Open Folder...` and select the folder containing `code/` and `data/` as siblings.
4. `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → **"Julia: Start REPL"**.
5. In the REPL, one time only:
   ```julia
   using Pkg
   Pkg.add(["CSV", "DataFrames", "JuMP", "HiGHS", "Turing"])
   ```
   `LinearAlgebra`, `Printf`, `Random`, and `Statistics` are built into Julia already.

**Running:**
```julia
include("code/6_OneShot_main.jl")   # auto-runs once with every default

# Re-run with different settings, without restarting Julia:
res = run_scenario_0(
    mode            = :normal,      # :normal / :near_mean / :high / :low / :spread_wide
                                     # (sampling shape for the plant), or :live_data to
                                     # draw from live_powers.csv instead
    input_dir       = "data/input_data",
    time_limit_sec  = Inf,          # solver time limit per day; set this for large
                                     # multi-MCS scenarios (see Section 7)
    multi_activity  = false,        # currently has no effect anywhere in the pipeline
    plant           = :sampled,     # :sampled (stochastic) or :mean (deterministic,
                                     # realized == planned — use this for validation)
    n_day_run       = 1,            # number of days to simulate back-to-back
    out_dir         = "output/normal",
    detailed_output = true,         # write the CSVs, not just print the summary
    seed            = 1,
)
```
Clicking VS Code's "Run" button on `6_OneShot_main.jl` does the same thing as the `include(...)` line above.

To skip the auto-run when only reusing the function/module definitions elsewhere (e.g. a script that compares Approaches 0/1/2), define `SCENARIO0_NO_AUTORUN = true` before including the file.

**Validation check:** run with `plant = :mean, n_day_run = 1`. Summing `A0_kpi_summary.csv`'s first 8 rows (`Total_Cost_USD` down through `Terminal_Shortfall_Penalty_USD`, excluding the total row itself) should match `A0_solve_log.csv`'s `objective` column within about 1e-4 — the shortfall penalty should be exactly 0 in this mode.

## 7. Additional changes compared to the paper

Everything below was found by comparing the code line by line against arXiv:2608.18494's equations, and confirmed with Avik. None of these are bugs — they are documented, deliberate departures.

**In the MILP itself (`3_MCSModel.jl`):**

1. **Idling is an explicit subactivity** with its own tracked power, added on top of the paper's formulation (the paper only identifies it as one of the 4 power-estimation subactivities in Section II, not as an optimization variable). While on shift, a CEV performs exactly one of the four activities, and can only charge while idle.
2. **A small tie-breaking term** (weight 1e-6, negligible against real costs) nudges the solver toward charging CEVs earlier in the window when multiple schedules are otherwise equally good.
3. **Terminal CEV SOE uses `>=`** where the paper's (10b) is an equality. Equivalent whenever `SOE_CEV_ini == SOE_CEV_max`, which holds in the sample data (Table IX).
4. **Constraint (13d)** is implemented as an arrival/departure balance that does not force the MCS to end the day at the node it started from — a deliberate relaxation. In practice the terminal MCS SOE condition and the cost of extra travel already bring it back to the grid node to recharge overnight.
5. **Rolling-window adaptations**, needed because `build_window_model` builds one window at a time rather than the whole horizon at once: carried-over MCS transit state across window boundaries, a starting-position condition at each window's first interval, remaining work (not total work) on the right-hand side of (14b), and cumulative history feeding (14c)–(14f).
6. **Solver settings:** single-threaded, symmetry detection off, and a 1% relative MIP gap (rather than solving to proven optimality) as a deliberate speed trade-off. The achieved gap is recorded per day in `A0_solve_log.csv`.

**In the simulation and KPI layer (`4_OneShot.jl`, `5_Output.jl`):**

7. **A terminal-shortfall penalty** is added to the total cost — not part of objective (4). Because Approach 0 never replans, a stochastic run can end the day with a CEV below its target SOE despite the plan promising otherwise; the shortfall is priced as if it were missed work hours, at the same `rho_miss` rate. This is zero whenever `plant = :mean`.
8. **The plant is stochastic** (`plant = :sampled`, the default): realized CEV activity power is drawn from a shared sample pool rather than fixed at the plan's mean. If realized work would drain a CEV below its SOE floor, the work is capped and the leftover time recorded as idle at zero cost (not the idle activity's own power, regardless of what `p_idling` is set to) — this reflects that the CEV physically cannot keep drawing power once its battery is empty. Any energy an already-full CEV can't accept is refunded back to the MCS that supplied it.
