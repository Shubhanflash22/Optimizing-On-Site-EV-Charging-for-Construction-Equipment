# Scenario 1 — Self-Improving Certainty-Equivalent MPC for MCS Dispatch

Standalone Julia implementation of *Scenario 1* — *"Approach 1: Deterministic
Certainty-Equivalent MPC"*. The formal model is in `math_model.tex`;
`constraints_explained.txt` is the plain-English map of every constraint.

| File | Language | MILP solver | Bayesian estimator |
|------|----------|-------------|--------------------|
| `Scenario_1.jl` | Julia | JuMP + HiGHS | Turing.jl (NUTS) |

It dispatches a **Mobile Charging Station (MCS)** to a fleet of **Construction EVs
(CEVs / excavators)**, **learning the per-activity power draw online** from the
realized energy consumed. It runs a **multi-day, cross-day receding horizon**: each
**15-min** re-solve plans across the rest of today **plus one or more future days**,
so the plan always "sees" tomorrow. Every day has **two phases** — a **daytime MPC**
(08:00–18:00), then a deterministic **overnight smart-charge** (18:00–08:00) that
refills the MCS in the cheapest hours. One extra **buffer day** is simulated and
dropped from all outputs so the last reported day still gets full lookahead; the CEV
batteries and any unfinished work carry across nights.

---

## Project layout

```
Receding_Horizon/
├── code/
│   ├── Scenario_1.jl                ← main model + closed-loop MPC (HiGHS)
│   ├── test_scenario_1.jl           ← edge-case test harness
│   └── export_synthetic_data.jl     ← writes the synthetic dataset to CSVs
├── data/
│   ├── input_data/                  ← the 7-CSV real dataset (input mode)
│   └── synthetic_data/              ← the built-in synthetic scenario as CSVs
├── output/
│   ├── input/                       ← results of the input-mode run
│   └── synthetic/                   ← results of the synthetic-mode run
└── docs/
    ├── README.md                    ← this file
    ├── how_to_run.txt               ← quick-start
    ├── constraints_explained.txt     ← plain-English map of every constraint
    ├── constraints_code_vs_model.txt ← constraint comparison table
    ├── math_model.tex                ← formal mathematical companion
    ├── synthetic_data_explained.md  ← the synthetic scenario in plain words
    └── output_files_explained.md    ← what every output file contains
```

The scripts live in `code/` and resolve data/output **relative to that folder**
(`../data/…`, `../output/…`), so they run from anywhere. Quick start:
**`docs/how_to_run.txt`**.

---

## What problem it solves

Heavy electric excavators draw power that depends on what they are doing
(digging, loading/swinging, traveling, **idling**). A single mobile charger must decide
**when to charge from the grid, when/where to drive, and which CEV to top up**,
so that work gets done, no on-site battery dies, and grid cost / carbon /
peak-demand charges are minimized.

The activity power draws are **uncertain**. The MPC collapses that uncertainty to
a working estimate (the posterior mean), acts on it, then **continuously refines
the estimate** from telemetry.

---

## The two feedback loops

Unlike the original one-shot 24 h solve (open loop), this closes the loop on
**both** state and model parameters.

| Loop | What is fed back |
|------|------------------|
| **State feedback** | Realized `SOE_MCS` / `SOE_CEV` re-initialize each window |
| **Parameter feedback** | Realized 15-min energy drop → Bayesian re-fit of activity powers |

### Per-step cycle (every 15 min)

```
        +-------------------------------------------------------------+
        |                                                             |
        v                                                             |
[current SOE + power estimate]                                        |
        |                                                             |
        v                                                             |
  solve MILP over window [k, k+H]  ──> plan                           |
        |                                                             |
        v                                                             |
  apply ONLY interval k's decisions                                   |
        |                                                             |
        v                                                             |
  observe realized energy drop + activity-time mix ("to the dot")     |
        |                                                             |
        +--> update SOE state ───────────────────────────────────────+
        |                                                             |
        +--> Bayesian re-fit on all data so far ─────────────────────-+
                       (improves the power estimate for next solve)
```

---

## The estimator uses the EXACT offline Bayesian regression

The online estimator is **not** a Kalman/Gaussian shortcut. It uses the same
model as the offline Bayesian fit that produced the priors:

```
x_a    ~ TruncatedNormal(mu_a, sigma_a, lower = 0)   # per-activity power (kW)
sigma  ~ HalfNormal(std(b))                          # observation noise
b_i    ~ Normal( (A x)_i , sigma )                   # energy-balance rows
```

inferred by **NUTS / MCMC** (via **Turing.jl**) — the same priors, likelihood, and
sampler family as the offline Bayesian posterior that seeds it.

- `prior_mu` / `prior_sigma` are the offline Bayesian posterior; they seed the online
  estimator and define the median profile used on step 1.
- **Online use** = accumulate observation rows `(A, b)` as telematics arrives and
  **re-fit on all data so far** (Bayesian updating with a fixed prior). Re-fit
  cadence is controlled by `refit_every` (re-running NUTS every 15 min is exact
  but slow, so it defaults to periodic).

Each observation `(a, b)`:

- `a` = realized **activity-hours** that interval over `[dig, load, travel, idle]`,
  e.g. `[0.10, 0, 0, 0.05]` for "10 min digging + 5 min idling". A CEV is always
  doing exactly one activity per sub-slot, and **idling is the residual** (it
  fills charging time, lunch breaks, and gaps), so every interval yields a row.
- `b` = realized **work energy** (kWh). In simulation `b = a · true_powers +
  noise`; on real hardware `b = charging_received·Δt − ΔSOC·battery_capacity`.

> **Multi-activity intervals improve conditioning.** A 15-min window that mixes
> digging and traveling produces a regression row with two non-zero entries —
> mixed rows separate activities faster than pure single-activity rows.

---

## Two data modes

Both implementations take a `mode`:

| Mode | Behavior |
|------|----------|
| `synthetic` (default) | Builds an artificial dataset in code. Runs out-of-the-box. |
| `input` | Loads a CSV dataset from a directory; **raises a descriptive error if any file / column / parameter is missing.** |

### Input dataset (`data/input_data/`)

The **7-CSV dataset schema** (full column details further below):

| File | Columns |
|------|---------|
| `parameters.csv` | `Parameter,Value,Unit,Description` — `delta_T, k_trv, rho_miss, rho_labor, lambda_demand_NC, lambda_demand_OP, carbon_price_per_ton, p_digging, p_loading_swinging, p_traveling` + our extras (`p_idling, scale, kappa_wt, t_limit_rest, day_end_hour, prior_sigma_frac, obs_noise_std, co2_unit_scale`) |
| `ev_data.csv` | `<id>,SOE_min,SOE_max,SOE_ini,ch_rate,work_cap` |
| `mcs_data.csv` | `<id>,SOE_min,SOE_max,SOE_ini,CH_MCS,DCH_MCS,C_MCS_plug,DCH_MCS_plug,eta_ch_dch` |
| `place.csv` | `site,<one e<i> column per CEV>,hours_digging,hours_loading_swinging` (node with no CEV = grid) |
| `travel_time.csv` | `Node,<dest cols...>` (square matrix; values in 15-min intervals) |
| `time_data.csv` | `<time>,<t-id>,lambda_CO2,lambda_buy,intensity_tons_emissions` (96 rows; `n_int`/`t_start` derived from it) |
| `work_flexible.csv` | `Location,EV,<one column per interval>` (per-interval kW work cap; `0` = no work) |

The shipped `data/input_data/` is a simple reference dataset (1 grid + 1 site, 1 MCS,
1 CEV, 96 intervals). Activity powers are **known constants** in `parameters.csv`;
the Bayesian estimator is seeded from them and essentially just confirms them.

---

## Requirements & run

Install the Julia packages once:

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Plots", "DataFrames", "CSV", "Turing"])
```

Run it from the `code/` folder (synthetic mode auto-runs on launch):

```bash
cd code
julia Scenario_1.jl                      # synthetic mode -> output/synthetic/
```

Or call the entry point directly with options (from `code/`):

```julia
include("Scenario_1.jl")
run_scenario_1(; mode = :synthetic)                       # -> output/synthetic/
run_scenario_1(; mode = :input, refit_every = 8)          # -> output/input/
```

> **Paths auto-resolve via `@__DIR__`**, so `julia Scenario_1.jl` and
> `run_scenario_1(; mode = :input)` work from **any** working directory.
> `input_dir` defaults to `../data/input_data`; `out_dir` defaults to
> `../output/<mode>` (so `input` mode writes `output/input/`, `synthetic` mode
> writes `output/synthetic/`) — both relative to `code/`.

### Options

| Julia kwarg | Default | Meaning |
|-------------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` build or `:input` CSV |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`); falls back to other layouts if absent |
| `n_days` | `nothing` | number of days to KEEP in the reported results (defaults to the dataset's `n_days`). One extra **buffer** day is always simulated and dropped, so the last kept day gets full lookahead. |
| `lookahead_days` | 1 | **cross-day** receding-horizon depth: each window spans the rest of the current day PLUS this many future daytime blocks (capped at the buffer day). This is what makes the window span multiple days. |
| `shrinking` / `H` | `true` / 16 | legacy within-day flags; the horizon is now the cross-day window controlled by `lookahead_days`, so these no longer bound the window. |
| `time_limit_sec` | 60 | solver seconds per window |
| `multi_activity` | `true` | allow sub-interval activity mixing |
| `require_site_visit` | `false` | MCS must visit a site (original flag) |
| `single_visit_per_site` | `false` | at most one visit per site (original flag) |
| `refit_every` | 8 | re-fit the Bayesian model every N intervals |
| `mcmc_samples` | 500 | NUTS posterior samples |
| `term_tol` | `0.1` | margin ε (kWh) on the hard CEV terminal: `SOE_end ≥ SOE_ini − ε`. `ε = 0` is the exact equality (but goes infeasible under estimator drift); `ε = 0.1` is the smallest value that keeps all 40 windows feasible on the input dataset. |
| `soft_prec` / `soft_pace` / `soft_term` | `false` | manually make precedence (12d) / pacing (13) / CEV terminal (8b) soft instead of hard. All hard by default and there is **no automatic fallback** — set these yourself if you want full relaxation. |
| `out_dir` | `../output/<mode>` | output folder (relative to `code/`): `output/input` or `output/synthetic` |

> **Synthetic dataset as CSVs:** `julia export_synthetic_data.jl` writes the built-in
> synthetic problem into `../data/synthetic_data/` using the exact same 7-CSV schema as
> `../data/input_data/`, so the two datasets can be diffed or swapped 1:1
> (`run_scenario_1(mode = :input, input_dir = "../data/synthetic_data")`).

---

## Outputs

Written to the output folder (`../output/<mode>`, e.g. `output/input/` or `output/synthetic/`):

There are **two CSVs by audience** — a simple one for the people on site, and a
detailed one for analysts:

| File | Audience | Contents |
|------|----------|----------|
| `worker_schedule.csv` | **Site workers** | Dead-simple, one row per 15 min of the **daytime** plan across all kept days: `time` (day-tagged, e.g. `D2 08:15`), each CEV's `..._activity` (Digging / Loading/Swinging / Traveling / Idle), each CEV's `..._plug_in_charge` (Yes/No), and `MCS_charge_from_grid` (Yes/No). Nothing else. |
| `closed_loop_trajectory.csv` | **Analysts** | Everything for the daytime, all kept days: a `day` tag and continuous `gstep`, within-day `k`/`clock`, grid/discharge/work power, SOEs, MCS node, time-of-use price, CO₂, **power estimates** `est_*`, uncertainty `unc_*`, observation count `n_obs`. The dropped buffer day is not included. |
| `overnight_mcs_charge_day<N>.csv` | **Analysts** | **Phase 2** overnight (18:00–08:00) MCS smart-charge, **one file per kept day**: per-interval price, charge power, SOE, and a charging flag — the cheapest-hours refill back to the MCS start level. |
| `replan_grids/day<N>/*.csv` | **Analysts** | Per-step **forward plans**, in **one subfolder per kept day**: one file each for planned grid draw, MCS SOE, and per-CEV SOE/activity (that day's own intervals). Rows = the 15-min re-plan step; columns = interval. Reading **across a row** = the whole forward plan made at that step; reading **down a column** = how the plan for one interval is **revised** as new state + Bayesian info arrive (the diagonal is what actually gets applied). |

Plus the analyst plots:

| File | Contents |
|------|----------|
| `01_grid_draw_vs_price.png` | Realized grid charging vs. time-of-use price |
| `02_state_of_energy.png` | MCS and CEV state-of-energy trajectories |
| `03_work_power.png` | Realized CEV work power |
| `04_power_estimate_convergence.png` | Online estimates (with uncertainty bands) converging to the hidden true powers |

Console prints prior vs. final estimate vs. truth and KPIs (energy, cost, CO₂,
peak demand, missed work).

> **Plain-English rules:** `constraints_explained.txt` describes every constraint
> the optimiser uses (energy-neutral MCS & CEV, two-phase daytime/overnight, rest
> rule, travel pacing, precedence, etc.) in non-technical language, with each marked
> HARD (never broken) or SOFT (a penalised goal). Precedence (12d), pacing (13) and the
> CEV terminal (8b) are HARD by default; only missed-work carries the model's own slack.

---

## Input data files (input mode, folder `data/input_data/`)

Seven CSVs describe the whole problem. **Every column below is required** (the
loader raises a clear error if a file, column, or parameter is missing); **extra
columns are ignored**. IDs (`i1`, `e1`, `m1`, …) are arbitrary strings that only
need to be **consistent across files**. Energies are **kWh**, powers **kW**, time
in **hours** (`delta_T`) and **15-min intervals**.

> The shipped `data/input_data/` is a simple reference dataset (1 grid + 1 site, 1 MCS,
> 1 CEV, 96 intervals). Copy it and edit values in place.

### `mcs_data.csv` — the Mobile Charging Station(s)
One row per MCS; the **first column is the id** (e.g. `m1`).
| Column | Unit | Notes |
|--------|------|-------|
| `SOE_min` / `SOE_max` / `SOE_ini` | kWh | reserve floor / capacity / start (start = overnight energy-neutral target) |
| `CH_MCS` | kW | max grid charge rate |
| `DCH_MCS` | kW | max total discharge rate (to CEVs) |
| `C_MCS_plug` | int | number of plugs (CEVs charged at once) |
| `DCH_MCS_plug` | kW | max rate per plug |
| `eta_ch_dch` | fraction | round-trip efficiency (e.g. 0.95) |

### `ev_data.csv` — the Construction Electric Vehicles
One row per CEV; the **first column is the id** (e.g. `e1`). Its site is set in `place.csv`.
| Column | Unit | Notes |
|--------|------|-------|
| `SOE_min` / `SOE_max` / `SOE_ini` | kWh | floor / capacity / start (start = energy-neutral target by 18:00) |
| `ch_rate` | kW | max rate the CEV can receive |
| `work_cap` | kW | nominal work-power cap (also encoded per-interval in `work_flexible.csv`) |

### `place.csv` — nodes, CEV→site assignment, and work demand
One row per node. **The node with no CEV assigned is the grid; nodes with an
assigned CEV are sites.** One `e<i>` column per CEV (1 = that CEV works here).
| Column | Notes |
|--------|-------|
| `site` | node id (e.g. `i1` = grid, `i2` = work site) |
| `e1`, `e2`, … | 1 if that CEV is assigned to this node, else 0 |
| `hours_digging` | total digging hours required at this site |
| `hours_loading_swinging` | total loading/swinging hours (precedence: cumulative loading ≤ `scale` × digging) |

### `travel_time.csv` — MCS travel times (matrix)
A square matrix: first column = origin node, remaining columns = destination nodes.
Values are **travel time in 15-min intervals** (e.g. `1` = one interval); diagonal 0.

### `time_data.csv` — the day's price + carbon profile
**`n_int` rows** (96), in chronological order. `n_int` and `t_start` are **derived
from this file** (the first time label is interval 1's end-time, so `8:15` ⇒ `t_start = 8`).
| Column | Unit | Notes |
|--------|------|-------|
| (col 1) | clock | interval end-time label (e.g. `8:15:00`) |
| (col 2) | id | `t1 … t96` |
| `lambda_buy` | $/kWh | time-of-use electricity price |
| `intensity_tons_emissions` | ton/kWh | grid carbon intensity (used for carbon cost) |
| `lambda_CO2` | — | alternate carbon column (used only if `intensity_tons_emissions` is absent) |

### `work_flexible.csv` — per-interval work availability / cap
One row per (`Location`, `EV`) pair, then **one column per interval** (96). Each
value is the CEV's **work-power cap** in that interval (e.g. `7`), or `0` when no
work is allowed — this is how the lunch break and after-hours are encoded.

### `parameters.csv` — scalar settings (`Parameter,Value,Unit,Description` rows)
Core scalar settings:
| key | Unit | Meaning |
|-----|------|---------|
| `delta_T` | h | interval length (0.25) |
| `k_trv` | kWh/h | MCS energy use while traveling (`k_trv·Δt` per in-transit interval) |
| `rho_miss` | — | penalty per hour of unfinished work |
| `rho_labor` | $/h | MCS towing labour (`rho_labor·Δt·Σ y_trv`) |
| `lambda_demand_NC` / `lambda_demand_OP` | $/kW | whole-day / on-peak (16–21) demand charges |
| `carbon_price_per_ton` | $/ton | carbon price (objective uses `price/1000 · intensity · kWh`) |
| `p_digging` / `p_loading_swinging` / `p_traveling` | kW | known activity powers |

Extra keys we add in the **same file** (so it stays one dataset):
| key | Default | Meaning |
|-----|---------|---------|
| `p_idling` | 0 | idle power draw (idle = no work) |
| `scale` | 2 | precedence factor (loading ≤ scale × digging) |
| `kappa_wt` | 4 | travel pacing (productive intervals per CEV travel) |
| `t_limit_rest` | 999 | rest-rule limit; **999 = off** (set ~1 to enable our rest rule) |
| `day_end_hour` | 18 | end of the Phase-1 daytime horizon (MCS home by then) |
| `prior_sigma_frac` | 0.2 | Bayesian prior std as a fraction of each activity power |
| `obs_noise_std` | 0.05 | telemetry noise std (simulation only) |
| `co2_unit_scale` | 1 | multiplier on the carbon-intensity column |

**Notes**
- `n_int` and `t_start` are **derived** from `time_data.csv` (don't set them).
- Node types are **inferred**: the node with no assigned CEV = grid; the rest = sites.
- Activity powers are **known constants**; the Bayesian estimator is seeded from
  them (`prior_sigma_frac`) and essentially just confirms them.
- ⚠️ Keep SOE values consistent (`min < ini < max`) and rates non-negative, or the
  MILP can be infeasible.

---

## File layout

| Section | Purpose |
|---------|---------|
| 1. Data (`build_default_data`) | Synthetic dataset (sets, params, prices, work demand) |
| 1a. Input loader (`load_input_data` / `load_data`) | CSV mode with missing-data errors |
| 1b. Estimator (`BayesianActivityEstimator`, `observe`/`refit`) | Exact TruncatedNormal + HalfNormal + NUTS regression |
| 2. Window MILP (`build_window_model` / `build_and_solve_window`) | **Faithful implementation** of the full window model over the cross-day window (rest of today + `lookahead_days`) |
| 3. MPC loop (`realized_activity_durations`, `advance_mcs_state`, `run_scenario_1`) | Multi-day, cross-day receding-horizon loop with state + parameter feedback and per-night overnight recharge |
| 4. Plotting (`make_plots`) | Output figures |
| 5. Entry point | runs `run_scenario_1` |

---

## Adapting to real data

1. **Dataset** — use `input` mode and fill `data/input_data/` with your CSVs.
2. **Prior** — set the activity powers (`p_digging` / `p_loading_swinging` /
   `p_traveling`) and `prior_sigma_frac` in `parameters.csv` to your offline values.
3. **Telemetry** — replace `realized_activity_durations` with the **actual**
   per-activity seconds from your CV pipeline, and set `b` from the **measured**
   SOC drop. The `true_power` column / `true_powers` only exist to simulate
   ground-truth telematics in the demo.

---

## Accuracy: faithful to the formal model

The window MILP follows the formal model (`math_model.tex`) — same
objective and the full constraint set: grid-connection exclusivity, plug/presence
logic, routing (departure/arrival indicators, presence partition, flow
conservation), travel energy `k_trv·Δt·y`, digging→loading precedence (12d),
the **rest rule** (12e) and **travel pacing** (13), peak-demand trackers, and the
optional `require_site_visit` / `single_visit_per_site` rules. Labour is a
**per-hour MCS towing** cost (`rho_labor·Δt·Σ y_trv`).

The controller is a **multi-day, cross-day receding horizon**. We simulate
`n_days` reported days plus **one buffer day** that is dropped from all outputs (it
absorbs the artificial end-of-horizon wrap-up so it never distorts a reported day).
Each 15-min step re-solves a window that spans the **rest of the current day plus
`lookahead_days` future daytime blocks** (global interval index `g = (day−1)·nK + k`),
so the plan always "sees" tomorrow. State **flows across days**: the CEV battery SOE
and any unfinished work carry straight through each night. It runs in **two phases**:

- **Phase 1 — daytime MPC (08:00–18:00 per day, 40 intervals/day):** the cross-day
  window MILP. Productive work (dig/load/travel) happens 08:00–12:00 and 14:00–17:00;
  the 12:00–14:00 lunch and 17:00–18:00 wind-down are non-productive but the CEV
  stays on-site and **may charge**. **Work quota is a daily, cumulative, soft target**:
  each morning a fresh `daily_dig`/`daily_load` arrives, and any shortfall is penalised
  (`rho_miss`) and **rolls over to the next day** (the target is cumulative). The
  **CEV energy-neutral terminal (8b)** — `SOE_end ≥ soe_ini − term_tol` — is applied
  **only at the true horizon end** (the buffer day's 18:00), so kept days are not forced
  back to their start SOE. The MCS must be **parked at a grid node by every 18:00**
  (hard, 10e) and draws from the grid during the day **only when needed** to stay above
  its 20% floor.
- **Phase 2 — overnight smart-charge (18:00–08:00, every night):** deterministic, not
  an MILP. The MCS (now at the grid) is refilled back to `soe_ini` in the **cheapest**
  15-min slots — closing its nightly energy-neutral cycle. In the cross-day window this
  overnight refill is modelled by resetting the MCS SOE to `soe_ini` at each night
  boundary; the realized cost is written to `overnight_mcs_charge_day*.csv`.

The MCS is therefore **energy-neutral every night** (recharged to full each morning),
while the **CEV battery carries over continuously** (no nightly reset) and only returns
to its start level at the very end of the horizon.

The model is **hard by default**: the objective is exactly Eq. (1)
(energy + carbon + demand charges + the model's own missed-work slack `rho_miss` + MCS
towing labour), and precedence (12d), travel pacing (13) and the CEV energy-neutral
terminal (8b) are **hard constraints**. State is handed off exactly between solves:

- **SOE** (MCS & CEV) re-initialize each window.
- **MCS routing** is carried over, including **in-transit** trips that span the
  apply boundary (`mcs_transit = (i, j, remaining)`; `advance_mcs_state`).
- **Daily peaks** (`peak_nc` / `peak_op`) carry over so demand charges reflect
  the whole day.
- **Completed work** enters via remaining demand `rem_*` and per-CEV cumulative
  hours `cum_*_e` (seeding precedence).

Remaining notes (not simplifications of the model, just MPC realities):

- The MILP plans at 15-min single-activity granularity; **learning and
  execution** run at true sub-interval (multi-activity) fidelity.
- Re-running NUTS every 15 min is exact but costly; `refit_every` trades fidelity
  for speed. Likewise the cross-day window re-solves the MILP each step — raise
  `time_limit_sec` or lower `lookahead_days` if runtime matters.
- The window MILP is solved with the **hard constraints only** — there is
  **no fallback**. If a re-plan from the realized carry-in state cannot satisfy them (in
  practice only the CEV energy-neutral terminal 8b can go infeasible, when a drifted
  state can't be refilled to exactly `soe_ini` in the time left), that interval is
  reported **INFEASIBLE** and the plant simply **holds state** (no work / no charging)
  for it. The number of infeasible/held windows is reported at the end of each run.
  Relaxation is available only manually via `soft_prec` / `soft_pace` / `soft_term`
  (all default `false` = hard).
- Limiting Phase 1 to the **daytime** (≤40 intervals) keeps the heavy MILP small;
  the overnight is handled by the near-free deterministic Phase-2 smart-charge.
- With a short `time_limit_sec`, HiGHS returns weaker incumbents for the daytime
  MILP (more idling, higher missed work); raise it for higher-quality schedules.
- **Edge-case tests:** run `julia test_scenario_1.jl` to verify the design rules
  (energy-neutral MCS & CEV, rest rule, travel pacing, 20% floors, MCS home at
  18:00, overnight refill at the cheapest hours, and always-feasible windows incl.
  mid-trip carry-in, depleted CEVs, heavy demand, and non-full start levels).

---

## Relation to Scenario 2

Both scenarios run the **same Bayesian regression** — the estimator fits the full
posterior over the per-activity powers with NUTS/MCMC and exposes both its mean
(`mu`) and standard deviation (`sd`), re-fitting online from telemetry. The
difference is **only in how the MILP uses that posterior**:

- **Scenario 1 (this code) — certainty-equivalent:** collapse the posterior to its
  **mean** (`mu`) and solve a single MILP on that point estimate each step. The
  `sd` is tracked (and plotted as uncertainty bands) but the optimizer ignores it.
- **Scenario 2 — scenario-based stochastic MPC:** **sample multiple power scenarios**
  from the *same* posterior and optimize over all of them, so the uncertainty (`sd`)
  actually shapes the dispatch instead of being discarded.

So it is not "point estimate vs. Bayesian" — both are Bayesian. It is
**mean-only vs. distribution-aware optimization**, and the posterior `sd` the
estimator already computes is the ready hook for Scenario 2.
