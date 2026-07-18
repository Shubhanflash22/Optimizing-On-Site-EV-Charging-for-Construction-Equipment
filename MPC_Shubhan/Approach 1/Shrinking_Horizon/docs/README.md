# Shrinking Horizon — Certainty-Equivalent MPC for MCS Dispatch

A Julia implementation of **Scenario 1 (Approach 1: Deterministic Certainty-Equivalent
MPC)** on a **single-day, full-24 h SHRINKING horizon**. It dispatches a **Mobile Charging
Station (MCS)** — a battery on wheels — to a fleet of **Construction EVs (CEVs / electric
excavators)**, deciding every 15 minutes *when to buy grid power, where to drive the MCS,
and which excavator to top up*, at minimum operating cost.

> **This README is the single deep reference.** It documents every code file, every input
> column, every constraint (each marked **HARD**/**SOFT**), and every output file. The formal
> LaTeX model is in `math_model.tex`; a line-by-line code-vs-model audit is in
> `constraints_code_vs_model.txt`.

| Program | Language | MILP solver | Power model |
|---------|----------|-------------|-------------|
| `6_Shrinking_Horizon_main.jl` (+ numbered modules `1_`…`5_`) | Julia | JuMP + HiGHS | Fixed calibrated Bayesian prior (μ, σ) |

---

## Table of contents
1. [The problem in 60 seconds](#1-the-problem-in-60-seconds)
2. [Time, the work shift & the horizon](#2-time-the-work-shift--the-horizon)
3. [Project layout — every file](#3-project-layout--every-file)
4. [Requirements & how to run](#4-requirements--how-to-run)
5. [`run_scenario_1` options](#5-run_scenario_1-options)
6. [How the controller works](#6-how-the-controller-works)
7. [Input-data schema — every file & column](#7-input-data-schema--every-file--column)
8. [The optimization model — every variable & constraint](#8-the-optimization-model--every-variable--constraint)
9. [Outputs — every file & column](#9-outputs--every-file--column)
10. [Adapting to real data](#10-adapting-to-real-data)
11. [Relation to Avik's reference & Scenario 2](#11-relation-to-aviks-reference--scenario-2)

---

## 1. The problem in 60 seconds

Heavy electric excavators draw power that depends on what they are doing —
**digging, loading/swinging, traveling, idling**. One mobile charger must keep every on-site
battery alive and get the day's work done at the least cost (time-of-use energy + carbon +
peak-demand charges + towing labour).

The controller runs a **state-feedback MPC loop**: every 15 min it re-solves a MILP from the
*measured* state (battery levels, MCS position, work done so far), applies only the **first**
interval, then re-measures and re-solves. This rejects the disturbance that **real** power
draw differs from the **planned** draw.

**Power model — fitted offline once per run (step 0).** Before the MPC loop, a **pure-Julia
Bayesian regression** (`0_Regression.jl`, step 0) reads the soil task-recording `.xlsx` files,
fits `N(μ, σ)` per activity, and writes the posterior **mean** (`p_*`) and **per-activity SD**
(`sigma_*`) into `parameters.csv` (see §6.4). The MILP then plans on the mean `μ`
(certainty-equivalent). The **stochastic plant** draws a fresh per-excavator power sample from
`N(μ, σ)` each interval, so realized consumption wobbles around the plan. The model is **not**
re-fit *during* the day (the fit is offline; step 0 can be skipped with `run_regression=false`).

**Shrinking horizon over the full day.** Each 15-min re-solve plans the **entire remaining
24 h** `[k0 … 08:00 next day]`, so the window **shrinks** as the day progresses (96 intervals
at 08:00, 1 interval at 07:45 next day). There is **one** optimisation per step and **one**
horizon — the overnight MCS recharge is scheduled *inside* the same MILP (no separate phase).

---

## 2. Time, the work shift & the horizon

All timing comes from `2_DataLoader.jl`.

* **Interval:** `delta_T = 0.25 h` (15 minutes). A full day is `n_int = 96` intervals.
* **Day start:** `t_start = 08:00`. Interval `k = 1` covers 08:00–08:15, `k = 2` 08:15–08:30, …
* **Horizon:** the **full 24 h**, `n_day = n_int = 96` (`K = 1…96`). The day runs
  **08:00 → 08:00 next day**. There is no return buffer and no `day_end_hour` parameter.
* **Work shift (synthetic):** `work_start_hour = 8`, `work_end_hour = 17`, with a
  `lunch 12:00–14:00`. So **productive work is available 08:00–12:00 and 14:00–17:00 only**;
  outside that the work-availability cap `R_work` is 0. (In `:input` the availability comes
  from `work_flexible.csv`.)

After the last work interval the CEVs can only idle/charge, and the MCS drives back to a grid
node and refills its own battery — all still planned by the single 24 h MILP.

---

## 3. Project layout — every file

```
Shrinking_Horizon/
├── code/
│   ├── 0_Regression.jl          (step 0: offline Bayesian power fit -> parameters.csv)
│   ├── 1_Common.jl
│   ├── 2_DataLoader.jl
│   ├── 3_MCSModel.jl
│   ├── 4_MPCLoop.jl
│   ├── 5_Output.jl
│   └── 6_Shrinking_Horizon_main.jl
├── data/input_data/            (7-CSV real dataset; :input mode)
├── Dummy/                      (multi-case stress harness: generate_and_run.jl + C01..C11 cases + summaries)
├── output/{input, synthetic}/  (regenerated every run)
└── docs/{README.md, math_model.tex, constraints_code_vs_model.txt}
```

> **External reference (not bundled).** The docs cite Avik's single-shot model
> (`MCS_OPTIMAL_v4_real.jl`) and the original Python Bayesian script for provenance only;
> those files live outside this folder and are **not** part of this submission bundle.

The soil task-recording `.xlsx` files that step 0 reads live in a **separate data folder**
(default `C:\Users\shubh\Desktop\Bayesian Regression`, overridable via `regression_data_dir`).
The include order is `1_Common.jl` first (so the regression can reuse its Turing model), then
`0_Regression.jl`, then `2_`…`5_`, then the driver. The **module name inside** each file is
unchanged (`module Common`, `module Regression`, …), so `using .Common` etc. still work.

| File | Module | Responsibility |
|------|--------|----------------|
| `0_Regression.jl` | `Regression` | **Step 0** (pure Julia, no Python): reads the soil `.xlsx` task files, builds the cumulative-ΔSoC energy-balance equations, fits the **same** Turing model as `1_Common.jl` (4 pooled NUTS chains), and writes the posterior mean + per-activity SD into `parameters.csv`. Fail-soft: warns and keeps the existing CSV if `XLSX.jl` / the data folder is missing. Needs `XLSX.jl`. |
| `1_Common.jl` | `Common` | Pure helpers: `normalize_travel_steps`, `in_peak`, `clock_label`/`build_time_labels`, STEP-plot builders — **plus** the `BayesianActivityEstimator` (a `TruncatedNormal(≥0)` Turing model with `observe!`/`refit!`/NUTS; `refit!` takes `nchains` for pooled multi-chain sampling). **Step 0 calls this to fit the powers offline** (§6.3); it is not re-fit *online* during the loop. |
| `2_DataLoader.jl` | `DataLoader` | Loads the whole scenario into one immutable `NamedTuple d`. `build_default_data()` (`:synthetic`) and `load_input_data(dir)` (`:input`, 7 CSVs) behind `load_data(mode)`. Full 24 h horizon (`n_day = n_int`), lumpsum work, idle power pinned to 0. |
| `3_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **single 24 h window MILP** over `[k0 … n_day]`. Nomenclature matches Avik's `MCS_OPTIMAL_v4_real.jl`. HiGHS is configured crash-tolerant (see §6.5). |
| `4_MPCLoop.jl` | `MPCLoop` | The closed loop `run_mpc(d; shrinking, H, …)`: for each 15-min step solve the (shrinking) window, apply the first interval, draw the stochastic plant's realized power, advance the real state. Returns one `res` NamedTuple. |
| `5_Output.jl` | `Output` | Every artefact from `res` via `write_outputs(res, out_dir)`: STEP figures (PNG) + CSVs, the KPI report, the replanning grids, and the plan-vs-actual **financial** + **activity** reports. |
| `6_Shrinking_Horizon_main.jl` | — | Thin orchestrator. `run_scenario_1(; mode, …)` = load → `run_mpc` → `write_outputs` → print summary. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |

---

## 4. Requirements & how to run

Install once:

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Plots", "DataFrames", "CSV", "Turing", "XLSX"])
```

`XLSX` is only needed by the **step-0** regression (reading the `.xlsx` task files); everything
else runs without it (step 0 then warns and reuses the existing `parameters.csv`).

Use a **plain terminal** (PowerShell), not the VS Code Julia REPL. Work from `code/`.

**Synthetic mode** (built-in demo — auto-runs on launch):

```bash
cd code
julia 6_Shrinking_Horizon_main.jl         # -> ../output/synthetic/
```

**Input mode** (real CSVs) — from a `julia>` prompt:

```julia
SCENARIO1_NO_AUTORUN = true               # stop the auto synthetic run on include
include("6_Shrinking_Horizon_main.jl")
run_scenario_1(mode = :input)             # step 0 refits parameters.csv, then -> ../output/input/
run_scenario_1(mode = :input, run_regression = false)   # skip step 0, reuse last parameters.csv
```

> **Step 0 runs by default in `:input` mode.** It fits the Bayesian power model (4 NUTS chains)
> and overwrites `p_*`/`sigma_*` in `parameters.csv` before the MPC. For parallel chains start
> Julia with threads (`julia -t 4 …`); with one thread the 4 chains run serially. `:synthetic`
> never runs step 0 (it builds its powers in code).

> Red `[ Info: [Turing] … ]` text is **not** an error — PowerShell colours Julia's stderr
> info logs red. If the KPI summary prints, the run succeeded. The full-24 h MILP re-solved
> 96 times takes roughly ~3 min (synthetic) / ~1 min (input) at defaults.

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `shrinking` | `true` | `true` = shrinking horizon (each step solves `[k0 … n_day]`); `false` = fixed lookahead of `H` intervals |
| `H` | `16` | fixed lookahead length in intervals (only used when `shrinking = false`) |
| `time_limit_sec` | `60.0` | HiGHS seconds per window solve |
| `multi_activity` | `false` | if `true`, a 15-min interval realizes a **mix** (planned activity for a 60–100 % random fraction, idle for the rest); if `false`, the whole interval realizes the single planned activity |
| `require_site_visit` | `false` | force the MCS to visit at least one site |
| `single_visit_per_site` | `false` | at most one visit per site |
| `mcmc_samples` | `500` | NUTS samples for the (dormant) online `refit!` path — **not** step 0 |
| `out_dir` | `../output/<mode>` | output folder |
| `run_regression` | `true` | run **step 0** (the offline Bayesian fit) before the MPC; only acts in `:input` mode |
| `regression_data_dir` | `…/Bayesian Regression` | folder holding the soil `.xlsx` task files step 0 reads |
| `regression_samples` | `2000` | NUTS draws **per chain** for step 0 |
| `regression_chains` | `4` | NUTS chains (pooled) for step 0, matching the Python reference |
| `seed` | `1` | RNG seed for the stochastic plant |

There are **no** `soft_*`, `term_tol`, or `refit_every` options — the model is fully hard and
the power model is fixed.

---

## 6. How the controller works

### 6.0 Architecture — one feedback loop, a stochastic plant

Two deliberately separated "worlds": a hidden **PLANT** (reality) and the **CONTROLLER** (the
brain). The brain plans on its fixed best guess `μ`; the plant reacts using a **sampled**
power `p_true`. The realized power drives both the battery drain and the analyst log.

```
                     ┌──────────────────────────────────────────────────┐
                     │                CONTROLLER  (brain)                 │
        plan:        │   3_MCSModel.build_window_model(state, μ)          │
   charge / route /  │   → single 24 h SHRINKING MILP over [k0 … n_day]   │
   work / plug       │   → HiGHS solves; APPLY only interval k0           │
        ▲            └───────────────┬──────────────────────▲────────────┘
        │                            │ apply k0             │ fixed mean μ
        │ measured state             ▼                       │ (calibrated prior)
        │ (SOE, position,   ┌──────────────────────────┐     │
        │  work, peaks)     │       PLANT (reality)     │     │
        └───────────────────┤ each step DRAW per CEV:   │─────┘
                            │   p_true ~ N(μ, σ)        │
                            │ batteries drain @ p_true  │
                            └──────────────────────────┘
  STATE-FEEDBACK loop : plant state after applying k0 ──► next MILP re-solve (k0+1)
```

**Fork B (fixed-curve plant).** The plant samples `p_true[e] = max.(μ + σ .* randn, 0)` from
the **fixed** calibrated model each interval. Idle has `σ = 0`, so its draw collapses to 0 (no
power lost while idling). The same `p_true[e]` drives the CEV battery drain, so reported energy
matches energy actually spent. (A "Fork A" against a separate hidden truth `N(true_powers,
true_sigma)` is a one-line edit in `4_MPCLoop.jl` step 2.5, kept for experimentation.)

**Low-level design — one 15-min step of `run_mpc`** (line numbers into `4_MPCLoop.jl`):

```
 STEP k0  (one 15-min interval)                                   4_MPCLoop.jl
 ──────────────────────────────────────────────────────────────────────────────
 (0) READ plant state:  soe_mcs, soe_cev, mcs_node, mcs_transit,
                        rem_dig, rem_load, hist, peak_nc/op
 (1) OPTIMISE  build_window_model(d, k0:nK, …state…, μ)
        └─ infeasible under HARD constraints → hold state, continue (no fallback)
 (2) APPLY interval k0 only:  grid draw, MCS discharge, route, plug decisions;
        write the forward plan into the replan grids
 (2.5) DRAW plant power PER EXCAVATOR (Fork B, fixed curve):
        p_true[e] = max.(μ + σ .* randn, 0)
 (3) SIMULATE realized activity split (60–100% planned, rest idle in multi mode)
 (4) ADVANCE plant:
        soe_mcs ← model SOE_MCS[k0+1]            (planned)
        soe_cev ← soe_cev + charged − dot(a_real, p_true[e])
        rem_dig / rem_load −= realized;  push a_real onto hist
 ──────────────────────────────────────────────────────────────────────────────
```

### 6.1 State carried between solves
SOE (MCS + CEV), MCS routing **including in-transit trips** that straddle the apply boundary,
the demand peaks, remaining lumpsum work per site, per-CEV cumulative dig/load/travel (seed the
precedence/pacing counters), and the per-CEV **applied Work/Break history** (seeds the rest-rule
seam so a work-run cannot leak across the every-15-min re-solves).

### 6.2 Single horizon (no phases)
The window MILP of §8 covers the whole 24 h. The MCS/CEV terminal rules (Eq. 8a/8b) are enforced
at the final boundary of the window, so the MCS's overnight refill is scheduled by the optimiser
itself. There is **no** separate overnight phase.

### 6.3 Step 0 — the offline Bayesian fit (`0_Regression.jl`)
Before the loop, step 0 fits the power model **from the raw data, in Julia** (a full port of
Avik's `Tasks_energy_loading_swinging_bayesian.py`; no Python needed). It:
1. reads the 12 soil `.xlsx` task files from `regression_data_dir`;
2. walks the task rows, bucketing activity durations until the cumulative `|ΔSoC|` reaches 3 %,
   emitting one energy-balance equation per bucket (`A·x = b`, hours × power = kWh);
3. fits `b ~ Normal(A·x, s)` with `TruncatedNormal(μ₀, σ₀; ≥0)` priors on the powers and a
   half-normal noise, using **4 pooled NUTS chains** (the same Turing model as §6.4);
4. writes the posterior **mean** → `p_digging/p_loading_swinging/p_traveling` and the posterior
   **SD** → `sigma_digging/sigma_loading_swinging/sigma_traveling` in `parameters.csv`.

Idle is **pinned to 0 kW** (a deliberate deviation from the Python, which estimates a small idle
power — the fleet model treats idle as no-draw). Step 0 is **fail-soft** and **skippable**
(`run_regression=false`). Only the *fit* is here; the MPC never re-fits online.

### 6.4 The power model (`1_Common.jl`)
The `BayesianActivityEstimator` holds the `TruncatedNormal(≥0)` model for the four activity
powers and the `observe!`/`refit!` (NUTS) machinery. Step 0 (§6.3) calls this to fit the powers
offline; the fitted `μ` is what the MILP consumes and `σ` is the plant's **per-activity** spread.
The same `refit!` path could also run *online* (Fork A) but is not called during the loop.

### 6.5 Solver robustness
HiGHS is pinned to `threads = 1`, `parallel = off`, `mip_heuristic_effort = 0`,
`mip_detect_symmetry = false`, `mip_rel_gap = 1e-2` (the parallel/heuristic sub-solvers were
disabled because they intermittently segfault on Windows). `optimize!` is wrapped in
`try/catch` so a rare native fault degrades to "no solution → hold state".

---

## 7. Input-data schema — every file & column

Files live in `data/input_data/`. Every listed column is **required**; extra columns are
ignored. IDs are arbitrary strings that must be **consistent across files**. Energies **kWh**,
powers **kW**, times in **hours** and **15-min intervals**.

### `parameters.csv` — two columns `Parameter, Value`
| Parameter | Req? | Meaning |
|-----------|------|---------|
| `delta_T` | ✔ | interval length in hours (0.25) |
| `k_trv` | ✔ | travel energy per arc (kWh) charged to the MCS while in transit |
| `rho_miss` | ✔ | $/hour penalty for unfinished dig/load work (soft) |
| `rho_labor` | ✔ | $/hour labour cost of the MCS being in transit (towing) |
| `lambda_demand_NC` | ✔ | $/kW non-coincident demand charge |
| `lambda_demand_OP` | ✔ | $/kW on-peak demand charge |
| `p_digging`, `p_loading_swinging`, `p_traveling` | ✔ | calibrated activity powers (kW) — the fixed model mean |
| `carbon_price_per_ton` | opt (0) | $/tonne CO₂ |
| `p_idling` | opt (0) | idle power (kW), 4th activity mean (0 = no idle drain) |
| `scale` | opt (2) | precedence ratio: max cumulative load / cumulative dig |
| `t_limit_rest` | opt (1.0) | rest rule: max hours of continuous work per rolling window |
| `sigma_digging`, `sigma_loading_swinging`, `sigma_traveling` | opt | **per-activity** plant σ (kW) — the posterior SD written by step 0; the plant samples `N(μ, σ)` per activity |
| `prior_sigma_frac` | opt (0.2) | **fallback** σ as a fraction of each power mean (min 0.05), used only when the `sigma_*` rows are absent; idle stays 0 |
| `obs_noise_std` | opt (0.05) | telematics energy-measurement noise std (used only by the dormant online learner) |
| `co2_unit_scale` | opt (1.0) | multiplier for the CO₂ column |

> `p_*` and `sigma_*` are **overwritten by step 0** on every `:input` run (unless
> `run_regression=false`). Edit them by hand only when running with step 0 disabled.

> `day_end_hour`, `kappa_wt`, `n_days`, and `work_by_day.csv` are **not** used (single-day,
> full-24 h, lumpsum model; travel pacing uses the fixed `work_per_travel = 4`).

### `ev_data.csv` — one row per CEV
`id, SOE_min, SOE_max, SOE_ini, ch_rate`. `SOE_ini` is the **end-of-day floor target** (the CEV
must finish **at or above** it); `ch_rate` = CEV charge-acceptance (kW).

### `mcs_data.csv` — one row per MCS
`id, SOE_min, SOE_max, SOE_ini, CH_MCS, DCH_MCS, C_MCS_plug, DCH_MCS_plug, eta_ch_dch`.
`CH_MCS`/`DCH_MCS` = grid-charge / total-discharge caps; `C_MCS_plug` = simultaneous plugs;
`DCH_MCS_plug` = per-plug cap; `eta_ch_dch` = round-trip efficiency.

### `place.csv` — one row per node
`site, <one column per CEV id>, hours_digging, hours_loading_swinging`. A `1` in a CEV column
marks that node as that CEV's site; a node with no CEV assigned is the **grid**. The two work
columns are the **lumpsum** dig/load requirement per site for the day.

### `travel_time.csv` — square matrix
Row header + one column per node; entry `[i,j]` = travel time from `i` to `j` **in intervals**.

### `time_data.csv` — one row per full-day interval (`n_int` rows)
A time-label column, `lambda_buy` ($/kWh), `intensity_tons_emissions` (CO₂ per kWh). `t_start`
is inferred as `first-row-clock − delta_T`.

### `work_flexible.csv` — availability
`Location, EV,` then **one column per full-day interval** giving the kW work cap (0 = no work).
These caps drive `R_work` over the full 24 h horizon.

---

## 8. The optimization model — every variable & constraint

Built in `build_window_model` over the shrinking window `K = [k0 … n_day]` (boundaries `Tb`).
Every rule is **HARD** unless marked **SOFT**. Variable names match Avik's
`MCS_OPTIMAL_v4_real.jl`.

### 8.1 Decision variables
**Continuous (≥ 0):** `P_ch_MCS`, `P_dch_MCS`, `P_MCS_CEV`, `P_work`, totals
`P_ch_tot`/`P_dch_tot`, travel energy `L_trv`/`L_trv_tot`, state `SOE_MCS[m,t]`/`SOE_CEV[e,t]`,
peaks `P_peak_NC`/`P_peak_OP`, and the single slack `s_miss_work[i,a]` (per site/activity).

**Binary:** `u[e,i,a,k]`, `mu[i,e,k]`, `rho[m,i,e,k]`, `z[m,i,k]`, `g_ch[m,i,k]`, `x[m,i,j,k]`,
`y_trv[m,i,j,k]`, `beta_arr`/`beta_dep`.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ missed work` (SOFT, ρ_miss·Σ s_miss_work) `+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP)
`+ towing labour` (ρ_labor·Δt·Σ y_trv). Six terms, identical to Avik.

### 8.3 Power flow & where power may go (HARD)
* `P_ch_tot = Σ_grid P_ch_MCS`; `P_dch_tot = Σ_site P_dch_MCS`; discharge forbidden at grid,
  charge forbidden at sites.
* `P_dch_MCS = Σ_e P_MCS_CEV` and `≤ DCH_MCS·z`.
* **Grid exclusivity:** `P_ch_MCS ≤ CH_MCS·g_ch`, `g_ch ≤ z`, ≤ 1 MCS charging per grid node.
* **Plug limits:** `P_MCS_CEV ≤ DCH_MCS_plug·rho`; `Σ_m P_MCS_CEV ≤ CH_CEV[e]·mu`.

### 8.4 Peak-demand trackers (E1, HARD)
`P_peak_NC ≥` carried-in peak and `≥ Σ_m P_ch_tot[m,k]` (all k); `P_peak_OP` likewise on the
**on-peak** k only.

### 8.5 Travel energy (HARD)
`y_trv` is 1 while a trip is in flight (its `tau_trv` intervals), or forced for a carried-in
in-transit trip. `L_trv = k_trv·Δt·y_trv`; `L_trv_tot = Σ L_trv`.

### 8.6 Battery dynamics & bounds (HARD)
* Initial: `SOE_MCS[first] = soe_mcs0`, `SOE_CEV[first] = soe_cev0` (measured carry-in).
* **MCS:** `SOE_MCS[k+1] = SOE_MCS[k] + η·P_ch_tot·Δt − P_dch_tot·Δt/η − L_trv_tot`.
* **CEV:** `SOE_CEV[k+1] = SOE_CEV[k] + Σ P_MCS_CEV·Δt − Σ P_work·Δt`.
* **Bounds:** every boundary clamped to `[SOE_min, SOE_max]` for MCS and CEV.

### 8.7 Terminal energy targets (Eq. 8a / 8b, HARD)
Applied at the final window boundary (the window always reaches day-end):
* **MCS (8a, exact):** `SOE_MCS[end] == SOE_MCS_ini` — energy-neutral, ready for the next day;
  the overnight refill is scheduled inside this MILP.
* **CEV (8b, floor):** `SOE_CEV[end] ≥ SOE_CEV_ini` — **overcharging is allowed**. Because a CEV
  cannot discharge, a hard equality would be unrecoverable whenever the stochastic plant lets a
  CEV drift above target; the floor keeps the terminal reachable while guaranteeing the fleet
  ends at least as charged as it began.

### 8.8 Routing / presence (Eq. 10, HARD)
* **Presence partition:** `Σ_i z + Σ_{i≠j} y_trv = 1` — parked at one node or in transit on one arc.
* `rho ≤ A`, `rho ≤ z`, `Σ_e rho ≤ C_MCS_plug`.
* **Departure/arrival:** `beta_dep = Σ_j x`; `beta_arr` from finishing trips (or carried-in
  arrival); `beta_arr − beta_dep = z[k] − z[k−1]`; `beta_arr + beta_dep ≤ 1`; flow balance
  (generalised for the MPC's carried-in start position).
* **(10e) Home by day-end:** `Σ_grid z[m,i,n_day] = 1` at the final interval.

### 8.9 Activity scheduling (Eq. 11, HARD)
* Exactly one activity per assigned CEV: `Σ_a u = A`; `u ≤ A`.
* Work capped & not while charging: `P_work ≤ R_work·A·(1−mu)`.
* Charging ⇒ idling: `mu ≤ u[idle]`.
* `P_work = Σ_a p_a·u`. Every `p_a` is a **constant**; idle's `p_idle = 0`, so an idling CEV
  draws no power. (The MILP uses the 4-activity encoding `B = [dig, load, trv, idle]`; Avik's
  reference uses the equivalent 3-activity form with `Σu ≤ 1`, `Σu + mu ≤ 1`.)

### 8.10 Work quota (Eq. 12c, SOFT) — lumpsum
A single dig/load requirement per site (`hours_digging`/`hours_loading_swinging`). Shortfall
`s_miss_work ≥ requirement − done` is penalised (`rho_miss`). No hard "no-working-ahead" cap:
with a single day there is no incentive to overshoot.

### 8.11 Precedence (Eq. 12d, HARD)
Cumulative loading `≤ scale ·` cumulative digging, in raw interval counts exactly as in Avik,
seeded by the realized work carried in from earlier windows.

### 8.12 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4) and `rest_win = 5`: over any 5 consecutive
intervals a CEV does **at most 4** work intervals (≥ 1 idle break). Two parts: within-window
5-windows, **plus a seam** seeded with the applied Work/Break history so a work-run cannot leak
across the every-15-min re-solves.

### 8.13 Travel pacing (Eq. 13, HARD)
Exactly as in Avik with `work_per_travel = 4`: for each `(site, CEV)`, the two-sided band
`W(k) − 4 ≤ 4·V(k) ≤ W(k)` on cumulative travel `V` vs cumulative useful work `W` (dig + load),
seeded with the travel/work already applied in earlier windows.

> **No fallback.** Windows use the hard constraints only; an infeasible re-plan is reported
> **INFEASIBLE** and the plant **holds state**. Under the Fork B stochastic plant **both**
> `:synthetic` and `:input` currently solve with **zero** infeasible windows: the MCS returns
> to its exact initial SOE and every CEV finishes at or above its start-of-day level.

---

## 9. Outputs — every file & column

Written by `5_Output.jl`, regenerated every run into `output/<mode>/`.

### 9.1 Figures (PNG) + matching CSVs
| File | Contents |
|------|----------|
| `01_total_grid_power_profile.png/.csv` | total grid charging (+) vs CEV discharging (−) |
| `02_work_profiles_by_site.png/.csv` | realized work power, one panel per site |
| `03_mcs_state_of_energy.png/.csv` | MCS SOE with min/max guides |
| `04_cev_state_of_energy.png/.csv` | CEV SOE with min/max guides |
| `05_electricity_prices_emissions.png` / `05_electricity_prices.csv` | price (left) + CO₂ (right) |
| `06_mcs_location_trajectory.png/.csv` | MCS node over time (0 = transit) |
| `07_mcs_optimization_summary.png` + `07_mcs_cev_soe.csv` | combined overview + long-form per-interval MCS+CEV table |
| `08_kpi_metrics_summary.png` + `08_cost_kpi_metrics.csv` | KPI bar summary + metric/value totals |
| `09_mcs_<m>_power_profile.png/.csv` | per-MCS charging/discharging |

### 9.2 Reports
**`08_cost_kpi_metrics.csv`** — metric/value rows: `Total_Cost_USD`, `Total_Energy_Cost_USD`,
`Total_CO2_Cost_USD`, `NC_demand_charge_USD`, `OP_demand_charge_USD`, `Missed_Work_Penalty_USD`,
`Travel_Labour_USD`, `Total_Grid_Energy_kWh`, `Total_CO2_Emissions_kg`, `NCD_Peak_kW`,
`OPD_Peak_kW`, `Missed_Work_hour`, `MCS_Transit_hour`, `Infeasible_windows`, `MPC_loop_time_s`.

**`replan_grids/*.csv` (+ `.html`)** — five grids: `plan_grid_kW`, `plan_mcs_soe`,
`plan_mcs_activity`, `plan_cev<e>_soe`, `plan_cev<e>_activity`. **Rows** = the 15-min re-plan
step; **columns** = the interval being planned. Across a row = the whole plan made at that step;
down a column = how one interval's plan is revised as new state arrives; the diagonal is what was
applied. The `.html` colours past (green) vs pending (yellow).

**`plan_vs_actual_*`** — the **first optimisation at 08:00** (the all-“pending”/yellow row of the
replan grid, i.e. the whole-day plan made before any disturbance) vs the **realised** day (the
“done”/green diagonal, after closed-loop re-planning against the stochastic plant). Two families:

*Financial* (cost/energy plan vs realised):
- `plan_vs_actual_costs.png` — grouped bar of the cost components (yellow = plan @08:00, green = realised).
- `plan_vs_actual.html` — the summary table (planned vs realised vs Δ vs %Δ for grid energy, energy
  cost, CO₂ + cost, NCD/OPD peaks + charges, missed work + penalty, transit + labour, and **total
  cost**) plus the literal yellow (plan) vs green (realised) per-interval grid-power rows.

*Activity* (planned vs realised activity per 15-min interval, one label per CEV + the MCS; every
interval where the realised activity differs from the plan is highlighted):
- `plan_vs_actual_activity.png` — timeline heatmap; per CEV and the MCS a two-row band (Planned over
  Actual), every changed interval boxed in red, plus a shared activity-colour legend.
- `plan_vs_actual_side_by_side.html` — table laid out as **all Planned columns, then all Actual
  columns** (one column per CEV + MCS in each block); changed Actual cells outlined red.
- `plan_vs_actual_by_entity.html` — the same data **grouped per unit**: each CEV/MCS shows its
  `Planned @ 08:00` column immediately beside its `Actual` column; changed Actual cells outlined red.

*Activity labels (reporting only — derived from the solution, no model change):*
- **`plan_cev<e>_activity`** — combined per-excavator label: `Digging` / `Loading/Swinging` /
  `Traveling` / **`Charging`** (real power delivered, `Σ_m P_MCS_CEV > 0`) / `Idle` (a genuine
  break). Charging is keyed off *delivered power*, not the plug-in permission bit `mu` (which the
  MILP may leave =1 with zero flow), so it always agrees with the MCS grid. Charging is shown
  explicitly instead of the idle slot it occupies, so the crew sees when to plug in.
- **`plan_mcs_activity`** — MCS status: **`Charging (grid)`** (pulling from the grid) /
  **`Serving CEV`** (discharging into an excavator) / `Traveling` (in transit) / `Idle` (parked,
  doing nothing).

The dummy stress harness (`Dummy/generate_and_run.jl`) additionally writes, per case, a
**`comparison.html`** — the applied plan (grid diagonal) shown interval-by-interval with every
CEV's activity beside the MCS status, so `Charging` (CEV) always lines up with `Serving CEV`
(MCS). A compact grouped version of all cases is collected in `Dummy/comparisons_grouped.txt`.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7.
2. **Power model** — set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling` and
   `prior_sigma_frac` in `parameters.csv` to your calibrated values.
3. **Telemetry** — in `4_MPCLoop.jl`, replace the simulated `realized_activity_durations` and
   the `p_true` sampling block at step (2.5) with the actual per-activity seconds + measured
   interval energy from your pipeline. The `p_true` / `true_powers` machinery only generates
   ground truth for the demo and disappears at go-live.

---

## 11. Relation to Avik's reference & Scenario 2

* **Avik's single-shot model** (`Avik/MCS_OPTIMAL_v4_real.jl`) solves the whole day in **one**
  MILP with deterministic powers and exact terminal equalities. This code is the **MPC** version
  of the same formulation: identical variable names, objective, routing, battery, precedence
  (raw-count) and travel-pacing (`work_per_travel = 4`); the differences are the shrinking-horizon
  re-solves, per-window history seeds, the carried-in MCS start position, and the CEV terminal
  **floor** (`≥`, overcharge allowed) instead of exact equality. On the real data the closed loop
  reproduces Avik's single-shot optimum to within ~0.2 %.
* **Scenario 1 vs 2.** Scenario 1 (this code) is certainty-equivalent (plans on the mean `μ`).
  Scenario 2 would sample multiple power scenarios from `N(μ, σ)` — the fixed `σ` is the hook.
