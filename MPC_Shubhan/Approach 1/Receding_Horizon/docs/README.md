# Receding Horizon — Multi-Day Certainty-Equivalent MPC for MCS Dispatch

A Julia implementation of **Scenario 1 (Approach 1: Deterministic Certainty-Equivalent
MPC)** on a **multi-day, cross-day RECEDING horizon**. It dispatches a **Mobile Charging
Station (MCS)** — a battery on wheels — to a fleet of **Construction EVs (CEVs / electric
excavators)**, deciding every 15 minutes *when to buy grid power, where to drive the MCS,
and which excavator to top up*, at minimum operating cost, across a run of work days.

> **This README is the single deep reference.** It documents every code file, every input
> column, every constraint (each marked **HARD**/**SOFT**), and every output file. The formal
> LaTeX model is in `math_model.tex`; a line-by-line code-vs-model audit is in
> `constraints_code_vs_model.txt`.

| Program | Language | MILP solver | Power model |
|---------|----------|-------------|-------------|
| `6_Receding_Horizon_main.jl` (+ numbered modules `1_`…`5_`) | Julia | JuMP + HiGHS | Fixed calibrated Bayesian prior (μ, σ) |

---

## Table of contents
1. [The problem in 60 seconds](#1-the-problem-in-60-seconds)
2. [Time, the work shift & the multi-day horizon](#2-time-the-work-shift--the-multi-day-horizon)
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
battery alive and get each day's work done at the least cost (time-of-use energy + carbon +
peak-demand charges + towing labour), day after day.

The controller runs a **state-feedback MPC loop**: every 15 min it re-solves a MILP from the
*measured* state (battery levels, MCS position, work done so far), applies only the **first**
interval, then re-measures and re-solves. This rejects the disturbance that **real** power
draw differs from the **planned** draw.

**Power model — fixed, calibrated once.** The four activity powers are a **fixed** Bayesian
estimate `N(μ, σ)` (the calibrated posterior of an offline regression; see §6.3). The MILP
plans on the mean `μ` (certainty-equivalent). The **stochastic plant** draws a fresh
per-excavator power sample from `N(μ, σ)` each interval, so realized consumption wobbles
around the plan. The model is **not** re-fit during the run (no online learning).

**Multi-day cross-day RECEDING horizon.** The simulation runs `n_days` **reported** days plus
**one dropped BUFFER day** (`D_total = n_days + 1`). Each 15-min re-solve optimises a
**cross-day window** = the rest of *today's* daytime block **plus** `lookahead_days` future
daytime blocks. Because only the first interval is applied and the window slides forward, the
horizon **recedes**. Between days, a separate **deterministic overnight recharge (Phase 2)**
refills the MCS to full at the cheapest hours; the CEV battery and any unfinished work carry
over into the next day. The buffer day is dropped from every reported output so the CEV
terminal wrap-up never lands on a reported day.

---

## 2. Time, the work shift & the multi-day horizon

All timing comes from `2_DataLoader.jl`.

* **Interval:** `delta_T = 0.25 h` (15 minutes).
* **Daytime block:** each day's optimised horizon is the **daytime** interval set `K`
  (`n_day` intervals, e.g. 40 = 08:00→18:00), inferred from the last interval with any work
  availability. `day_end_hour = t_start + n_day·delta_T`.
* **Day start:** `t_start = 08:00`. Interval `k = 1` covers 08:00–08:15, `k = 2` 08:15–08:30, …
* **Reported vs simulated days:** `n_days` days are **kept**; the loop simulates
  `D_total = n_days + 1` and drops the last (buffer) day. Per-interval realized arrays are
  captured over the **kept** days, concatenated end to end.
* **Cross-day window:** at day `dy`, step `k0`, the window spans `[global k0 … end of day
  min(D_total, dy + lookahead_days)]`. `lookahead_days = 1` means "rest of today + all of the
  next daytime block".
* **Nights:** handled *outside* the daytime MILP by `phase2_overnight_charge` — a deterministic
  smart-charge that refills the MCS to `SOE_MCS_ini` at the cheapest overnight hours. The CEV
  battery is **not** recharged overnight beyond what the daytime plan delivered; it carries over.
* **Work shift (synthetic):** productive work is available inside the daytime block only;
  outside it the work-availability cap `R_work` is 0. (In `:input` the availability comes from
  `work_flexible.csv`.)

---

## 3. Project layout — every file

```
Receding_Horizon/
├── code/
│   ├── 1_Common.jl
│   ├── 2_DataLoader.jl
│   ├── 3_MCSModel.jl
│   ├── 4_MPCLoop.jl
│   ├── 5_Output.jl
│   └── 6_Receding_Horizon_main.jl
├── data/input_data/            (7-CSV real dataset; optional work_by_day.csv)
├── output/{input, synthetic}/
├── docs/{README.md, math_model.tex, constraints_code_vs_model.txt}
└── Dummy/generate_and_run.jl   (multi-case stress harness)
```

The number prefix `1_`…`6_` is exactly the order the entry point `include`s them
(dependencies first). The **module name inside** each file is unchanged (`module Common`, …),
so `using .Common` etc. still work.

| File | Module | Responsibility |
|------|--------|----------------|
| `1_Common.jl` | `Common` | Pure helpers: `normalize_travel_steps`, `in_peak`, `clock_label`/`clock_day_label`/`build_time_labels`/`build_time_labels_days`, multi-day x-ticks, STEP-plot builders — **plus** the `BayesianActivityEstimator`. In this pipeline the estimator is used in **calibrated-prior mode**: `μ,σ` are the fixed prior; the `observe!`/`refit!` fitting path is present but **dormant** (never called). |
| `2_DataLoader.jl` | `DataLoader` | Loads the whole scenario into one immutable `NamedTuple d`. `build_default_data()` (`:synthetic`) and `load_input_data(dir)` (`:input`, 7 CSVs) behind `load_data(mode)`. Infers the daytime block (`n_day`, `day_end_hour`), reads `n_days` + per-day work quotas (`dig_by_day`/`load_by_day`, from optional `work_by_day.csv`), idle power pinned to 0. |
| `3_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **cross-day window MILP** over `K_win`; `phase2_overnight_charge(...)` is the deterministic overnight recharge. HiGHS is configured crash-tolerant (see §6.4). |
| `4_MPCLoop.jl` | `MPCLoop` | The multi-day closed loop `run_mpc(d; n_days, lookahead_days, …)`: for each day and each 15-min step solve the cross-day window, apply the first interval, draw the stochastic plant's realized power, advance the real state; each night run Phase 2 and reset the MCS to full; drop the buffer day. Returns one `res` NamedTuple. |
| `5_Output.jl` | `Output` | Every artefact from `res` via `write_outputs(res, out_dir)`: STEP figures (PNG) + CSVs over the concatenated kept horizon, KPI/cost reports, per-window solver diagnostics, worker schedule, per-day overnight tables, and per-day replanning grids. |
| `6_Receding_Horizon_main.jl` | — | Thin orchestrator. `run_scenario_1(; mode, …)` = load → `run_mpc` → `write_outputs` → print summary. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |

---

## 4. Requirements & how to run

Install once:

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Plots", "DataFrames", "CSV", "Turing"])
```

Use a **plain terminal** (PowerShell), not the VS Code Julia REPL. Work from `code/`.

**Synthetic mode** (built-in demo — auto-runs on launch):

```bash
cd code
julia 6_Receding_Horizon_main.jl          # -> ../output/synthetic/
```

**Input mode** (real CSVs) — from a `julia>` prompt:

```julia
SCENARIO1_NO_AUTORUN = true               # stop the auto synthetic run on include
include("6_Receding_Horizon_main.jl")
run_scenario_1(mode = :input)             # -> ../output/input/
run_scenario_1(mode = :input, n_days = 1) # faster: keep 1 day (+ buffer)
```

> Red `[ Info: [Turing] … ]` text is **not** an error — PowerShell colours Julia's stderr
> info logs red. If the KPI summary prints, the run succeeded. Cost scales with
> `(n_days + 1) · n_day` window solves; keep `n_days` and `time_limit_sec` modest while iterating.

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `n_days` | `nothing` | reported days to **keep** (a buffer day is always simulated + dropped). `nothing` → use `d.n_days` from the data |
| `lookahead_days` | `1` | cross-day window depth: rest of today + this many future daytime blocks |
| `time_limit_sec` | `60.0` | HiGHS seconds per window solve |
| `multi_activity` | `false` | if `true`, a 15-min interval realizes a **mix** (planned activity for a 60–100 % random fraction, idle for the rest); if `false`, the whole interval realizes the single planned activity |
| `require_site_visit` | `false` | force the MCS to visit at least one site |
| `single_visit_per_site` | `false` | at most one visit per site |
| `soft_prec` | `false` | relax precedence (Eq. 12d) to a penalised slack |
| `soft_pace` | `false` | relax travel pacing (Eq. 13) to a penalised slack |
| `soft_term` | `false` | make the daily CEV terminal target soft |
| `term_tol` | `0.1` | tolerance band for the (soft) terminal target |
| `mcmc_samples` | `500` | NUTS posterior samples (only used if the dormant `refit!` path is enabled) |
| `out_dir` | `../output/<mode>` | output folder |
| `seed` | `1` | RNG seed for the stochastic plant |

> There is **no** `shrinking`/`H` (that was the single-day variant) and **no** `refit_every`
> (the power model is fixed — no online learning).

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
   charge / route /  │   → CROSS-DAY window MILP over [k0 … lookahead end]│
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
  Each NIGHT : phase2_overnight_charge refills the MCS to full; state carries to next day.
```

**Fork B (fixed-curve plant).** The plant samples `p_true[e] = max.(μ + σ .* randn, 0)` from
the **fixed** calibrated model each interval. Idle has `σ = 0`, so its draw collapses to 0 (no
power lost while idling). The same `p_true[e]` drives the CEV battery drain, so reported energy
matches energy actually spent. (A "Fork A" against a separate hidden truth `N(true_powers,
true_sigma)` plus online `observe!`/`refit!` learning is a small edit in `4_MPCLoop.jl`, kept
for experimentation; the shipped loop does not learn online.)

**Low-level design — one 15-min step of `run_mpc`** (`4_MPCLoop.jl`):

```
 STEP (day dy, k0)  (one 15-min interval)                        4_MPCLoop.jl
 ──────────────────────────────────────────────────────────────────────────────
 (0) READ plant state:  soe_mcs, soe_cev, mcs_node, mcs_transit,
                        rem_dig, rem_load, hist (current day), peak_nc/op
 (1) OPTIMISE  build_window_model(d, K_win, …state…, μ)
        └─ hard-infeasible → re-solve once with SOFT terminal + relaxed reserve
        └─ still infeasible → hold state, continue
 (2) APPLY interval k0 only:  grid draw, MCS discharge, route, plug decisions;
        write the current-day slice of the forward plan into the replan grids
 (2.5) DRAW plant power PER EXCAVATOR (Fork B, fixed curve):
        p_true[e] = max.(μ + σ .* randn, 0)
 (3) SIMULATE realized activity split (60–100% planned, rest idle in multi mode)
 (4) ADVANCE plant:
        soe_mcs ← applied flows (η·charge − discharge/η − travel)
        soe_cev ← soe_cev + charged − dot(a_real, p_true[e])
        rem_dig / rem_load −= realized;  push (activity, a_real) onto hist
 END OF DAY : missed-work snapshot; phase2_overnight_charge(soe_mcs);
              soe_mcs ← SOE_MCS_ini (full); hist reset; carry soe_cev + rem_* over.
 ──────────────────────────────────────────────────────────────────────────────
```

### 6.1 State carried between solves
SOE (MCS + CEV), MCS routing **including in-transit trips** that straddle the apply boundary,
the demand peaks, remaining per-site work (`rem_dig`/`rem_load`, replenished by each day's quota
and rolled over if unfinished), and the per-CEV **applied activity history** `hist` for the
**current day** (reset each morning). `hist` seeds the precedence/pacing counters and the
rest-rule seam so a work-run cannot leak across the every-15-min re-solves; a night is a long
break, so those counters restart.

### 6.2 Two mechanisms: daytime MILP + overnight phase
The window MILP of §8 covers only **daytime** blocks. The CEV terminal target (Eq. 8b) is
enforced at each day-end boundary inside the window; the MCS is **not** required to be
energy-neutral inside the MILP — instead each night the deterministic
`phase2_overnight_charge` refills it to `SOE_MCS_ini` at the cheapest hours and reports the
overnight energy/cost. This keeps the daytime MILP small and the overnight recharge optimal
against the night price curve.

### 6.3 The power model (`1_Common.jl`)
The `BayesianActivityEstimator` holds the calibrated `TruncatedNormal(≥0)` prior for the four
activity powers. Here `μ,σ` are used **as the fixed model** (an offline regression supplies the
calibrated values that seed the prior). `observe!`/`refit!` (NUTS) exist for future
online-learning experiments but are **not** called during a run; `μ` is what the MILP consumes
and `σ` is the plant's per-activity spread.

### 6.4 Solver robustness
HiGHS is pinned to `threads = 1`, `parallel = off`, `mip_heuristic_effort = 0`,
`mip_detect_symmetry = false`, `mip_rel_gap = 1e-2` (the parallel/heuristic sub-solvers were
disabled because they intermittently segfault on Windows). `optimize!` is wrapped in
`try/catch` so a rare native fault degrades to "no solution → soft re-solve → hold state".

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
| `prior_sigma_frac` | opt (0.2) | plant σ as a fraction of each power mean (min 0.05); idle stays 0 |
| `obs_noise_std` | opt (0.05) | telematics energy-measurement noise std (used only by the dormant learner) |
| `co2_unit_scale` | opt (1.0) | multiplier for the CO₂ column |
| `n_days` | opt (2) | reported days to keep (a buffer day is always simulated + dropped) |

> `day_end_hour` is **inferred** from work availability (no parameter). `kappa_wt` is derived.
> Per-day work quotas come from the optional `work_by_day.csv` (below); if absent, the single
> `place.csv` quota is repeated for every day.

### `ev_data.csv` — one row per CEV
`id, SOE_min, SOE_max, SOE_ini, ch_rate`. `SOE_ini` is the **daily end-of-day floor target** (the
CEV must finish each daytime block **at or above** it); `ch_rate` = CEV charge-acceptance (kW).

### `mcs_data.csv` — one row per MCS
`id, SOE_min, SOE_max, SOE_ini, CH_MCS, DCH_MCS, C_MCS_plug, DCH_MCS_plug, eta_ch_dch`.
`CH_MCS`/`DCH_MCS` = grid-charge / total-discharge caps; `C_MCS_plug` = simultaneous plugs;
`DCH_MCS_plug` = per-plug cap; `eta_ch_dch` = round-trip efficiency. `SOE_ini` is the level the
overnight phase refills to each night.

### `place.csv` — one row per node
`site, <one column per CEV id>, hours_digging, hours_loading_swinging`. A `1` in a CEV column
marks that node as that CEV's site; a node with no CEV assigned is the **grid**. The two work
columns are the default per-site dig/load requirement, used for **every** day unless
`work_by_day.csv` overrides it.

### `work_by_day.csv` — OPTIONAL per-day work quota
`site, day, hours_digging, hours_loading_swinging`. One row per (site, day). If present, it
overrides `place.csv` per reported day `1…n_days`; missing days get zero fresh work. If the file
is absent, the `place.csv` quota is repeated each day.

### `travel_time.csv` — square matrix
Row header + one column per node; entry `[i,j]` = travel time from `i` to `j` **in intervals**.

### `time_data.csv` — one row per **daytime** interval
A time-label column, `lambda_buy` ($/kWh), `intensity_tons_emissions` (CO₂ per kWh) over the
daytime block. `t_start` is inferred as `first-row-clock − delta_T`. The overnight price curve
used by Phase 2 is derived from the same daily profile.

### `work_flexible.csv` — availability
`Location, EV,` then **one column per daytime interval** giving the kW work cap (0 = no work).
These caps drive `R_work` and, via the last non-zero column, infer `day_end_hour`.

---

## 8. The optimization model — every variable & constraint

Built in `build_window_model` over the cross-day window `K_win = [k0 … lookahead day-end]`
(boundaries `Tb`). Every rule is **HARD** unless marked **SOFT**. Variable names match Avik's
`MCS_OPTIMAL_v4_real.jl`.

### 8.1 Decision variables
**Continuous (≥ 0):** `P_ch_MCS`, `P_dch_MCS`, `P_MCS_CEV`, `P_work`, totals
`P_ch_tot`/`P_dch_tot`, travel energy `L_trv`/`L_trv_tot`, state `SOE_MCS[m,t]`/`SOE_CEV[e,t]`,
peaks `P_peak_NC`/`P_peak_OP`, and slacks (`s_miss_dig`/`s_miss_load` per (site, day-block);
plus terminal `s_term_cev` / precedence `s_prec` / pacing `s_pace_hi`,`s_pace_lo` slacks when the
matching `soft_*` lever is on).

**Binary:** `u[e,i,a,k]`, `mu[i,e,k]`, `rho[m,i,e,k]`, `z[m,i,k]`, `g_ch[m,i,k]`, `x[m,i,j,k]`,
`y_trv[m,i,j,k]`, `beta_arr`/`beta_dep`.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ missed work` (SOFT, ρ_miss·Σ (s_miss_dig + s_miss_load) over sites × day-blocks)
`+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP)
`+ towing labour` (ρ_labor·Δt·Σ y_trv), plus any active `soft_*` penalties. The overnight
recharge cost is reported separately by Phase 2 (not inside this objective).

### 8.3 Power flow & where power may go (HARD)
* `P_ch_tot = Σ_grid P_ch_MCS`; `P_dch_tot = Σ_site P_dch_MCS`; discharge forbidden at grid,
  charge forbidden at sites.
* `P_dch_MCS = Σ_e P_MCS_CEV` and `≤ DCH_MCS·z`.
* **Grid exclusivity:** `P_ch_MCS ≤ CH_MCS·g_ch`, `g_ch ≤ z`, ≤ 1 MCS charging per grid node.
* **Plug limits:** `P_MCS_CEV ≤ DCH_MCS_plug·rho`; `Σ_m P_MCS_CEV ≤ CH_CEV[e]·mu`.

### 8.4 Peak-demand trackers (E1, HARD)
`P_peak_NC ≥` carried-in peak and `≥ Σ_m P_ch_tot[m,k]` (all k); `P_peak_OP` likewise on the
**on-peak** k only. Peaks carry across the day's re-solves.

### 8.5 Travel energy (HARD)
`y_trv` is 1 while a trip is in flight (its `tau_trv` intervals), or forced for a carried-in
in-transit trip. `L_trv = k_trv·Δt·y_trv`; `L_trv_tot = Σ L_trv`.

### 8.6 Battery dynamics & bounds (HARD)
* Initial: `SOE_MCS[first] = soe_mcs0`, `SOE_CEV[first] = soe_cev0` (measured carry-in).
* **MCS:** `SOE_MCS[k+1] = SOE_MCS[k] + η·P_ch_tot·Δt − P_dch_tot·Δt/η − L_trv_tot`. At a
  night boundary inside the window the MCS SOE is bridged to `SOE_MCS_ini` (the overnight refill
  is realized by Phase 2 in the loop).
* **CEV:** `SOE_CEV[k+1] = SOE_CEV[k] + Σ P_MCS_CEV·Δt − Σ P_work·Δt` (no overnight recharge).
* **Bounds:** every boundary clamped to `[SOE_min, SOE_max]` for MCS and CEV.

### 8.7 Terminal energy targets & keep-up reserve
* **CEV daily terminal (Eq. 8b):** at each day-end boundary in the window,
  `SOE_CEV[day-end] ≥ SOE_CEV_ini` — **overcharging is allowed** (a CEV cannot discharge, so a
  hard equality would be unrecoverable once the stochastic plant lets a CEV drift above target).
  Made soft (with `term_tol`) under `soft_term`, or by the loop's infeasibility fallback.
* **Keep-up reserve (HARD, relaxable):** a per-window reserve keeps enough MCS energy on hand
  to top the fleet; dropped when the loop re-solves with `soft_reserve` after a hard
  infeasibility, so a drifted low-battery state can still charge toward target instead of
  freezing.
* **MCS:** no in-MILP energy-neutral equality — the overnight Phase 2 restores it each night.
* **Anti-hoarding over-charge cap (HARD, always on):** each CEV is capped at
  `SOE_ini + overcharge_frac·(SOE_max − SOE_ini)` at every boundary after the carried-in start,
  so the single MCS cannot park at the near site charging one CEV far above target while the far
  one starves. It is an upper bound below `SOE_max`, so it can never make the window infeasible.
* **Daytime health floor (HARD, relaxable):** no CEV may drop more than `drawdown_kwh` below its
  start-of-day level at any daytime boundary, forcing the MCS to **shuttle** between CEVs rather
  than camp at one. Dropped under `soft_reserve` in the loop fallback.

### 8.8 Routing / presence (Eq. 10, HARD)
* **Presence partition:** `Σ_i z + Σ_{i≠j} y_trv = 1` — parked at one node or in transit on one arc.
* `rho ≤ A`, `rho ≤ z`, `Σ_e rho ≤ C_MCS_plug`.
* **Departure/arrival:** `beta_dep = Σ_j x`; `beta_arr` from finishing trips (or carried-in
  arrival); `beta_arr − beta_dep = z[k] − z[k−1]`; `beta_arr + beta_dep ≤ 1`; flow balance
  (generalised for the MPC's carried-in start position).
* **Home by day-end:** the MCS is at a grid node at each day-end boundary in the window.

### 8.9 Activity scheduling (Eq. 11, HARD)
* Exactly one activity per assigned CEV: `Σ_a u = A`; `u ≤ A`.
* Work capped & not while charging: `P_work ≤ R_work·A·(1−mu)`.
* Charging ⇒ idling: `mu ≤ u[idle]`.
* `P_work = Σ_a p_a·u`. Every `p_a` is a **constant** (`μ`); idle's `p_idle = 0`, so an idling
  CEV draws no power. (The MILP uses the 4-activity encoding `B = [dig, load, trv, idle]`.)

### 8.10 Work quota (Eq. 12c, SOFT) — per-day, rolling over
Each reported day issues its **own** per-site dig/load quota (`dig_by_day`/`load_by_day`), added
to the carried-over remainder `rem_dig`/`rem_load` each morning. The quota is **cumulative** and
pinned from **both** sides against the running work done through the end of each day-block:
- **lower (SOFT):** any shortfall `s_miss_*` is penalised (`rho_miss`) and **rolls over** into the
  next day (it is re-charged there against the still-remaining requirement);
- **upper (HARD) — no working ahead:** cumulative work through the end of a day may not exceed the
  cumulative quota, so a day can catch up earlier unfinished work but can never borrow from a
  **future** day.

The dropped buffer day gets **no** fresh quota.

### 8.11 Precedence (Eq. 12d, HARD / SOFT)
Cumulative loading `≤ scale ·` cumulative digging, in raw interval counts exactly as in Avik,
seeded by the realized work carried in for the **current day**. Relaxed to a penalised slack
under `soft_prec`.

### 8.12 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4) and `rest_win = 5`: over any 5 consecutive
intervals a CEV does **at most 4** work intervals (≥ 1 idle break). Two parts: within-window
5-windows, **plus a seam** seeded with the current day's applied activity history so a work-run
cannot leak across the every-15-min re-solves. Reset each morning (a night is a long break).

### 8.13 Travel pacing (Eq. 13, HARD / SOFT)
As in Avik with `work_per_travel = 4`: for each `(site, CEV)`, the two-sided band
`W(k) − 4 ≤ 4·V(k) ≤ W(k)` on cumulative travel `V` vs cumulative useful work `W` (dig + load),
seeded with the travel/work already applied this day. Relaxed under `soft_pace`.

> **Graceful fallback.** A hard-infeasible window (daily terminal + reserve jointly unreachable
> from a drifted state) triggers a single re-solve with a **soft terminal + relaxed reserve** so
> the MCS still charges toward target; any residual shortfall surfaces as a penalty. Only if that
> too fails does the plant **hold state**. Counts are reported as `Softened_windows` /
> `Infeasible_windows`.

---

## 9. Outputs — every file & column

Written by `5_Output.jl`, regenerated every run into `output/<mode>/`, over the **kept**
concatenated multi-day horizon (buffer day already dropped). Figures use multi-day x-ticks;
per-interval daily profiles (price, CO₂) are indexed by within-day position.

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
| `mcs_<m>_power_profile.png/.csv` | per-MCS charging/discharging |
| `11_power_estimate_convergence.png` | fixed activity powers `μ ± σ` vs hidden truth (flat under Fork B) |

### 9.2 Reports
**`08_cost_emissions_timeseries.csv` / `08_cost_emissions_summary.png`** — per-interval grid
energy / cost / CO₂ with running cumulatives.

**`09_cost_kpi_metrics.csv` / `09_kpi_metrics_summary.png`** — metric/value rows:
`Total_Cost_USD`, `Total_Energy_Cost_USD`, `Total_CO2_Cost_USD`, `NC_demand_charge_USD`,
`OP_demand_charge_USD`, `Missed_Work_Penalty_USD`, `Travel_Labour_USD`, `Total_Grid_Energy_kWh`,
`Total_CO2_Emissions_kg`, `NCD_Peak_kW`, `OPD_Peak_kW`, `Missed_Work_hour`, `MCS_Transit_hour`,
`Overnight_Recharge_kWh`, `Overnight_Cost_USD`, `Softened_windows`, `Infeasible_windows`,
`MPC_loop_time_s`.

**`10_mip_convergence.csv`** — per-window solver diagnostics (day, step, clock, status,
objective, gap %, solve time).

**`closed_loop_trajectory.csv`** — one row per applied interval: `day`, `gstep`, `k` (within-day),
`clock` (`D<day> HH:MM`), `price`, `co2`, `grid_kW`, `dch_kW`, `work_kW`, `soe_mcs`, `soe_cev1`,
`soe_cev2` (`NaN` if absent), `mcs_node` (0 = transit), `est_*`, `unc_*`, `n_obs`. Held/infeasible
intervals appear with `grid=dch=work=0`.

**`overnight_mcs_charge_day<d>.csv`** — the deterministic Phase-2 overnight recharge schedule,
one file per kept day.

**`worker_schedule.csv`** — plain-words site instructions: `time` (`D<day> HH:MM`), per-CEV
`activity` and `plug_in_charge` (`Yes` only when real power is delivered), and
`MCS_charge_from_grid`.

**`replan_grids/day<d>/*.csv` (+ `.html`)** — per kept day, five grids: `plan_grid_kW`,
`plan_mcs_soe`, `plan_mcs_activity`, `plan_cev<e>_soe`, `plan_cev<e>_activity`. **Rows** = the
15-min re-plan step; **columns** = the interval being planned. Across a row = the whole plan made
at that step; down a column = how one interval's plan is revised as new state arrives; the
diagonal is what was applied. The `.html` colours past (green) vs pending (yellow).

*Activity labels (reporting only — derived from the solution, no model change):*
- **`plan_cev<e>_activity`** — combined per-excavator label: `Digging` / `Loading/Swinging` /
  `Traveling` / **`Charging`** (real power delivered, `Σ_m P_MCS_CEV > 0`) / `Idle` (a genuine
  break). Charging is keyed off *delivered power*, not the plug-in permission bit `mu` (which the
  MILP may leave =1 with zero flow), so it always agrees with the MCS grid.
- **`plan_mcs_activity`** — MCS status: **`Charging (grid)`** / **`Serving CEV`** / `Traveling` /
  `Idle`.

The dummy stress harness (`Dummy/generate_and_run.jl`) additionally writes, per case, a
**`comparison.html`** — the applied plan (grid diagonal) shown interval-by-interval with every
CEV's activity beside the MCS status, so `Charging` (CEV) always lines up with `Serving CEV`
(MCS). A compact grouped version of all cases is collected in `Dummy/comparisons_grouped.txt`.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7. Set
   `n_days` and (optionally) `work_by_day.csv` for the multi-day work plan.
2. **Power model** — set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling` and
   `prior_sigma_frac` in `parameters.csv` to your calibrated values.
3. **Telemetry** — in `4_MPCLoop.jl`, replace the simulated `realized_activity_durations` and
   the `p_true` sampling block at step (2.5) with the actual per-activity seconds + measured
   interval energy from your pipeline. The `p_true` / `true_powers` machinery only generates
   ground truth for the demo and disappears at go-live. (To *learn* online instead of using a
   fixed model, re-enable the `observe!`/`refit!` path — Fork A.)

---

## 11. Relation to Avik's reference & Scenario 2

* **Avik's single-shot model** (`MCS_OPTIMAL_v4_real.jl`) solves one day in **one** MILP with
  deterministic powers and exact terminal equalities. This code is the multi-day **MPC** version:
  identical variable names, objective, routing, battery, precedence (raw-count) and travel-pacing
  (`work_per_travel = 4`); the differences are the cross-day receding re-solves, per-window
  history seeds, the carried-in MCS start position, per-day work quotas with roll-over, the CEV
  terminal **floor** (`≥`, overcharge allowed) instead of exact equality, and the separate
  overnight recharge phase.
* **Scenario 1 vs 2.** Scenario 1 (this code) is certainty-equivalent (plans on the mean `μ`).
  Scenario 2 would sample multiple power scenarios from `N(μ, σ)` — the fixed `σ` is the hook.
