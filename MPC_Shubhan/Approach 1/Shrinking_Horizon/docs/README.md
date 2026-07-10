# Shrinking Horizon — Self-Improving Certainty-Equivalent MPC for MCS Dispatch

A Julia implementation of **Scenario 1 (Approach 1: Deterministic Certainty-Equivalent
MPC)** on a **single-day SHRINKING horizon**. It dispatches a **Mobile Charging Station
(MCS)** — a battery on wheels — to a fleet of **Construction EVs (CEVs / electric
excavators)**, deciding every 15 minutes *when to buy grid power, where to drive the MCS,
and which excavator to top up*, while **learning each activity's power draw online** from the
energy actually consumed.

> **This README is the single deep reference.** It documents every code file, every input
> column, every constraint (each marked **HARD**/**SOFT**), and every output file column by
> column. The formal LaTeX model is in `math_model.tex`; a line-by-line code-vs-model audit
> is in `constraints_code_vs_model.txt`.

| Program | Language | MILP solver | Estimator |
|---------|----------|-------------|-----------|
| `7_Shrinking_Horizon_main.jl` (+ numbered modules `1_`…`6_`) | Julia | JuMP + HiGHS | Turing.jl (NUTS/MCMC) |

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
11. [Relation to Receding Horizon & Scenario 2](#11-relation-to-receding-horizon--scenario-2)

---

## 1. The problem in 60 seconds

Heavy electric excavators draw power that depends on what they are doing —
**digging, loading/swinging, traveling, idling** — and those per-activity draws are
**uncertain**. One mobile charger must keep every on-site battery alive and get the day's
work done at the least cost (time-of-use energy + carbon + peak-demand charges + towing
labour).

The controller closes **two feedback loops at once**:

1. **State feedback (MPC).** Every 15 min it re-solves a MILP from the *measured* state
   (battery levels, MCS position, work done so far), applies only the **first** interval,
   then re-measures and re-solves.
2. **Parameter feedback (online learning).** After each interval it feeds the realized energy
   back into a **Bayesian regression** (TruncatedNormal prior + NUTS sampler) to refine its
   estimate of the four activity powers. The MILP plans on the **posterior mean**
   (certainty-equivalent).

**Shrinking horizon.** Unlike the sibling Receding version, this solves a **single day**. Each
15-min re-solve plans the *entire remaining daytime* `[k0 … 18:00]`, so the window **shrinks**
as the day progresses (40 intervals at 08:00, 1 interval at 17:45). No cross-day look-ahead,
no buffer day.

**Two phases.** Daytime (08:00–18:00) is the MILP MPC. The night (18:00–08:00) is a
deterministic **overnight smart-charge** that refills the MCS in the cheapest hours.

---

## 2. Time, the work shift & the horizon

All timing comes from `2_DataLoader.jl` and is **inferred from the data**, not hard-coded.

* **Interval:** `delta_T = 0.25 h` (15 minutes). A full day is `n_int = 96` intervals.
* **Day start:** `t_start = 08:00`. Interval `k = 1` covers 08:00–08:15, `k = 2` 08:15–08:30, …
* **Work shift (synthetic):** `work_start_hour = 8`, `work_end_hour = 17`, with a
  `lunch 12:00–14:00`. So **productive work is available 08:00–12:00 and 14:00–17:00 only**;
  during lunch and after 17:00 the work-availability cap `R_work` is 0.
* **Daytime horizon end (inferred):** the last interval with any work availability **plus a
  1-hour return buffer** (`RETURN_BUFFER_HOURS = 1.0`). Last work is 17:00, so the horizon
  ends at **18:00** → **`n_day = 40` daytime intervals** (k = 1…40).

**So yes: work can only happen until 5 pm; 5 pm–6 pm (k = 37…40) is a no-work window** where
each CEV can only idle/charge and the MCS must drive home to a grid node before the overnight
recharge.

---

## 3. Project layout — every file

```
Shrinking_Horizon/
├── code/
│   ├── 1_Common.jl
│   ├── 2_DataLoader.jl
│   ├── 3_BayesianEstimator.jl
│   ├── 4_MCSModel.jl
│   ├── 5_MPCLoop.jl
│   ├── 6_Output.jl
│   ├── 7_Shrinking_Horizon_main.jl
│   └── Scenario_1.jl            (legacy standalone; reference only, NOT used)
├── data/{input_data, synthetic_data}/
├── output/{input, synthetic}/
└── docs/{README.md, math_model.tex, constraints_code_vs_model.txt}
```

The number prefix `1_`…`7_` is exactly the order the entry point `include`s them
(dependencies first). The **module name inside** each file is unchanged (`module Common`, …),
so `using .Common` etc. still work.

| File | Module | Responsibility |
|------|--------|----------------|
| `1_Common.jl` | `Common` | Pure helpers: `normalize_travel_steps` (round travel times to whole intervals), `in_peak`, `clock_label` / `build_time_labels`, and the STEP-plot builders (`stepify_interval_values`, `stepify_boundary_values`) so figures are steps, not smooth lines. |
| `2_DataLoader.jl` | `DataLoader` | Loads the whole scenario into one immutable `NamedTuple d`. `build_default_data()` (`:synthetic`) and `load_input_data(dir)` (`:input`, 7 CSVs) behind `load_data(mode)`. **Infers** `n_day` / `day_end_hour` from work availability. Single-day, **lumpsum** work. |
| `3_BayesianEstimator.jl` | `BayesianEstimator` | The learning half. Turing model with a **TruncatedNormal(≥0)** prior on each of 4 activity powers + a HalfNormal noise std, and a Normal likelihood linking predicted energy `A·x` to measured energy `b`. `observe!` appends a datum; `refit!` re-runs **NUTS** and refreshes `mu` (→ MILP) and `sd` (→ figure). |
| `4_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **single-day window MILP** over `[k0 … n_day]`. `phase2_overnight_charge(...)` is the deterministic cheapest-hours overnight refill. HiGHS is configured crash-tolerant (see §6.5). |
| `5_MPCLoop.jl` | `MPCLoop` | The closed loop `run_mpc(d; shrinking, H, …)`: for each 15-min step solve the (shrinking) window, apply the first interval, simulate the true activity mix + noise, feed the learner, advance the real state. Returns one `res` NamedTuple. |
| `6_Output.jl` | `Output` | Every artefact from `res` via `write_outputs(res, out_dir)`: STEP figures (PNG) + CSVs, KPI/cost reports, worker schedule, detailed trajectory, the overnight table and the replanning grids. |
| `7_Shrinking_Horizon_main.jl` | — | Thin orchestrator. `run_scenario_1(; mode, …)` = load → `run_mpc` → `write_outputs` → print summary. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |
| `Scenario_1.jl` | — | Legacy all-in-one version; the numbered pipeline does not use it. |

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
julia 7_Shrinking_Horizon_main.jl         # -> ../output/synthetic/
```

**Input mode** (your real CSVs) — from a `julia>` prompt:

```julia
SCENARIO1_NO_AUTORUN = true               # stop the auto synthetic run on include
include("7_Shrinking_Horizon_main.jl")
run_scenario_1(mode = :input)             # -> ../output/input/
```

One-liner (PowerShell):

```powershell
julia --% -e "SCENARIO1_NO_AUTORUN=true; include(\"7_Shrinking_Horizon_main.jl\"); run_scenario_1(mode=:input)"
```

> Red `[ Info: [Turing] … ]` / `NativeCommandError` text is **not** an error — PowerShell
> colours Julia's stderr info logs red. If the KPI summary prints, the run succeeded. This
> single-day model is fast (synthetic ~30 s, input ~8 s at defaults).

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `shrinking` | `true` | `true` = accurate shrinking horizon (each step solves `[k0 … 18:00]`); `false` = fixed lookahead of `H` intervals |
| `H` | `16` | fixed lookahead length in intervals (only used when `shrinking = false`) |
| `time_limit_sec` | `60` | HiGHS seconds per 15-min window |
| `term_tol` | `0.1` | margin ε (kWh) on the hard CEV terminal `SOE_end ≥ SOE_ini − ε` |
| `multi_activity` | `false` | if `true`, a 15-min interval realizes a **mix** (planned activity for a 60–100 % random fraction, idle for the rest); if `false`, the whole interval realizes the single planned activity |
| `require_site_visit` | `false` | force the MCS to visit at least one site |
| `single_visit_per_site` | `false` | at most one visit per site |
| `refit_every` | `8` | re-fit the Bayesian model every N applied intervals |
| `mcmc_samples` | `500` | NUTS posterior samples |
| `soft_prec` / `soft_pace` / `soft_term` | `false` | make precedence (12d) / pacing (13) / CEV terminal (8b) **soft** (penalised) instead of hard. All hard by default; **no automatic fallback**. |
| `out_dir` | `../output/<mode>` | output folder |
| `seed` | `1` | RNG seed (telematics noise + NUTS) |

---

## 6. How the controller works

### 6.1 The single-day loop (`run_mpc` in `5_MPCLoop.jl`)

```
for k0 in 1 … 40:
    K_win = shrinking ? (k0 … 40) : (k0 … min(k0+H-1, 40))   # window shrinks
    model = build_window_model(state, estimate, …)           # solve MILP
    if infeasible:  record INFEASIBLE, HOLD STATE, continue   # no fallback
    apply interval k0's decisions; log grid/dch/work/SOE/node
    realize the true activity mix (+noise); observe!(estimator)
    every refit_every steps: refit!(estimator)   # NUTS
    advance real MCS energy+position and CEV energy; update remaining work
run Phase-2 overnight charge
```

### 6.2 State carried between solves
SOE (MCS + CEV), MCS routing **including in-transit trips** that straddle the apply boundary,
the demand peaks, remaining lumpsum work per site, per-CEV cumulative dig/load/travel (seed the
precedence/pacing counters), and the per-CEV **applied Work/Break history** (seeds the rest-rule
seam so a work-run cannot leak across the every-15-min re-solves).

### 6.3 The two phases
* **Phase 1 — daytime MPC (08:00–18:00):** the window MILP of §8.
* **Phase 2 — overnight smart-charge (18:00–08:00):** `phase2_overnight_charge` — deterministic
  greedy fill. It computes the MCS deficit vs its start level and buys it back in the **cheapest
  overnight slots first**, respecting charge rate and efficiency. Both batteries are therefore
  energy-neutral over the full 24 h cycle. Not a MILP.

### 6.4 The online estimator (`3_BayesianEstimator.jl`)
Each realized interval yields `(a, b)`: `a` = hours on each activity, `b` = measured energy
`= a·true_powers + noise`. The Turing model puts a **TruncatedNormal(mean = prior,
σ = prior_sigma, lower = 0)** on each power and a HalfNormal on the noise std, with likelihood
`b ~ Normal(A·x, s)`. `refit!` runs **NUTS(0.9)**; `mu = posterior mean` (fed to the MILP),
`sd = posterior std` (plotted, not used by the optimizer — the hook for Scenario 2).

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
| `p_digging`, `p_loading_swinging`, `p_traveling` | ✔ | known offline activity powers (kW) — seed the estimator prior mean |
| `carbon_price_per_ton` | opt (0) | $/tonne CO₂ |
| `p_idling` | opt (0) | idle power (kW), 4th prior mean |
| `scale` | opt (2) | precedence ratio: max cumulative load / cumulative dig |
| `t_limit_rest` | opt (1.0) | rest rule: max hours of continuous work per rolling window |
| `kappa_wt` | opt (4) | travel-pacing ratio |
| `prior_sigma_frac` | opt (0.2) | prior σ as a fraction of each prior mean (min 0.05) |
| `obs_noise_std` | opt (0.05) | telematics energy-measurement noise std |
| `co2_unit_scale` | opt (1.0) | multiplier for the CO₂ column |

> `day_end_hour` is **deliberately not read** — it is inferred from `work_flexible.csv`.
> There is **no** `n_days` and **no** `work_by_day.csv` (single-day, lumpsum model).

### `ev_data.csv` — one row per CEV
`id, SOE_min, SOE_max, SOE_ini, ch_rate`. `SOE_ini` is the **end-of-day energy-neutral
target**; `ch_rate` = CEV charge-acceptance (kW).

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
The inferred horizon = last interval with any nonzero cap + the 1-hour return buffer.

---

## 8. The optimization model — every variable & constraint

Built in `build_window_model` over the shrinking window `K = [k0 … n_day]` (boundaries `Tb`).
Every rule is **HARD** unless marked **SOFT**; by default all `soft_*` slacks are pinned to 0.

### 8.1 Decision variables
**Continuous (≥ 0):** `P_ch_MCS`, `P_dch_MCS`, `P_MCS_CEV`, `P_work`, totals
`P_ch_tot`/`P_dch_tot`, travel energy `L_trv`/`L_trv_tot`, state `SOE_MCS[m,t]`/`SOE_CEV[e,t]`,
peaks `P_peak_NC`/`P_peak_OP`, slacks `s_miss_dig`/`s_miss_load` (per site), `s_prec`, `s_pace_hi`/
`s_pace_lo`, `s_term_cev` (per e).

**Binary:** `u[e,i,a,k]`, `mu[i,e,k]`, `rho[m,i,e,k]`, `z[m,i,k]`, `g_ch[m,i,k]`, `x[m,i,j,k]`,
`y_trv[m,i,j,k]`, `beta_arr`/`beta_dep`.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP) `+ missed work` (SOFT, ρ_miss·Σ slacks)
`+ towing labour` (ρ_labor·Δt·Σ y_trv), plus weighted `soft_*` slacks if enabled.

### 8.3 Power flow & where power may go (HARD)
* `P_ch_tot = Σ_grid P_ch_MCS`; `P_dch_tot = Σ_site P_dch_MCS`; discharge forbidden at grid,
  charge forbidden at sites.
* `P_dch_MCS = Σ_e P_MCS_CEV` and `≤ DCH_MCS·z`.
* **Grid exclusivity:** `P_ch_MCS ≤ CH_MCS·g_ch`, `g_ch ≤ z`, ≤ 1 MCS charging per grid node.
* **Plug limits:** `P_MCS_CEV ≤ DCH_MCS_plug·rho`; `Σ_m P_MCS_CEV ≤ CH_CEV[e]·mu`.

### 8.4 Peak-demand trackers (E1, HARD)
`P_peak_NC ≥` carried-in and `≥ Σ_m P_ch_tot[m,k]` (all k); `P_peak_OP` likewise on **on-peak**
k only.

### 8.5 Travel energy (HARD)
`y_trv` is 1 while a trip is in flight (its `tau_trv` intervals), or forced for a carried-in
in-transit trip. `L_trv = k_trv·Δt·y_trv`; `L_trv_tot = Σ L_trv`.

### 8.6 Battery dynamics & bounds (HARD)
* Initial: `SOE_MCS[first] = soe_mcs0`, `SOE_CEV[first] = soe_cev0` (measured carry-in).
* **MCS:** `SOE_MCS[k+1] = SOE_MCS[k] + η·P_ch_tot·Δt − P_dch_tot·Δt/η − L_trv_tot`.
* **CEV:** `SOE_CEV[k+1] = SOE_CEV[k] + Σ P_MCS_CEV·Δt − Σ P_work·Δt`.
* **Bounds:** every boundary clamped to `[SOE_min, SOE_max]` for MCS and CEV.

### 8.7 Routing / presence (Eq. 10, HARD)
* **Presence partition:** `Σ_i z + Σ_{i≠j} y_trv = 1` — parked at one node or in transit on one
  arc.
* `rho ≤ A`, `rho ≤ z`, `Σ_e rho ≤ C_MCS_plug`.
* **Departure/arrival:** `beta_dep = Σ_j x`; `beta_arr` from finishing trips (or carried-in
  arrival); `beta_arr − beta_dep = z[k] − z[k−1]`; `beta_arr + beta_dep ≤ 1`; global flow
  balance.
* **(10e) Home by 18:00:** `Σ_grid z[m,i,n_day] = 1` at the day-end interval.

### 8.8 Activity scheduling (Eq. 11, HARD)
* Exactly one activity per assigned CEV: `Σ_a u = A`; `u ≤ A`.
* Work capped & not while charging: `Σ_{dig,load,trv} p_a·u ≤ R_work·A·(1−mu)`.
* Charging ⇒ idling: `mu ≤ u[idle]` (E4).
* `P_work = Σ_a p_a·u`.

### 8.9 Work quota (Eq. 12c, SOFT) — lumpsum
A single dig/load requirement per site (`hours_digging`/`hours_loading_swinging`). Shortfall
`s_miss ≥ requirement − done` is penalised (`rho_miss`). **There is no hard "no-working-ahead"
cap** (unlike the Receding sibling): with a single day and daily neutrality there is no
incentive to overshoot, so the model naturally hits the quota.

### 8.10 Precedence (Eq. 12d, HARD)
Cumulative loading `≤ scale ·` cumulative digging (`+ s_prec`, pinned 0), seeded by realized
work carried in.

### 8.11 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4) and `rest_win = 5`: over any 5 consecutive
intervals a CEV does **at most 4** work intervals (≥ 1 idle break). Two parts: within-window
5-windows, **plus a seam** seeded with the applied Work/Break history so a work-run cannot leak
across the every-15-min re-solves.

### 8.12 Travel pacing (Eq. 13, HARD)
`kappa·trv_cum ≤ work_cum + s_pace_hi` and `kappa·trv_cum ≥ work_cum − kappa − s_pace_lo`
(slacks pinned 0) — travel roughly proportional to productive work, two-sided.

### 8.13 CEV energy neutrality (Eq. 8b, HARD)
At the day-end boundary: `SOE_CEV[e, last] ≥ SOE_CEV_ini[e] − term_tol` (or a two-sided
penalised band under `soft_term`). Each excavator ends the day at its start level.

### 8.14 Keep-up reserve (E7, HARD)
A per-boundary **lower bound** on each CEV's SOE, built backward from 18:00 so the terminal
stays reachable. The MCS must leave `tgrid` intervals before 18:00, so the **last chargeable
interval** is `Lc = n_day − tgrid`; after `Lc` the CEV idles and drains. The bound is the least
SOE from which the terminal is still reachable at the net charge rate
`chg_net = min(CH_CEV, DCH_MCS_plug)·Δt − idle_drain`. It binds only in the late tail, so it
never distorts productive hours and makes the hard terminal recursively feasible.

> **No fallback.** Windows use the hard constraints only; an infeasible re-plan is reported
> **INFEASIBLE** and the plant **holds state**. In the current runs both `:synthetic` and
> `:input` solve with **zero** infeasible windows and 0.00 h missed work.

---

## 9. Outputs — every file & column

Written by `6_Output.jl`, regenerated every run.

### 9.1 Schedules
**`worker_schedule.csv`** — for the site crew, one row per 15 min: `time` (clock),
`CEV<e>_activity` (`Digging`/`Loading/Swinging`/`Traveling`/`Idle`),
`CEV<e>_plug_in_charge` (Yes/No), `MCS_charge_from_grid` (Yes/No).

**`closed_loop_trajectory.csv`** — for analysts, one row per applied interval: `k` (1…40),
`clock`, `price`, `co2`, `grid_kW`, `dch_kW`, `work_kW`, `soe_mcs`, `soe_cev1`, `soe_cev2`
(`NaN` if absent), `mcs_node` (0 = transit), `est_dig/est_load/est_trv/est_idle`, `unc_*`,
`n_obs`. Held/infeasible intervals appear with `grid=dch=work=0`, `Idle`.

**`overnight_mcs_charge.csv`** — the single Phase-2 refill: `k`, `clock`, `price`, and per MCS
`MCS<m>_charge_kW`, `MCS<m>_soe_kWh`, `MCS<m>_charging` (Yes/No).

**`replan_grids/*.csv` (+ `.html`)** — four grids: `plan_grid_kW`, `plan_mcs_soe`,
`plan_cev<e>_soe`, `plan_cev<e>_activity`. **Rows** = the 15-min re-plan step; **columns** = the
interval being planned. Across a row = the whole plan made at that step; down a column = how one
interval's plan is revised as new state + learning arrive; the diagonal is what was applied. The
`.html` colours past (green) vs pending (yellow).

### 9.2 Figures (PNG) + matching CSVs
| File | Contents |
|------|----------|
| `01_total_grid_power_profile.png/.csv` | total grid charging (+) vs CEV discharging (−) |
| `02_work_profiles_by_site.png/.csv` | realized work power, one panel per site |
| `03_mcs_state_of_energy.png/.csv` | MCS SOE with min/max guides |
| `04_cev_state_of_energy.png/.csv` | CEV SOE with min/max guides |
| `05_electricity_prices_emissions.png` / `05_electricity_prices.csv` | price (left) + CO₂ (right) *(CSV is named `05_electricity_prices.csv`)* |
| `06_mcs_location_trajectory.png/.csv` | MCS node over time (0 = transit) |
| `07_mcs_optimization_summary.png` + `07_mcs_cev_soe.csv` | combined overview + long-form per-interval MCS+CEV table |
| `mcs_<m>_power_profile.png/.csv` | per-MCS charging/discharging |
| `11_power_estimate_convergence.png` | online estimates (± uncertainty) → hidden truth |

### 9.3 Cost / KPI reports
* **`08_cost_emissions_timeseries.csv`** — per interval `Grid_Energy_kWh`, `Energy_Cost_USD`,
  `CO2_Emissions_kg` + running cumulatives; **`08_cost_emissions_summary.png`** plots the
  cumulatives.
* **`09_cost_kpi_metrics.csv`** — metric/value rows (total & component costs, NCD/OPD peaks,
  missed work, transit hours, overnight recharge kWh/cost, infeasible windows, loop time);
  **`09_kpi_metrics_summary.png`** is a two-panel bar chart.
* **`10_mip_convergence.csv`** — per applied window: `step`, `clock`, `status`, `objective`,
  `gap_percent`, `solve_time_s`.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7.
2. **Prior** — set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling` and
   `prior_sigma_frac` in `parameters.csv`.
3. **Telemetry** — in `5_MPCLoop.jl`, replace the simulated `realized_activity_durations` and
   the faked meter `b = a·true_powers + noise` with the actual per-activity seconds + measured
   interval energy from your pipeline. `true_powers` only generates ground truth for the demo.

---

## 11. Relation to Receding Horizon & Scenario 2

* **Shrinking vs Receding.** This version solves **one day** and shrinks the window toward
  18:00: no cross-day look-ahead, no buffer day, one lumpsum work quota, one end-of-day
  terminal, and **no** no-working-ahead cap. The sibling `Receding_Horizon/` chains days with a
  cross-day window, a per-day work schedule, daily battery realignment, and a hard
  no-working-ahead cap. See `../../README.md` for the full comparison.
* **Scenario 1 vs 2.** Both use the **same Bayesian regression**. Scenario 1 (this code) is
  certainty-equivalent (posterior mean, one MILP). Scenario 2 would sample multiple power
  scenarios from the same posterior — the posterior `sd` is the hook.
