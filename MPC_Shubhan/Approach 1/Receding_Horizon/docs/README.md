# Receding Horizon — Self-Improving Certainty-Equivalent MPC for MCS Dispatch

A Julia implementation of **Scenario 1 (Approach 1: Deterministic Certainty-Equivalent
MPC)** on a **multi-day, cross-day RECEDING horizon**. It dispatches a **Mobile Charging
Station (MCS)** — a battery on wheels — to a fleet of **Construction EVs (CEVs / electric
excavators)**, deciding every 15 minutes *when to buy grid power, where to drive the MCS,
and which excavator to top up*, while **learning each activity's power draw online** from
the energy actually consumed.

> **This README is the single deep reference.** It documents every code file, every input
> column, every constraint (each marked **HARD**/**SOFT**), and every output file column by
> column. The formal LaTeX model is in `math_model.tex`; a line-by-line code-vs-model audit
> is in `constraints_code_vs_model.txt`.

| Program | Language | MILP solver | Estimator |
|---------|----------|-------------|-----------|
| `7_Receding_Horizon_main.jl` (+ numbered modules `1_`…`6_`) | Julia | JuMP + HiGHS | Turing.jl (NUTS/MCMC) |

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
11. [Relation to Shrinking Horizon & Scenario 2](#11-relation-to-shrinking-horizon--scenario-2)

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
   then re-measures and re-solves. Classic receding-horizon control.
2. **Parameter feedback (online learning).** After each interval it feeds the realized
   energy back into a **Bayesian regression** (TruncatedNormal prior + NUTS sampler) to
   refine its estimate of the four activity powers. The MILP then plans on the **posterior
   mean** (certainty-equivalent).

**Cross-day lookahead.** Each 15-min window spans *the rest of today plus `lookahead_days`
future daytime blocks*, so the plan always "sees" tomorrow. The run simulates `n_days`
reported days **plus one extra "buffer" day** that is dropped from every output — it exists
only to give the last reported day a full day of look-ahead.

**Two phases per day.** Daytime (08:00–18:00) is the MILP MPC. The night (18:00–08:00) is a
deterministic **overnight smart-charge** that refills the MCS in the cheapest hours.

---

## 2. Time, the work shift & the horizon

All timing comes from `2_DataLoader.jl` and is **inferred from the data**, not hard-coded as
a horizon parameter.

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
recharge. (In the current synthetic run a few early-day wind-down windows are softened by the
graceful fallback — see §8.16 — but missed work stays 0.00 h and every daily terminal is met.)

**Global vs within-day index.** Across days the controller lays each day's 40-interval block
end to end. For a global index `g`: `wd(g) = mod(g−1, n_day)+1` is its position *within its
day* (1…40, used to index the same-every-day price / carbon / work-availability profiles),
and `dayof(g) = div(g−1, n_day)+1` is which day it belongs to.

---

## 3. Project layout — every file

```
Receding_Horizon/
├── code/
│   ├── 1_Common.jl
│   ├── 2_DataLoader.jl
│   ├── 3_BayesianEstimator.jl
│   ├── 4_MCSModel.jl
│   ├── 5_MPCLoop.jl
│   ├── 6_Output.jl
│   ├── 7_Receding_Horizon_main.jl
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
| `1_Common.jl` | `Common` | Pure helpers used everywhere: `normalize_travel_steps` (round travel times to whole intervals), `in_peak` (is an interval in the on-peak window?), `clock_label` / `clock_day_label` / `build_time_labels_days` (clock strings), `multiday_xticks`, and the **STEP-plot builders** (`stepify_interval_values`, `stepify_boundary_values`) so figures are drawn as steps, not smooth lines. No optimisation or IO logic. |
| `2_DataLoader.jl` | `DataLoader` | Loads the whole scenario into one immutable `NamedTuple d`. Two entry points behind `load_data(mode)`: `build_default_data()` (`:synthetic`) and `load_input_data(dir)` (`:input`, the 7 CSVs). **Infers** `n_day` / `day_end_hour` from work availability. Everything downstream reads `d` and cannot tell where the numbers came from. |
| `3_BayesianEstimator.jl` | `BayesianEstimator` | The learning half. A Turing probabilistic model with a **TruncatedNormal(≥0)** prior on each of the 4 activity powers + a HalfNormal noise std, and a Normal likelihood linking predicted energy `A·x` to measured energy `b`. `observe!` appends one `(activity-hours, energy)` datum; `refit!` re-runs **NUTS** on all data so far and refreshes `mu` (posterior mean → the MILP) and `sd` (posterior std → the convergence figure). Knows nothing about the optimiser. |
| `4_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **cross-day window MILP** (JuMP + HiGHS). `phase2_overnight_charge(...)` is the deterministic cheapest-hours overnight refill. HiGHS is configured crash-tolerant (see §6.5). |
| `5_MPCLoop.jl` | `MPCLoop` | The closed loop `run_mpc(d; …)`: for each day and each 15-min step it solves the window, applies the first interval, simulates the true within-interval activity mix + telematics noise, feeds the learner, advances the real state, and (each night) runs Phase 2. Captures the realized trajectory and returns one big `res` NamedTuple. |
| `6_Output.jl` | `Output` | Every on-disk artefact from `res` via `write_outputs(res, out_dir)`: the STEP figures (PNG) + their CSVs, the KPI/cost reports, the worker schedule, the detailed trajectory, the overnight tables and the per-day replanning grids. |
| `7_Receding_Horizon_main.jl` | — | Thin orchestrator. Defines `run_scenario_1(; mode, …)` = load → `run_mpc` → `write_outputs` → print summary. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |
| `Scenario_1.jl` | — | Legacy all-in-one version kept for reference; the numbered pipeline does not use it. |

---

## 4. Requirements & how to run

Install once:

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Plots", "DataFrames", "CSV", "Turing"])
```

Use a **plain terminal** (PowerShell), not the VS Code Julia REPL — a long run is more stable
there. Work from the `code/` folder.

**Synthetic mode** (built-in demo data — auto-runs on launch):

```bash
cd code
julia 7_Receding_Horizon_main.jl          # -> ../output/synthetic/
```

**Input mode** (your real CSVs in `../data/input_data/`) — from a `julia>` prompt:

```julia
SCENARIO1_NO_AUTORUN = true               # stop the auto synthetic run on include
include("7_Receding_Horizon_main.jl")
run_scenario_1(mode = :input)             # -> ../output/input/
```

One-liner (PowerShell `--%` passes the quotes through unchanged):

```powershell
julia --% -e "SCENARIO1_NO_AUTORUN=true; include(\"7_Receding_Horizon_main.jl\"); run_scenario_1(mode=:input)"
```

> Red `[ Info: [Turing] … ]` / `NativeCommandError` text is **not** an error — PowerShell
> colours Julia's stderr info logs red. If the KPI summary prints, the run succeeded. Run one
> job at a time; each solve uses significant CPU/RAM. **Note:** at default `time_limit_sec=60`
> the cross-day MILP is large and the synthetic run can take ~20–25 min (the `:input` run is
> ~1 min).

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `n_days` | `nothing` | reported days to KEEP (defaults to the dataset's `n_days`). One extra **buffer** day is always simulated and dropped. |
| `lookahead_days` | `1` | cross-day window depth: each window = rest of today + this many future daytime blocks (capped at the buffer day) |
| `time_limit_sec` | `60` | HiGHS seconds per 15-min window (lower = faster, weaker incumbents) |
| `term_tol` | `0.1` | margin ε (kWh) on the hard CEV terminal `SOE_end ≥ SOE_ini − ε`. `ε = 0` is exact equality but goes infeasible under estimator drift; `0.1` is the smallest value keeping windows feasible. |
| `multi_activity` | `false` | if `true`, a 15-min interval realizes a **mix** (the planned activity for a 60–100 % random fraction, idle for the rest); if `false`, the whole interval realizes the single planned activity |
| `require_site_visit` | `false` | force the MCS to visit at least one site |
| `single_visit_per_site` | `false` | at most one visit per site |
| `refit_every` | `8` | re-fit the Bayesian model every N applied intervals |
| `mcmc_samples` | `500` | NUTS posterior samples |
| `soft_prec` / `soft_pace` / `soft_term` | `false` | make precedence (12d) / pacing (13) / CEV terminal (8b) **soft** (penalised) instead of hard. All hard by default; a hard-infeasible window triggers the **graceful soft-terminal fallback** (§8.16), not a dead hold. |
| `out_dir` | `../output/<mode>` | output folder |
| `seed` | `1` | RNG seed (telematics noise + NUTS) |

> **Anti-hoarding levers (internal `build_window_model` defaults, not `run_mpc` args):**
> `overcharge_frac = 0.5` (E8 over-charge cap, §8.15) and `drawdown_kwh = 10.0` (E9 daytime
> health floor, §8.16). Together they force the single MCS to shuttle between the two CEVs
> instead of camping at one. Edit them in `4_MCSModel.jl` if you need looser/tighter balancing.

---

## 6. How the controller works

### 6.1 The multi-day loop (`run_mpc` in `5_MPCLoop.jl`)

```
for day in 1 … n_days+1 (last = buffer):
    add this day's fresh work quota to the rolling remaining-work totals
    reset the per-CEV Work/Break history (a night is a long break)
    for k0 in 1 … 40:
        K_win = g0 … min(D_total, day+lookahead_days)*40        # cross-day window
        model = build_window_model(state, estimate, …)          # solve MILP (hard)
        if hard-infeasible: re-solve soft_term+soft_reserve      # graceful fallback -> Softened
        if still infeasible: record INFEASIBLE, HOLD STATE, continue
        apply interval k0's decisions; log grid/dch/work/SOE/node
        realize the true activity mix (+noise); observe!(estimator)
        every refit_every steps: refit!(estimator)   # NUTS
        advance real MCS energy+position and CEV energy; update remaining work
    run Phase-2 overnight charge; reset MCS to full for next morning
drop the buffer day from all reported outputs
```

### 6.2 State carried between solves
SOE (MCS + CEV), MCS routing **including in-transit trips** that straddle the apply boundary
(`mcs_transit`), the daily demand peaks (`peak_nc`, `peak_op`), remaining work per site
(`rem_dig`/`rem_load`, which roll over across days), per-CEV cumulative dig/load/travel for the
current day (seed the precedence/pacing counters), and the per-CEV **applied Work/Break
history** `work_hist` (seeds the rest-rule seam). CEV SOE and remaining work cross the night;
the MCS is reset to full and re-parked at the grid each morning.

### 6.3 The two phases
* **Phase 1 — daytime MPC (08:00–18:00):** the window MILP of §8.
* **Phase 2 — overnight smart-charge (18:00–08:00):** `phase2_overnight_charge` — a
  deterministic greedy fill. It computes the MCS energy deficit vs its start level and buys it
  back in the **cheapest overnight slots first** (sorted by price), respecting the charge rate
  and efficiency. Not a MILP.

### 6.4 The online estimator (`3_BayesianEstimator.jl`)
Each realized interval yields a datum `(a, b)`: `a` = hours on each activity, `b` = measured
energy `= a·true_powers + noise`. The Turing model puts a **TruncatedNormal(mean = prior,
σ = prior_sigma, lower = 0)** on each power and a HalfNormal on the noise std, with a Normal
likelihood `b ~ Normal(A·x, s)`. `refit!` runs **NUTS(0.9)** for `mcmc_samples` draws over all
data so far; `mu = posterior mean` (fed to the MILP), `sd = posterior std` (plotted). The MILP
never sees `sd` — that is the hook a future Scenario 2 (stochastic MPC) would use.

### 6.5 Solver robustness
HiGHS is pinned to `threads = 1`, `parallel = off`, `mip_heuristic_effort = 0`,
`mip_detect_symmetry = false`, `mip_rel_gap = 1e-2`. The parallel/heuristic sub-solvers were
disabled because they intermittently segfault on Windows; the 1 % gap keeps solves fast.
`optimize!` is wrapped in `try/catch` so a rare native fault degrades to "no solution → hold
state" rather than crashing the run.

---

## 7. Input-data schema — every file & column

Files live in `data/input_data/`. Every listed column is **required** (the loader raises a
clear error on a missing file/column/parameter); extra columns are ignored. IDs (`e1`, `m1`,
site names, …) are arbitrary strings that only need to be **consistent across files**.
Energies **kWh**, powers **kW**, times in **hours** and **15-min intervals**.

### `parameters.csv` — two columns `Parameter, Value`
| Parameter | Req? | Meaning |
|-----------|------|---------|
| `delta_T` | ✔ | interval length in hours (0.25) |
| `k_trv` | ✔ | travel energy per arc (kWh) charged to the MCS while in transit |
| `rho_miss` | ✔ | $/hour penalty for unfinished dig/load work (soft) |
| `rho_labor` | ✔ | $/hour labour cost of the MCS being in transit (towing) |
| `lambda_demand_NC` | ✔ | $/kW non-coincident demand charge on the highest grid draw |
| `lambda_demand_OP` | ✔ | $/kW on-peak demand charge on the highest grid draw during peak hours |
| `p_digging`, `p_loading_swinging`, `p_traveling` | ✔ | known offline activity powers (kW) — seed the estimator prior mean |
| `carbon_price_per_ton` | opt (0) | $/tonne CO₂ |
| `p_idling` | opt (0) | idle power (kW), 4th prior mean |
| `scale` | opt (2) | precedence ratio: max cumulative load / cumulative dig |
| `t_limit_rest` | opt (1.0) | rest rule: max hours of continuous work per rolling window |
| `kappa_wt` | opt (4) | travel-pacing ratio (~1 travel per `kappa_wt` productive intervals) |
| `prior_sigma_frac` | opt (0.2) | prior σ as a fraction of each prior mean (min 0.05) |
| `obs_noise_std` | opt (0.05) | telematics energy-measurement noise std |
| `co2_unit_scale` | opt (1.0) | multiplier to convert the CO₂ column into consistent units |
| `n_days` | opt (2) | reported days to keep |

> `day_end_hour` is **deliberately not read** — it is inferred from `work_flexible.csv`.

### `ev_data.csv` — one row per CEV
`id, SOE_min, SOE_max, SOE_ini, ch_rate`. `SOE_ini` is the **daily energy-neutral target**
(each excavator must return to it by every 18:00). `ch_rate` = CEV charge-acceptance (kW).

### `mcs_data.csv` — one row per MCS
`id, SOE_min, SOE_max, SOE_ini, CH_MCS, DCH_MCS, C_MCS_plug, DCH_MCS_plug, eta_ch_dch`.
`CH_MCS`/`DCH_MCS` = grid-charge / total-discharge power caps; `C_MCS_plug` = number of
simultaneous plugs; `DCH_MCS_plug` = per-plug power cap; `eta_ch_dch` = round-trip efficiency.

### `place.csv` — one row per node
`site, <one column per CEV id>, hours_digging, hours_loading_swinging`. A `1` in a CEV column
marks that node as that CEV's site (builds the assignment matrix `A`); a node with **no** CEV
assigned is the **grid**. The two work columns are the **default daily quota** used only when
`work_by_day.csv` is absent.

### `work_by_day.csv` *(optional)* — per-day quota
`site, day, hours_digging, hours_loading_swinging`, one row per site per reported day. When
present, day `D` uses its own quota; when absent, the `place.csv` quota repeats every day. The
buffer day always gets no fresh work.

### `travel_time.csv` — square matrix
Row header + one column per node; entry `[i,j]` = travel time from node `i` to `j` **in
intervals** (rounded to whole intervals by `normalize_travel_steps`).

### `time_data.csv` — one row per full-day interval (`n_int` rows)
A time-label column, `lambda_buy` ($/kWh price), `intensity_tons_emissions` (CO₂ per kWh).
`t_start` is inferred as `first-row-clock − delta_T`, and the full-day series drive the
per-interval price/carbon used every day.

### `work_flexible.csv` — availability
`Location, EV,` then **one column per full-day interval** giving the kW work cap (0 = no work
allowed in that interval). This is the source of the inferred horizon: the last interval with
any nonzero cap + the 1-hour return buffer = `n_day`.

---

## 8. The optimization model — every variable & constraint

Built in `build_window_model` over the window `K` (global indices) with boundaries `Tb`.
`wd(k)`/`dayof(k)` map a global index to its within-day slot / day. Every rule below is
**HARD** unless marked **SOFT**. By default all `soft_*` slacks are pinned to 0.

### 8.1 Decision variables
**Continuous (≥ 0):** `P_ch_MCS[m,i,k]` grid→MCS, `P_dch_MCS[m,i,k]` MCS→site,
`P_MCS_CEV[m,i,e,k]` MCS→CEV, `P_work[i,e,k]` CEV work power, totals `P_ch_tot`/`P_dch_tot`,
travel energy `L_trv[m,i,j,k]`/`L_trv_tot[m,k]`, state `SOE_MCS[m,t]`/`SOE_CEV[e,t]` on
boundaries, peaks `P_peak_NC`/`P_peak_OP`, and slacks `s_miss_dig/s_miss_load` (per site,day),
`s_prec` (per site,k), `s_pace_hi/s_pace_lo` (per e,k), `s_term_cev` (per e,day).

**Binary:** `u[e,i,a,k]` activity choice, `mu[i,e,k]` CEV charging, `rho[m,i,e,k]` plugged,
`z[m,i,k]` parked at node, `g_ch[m,i,k]` grid-connected, `x[m,i,j,k]` departs i→j,
`y_trv[m,i,j,k]` in transit, `beta_arr`/`beta_dep` arrival/departure indicators.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP) `+ missed work` (SOFT, ρ_miss·Σ slacks)
`+ towing labour` (ρ_labor·Δt·Σ y_trv). If any `soft_*` flag is on, its weighted slack
(`W_prec=800`, `W_pace=100`, `W_term=150`) is added.

### 8.3 Power flow & where power may go (HARD)
* `P_ch_tot[m,k] = Σ_{grid} P_ch_MCS`; `P_dch_tot[m,k] = Σ_{site} P_dch_MCS`.
* Discharge forbidden at grid nodes; charge forbidden at site nodes.
* `P_dch_MCS[m,i,k] = Σ_e P_MCS_CEV[m,i,e,k]` and `≤ DCH_MCS·z[m,i,k]` (can only discharge
  where parked).
* **Grid exclusivity:** `P_ch_MCS ≤ CH_MCS·g_ch`, `g_ch ≤ z`, and ≤ 1 MCS charging per grid
  node per interval.
* **Plug limits:** `P_MCS_CEV ≤ DCH_MCS_plug·rho` (per plug) and
  `Σ_m P_MCS_CEV ≤ CH_CEV[e]·mu[i,e,k]` (a CEV only accepts power when its charging flag `mu`
  is on).

### 8.4 Peak-demand trackers (E1, HARD)
`P_peak_NC ≥` carried-in peak and `≥ Σ_m P_ch_tot[m,k]` for **all** k;
`P_peak_OP ≥` carried-in and `≥ Σ_m P_ch_tot[m,k]` for **on-peak** k only. An optional hard
`peak_demand_limit` cap is available (off by default).

### 8.5 Travel energy (HARD)
`y_trv[m,i,j,k]` is 1 while a trip launched at `x[m,i,j,·]` is in flight (its `tau_trv`
intervals), or forced to 1 for an in-transit trip carried in from the previous solve.
`L_trv[m,i,j,k] = k_trv·Δt·y_trv`; `L_trv_tot = Σ L_trv`.

### 8.6 Battery dynamics & bounds (HARD)
* Initial: `SOE_MCS[·,first] = soe_mcs0`, `SOE_CEV[·,first] = soe_cev0` (measured carry-in).
* **MCS within a day:** `SOE_MCS[k+1] = SOE_MCS[k] + η·P_ch_tot·Δt − P_dch_tot·Δt/η − L_trv_tot`.
* **Overnight bridge:** at each night boundary `SOE_MCS[k+1] = SOE_MCS_ini` (recharged).
* **CEV (all intervals, no night reset):**
  `SOE_CEV[k+1] = SOE_CEV[k] + Σ P_MCS_CEV·Δt − Σ P_work·Δt`.
* **Bounds:** every boundary is clamped to `[SOE_min, SOE_max]` for MCS and CEV.

### 8.7 Routing / presence (Eq. 10, HARD)
* **Presence partition:** each interval `Σ_i z[m,i,k] + Σ_{i≠j} y_trv[m,i,j,k] = 1` — the MCS
  is parked at exactly one node **or** in transit on exactly one arc.
* `rho ≤ A` (can only plug a CEV at its own site) and `rho ≤ z` (only where parked);
  `Σ_e rho ≤ C_MCS_plug`.
* **Departure/arrival:** `beta_dep[m,i,k] = Σ_j x[m,i,j,k]`; `beta_arr` set by trips that
  finish at `k` (or a carried-in arrival). `beta_arr − beta_dep = z[k] − z[k−1]` (flow
  conservation) and `beta_arr + beta_dep ≤ 1`.
* Global flow: total arrivals − departures at a node = end-parked − started-here.
* **(10e) Home by every 18:00:** `Σ_{grid} z[m,i,eve] = 1` at each day's last interval.

### 8.8 Activity scheduling (Eq. 11, HARD)
* Exactly one activity per assigned CEV: `Σ_a u[e,i,a,k] = A[i,e]`; `u ≤ A`.
* Work capped by availability & not while charging:
  `Σ_{dig,load,trv} p_a·u ≤ R_work(i,e,wd(k))·A·(1−mu)`.
* Charging ⇒ idling: `mu[i,e,k] ≤ u[e,i,idle,k]` (E4).
* Definition: `P_work[i,e,k] = Σ_a p_a·u`.

### 8.9 Work quota (Eq. 12c) — cumulative, pinned BOTH ways
For each day-block `dy` in the window, with cumulative target `tgt` (window-start remaining +
each later morning's fresh quota) and cumulative work done through the end of `dy`:
* **SOFT lower:** `s_miss ≥ tgt − done` — a shortfall is penalised (`rho_miss`) and, being
  cumulative, **rolls over** to the next day.
* **HARD upper (no working ahead):** `done ≤ tgt` — cumulative work may not exceed the day's
  cumulative quota, so a day cannot borrow a future day's work. Earlier leftover can still be
  **caught up** (it is inside the same cumulative budget). Working *less* is always feasible,
  so this cap can never by itself make a window infeasible.

### 8.10 Precedence (Eq. 12d, HARD)
Within each day-block: cumulative loading `≤ scale ·` cumulative digging (`+ s_prec`, pinned 0).
Counters restart each morning and are seeded by the current day's realized work
(`cum_load_site`/`cum_dig_site`).

### 8.11 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4 at defaults) and window `rest_win = rest_cap+1`
(=5): over any 5 consecutive intervals a CEV does **at most 4** work intervals (≥ 1 idle
break). Two parts:
* **(a) within-window:** every 5-window lying fully inside one day-block.
* **(b) seam:** windows straddling the window start are seeded with the applied Work/Break
  flags `work_hist` of the current day, so a work-run cannot leak across the every-15-min
  re-solves (the "4 at the tail of one window + 4 at the head of the next = 8-in-a-row"
  artefact). The `o = rest_cap` seam is the binding one.

### 8.12 Travel pacing (Eq. 13, HARD)
Per day, with `kappa = kappa_wt`: `kappa·trv_cum ≤ work_cum + s_pace_hi` and
`kappa·trv_cum ≥ work_cum − kappa − s_pace_lo` (both slacks pinned 0). Keeps travel roughly
proportional to productive work (~1 travel per `kappa` productive intervals), two-sided.

### 8.13 Daily CEV energy neutrality (Eq. 8b, HARD)
At **every** evening boundary `eve+1` (each 18:00 in the window):
`SOE_CEV[e, eve+1] ≥ SOE_CEV_ini[e] − term_tol`. So each reported day is energy-neutral
(start ≈ end) rather than the battery drifting across days. With `soft_term` it becomes a
two-sided penalised band instead.

### 8.14 Keep-up reserve (E7, HARD)
A per-boundary **lower bound** on each CEV's SOE, built backward from each day's 18:00 so the
daily terminal (8.13) stays reachable. Because a CEV charges only while the MCS is parked at
its site, and the MCS must depart `tgrid` intervals before 18:00 to be home, the **last
interval it can charge** is `Lc = Gd − tgrid`. After `Lc` the CEV idles and its SOE only
drains (idle draw). The bound is the least SOE from which the terminal is still reachable at
the net charge rate `chg_net = min(CH_CEV, DCH_MCS_plug)·Δt − idle_drain`. It binds only in
each day's late tail, so it never distorts productive hours.

### 8.15 Anti-hoarding over-charge cap (E8, HARD, always on)
`SOE_CEV[e, t] ≤ SOE_CEV_ini[e] + overcharge_frac·(SOE_max − SOE_ini)` at every daytime
boundary (the fixed initial boundary is skipped). A CEV only needs to **return** to its start
level by 18:00, so charging it far above that is wasted effort that keeps the single MCS parked
at the **near** CEV while the **far** one is neglected (we saw CEV1 pushed to ~86 kWh while CEV2
drained to ~32). This is an upper bound — the terminal target `SOE_ini` always lies below it —
so it can **never** make a window infeasible and is applied even under the soft-reserve fallback.
Default `overcharge_frac = 0.5`.

### 8.16 Daytime health floor (E9, HARD, relaxable)
`SOE_CEV[e, t] ≥ SOE_CEV_ini[e] − drawdown_kwh` at every daytime boundary. The keep-up reserve
(8.14) only guarantees each CEV can **recover by its own 18:00**, which still permits a deep
mid-day drain plus a last-minute dash — i.e. the MCS camps at one site and neglects the other.
This floor forbids any CEV from draining more than `drawdown_kwh` below its start level, so the
single MCS must keep **both** CEVs healthy and therefore **shuttle** between them. It is relaxed
under the soft-reserve fallback (and skips the fixed initial boundary) so a drifted state can
never freeze the plant. Default `drawdown_kwh = 10.0`.

> **Graceful fallback (no dead intervals).** Windows use the hard constraints first. If a
> re-plan is hard-infeasible (early-day estimator bias + one MCS shared across two CEVs), the
> loop re-solves **once** with `soft_term = true` **and** `soft_reserve = true` (the terminal
> becomes a penalised band and the E7/E9 floors are dropped), so the MCS still charges toward
> target instead of holding state. Such windows are counted as **`Softened_windows`**; only if
> even that soft re-solve fails does the plant hold state (`Infeasible_windows`).
>
> In the current default runs the `:input` dataset solves with **0 softened / 0 infeasible**,
> and the `:synthetic` scenario has **3 softened** windows (early-day wind-down) and **0
> infeasible**. Missed work is **0.00 h** in both, both CEVs are shuttle-served, and every daily
> terminal is met — the softened windows carry no shortfall.

---

## 9. Outputs — every file & column

All outputs cover the **kept days 1…n_days only** (the buffer day is dropped), written by
`6_Output.jl` and regenerated every run.

### 9.1 Schedules
**`worker_schedule.csv`** — for the site crew, one row per applied 15 min:
* `time` — day-tagged clock (e.g. `D2 08:15`).
* `CEV<e>_activity` — `Digging` / `Loading/Swinging` / `Traveling` / `Idle` (or `Off (home)`).
* `CEV<e>_plug_in_charge` — `Yes/No`: plug this CEV into the MCS this slot?
* `MCS_charge_from_grid` — `Yes/No`: should the MCS draw from the grid this slot?

**`closed_loop_trajectory.csv`** — for analysts, one row per applied interval:
`day`, `gstep` (continuous across kept days), `k` (within-day 1…40), `clock`, `price`, `co2`,
`grid_kW` (MCS draw from grid), `dch_kW` (MCS → CEVs), `work_kW` (realized CEV work power),
`soe_mcs`, `soe_cev1`, `soe_cev2` (`NaN` if that CEV doesn't exist), `mcs_node` (0 = transit),
`est_dig/est_load/est_trv/est_idle` (online estimates), `unc_*` (posterior std), `n_obs`
(telematics observations so far). Held/infeasible intervals appear with
`grid=dch=work=0`, `Idle`.

**`overnight_mcs_charge_day<N>.csv`** — Phase-2 refill for each kept day: `k`, `clock`,
`price`, and per MCS `MCS<m>_charge_kW`, `MCS<m>_soe_kWh`, `MCS<m>_charging` (Yes/No).

**`replan_grids/day<N>/*.csv` (+ `.html`)** — four grids per kept day: `plan_grid_kW`,
`plan_mcs_soe`, `plan_cev<e>_soe`, `plan_cev<e>_activity`. **Rows** = the 15-min re-plan step
(labelled by the clock it was made at); **columns** = the interval being planned. Reading
**across a row** = the whole plan made at that step; **down a column** = how one interval's
plan is revised as new state + learning arrive; the **diagonal** is what was actually applied.
The `.html` colours past (green, fixed) vs pending (yellow).

### 9.2 Figures (PNG) + matching CSVs
| File | Contents |
|------|----------|
| `01_total_grid_power_profile.png/.csv` | total grid charging (+) vs CEV discharging (−); CSV has charging/discharging/net kW |
| `02_work_profiles_by_site.png/.csv` | realized work power, one panel per site; CSV has per-site + total kW |
| `03_mcs_state_of_energy.png/.csv` | MCS SOE with min/max guide lines |
| `04_cev_state_of_energy.png/.csv` | CEV SOE with min/max guide lines |
| `05_electricity_prices_emissions.png` / `05_electricity_prices.csv` | price (left axis) + CO₂ factor (right axis) *(note the CSV is named `05_electricity_prices.csv`)* |
| `06_mcs_location_trajectory.png/.csv` | MCS node over time (0 = transit); CSV adds a node-type label |
| `07_mcs_optimization_summary.png` + `07_mcs_cev_soe.csv` | combined multi-panel overview; the CSV is a long-form per-interval MCS+CEV table (charge/discharge/travel kW, SOE start/end) |
| `mcs_<m>_power_profile.png/.csv` | per-MCS charging/discharging |
| `11_power_estimate_convergence.png` | the online estimates (± uncertainty ribbon) converging to the dashed hidden truth |

### 9.3 Cost / KPI reports
* **`08_cost_emissions_timeseries.csv`** — per interval: `Grid_Energy_kWh`, `Energy_Cost_USD`,
  `CO2_Emissions_kg`, and running cumulatives. **`08_cost_emissions_summary.png`** plots the
  cumulative cost (+ CO₂).
* **`09_cost_kpi_metrics.csv`** — one row per metric: `Total_Cost_USD`, `Total_Energy_Cost_USD`,
  `Total_CO2_Cost_USD`, `NC_demand_charge_USD`, `OP_demand_charge_USD`,
  `Missed_Work_Penalty_USD`, `Travel_Labour_USD`, `Total_Grid_Energy_kWh`,
  `Total_CO2_Emissions_kg`, `NCD_Peak_kW`, `OPD_Peak_kW`, `Missed_Work_hour`,
  `MCS_Transit_hour`, `Overnight_Recharge_kWh`, `Overnight_Cost_USD`, `Infeasible_windows`,
  `MPC_loop_time_s`. **`09_kpi_metrics_summary.png`** is a two-panel bar chart (costs + peaks).
* **`10_mip_convergence.csv`** — per applied window: `day`, `step`, `clock`, `status`,
  `objective`, `gap_percent`, `solve_time_s`.

The console also prints prior → final estimate → hidden truth, any infeasible-window count,
and the KPI summary.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7.
2. **Prior** — set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling` and
   `prior_sigma_frac` in `parameters.csv` to your offline values.
3. **Telemetry** — in `5_MPCLoop.jl`, `realized_activity_durations` currently *simulates* the
   within-interval activity split and `b_obs = a·true_powers + noise` fakes the meter. Replace
   both with the **actual** per-activity seconds and the measured interval energy from your
   CV/telematics pipeline. `true_powers` exists only to generate ground truth in the demo.

---

## 11. Relation to Shrinking Horizon & Scenario 2

* **Shrinking vs Receding.** The sibling `Shrinking_Horizon/` solves a **single day** and
  shrinks the window toward 18:00 (no cross-day look-ahead, no buffer day, one lumpsum work
  quota, one end-of-day terminal, and it does **not** add the no-working-ahead cap). This
  Receding version chains days with a cross-day window, a per-day work schedule, daily battery
  realignment, and the hard no-working-ahead cap. See `../../README.md` for the full compare.
* **Scenario 1 vs 2.** Both use the **same Bayesian regression**. Scenario 1 (this code) is
  *certainty-equivalent*: it collapses the posterior to its mean and solves one MILP. Scenario
  2 would sample multiple power scenarios from the *same* posterior and optimise over all of
  them — the posterior `sd` this estimator already computes is the hook.
