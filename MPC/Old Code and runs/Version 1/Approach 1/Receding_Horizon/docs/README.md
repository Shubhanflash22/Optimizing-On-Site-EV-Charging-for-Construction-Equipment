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
11. [Corrections & open items](#11-corrections--open-items)
12. [Relation to Avik's reference & Scenario 2](#12-relation-to-aviks-reference--scenario-2)

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
plans on the mean `μ` (certainty-equivalent). The **stochastic plant** draws each excavator's
realized power from a pool of samples pre-drawn from `N(μ, σ)` (§6.0), one draw per (excavator,
activity) occurrence, so realized consumption wobbles around the plan. The model is **not**
re-fit during the run (no online learning).

**Multi-day cross-day RECEDING horizon.** The simulation runs `n_days` **reported** days plus
**one dropped BUFFER day** (`D_total = n_days + 1`). Each 15-min re-solve optimises a
**cross-day window** = the rest of *today's* daytime block **plus** `lookahead_days` future
daytime blocks. Because only the first interval is applied and the window slides forward, the
horizon **recedes**. The day-block is the **full 24 h** (`n_day = n_int = 96`), so the overnight
recharge is scheduled *inside the same MILP* — there is no separate phase, and no
`phase2_overnight_charge` routine exists. The MCS terminal equality at the next day-start forces
the refill and the TOU price puts it in the cheapest hours; the CEV battery and any unfinished
work carry over into the next day. The buffer day is dropped from every reported output so the CEV
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
* **Nights:** ordinary intervals of the same MILP. `n_day = n_int = 96`, so the overnight hours
  are optimised alongside the day, and the MCS terminal equality at the next day-start
  (`SOE_MCS[b_term] == SOE_MCS_ini`, §8.7) forces the recharge; the TOU price then puts it in the
  cheapest hours. There is **no** `phase2_overnight_charge` routine in the code.
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
| `3_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **cross-day window MILP** over `K_win`, overnight hours included. HiGHS is configured crash-tolerant (see §6.4). |
| `4_MPCLoop.jl` | `MPCLoop` | The multi-day closed loop `run_mpc(d, pool; n_days, lookahead_days, …)` (Approach 1): for each day and each 15-min step solve the cross-day window, apply the first interval, draw the stochastic plant's realized power, advance the real state; drop the buffer day. **Plus** `run_one_shot(d, pool; n_days, …)` (Approach 0): solve each day's own window **once**, at that day's 8:00, and replay it open-loop for the day (no re-solves); errors immediately if a day's own MILP is infeasible. Both take a `plant` switch (`:sampled` / `:mean`, §6.5) and share the apply/simulate/advance logic (`apply_and_simulate!`), so they're directly comparable. Each returns one `res` NamedTuple carrying `approach`, `plant`, `n_infeasible` and `n_capped`. |
| `5_Output.jl` | `Output` | Every artefact from `res` via `write_outputs(res, out_dir)`: STEP figures (PNG) + CSVs over the concatenated kept horizon, KPI/cost reports, per-window solver diagnostics, worker schedule, and per-day replanning grids. **Plus** `write_approach_comparison(res0, res1, out_dir)`: an additive, numbers-only report — totals, a per-day breakdown and a run-diagnostics table — comparing Approach 0's and Approach 1's fully-realized outcomes, self-labelled by Approach 0's plant mode; not called from `write_outputs`, so it never changes the existing artefact set. |
| `6_Receding_Horizon_main.jl` | — | Thin orchestrator. `run_scenario_1(; mode, …)` = load → generate the shared `ActivityPowerPool` → `run_one_shot` (Approach 0, under `approach0_plant`) → `run_mpc` (Approach 1, always `:sampled`) → `write_outputs` + `write_approach_comparison` → print the KPI block and the two-run approach summary. Also mirrors console output to `run_log.txt` in `out_dir`. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |

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
> info logs red. If the KPI summary prints, the run succeeded. Each run now does Approach 0's
> `n_days` day-solves **plus** Approach 1's `(n_days + 1) · n_day` window re-solves.
> `time_limit_sec` now defaults to `Inf` (solve every window to the MIP gap, not a wall-clock
> cap); keep `n_days` modest and/or pass a finite `time_limit_sec` while iterating.

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `n_days` | `nothing` | reported days to **keep** (a buffer day is always simulated + dropped). `nothing` → use `d.n_days` from the data |
| `lookahead_days` | `1` | cross-day window depth: rest of today + this many future daytime blocks |
| `time_limit_sec` | `Inf` | HiGHS seconds per solve — each window solve for Approach 1, each day's whole-window solve for Approach 0. `Inf` = no limit (solve to the MIP gap); pass a finite value to cap it |
| `multi_activity` | `false` | if `true`, a 15-min interval realizes a **mix** (planned activity for a 60–100 % random fraction, idle for the rest); if `false`, the whole interval realizes the single planned activity |
| `require_site_visit` | `false` | force the MCS to visit at least one site |
| `single_visit_per_site` | `false` | at most one visit per site |
| `approach0_plant` | `:sampled` | which plant Approach 0's per-day 08:00 plans are replayed under — `:sampled` (each day's plan drifting under the stochastic pool, no intra-day feedback) or `:mean` (deterministic; realized equals planned within each day, so the KPIs are the per-day MILPs' own optima). **Exactly one runs per call** (§6.5). Approach 1 is always `:sampled` |
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
power `p_true`. The realized power drives both the CEV battery drain and the analyst log — but see §6.6 for
what that does *not* cover (the MCS side follows the plan, and the CEV balance is energy-capped).

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
        └───────────────────┤ per occurrence, DRAW next:│─────┘
                            │   p_true ~ pool(μ, σ)     │
                            │ batteries drain @ p_true  │
                            └──────────────────────────┘
  STATE-FEEDBACK loop : plant state after applying k0 ──► next MILP re-solve (k0+1)
  Each NIGHT : ordinary MILP intervals; the terminal equality at the next day-start
               forces the recharge, priced by the TOU curve. State carries to next day.
```

**Fork B (fixed-curve plant, shared sample pool).** Before either approach runs, `1_Common.jl`
generates an `ActivityPowerPool`: `n_samples` pre-drawn samples per **(CEV, activity)** pair from
the fixed calibrated `N(μ, σ)`, truncated at 0. Each interval, a CEV realizing activity `a`
consumes the **next unused** sample for that `(e, a)` pair (`next_power!`) instead of drawing a
fresh random number — so the sample consumed depends on *how many times that CEV has done that
activity so far*, not on the interval index. Idle (`σ = 0`) is deterministic and never consumes a
slot. Both Approach 1 (`run_mpc`) and Approach 0 (`run_one_shot`, §6.6) draw from the **same
pool**, each with its **own** cursor (`new_cursor`), so identical occurrence sequences draw
identical numbers and any KPI gap between the two approaches reflects the control strategy, not
different randomness. The same drawn sample drives the CEV battery drain, so reported energy
matches energy actually spent. `n_samples` scales with the number of days in play (the
single-day rule of thumb is 20 occurrences/activity; the run generates `20*(n_days_keep+1)`, the
`+1` covering Approach 1's dropped buffer day). (A "Fork A" against a separate hidden truth
`N(true_powers, true_sigma)` plus online `observe!`/`refit!` learning is a small edit in
`4_MPCLoop.jl`'s `apply_and_simulate!`, kept for experimentation; the shipped loop does not learn
online.)

**Low-level design — one 15-min step of `run_mpc`** (`4_MPCLoop.jl`; steps (2)-(4) live in the
shared `apply_and_simulate!`, also called by `run_one_shot`, §6.6):

```
 STEP (day dy, k0)  (one 15-min interval)                        4_MPCLoop.jl
 ──────────────────────────────────────────────────────────────────────────────
 (0) READ plant state:  soe_mcs, soe_cev, mcs_node, mcs_transit,
                        rem_dig, rem_load, hist (current day), peak_nc/op
 (1) OPTIMISE  build_window_model(d, K_win, …state…, μ)
        └─ infeasible → HOLD state (no fallback, nothing relaxed, no second solve)
        └─ still infeasible → hold state, continue
 (2) APPLY interval k0 only:  grid draw, MCS discharge, route, plug decisions;
        write the current-day slice of the forward plan into the replan grids
 (2.5) DRAW plant power PER EXCAVATOR from the SHARED pool (Fork B):
        p_true[e][a] = next_power!(pool, cursor, e, a)   -- one draw per
        (CEV, activity) OCCURRENCE this interval, not a fresh randn every step
 (3) SIMULATE realized activity split (60–100% planned, rest idle in multi mode)
 (4) ADVANCE plant:
        soe_mcs ← model's own SOE_MCS[k0+1]      (read directly, not recomputed)
        soe_cev ← soe_cev + charged − dot(a_real, p_true[e])
        rem_dig / rem_load −= realized;  push (activity, a_real) onto hist
 END OF DAY : missed-work snapshot;
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

### 6.2 One mechanism: a single 24 h window MILP
The window MILP of §8 covers the **full 24 h**, nights included (`n_day = n_int = 96`). Both
terminal targets are pinned at `b_term`, the boundary at the next day-start: the MCS to an exact
equality with `SOE_MCS_ini` (Eq. 8a) and each CEV to a floor at `SOE_CEV_ini` (Eq. 8b). The
overnight recharge that satisfies the MCS equality is therefore an ordinary part of the same
optimisation, scheduled against the TOU price. One mechanism, one horizon, one solve per step.

> Earlier revisions of this README described a second mechanism — a deterministic
> `phase2_overnight_charge` running outside a daytime-only MILP, with no in-MILP MCS equality.
> That design is not in the code.

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

### 6.5 Plant modes, the one-shot-per-day baseline & the comparison report

**The `plant` switch.** `run_mpc` and `run_one_shot` both take `plant = :sampled | :mean`. The
MILP is *identical* either way; only what the plant does with the applied plan changes.

| `plant` | Realized power | Activity split | Consumes pool samples? |
|---------|----------------|----------------|------------------------|
| `:sampled` (default) | next unused draw from the shared pool | may be randomised (`multi_activity`) | yes — advances the cursor |
| `:mean` | pinned to the planning mean `μ` | forced to the single planned activity, full interval | **no** — cursor untouched |

Under `:mean` nothing is stochastic, so **realized == planned exactly** and the run needs no seed
to reproduce. Because it consumes no samples it cannot perturb a `:sampled` run sharing the same
pool, so the two can be run back to back off one pool.

`run_scenario_1` runs **two** approaches every call, against the **same** `ActivityPowerPool`
(§6.0/§6.3), with `approach0_plant` deciding which baseline Approach 0 is:
* **Approach 0** (`run_one_shot`) solves each **kept** day's own daytime window **once**, at that
  day's 8:00, from the real state carried over from the previous day's execution, and replays
  that fixed plan open-loop for the day — no re-solves within the day. It needs **no buffer
  day**: each day's own window already ends exactly at the next day's 8:00, unlike Approach 1's
  window, which always looks a fixed distance ahead of *right now* and so needs the buffer day to
  give its last reported day's window somewhere sane to end. Approach 0 and Approach 1 handle
  nights identically, since nights are just intervals of the window MILP; the only difference between
  days as Approach 1. Approach 0 has no fallback: if a day's own MILP is infeasible, it errors
  rather than holding state.
* **Approach 1** (`run_mpc`, always `:sampled`) is the existing cross-day receding closed loop.

Under `approach0_plant = :mean` Approach 0 realizes exactly what it planned *within each day*, so
its KPIs are the per-day MILPs' own optima, chained through the real end-of-day carry-over — the
deterministic reference. Under `:sampled` both approaches consume the same underlying samples via
independent cursors (§6.0), so the gap is a like-for-like measure of intra-day re-planning value.
Which you pick decides what the reported Δ *means*, so the report labels itself accordingly:

```
  A0(:sampled) ──► A1   =  value of intra-day re-planning     (like-for-like plant)
  A0(:mean)    ──► A1   =  drift AND re-planning together     (the net figure)
```

Run it both ways if you want the two separated; the difference between the two A0 numbers is the
cost of plan drift alone. Neither replay re-solves a MILP, so a second run is cheap.

`write_approach_comparison` writes this as a numbers-only `approach0_vs_approach1.html` (§9.2)
with a **totals** table over all kept days, a **per-day** breakdown (physical/additive quantities
only — demand charges and the missed-work penalty are whole-run concepts and aren't split per
day), and a **run-diagnostics** table (plant mode, infeasible windows, CEV SOE capped count, solve
time). It is **additive** and never alters the existing figure or report set.

### 6.6 What is *not* stochastic, and where the SOE trace is not exact
Two scope limits worth stating plainly, because "the stochastic plant" over-promises:

* **Only the CEV side is stochastic.** `apply_and_simulate!` advances the MCS with
  `soe_mcs[m] = value(SOE_MCS[m, g0+1])` — the *planned* value. MCS energy, grid draw and travel
  loss therefore follow the plan exactly, and so does every cost term. The disturbance enters
  only through CEV depletion, reaching the objective indirectly via what the next re-solve does
  about it.
* **The CEV balance is energy-capped.** Before crediting any realized dig/load/travel duration,
  `apply_and_simulate!` checks the energy actually available before `SOE_min` (`headroom = soe_cev +
  charged − SOE_min`) against what the sampled draw would cost. If the draw exceeds headroom, the
  realized duration is scaled down to exactly what's affordable — crediting only the completed
  fraction to `rem_dig`/`rem_load`, with the remainder of that interval becoming idle — *before* the
  SOE update is applied. This is a physical correction, not a numerical guard: no energy is created
  or destroyed, and `rem_dig`/`rem_load` honestly reflect what could actually be done. A residual
  `clamp(soe_cev, SOE_min, SOE_max)` remains afterward purely as a safety net and should not bind in
  normal operation. This is invisible to the MILP's feasibility status, so a run can report **0
  infeasible windows** and still have hit an energy shortfall mid-interval. Every capping event is
  counted (`res.n_capped`), printed at the end of the run, and shown per run in the comparison report
  and the sweep summary. **A non-zero capped count means some realized work was cut short by
  available charge that interval** — check `rem_dig`/`rem_load` at day's end for the resulting
  shortfall.

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
full 24 h day-block. `t_start` is inferred as `first-row-clock − delta_T`; the same daily profile
prices the overnight hours, which are ordinary intervals of the window MILP.

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
peaks `P_peak_NC`/`P_peak_OP`, and the slack `s_miss_work[i,a]` (per site & activity — **not** per
day-block). This is the model's only slack; every other constraint is hard and there are no
`soft_*` levers.

> `s_miss_work` is declared over all four activities and the objective sums all four, but only dig
> and load appear in a balance (§8.10). The travel/idle entries are free `≥0` variables carrying a
> positive cost, so the minimisation drives them to 0 — dead variables, not a missing quota. The
> source PDF's objective has the same quirk. Likewise `y_trv[m,i,i,k]` is never defined (the
> defining loop skips `i == j`) yet is summed into `L_trv` and the labour term; it only ever costs,
> so presolve fixes it to 0.

**Binary:** `u[e,i,a,k]`, `mu[i,e,k]`, `rho[m,i,e,k]`, `z[m,i,k]`, `g_ch[m,i,k]`, `x[m,i,j,k]`,
`y_trv[m,i,j,k]`, `beta_arr`/`beta_dep`.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ missed work` (SOFT, ρ_miss·Σ s_miss_work over sites × activities)
`+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP)
`+ towing labour` (ρ_labor·Δt·Σ y_trv). There are no `soft_*` penalty terms. The overnight
recharge is scheduled inside this MILP, so its cost is already carried by the energy and carbon
terms — there is no separate overnight accounting.

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
* **MCS:** `SOE_MCS[k+1] = SOE_MCS[k] + η·P_ch_tot·Δt − P_dch_tot·Δt/η − L_trv_tot`, run
  **unbroken across the whole window, nights included**. There is no reset bridge.
* **CEV:** `SOE_CEV[k+1] = SOE_CEV[k] + Σ P_MCS_CEV·Δt − Σ P_work·Δt`, likewise unbroken.
* **Bounds:** every boundary clamped to `[SOE_min, SOE_max]` for MCS and CEV (a safety net; the
  realized CEV work duration is capped by available energy *before* this bound is ever reached — see
  §6.7).

### 8.7 Terminal energy targets (Eq. 8a / 8b, HARD)
Both targets are pinned to `b_term = k_term + 1`, the boundary at the **next day-start**, where
`k_term = firstday · n_day`. They are applied **unconditionally** — the window is one day-block
long, so that boundary always falls inside it.

* **MCS (Eq. 8a):** `SOE_MCS[b_term] == SOE_MCS_ini` — an **exact equality**. The overnight
  recharge that satisfies it is scheduled *inside this same MILP*, driven by the TOU price, so
  the cheapest hours are chosen automatically.
* **CEV (Eq. 8b):** `SOE_CEV[b_term] ≥ SOE_CEV_ini` — a **floor**; overcharging is allowed. A CEV
  cannot discharge, so a hard equality would be unrecoverable once the stochastic plant lets a CEV
  drift above target. The floor keeps the terminal reachable while guaranteeing the fleet ends at
  least as charged as it began.

Both are **hard**, with no tolerance band and no soft variant. There is no keep-up reserve, no
anti-hoarding cap and no daytime health floor in the code — earlier revisions of this README
described all three, plus `soft_term`/`term_tol` levers, none of which exist.

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

### 8.10 Work quota (Eq. 12c, SOFT shortfall + implicit HARD cap)
Each reported day issues its **own** per-site dig/load quota (`dig_by_day`/`load_by_day`) — but
**the roll-over happens outside the MILP.** The closed loop adds each morning's quota to a running
backlog (`rem_dig .+= quota_dig(day)`) and subtracts realized work as it is applied. The MILP sees
only that backlog, and enforces **one** balance per site over the intervals up to `k_term`:

```julia
delta_T * sum(u[e, i, B[1], k] for e in E, k in K if k <= k_term) + s_miss_work[i, B[1]] == max(rem_dig[i], 0.0)
```

Note what this is and is not:
- `s_miss_work[N_c, B]` is a **single slack per (site, activity)** — it is *not* indexed by
  day-block, and there is **one** constraint per site, not one per day in the window.
- The shortfall is soft (priced at `rho_miss`) and rolls over via the backlog.
- Because `s_miss_work ≥ 0`, the equality **also implies** `delta_T·Σu ≤ rem_dig[i]` — a hard
  upper cap. That is how "no working ahead" is obtained: **implicitly**, and only up to `k_term`.
- Intervals in the window's tail **beyond `k_term` are excluded from the balance entirely** and
  carry no quota accounting. They are unconstrained rather than budgeted. Benign in practice —
  work costs energy and earns nothing until the next day's quota is issued — but it means the cap
  guarantees nothing about the tail.

The dropped buffer day gets **no** fresh quota.

> **Correction.** Earlier revisions of this README and `math_model.tex` described per-day-block
> slacks `s_miss_{i,d}` with a cumulative target `T_{i,d}` summing future days' quotas, split into
> separate SOFT-lower and HARD-upper inequalities. That formulation is not implemented; the code
> uses the single equality above with roll-over handled by the loop.

### 8.11 Precedence (Eq. 12d, HARD)
Cumulative loading `≤ scale ·` cumulative digging, in raw interval counts exactly as in Avik,
seeded by the realized work applied so far. **The counters run continuously and do not restart
each morning** — the shared activity history is never reset (see `4_MPCLoop.jl`, the day loop
explicitly carries `hist` over), so the seed aggregates every interval since the run began.
Hard; there is no precedence slack and no `soft_prec` lever.

Note the seed aggregates **per site** (`cum_*_site`) while travel pacing (§8.13) seeds **per CEV**
(`cum_*_e`). The two agree while `A[i,e]` is a one-to-one assignment, which it is in both
datasets; they would diverge if a site ever got two CEVs.

### 8.12 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4 at the defaults; the code **rounds**, it does not
take a ceiling, so the two differ for any `t_limit_rest` that is not a whole multiple of `Δt`)
and `rest_win = 5`: over any 5 consecutive
intervals a CEV does **at most 4** work intervals (≥ 1 idle break). Two parts: within-window
5-windows, **plus a seam** seeded with the applied activity history so a work-run cannot leak
across the every-15-min re-solves. **The counters do not reset each morning** — the shared history
runs continuously for the whole multi-day run. Because the day-block is the full 24 h, the
overnight intervals sit in that history as idle, so a night supplies its own break and the seam is
harmless in practice; but the "resets each morning" mechanism described in earlier revisions does
not exist.

### 8.13 Travel pacing (Eq. 13, HARD, no tolerance)
As in Avik with `work_per_travel = 4`: for each assigned `(site, CEV)`, the two-sided band
`W(k) − 4 ≤ 4·V(k) ≤ W(k)` on cumulative travel `V` vs cumulative useful work `W`. `V` and `W` are
now raw **applied interval counts** off the `u` indicator (`cum_trv_cnt_e`/`cum_work_cnt_e`), not
hours — a battery-shortage-capped interval still counts as one full travel/work interval, so no
tolerance is needed (supersedes D11 below).

Unlike precedence (which still seeds off the whole-run hours), the pacing seed is scoped to the
**current calendar day only**: `hist` never resets across days, so the count is taken from
`today_start = (firstday-1)*n_day + 1` onward, i.e. only intervals applied since this calendar
day's 8am boundary. A CEV's travel/work balance resets fresh every morning.

> **No graceful fallback.** Every constraint except `s_miss_work` is hard and there is no soft
> re-solve. An infeasible window makes the plant **hold state** for that interval, counted as
> `n_infeasible`; nothing is relaxed and no second solve is attempted. Earlier revisions described
> a soft-terminal / relaxed-reserve retry and a `Softened_windows` counter — neither exists.

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

**`approach0_vs_approach1.html`** — Approach 0 (one-shot-per-day plan, executed open-loop) vs
Approach 1 (closed-loop MPC), both **fully realized** over the same kept days. A totals table
(grid energy, energy/CO₂/demand/missed-work/travel costs, and total), a per-day breakdown so a
gap can be localized to a specific day, and a **run-diagnostics** table (plant mode, infeasible
windows, CEV SOE capped count, solve time). The Approach 0 column header and the explanatory
blurb **adapt to `res0.plant`**, so the report always states whether the Δ is a like-for-like
intra-day re-planning gap (`:sampled`) or the combined drift-plus-re-planning figure (`:mean`),
and points at the other mode (§6.5). Written by `write_approach_comparison`; additive only.

**`run_log.txt`** — everything printed to the console during the run (progress, KPI summary,
`@warn`s), mirrored to this file in addition to the terminal.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7. Set
   `n_days` and (optionally) `work_by_day.csv` for the multi-day work plan.
2. **Power model** — set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling` and
   `prior_sigma_frac` in `parameters.csv` to your calibrated values.
3. **Telemetry** — in `4_MPCLoop.jl`'s `apply_and_simulate!`, replace the simulated
   `realized_activity_durations` call and the `next_power!` draw from the shared
   `ActivityPowerPool` (`1_Common.jl`) with the actual per-activity seconds + measured interval
   energy from your pipeline. Note `d.true_powers` is a Fork-A leftover and is no longer read by
   the plant at all. The `ActivityPowerPool` / `true_powers` machinery only generates
   ground truth for the demo and disappears at go-live. (To *learn* online instead of using a
   fixed model, re-enable the `observe!`/`refit!` path — Fork A.)

---

## 11. Corrections & open items

Carried over from the audit of the single-day Shrinking sibling; the same code paths exist here.

**Fixed in the code:**
* The analyst log's `work_kW` was computed from `d.true_powers` (a Fork-A hidden-truth curve)
  while the batteries drained on the Fork-B pool draw `p_true`; in `:synthetic` those vectors
  differ, so the column contradicted the batteries it described. Now computed from `p_true`.
  Figures and KPI CSVs never read the column, so only `run_log.txt` was affected.
* The CEV SOE clamp used to silently create or destroy energy when it bit, because
  `apply_and_simulate!` credited the full realized activity duration before clamping the resulting
  SOE. Fixed: realized dig/load/travel duration is now capped by available headroom *before* it is
  credited (§6.6), so `rem_dig`/`rem_load` and the SOE trajectory stay physically honest. Events are
  counted as `n_capped` (renamed from `n_clamped`). The residual `clamp()` call is now only a
  never-should-bind safety net.
* The two-sided travel-pacing band (Eq. 13) could become spuriously infeasible: the sub-interval
  residue the capping fix above can leave in `cum_dig_e`/`cum_load_e` could land the band's floor
  and ceiling on opposite sides of an integer boundary with no whole-interval solution in between,
  permanently blocking further travel/work for a CEV even though the shortfall was physically
  meaningless (traced in detail via HiGHS IIS on `run10_SOE_14.80`). Fixed by adding
  `pacing_tol = 0.05` to the floor side only (§8.13) — large enough to absorb realistic capping
  residue, small enough (5% of one interval) that it cannot be mistaken for a free extra work unit.
  **Superseded:** pacing now seeds off applied interval counts instead of hours (scoped to the
  current calendar day, §8.13), which removes the fractional residue at its source, so
  `pacing_tol` was dropped entirely.

**Corrected in the docs:**
* `rest_cap` uses `round`, not `ceil` (§8.12) — `math_model.tex` said `⌈·⌉`.

**Open / instrumented, not resolved:**
* Only the CEV side of the plant is stochastic (§6.6).
* `s_miss_work` is declared over all four activities and summed in the objective, but only dig and
  load appear in a balance; the travel/idle entries are free `≥0` variables carrying a positive
  cost, so they are driven to 0 — dead variables, not a missing quota. Likewise `y_trv[m,i,i,k]`
  is never defined yet is summed into `L_trv` and the labour term; it only ever costs, so presolve
  fixes it to 0.
* Precedence seeds per **site**, travel pacing seeds per **CEV**. Equivalent while `A[i,e]` is a
  one-to-one assignment, which it is in both datasets; latent if a site ever gets two CEVs.

---

## 12. Relation to Avik's reference & Scenario 2

* **Avik's single-shot model** (`MCS_OPTIMAL_v4_real.jl`) solves one day in **one** MILP with
  deterministic powers and exact terminal equalities. This code is the multi-day **MPC** version:
  identical variable names, objective, routing, battery, precedence (raw-count) and travel-pacing
  (`work_per_travel = 4`); the differences are the cross-day receding re-solves, per-window
  history seeds, the carried-in MCS start position, per-day work quotas with roll-over, the CEV
  terminal **floor** (`≥`, overcharge allowed) instead of exact equality, and the separate
  overnight recharge phase. **Approach 0** (`run_one_shot`, §6.5) is this codebase's own
  reproduction of Avik's single-shot model **applied once per day** — same daytime window MILP,
  solved once instead of every 15 min — with the stochastic plant executed on top of it, so it's
  the direct apples-to-apples baseline for Approach 1 rather than a number computed elsewhere.
  Measured (`approach0_vs_approach1.html`): on **input** (1 kept day) Approach 1 matches
  Approach 0 to within ~0.3% ($193.97 vs $193.40); on **synthetic** (2 kept days) they diverge
  by **68%** ($233.85 vs $139.05), almost entirely from MCS transit — Approach 1 logs a
  consistent ~1.5 h/day of *extra* driving on both days, not a one-off — an unexplained,
  larger-scale echo of the smaller (28%) synthetic gap seen in the single-day
  Shrinking_Horizon sibling, worth investigating rather than assumed benign.
* **Scenario 1 vs 2.** Scenario 1 (this code) is certainty-equivalent (plans on the mean `μ`).
  Scenario 2 would sample multiple power scenarios from `N(μ, σ)` — the fixed `σ` is the hook.

## Changes 2, 3, 5 (this session)

* **Change 2 — earlier-charging tie-break (Issue 2).** Same `1e-6`-weighted `idx * mu` term as
  the Shrinking_Horizon sibling (see that doc for the `g_ch` → `mu` correction — the term originally
  targeted the wrong variable), added to `3_MCSModel.jl`'s objective. `idx` is the LOCAL position
  within the current re-solve window.
* **Change 3 — terminal SOE_CEV shortfall penalty (Issue 1).** Same design as the sibling doc.
  `rem_dig`/`rem_load`/`soe_cev` are snapshotted at the kept-day boundary (`rem_dig_kept` etc.)
  before the trailing buffer day mutates them further, so the shortfall reflects the kept-day
  numbers klog reports, not the buffer day.
* **Change 5 — `n_day_run` (currently 1).** The day-count knob is renamed `n_day_run` and now
  defaults to 1 (was a hardcoded 2-day synthetic default using two *different* day-1/day-2 quotas).
  When `n_day_run > 1`, every day now uses the *same* work requirement, repeated. Real-state
  day-to-day carry-over (`quota_dig`/`quota_load`, `soe_mcs`/`soe_cev` continuation) already existed
  here and is unchanged — this only renames/redefaults the day-count knob and its quota semantics.
