# Shrinking Horizon — Stochastic (Scenario-Based) MPC for MCS Dispatch

A Julia implementation of **Scenario 1 (Approach 2: Stochastic Scenario-Based
MPC)** on a **single-day, full-24 h SHRINKING horizon**. It dispatches a **Mobile Charging
Station (MCS)** — a battery on wheels — to a fleet of **Construction EVs (CEVs / electric
excavators)**, deciding every 15 minutes *when to buy grid power, where to drive the MCS,
and which excavator to top up*, at minimum operating cost.

> **About this document.** Sections 1–12 below are the deep reference for the underlying
> window MILP — sets, variables, every constraint, every input/output file — and the
> **physics, data schema, and output artefacts are identical between Approach 1
> (deterministic) and Approach 2 (stochastic)**. Where a claim in §1–12 depends on *which*
> controller is planning (e.g. "the MILP plans on the mean"), it's now annotated inline for
> both cases; the full account of what's different is in **§13**: instead of planning once
> on the posterior mean, every re-solve samples a small set of scenarios and plans one shared
> action that hedges across all of them. If you haven't read `Understanding_Stochastic_MPC.md`
> yet (five levels of depth, plus a worked numerical example), start there before this file —
> it explains the *idea*; this file documents the *implementation*.

> **This README is the single deep reference.** It documents every code file, every input
> column, every constraint (each marked **HARD**/**SOFT**), and every output file. The formal
> LaTeX model is in `math_model.tex`; a line-by-line code-vs-model audit is in
> `constraints_code_vs_model.txt`.

| Program | Language | MILP solver | Power model |
|---------|----------|-------------|-------------|
| `6_Shrinking_Horizon_main.jl` (+ numbered modules `1_`…`5_`, `2b_ScenarioSampler.jl`) | Julia | JuMP + HiGHS | Fixed calibrated Bayesian prior (μ, σ); Approach 2 samples `S` scenarios from it per re-solve |

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
12. [Corrections & open items](#12-corrections--open-items)
13. [Approach 2 — the stochastic extension](#13-approach-2--the-stochastic-extension)

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
(`sigma_*`) into `parameters.csv` (see §6.4). Approach 0's one-shot baseline plans on the mean
`μ` directly (certainty-equivalent); **this codebase's controller (Approach 2) instead samples a
small set of scenarios from this same posterior at every re-solve and hedges across all of them
at once** — see §13. The **stochastic plant**, either way, draws each excavator's realized power
from a pool of samples pre-drawn from `N(μ, σ)` (§6.0), one draw per (excavator, activity)
occurrence, so realized consumption wobbles around the plan. The model is **not** re-fit
*during* the day (the fit is offline; step 0 can be skipped with `run_regression=false`).

**Shrinking horizon over the full day.** Each 15-min re-solve plans the **entire remaining
24 h** `[k0 … 08:00 next day]`, so the window **shrinks** as the day progresses (96 intervals
at 08:00, 1 interval at 07:45 next day). There is **one** optimisation per step and **one**
horizon — the overnight MCS recharge is scheduled *inside* the same MILP (no separate phase).

**A one-shot baseline to measure against.** Every run also solves the whole day **once** at
08:00 and replays that fixed plan open-loop, under a plant you choose: **deterministic**
(realized power = the planning mean, so realized equals planned exactly — the MILP's own
optimum) or **stochastic** (the same plan drifting with no feedback). Which you pick decides
whether the reported gap is the value of re-planning alone or drift and re-planning combined
(§6.6).

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
│   ├── 6_Shrinking_Horizon_main.jl
│   └── run_soe_sweep.jl         (harness: sweeps SOE_CEV_ini, writes summary.html)
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
| `2b_ScenarioSampler.jl` | `ScenarioSampler` | **New in Approach 2.** Standalone module: `sample_scenarios(mu, sd, n_scenarios; rng)` draws `n_scenarios` independent power vectors from `N(μ,σ)⁺`; `equal_weights(n_scenarios)`; `DEFAULT_N_SCENARIOS = 5`. Used by `4_MPCLoop.jl`'s `run_mpc` every re-solve. See §13. |
| `3_MCSModel.jl` | `MCSModel` | The optimise half. `build_window_model(...)` builds & solves the **single 24 h window MILP** over `[k0 … n_day]` — used by Approach 0's `run_one_shot`, unchanged from Approach 1. **New in Approach 2:** `build_window_model_stochastic(...)` — the scenario-linked sibling `run_mpc` actually calls every re-solve (§13.3). Nomenclature matches Avik's `MCS_OPTIMAL_v4_real.jl`. HiGHS is configured crash-tolerant (see §6.5). |
| `4_MPCLoop.jl` | `MPCLoop` | `run_mpc(d, pool; shrinking, H, n_scenarios, …)`: the closed loop (Approach 2, **stochastic**) — for each 15-min step, sample `n_scenarios` scenarios, solve the scenario-linked (shrinking) window, apply the shared first-interval decision, draw the stochastic plant's realized power, advance the real state. **Plus** `run_one_shot(d, pool; …)`: Approach 0 — solve the full-day window **once**, certainty-equivalent, and replay it open-loop (no re-solves) — unchanged from Approach 1. Both take a `plant` switch (`:sampled` / `:mean`, §6.6) and share the apply/simulate/advance logic (`apply_and_simulate!`, reached from `run_mpc` via the thin `apply_and_simulate_stochastic!` wrapper, §13.4), so they're directly comparable. Each returns one `res` NamedTuple carrying `approach` (`0` or `2`), `plant`, `n_infeasible`, `n_capped`, and (for `run_mpc`) `n_scenarios`. |
| `5_Output.jl` | `Output` | Every artefact from `res` via `write_outputs(res, out_dir)`: STEP figures (PNG) + CSVs, the KPI report, the replanning grids, and the plan-vs-actual **financial** + **activity** reports. **Plus** `write_approach_comparison(res0, res1, out_dir; res0_mean = nothing)`: an additive, numbers-only report comparing the fully-realized outcomes of Approach 0 (both plant modes) and Approach 2, plus a per-run diagnostics table; not called from `write_outputs`, so it never changes the existing artefact set. |
| `run_soe_sweep.jl` | — | Sensitivity harness (not part of the controller). Sweeps `SOE_CEV_ini` across `NRUNS = 10` points from `SOE_min` to `SOE_max` at a fixed seed, re-running all three approach/plant combinations per point, keeping a subset of artefacts per run and writing `output/input_testing/summary.html`. Sets `SCENARIO1_NO_AUTORUN` before including the driver. |
| `6_Shrinking_Horizon_main.jl` | — | Thin orchestrator. `run_scenario_1(; mode, …)` = load → generate the shared `ActivityPowerPool` → `run_one_shot` twice (Approach 0 with `plant = :mean`, then `plant = :sampled`) → `run_mpc` (Approach 2) → `write_outputs` + `write_approach_comparison` → print the KPI block and the three-way approach summary. Also mirrors console output to `run_log.txt` in `out_dir`. **Auto-runs `:synthetic` on `include`** unless `SCENARIO1_NO_AUTORUN = true`. |

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
> info logs red. If the KPI summary prints, the run succeeded. Each run now does the single
> whole-day solve for Approach 0 **plus** the full-24 h MILP re-solved 96 times for Approach 2.
> `time_limit_sec` now defaults to `Inf` (solve every window to the MIP gap, not a wall-clock
> cap), so total runtime depends on solve difficulty — pass a finite `time_limit_sec` to bound it.

---

## 5. `run_scenario_1` options

| kwarg | default | meaning |
|-------|---------|---------|
| `mode` | `:synthetic` | `:synthetic` (built-in) or `:input` (CSV dataset) |
| `input_dir` | `../data/input_data` | dataset folder (relative to `code/`) |
| `shrinking` | `true` | `true` = shrinking horizon (each step solves `[k0 … n_day]`); `false` = fixed lookahead of `H` intervals. **`false` is experimental** — the terminal rules 8a/8b/10e are gated on the window reaching day-end, so under a fixed `H` they vanish from all but the last `H` windows with nothing replacing them (§8.7) |
| `H` | `16` | fixed lookahead length in intervals (only used when `shrinking = false`) |
| `approach0_plant` | `:sampled` | which plant Approach 0's one-shot 08:00 plan is replayed under — `:sampled` (the plan drifting under the stochastic pool, no feedback) or `:mean` (deterministic; realized equals planned, so the KPIs are the whole-day MILP's own optimum). **Exactly one runs per call** (§6.6) |
| `time_limit_sec` | `Inf` | HiGHS seconds per solve — each window solve for Approach 2, the single whole-day solve for Approach 0. `Inf` = no limit (solve to the MIP gap); pass a finite value to cap it |
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

`run_mpc` and `run_one_shot` additionally take `plant = :sampled | :mean` directly (§6.6).
`run_scenario_1` fixes Approach 2 to `:sampled`. Note `plant` only controls what the **plant**
does — it does **not** turn off scenario hedging: even `run_mpc(d, pool; plant = :mean)` still
samples `n_scenarios` scenarios and solves the non-anticipative model every step (§13), so it is
reproducible (fixed `seed` ⇒ fixed scenario draws) but **not** identical to Approach 0's
certainty-equivalent plan. For the old "is this a seam bug or is this randomness" diagnostic,
pass `n_scenarios = 1` as well — that collapses `build_window_model_stochastic` to exactly
`build_window_model` (§13.6) — see the diagnostic order note below.

---

## 6. How the controller works

### 6.0 Architecture — one feedback loop, a stochastic plant

Two deliberately separated "worlds": a hidden **PLANT** (reality) and the **CONTROLLER** (the
brain). The brain — Approach 2's `run_mpc` — plans on a **sampled set of scenarios** drawn from
the posterior `N(μ,σ)` every re-solve (§13), hedging across all of them at once
(non-anticipativity); Approach 0's `run_one_shot` instead plans once on the fixed best guess `μ`
directly. Either way, the plant reacts using a realized power `p_true` that is either **sampled**
from the pool or **pinned to `μ`**, depending on the plant mode (§6.6) — a completely separate
draw from the controller's own scenario sampling (§13.5). The realized power drives both the CEV
battery drain and the analyst log.

```
                     ┌──────────────────────────────────────────────────┐
                     │                CONTROLLER  (brain)                 │
        plan:        │   (0) sample S scenarios from N(μ,σ) [2b_Scenario- │
   charge / route /  │       Sampler.jl]  (1) 3_MCSModel.build_window_    │
   work / plug       │       model_stochastic(state, scenarios) → S      │
        ▲            │       linked 24 h MILPs, non-anticipative at k0   │
        │            │   → HiGHS solves; APPLY only the SHARED interval  │
        │            │     k0 decision (identical across every scenario) │
        │            └───────────────┬──────────────────────▲────────────┘
        │                            │ apply k0             │ posterior N(μ,σ)
        │ measured state             ▼                       │ (calibrated prior;
        │ (SOE, position,   ┌──────────────────────────┐     │  scenarios resampled
        │  work, peaks)     │       PLANT (reality)     │     │  fresh every step)
        └───────────────────┤ per occurrence, DRAW next:│─────┘
                            │   p_true ~ pool(μ, σ)     │
                            │ batteries drain @ p_true  │
                            └──────────────────────────┘
  STATE-FEEDBACK loop : plant state after applying k0 ──► next MILP re-solve (k0+1)
```

Note the diagram's two separate uses of `N(μ,σ)`: the **plant** pool (bottom right) is drawn
**once**, up front, and is what "reality" turns out to be; the **scenario set** the controller
plans against (top right) is resampled **fresh every re-solve** from a completely separate RNG
stream and thrown away the moment that window is solved. See §13 for the full detail — this
diagram shows the Approach 2 (stochastic) planning step; the certainty-equivalent single-`μ`
planning step it replaces is still exactly what Approach 0's `run_one_shot` does (§8, unchanged).

**Fork B (fixed-curve plant, shared sample pool).** Before either approach runs, `1_Common.jl`
generates an `ActivityPowerPool`: `n_samples` (20) pre-drawn samples per **(CEV, activity)**
pair from the fixed calibrated `N(μ, σ)`, truncated at 0. Each interval, a CEV realizing
activity `a` consumes the **next unused** sample for that `(e, a)` pair (`next_power!`) instead
of drawing a fresh random number — so the sample consumed depends on *how many times that CEV
has done that activity so far*, not on the interval index. Idle (`σ = 0`) is deterministic and
never consumes a slot. Both Approach 2 (`run_mpc`) and Approach 0 (`run_one_shot`, §6.6) draw
from the **same pool**, each with its **own** cursor (`new_cursor`), so identical occurrence
sequences draw identical numbers and any KPI gap between the two approaches reflects the control
strategy, not different randomness. The same drawn sample drives the CEV battery drain **and** the logged
work power, so the two agree. Two caveats on how far that goes: the MCS side follows the plan,
not the plant, and the CEV balance is energy-capped at its SOE floor — see §6.7. (A "Fork A" against a separate hidden truth
`N(true_powers, true_sigma)` is a one-line edit in `4_MPCLoop.jl`'s `apply_and_simulate!`, kept
for experimentation.)

**Low-level design — one 15-min step of `run_mpc`** (line numbers into `4_MPCLoop.jl`; steps
(2)–(4) live in the shared `apply_and_simulate!` — reached here via the thin
`apply_and_simulate_stochastic!` wrapper, §13.4 — also called by `run_one_shot`, §6.6):

```
 STEP k0  (one 15-min interval)                                   4_MPCLoop.jl
 ──────────────────────────────────────────────────────────────────────────────
 (0) READ plant state:  soe_mcs, soe_cev, mcs_node, mcs_transit,
                        rem_dig, rem_load, hist, peak_nc/op
 (0.5) SAMPLE  scenarios = sample_scenarios(est.mu, est.sd, n_scenarios;
                rng = rng_scenarios)   -- S fresh draws, own RNG stream (§13.5)
 (1) OPTIMISE  build_window_model_stochastic(d, k0:nK, …state…, scenarios,
                weights)  -- S linked MILPs, tied at k0 (non-anticipativity, §13.3)
        └─ infeasible under HARD constraints (in ANY scenario) → hold state, continue (no fallback)
 (2) APPLY interval k0 only — the SHARED (scenario-1) decision, since non-
        anticipativity guarantees every scenario agrees there:  grid draw, MCS
        discharge, route, plug decisions; write the forward plan into the replan
        grids (scenario-1's forward look for k > k0 — a diagnostic display only)
 (2.5) DRAW plant power PER EXCAVATOR — depends on the plant mode (§6.6):
        :sampled → p_true[e][a] = next_power!(pool, cursor, e, a)  -- one
                   draw per (CEV, activity) OCCURRENCE, not a fresh randn
        :mean    → p_true[e][a] = mu[a]        -- no draw, cursor untouched
 (3) SIMULATE realized activity split (60–100% planned, rest idle in multi
        mode; forced to the single planned activity in full under :mean)
 (4) ADVANCE plant:
        soe_mcs ← model SOE_MCS[k0+1]     (PLANNED, not stochastic — §6.7)
        a_real  ← capped by headroom = soe_cev + charged − SOE_min, if draw exceeds it
                                          (capping events counted — §6.7)
        soe_cev ← soe_cev + charged − dot(a_real, p_true[e]), then clamp(·, min, max) as a safety net
        rem_dig / rem_load −= realized (capped) amount;  push a_real onto hist
 ──────────────────────────────────────────────────────────────────────────────
```

Note step (2.5)'s `pool`/`cursor` — the realized plant draw — is a **completely different**
`N(μ,σ)` draw from step (0.5)'s `scenarios` — see the RNG-separation note under the diagram
above and §13.5. `run_one_shot` (Approach 0) skips (0.5) and (1) entirely (it solves the plain
`build_window_model` **once**, at 08:00, on `pool.mu` — no scenarios, no non-anticipativity)
and then replays steps (2)–(4) unchanged for every interval against that one fixed plan.

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

### 6.6 Plant modes, Approach 0's two baselines & the comparison report

**The `plant` switch.** `run_mpc` and `run_one_shot` both take `plant = :sampled | :mean`. The
MILP is *identical* either way; only what the plant does with the applied plan changes.

| `plant` | Realized power | Activity split | Consumes pool samples? |
|---------|----------------|----------------|------------------------|
| `:sampled` (default) | next unused draw from the shared pool | may be randomised (`multi_activity`) | yes — advances the cursor |
| `:mean` | pinned to the planning mean `μ` | forced to the single planned activity, full interval | **no** — cursor untouched |

Under `:mean` nothing is stochastic, so **realized == planned exactly** and the run needs no
seed to reproduce. Because it consumes no samples it cannot perturb a `:sampled` run sharing
the same pool, so the two can be run back to back off one pool.

**Two runs per call, and you choose the baseline.** `run_scenario_1` solves the whole day once
at 08:00, replays it open-loop under **one** plant mode selected by `approach0_plant`, and then
runs the closed loop:

* **Approach 0** (`run_one_shot(…, plant = approach0_plant)`) — either the **deterministic**
  baseline (`:mean`), whose KPIs *are* the whole-day MILP's own optimum and which is the clean
  reproduction of Avik's single-shot model; or the **stochastic** baseline (`:sampled`), the same
  fixed plan drifting with no feedback.
* **Approach 2** (`run_mpc`, always `:sampled`) re-plans every 15 min against the pool.

Which one you pick decides what the reported Δ *means*, so the report labels itself accordingly:

```
  A0(:sampled) ──► A1   =  value of re-planning against drift   (like-for-like plant)
  A0(:mean)    ──► A1   =  drift AND re-planning together       (the net figure)
```

Run it both ways if you want the two separated; the difference between the two A0 numbers is the
cost of plan drift alone. Neither replay re-solves the MILP, so a second run is cheap.

**Diagnostic order when the approaches disagree.** Run `run_mpc(d, pool; plant = :mean,
n_scenarios = 1)` first — `n_scenarios = 1` is required here, not just `plant = :mean`: with
`plant = :mean` alone the controller still hedges against `n_scenarios` sampled scenarios every
step (§13), so it will **not** reproduce `A0(:mean)` exactly even though the plant is
deterministic — that gap is real hedging cost, not a bug. With **both** `plant = :mean` and
`n_scenarios = 1`, `build_window_model_stochastic` collapses to exactly `build_window_model`
(§13.6) and, with `shrinking = true`, this fully certainty-equivalent closed loop *should*
reproduce `A0(:mean)` exactly; if it does not, the cause is a closed-loop seam or carry-in bug
(§6.1) rather than anything to do with randomness or hedging, and the sampled/hedged numbers are
not worth interpreting until it is found.

`write_approach_comparison(res0, res1, out_dir; res0_mean)` writes all of this as a numbers-only
`approach0_vs_approach1.html` (§9.2), with a **run-diagnostics table** underneath (plant mode,
infeasible windows, CEV SOE capped count, solve time). It is **additive** and never alters the
existing figure or report set. Approach 0 has no fallback: if the single 08:00 MILP itself is
infeasible, it errors rather than holding state.

### 6.7 What is *not* stochastic, and where the SOE trace is not exact
Two scope limits worth stating plainly, because "the stochastic plant" over-promises:

* **Only the CEV side is stochastic.** `apply_and_simulate!` advances the MCS with
  `soe_mcs[m] = value(SOE_MCS[m, k0+1])` — the *planned* value. MCS energy, grid draw and travel
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
| `sigma_digging`, `sigma_loading_swinging`, `sigma_traveling` | opt | **per-activity** plant σ (kW) — the posterior SD written by step 0; the plant samples `N(μ, σ)` per activity |
| `prior_sigma_frac` | opt (0.2) | **fallback** σ as a fraction of each power mean (min 0.05), used only when the `sigma_*` rows are absent; idle stays 0 |
| `obs_noise_std` | opt (0.05) | telematics energy-measurement noise std (used only by the dormant online learner) |
| `co2_unit_scale` | opt (1.0) | multiplier for the CO₂ column |

> `p_*` and `sigma_*` are **overwritten by step 0** on every `:input` run (unless
> `run_regression=false`). Edit them by hand only when running with step 0 disabled.

> `day_end_hour`, `kappa_wt`, `n_days`, and `work_by_day.csv` are **not** used (single-day,
> full-24 h, lumpsum model; travel pacing uses the fixed `work_per_travel = 4`).

### `ev_data.csv` — one row per CEV
`id, SOE_min, SOE_max, SOE_ini, ch_rate, eta_ch_dch_cev`. `SOE_ini` is the **end-of-day floor
target** (the CEV must finish **at or above** it); `ch_rate` = CEV charge-acceptance (kW);
`eta_ch_dch_cev` = the CEV's own charge-acceptance efficiency.

### `mcs_data.csv` — one row per MCS
`id, SOE_min, SOE_max, SOE_ini, CH_MCS, DCH_MCS, C_MCS_plug, DCH_MCS_plug, eta_ch_dch_mcs`.
`CH_MCS`/`DCH_MCS` = grid-charge / total-discharge caps; `C_MCS_plug` = simultaneous plugs;
`DCH_MCS_plug` = per-plug cap; `eta_ch_dch_mcs` = the MCS's own round-trip efficiency.

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

> This section documents `build_window_model`, which Approach 0's `run_one_shot` still calls
> directly and unchanged. Approach 2's `run_mpc` instead calls the sibling
> `build_window_model_stochastic` (§13.3), which reuses every variable/constraint documented
> below **per scenario**, plus a small non-anticipativity block layered on top — so everything
> in this section is still exactly the physics each scenario copy obeys, just read that way.

### 8.1 Decision variables
**Continuous (≥ 0):** `P_ch_MCS`, `P_dch_MCS`, `P_MCS_CEV`, `P_work`, totals
`P_ch_tot`/`P_dch_tot`, state `SOE_MCS[m,t]`/`SOE_CEV[e,t]`,
peaks `P_peak_NC`/`P_peak_OP`, and the slack `s_miss_work[i,a]`.

> `s_miss_work` is declared over all four activities and the objective sums all four, but only
> `B[1]` (dig) and `B[2]` (load) appear in a balance (§8.10). `s_miss_work[i,trv]` and
> `s_miss_work[i,idle]` are free `≥0` variables carrying a positive cost, so the minimisation
> drives them to 0 and the result is unaffected — they are dead variables, not a missing
> travel/idle quota. Likewise `y_trv[m,i,i,k]` is never defined (the defining loop skips
> `i == j`) yet is summed into the labour term; it only ever costs, so presolve
> fixes it to 0.

**Binary:** `u[e,i,a,k]`, `mu[i,e,k]`, `rho[m,i,e,k]`, `z[m,i,k]`, `x[m,i,j,k]`,
`y_trv[m,i,j,k]`, `beta_arr`/`beta_dep`.

### 8.2 Objective (Eq. 1) — minimise total operating cost
`energy` (Σ price·P_ch_tot·Δt) `+ carbon` (Σ (carbon_price/1000)·co2·P_ch_tot·Δt)
`+ missed work` (SOFT, ρ_miss·Σ s_miss_work) `+ demand` (λ_NC·P_peak_NC + λ_OP·P_peak_OP)
`+ towing labour` (ρ_labor·Δt·Σ y_trv). Six terms.

### 8.3 Power flow & where power may go (HARD)
* `P_ch_tot = Σ_grid P_ch_MCS`; `P_dch_tot = Σ_site P_dch_MCS`; discharge forbidden at grid,
  charge forbidden at sites.
* `P_dch_MCS = Σ_e P_MCS_CEV` and `≤ DCH_MCS·z`.
* **Grid charging:** `P_ch_MCS ≤ CH_MCS·z` — capped by the MCS's charge rate, and flows only
  where the MCS is physically parked.
* **Plug limits:** `P_MCS_CEV ≤ DCH_MCS_plug·rho`; `Σ_m P_MCS_CEV ≤ CH_CEV[e]·mu`.
* **Charging-mode consistency:** `mu[i,e,k] == Σ_m rho[m,i,e,k]` — a CEV is in charging mode
  exactly when it is plugged into an MCS.

### 8.4 Peak-demand trackers (E1, HARD)
`P_peak_NC ≥` carried-in peak and `≥ Σ_m P_ch_tot[m,k]` (all k); `P_peak_OP` likewise on the
**on-peak** k only.

### 8.5 Transit tracking (HARD)
`y_trv` is 1 while a trip is in flight (its `tau_trv` intervals), or forced for a carried-in
in-transit trip. Transit is costed purely as time (the labour term in §8.2) and does not draw
from either battery.

### 8.6 Battery dynamics & bounds (HARD)
* Initial: `SOE_MCS[first] = soe_mcs0`, `SOE_CEV[first] = soe_cev0` (measured carry-in).
* **MCS:** `SOE_MCS[k+1] = SOE_MCS[k] + η_mcs·P_ch_tot·Δt − P_dch_tot·Δt/η_mcs` — transit does
  not draw from the battery (§8.5).
* **CEV:** `SOE_CEV[k+1] = SOE_CEV[k] + η_cev·Σ P_MCS_CEV·Δt − Σ P_work·Δt`, where `η_cev` is the
  CEV's own charge-acceptance efficiency.
* **Bounds:** every boundary clamped to `[SOE_min, SOE_max]` for MCS and CEV (a safety net; the
  realized CEV work duration is capped by available energy *before* this bound is ever reached — see
  §6.7).

### 8.7 Terminal energy targets (Eq. 8a / 8b, HARD)
Applied at the final window boundary **only when that boundary is the end of the day** — the
code gates both terminal rules (8a and 8b) on
`is_terminal = last(K) == d.n_day`. On the shrinking horizon every window reaches day-end, so
they always bind. Under the experimental fixed-`H` lookahead (`shrinking = false`) they are
absent from all but the last `H` windows and **nothing replaces them** — there is no terminal
cost and no value-function approximation standing in for the discarded tail, so that mode is a
myopic controller that will drain the MCS early with no obligation to restore it. Add a
terminal cost before using fixed-`H` for reported results.
* **MCS (8a, exact):** `SOE_MCS[end] == SOE_MCS_ini` — energy-neutral, ready for the next day;
  the overnight refill is scheduled inside this MILP. Since transit costs no energy, the MCS can
  reach this target while charging and then depart for a site afterward at no cost — so the
  target alone governs recovery, with no separate requirement to remain at the grid node.
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

### 8.9 Activity scheduling (Eq. 11, HARD)
* Exactly one activity per assigned CEV: `Σ_a u = A`; `u ≤ A`.
* Work capped & not while charging: `P_work ≤ R_work·A·(1−mu)`.
* Charging ⇒ idling: `mu ≤ u[idle]`.
* `P_work = Σ_a p_a·u`. Every `p_a` is a **constant**; idle's `p_idle = 0`, so an idling CEV
  draws no power. (The MILP uses the 4-activity encoding `B = [dig, load, trv, idle]`; Avik's
  reference uses the equivalent 3-activity form with `Σu ≤ 1`, `Σu + mu ≤ 1`.)

### 8.10 Work quota (Eq. 12c, SOFT shortfall + implicit HARD cap) — lumpsum
A single dig/load requirement per site (`hours_digging`/`hours_loading_swinging`). The code
writes this as a **balance equality** against the work still outstanding:

```julia
delta_T * sum(u[e, i, B[1], k] for e in E, k in K) + s_miss_work[i, B[1]] == max(rem_dig[i], 0.0)
```

The shortfall `s_miss_work` is soft (priced at `rho_miss` in the objective) — but since
`s_miss_work ≥ 0`, the equality **also implies the hard bound** `delta_T·Σu ≤ rem_dig[i]`. A CEV
cannot dig or load more than the requirement still outstanding, and because `rem_*` is recomputed
from realized work at every step, the cap is re-imposed against the *remaining* requirement on
every re-solve.

> **Correction.** Earlier revisions of this README, `math_model.tex` and
> `constraints_code_vs_model.txt` all described this as an inequality with "no hard
> no-working-ahead cap". That was wrong. The cap is real; it is simply reached implicitly through
> the equality rather than written as a separate rule the way the receding sibling writes it.
> With a single day and daily neutrality there is little incentive to overshoot anyway, so the
> cap is rarely the binding reason work stops — but it *is* in the model.

### 8.11 Precedence (Eq. 12d, HARD)
Cumulative loading `≤ scale ·` cumulative digging, in raw interval counts exactly as in Avik,
seeded by the realized work carried in from earlier windows.

### 8.12 Rest rule (Eq. 12e, HARD)
With `rest_cap = round(t_limit_rest/Δt)` (=4 at the defaults; the code **rounds**, it does not
take a ceiling, so the two differ for any `t_limit_rest` that is not a whole multiple of `Δt`)
and `rest_win = 5`: over any 5 consecutive
intervals a CEV does **at most 4** work intervals (≥ 1 idle break). Two parts: within-window
5-windows, **plus a seam** seeded with the applied Work/Break history so a work-run cannot leak
across the every-15-min re-solves.

### 8.13 Travel pacing (Eq. 13, HARD, no tolerance)
Exactly as in Avik with `work_per_travel = 4`: for each `(site, CEV)`, the two-sided band
`W(k) − 4 ≤ 4·V(k) ≤ W(k)` on cumulative travel `V` vs cumulative useful work `W`. `V` and `W` are
now raw **applied interval counts** off the `u` indicator (`cum_trv_cnt_e`/`cum_work_cnt_e`), not
hours — a battery-shortage-capped interval still counts as one full travel/work interval, so no
tolerance is needed. Precedence (§8.11) still uses realized hours; only pacing switched to
interval counts. Applies to both the deterministic and the scenario-linked stochastic builder.

> **No fallback.** Windows use the hard constraints only; an infeasible re-plan is reported
> **INFEASIBLE** and the plant **holds state**. Under the Fork B stochastic plant **both**
> `:synthetic` and `:input` previously solved with **zero** infeasible windows: the MCS returns
> to its exact initial SOE and every CEV finishes at or above its start-of-day level.
>
> Read that alongside the **capped count** (§6.7). Zero infeasible windows means the *optimiser*
> never failed; it does not mean the realized trajectory was physical, because the CEV SOE guard
> can silently absorb an over-draw without any window becoming infeasible. Both numbers appear
> in the run diagnostics table of `approach0_vs_approach1.html`.

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

**`approach0_vs_approach1.html`** — **Approach 0** (one-shot 08:00 plan, open-loop) vs
**Approach 2** (closed loop), both fully realized over 08:00 → next-day 08:00. Same metric rows
as the KPI report (grid energy, energy/CO₂/demand/missed-work/travel costs, total) plus the Δ.
The column header and the explanatory blurb **adapt to `res0.plant`**, so the report always
states whether the Δ is a like-for-like re-planning gap (`:sampled`) or the combined
drift-plus-re-planning figure (`:mean`), and points at the other mode (§6.6). Underneath, a
**run-diagnostics table**: plant mode, infeasible windows, CEV SOE capped count and solve time per
run. Written by `write_approach_comparison(res0, res1, out_dir)`; additive only. Companion CSVs:
`10_approach0_timeline.csv`, `10_approach1_timeline.csv`.

**`run_log.txt`** — everything printed to the console during the run (progress, KPI summary,
`@warn`s), mirrored to this file in addition to the terminal.

---

## 10. Adapting to real data

1. **Dataset** — use `mode = :input` and fill `data/input_data/` with the CSVs in §7.
2. **Power model** — the normal path is to let **step 0** fit it: point `regression_data_dir` at
   your task recordings and it overwrites `p_*` and the per-activity `sigma_*` in
   `parameters.csv` on every `:input` run. To supply the model by hand instead, run with
   `run_regression = false` and set `p_digging`/`p_loading_swinging`/`p_traveling`/`p_idling`
   **and the per-activity `sigma_digging`/`sigma_loading_swinging`/`sigma_traveling`** rows.
   `prior_sigma_frac` is only a **fallback** for datasets predating the `sigma_*` rows (§7) —
   prefer the explicit per-activity values.
3. **Telemetry** — in `4_MPCLoop.jl`'s `apply_and_simulate!`, replace the simulated
   `realized_activity_durations` call and the `next_power!` draw from the shared
   `ActivityPowerPool` (`1_Common.jl`) with the actual per-activity seconds + measured interval
   energy from your pipeline. The `ActivityPowerPool` / `true_powers` machinery only generates
   ground truth for the demo and disappears at go-live.

---

## 11. Relation to Avik's reference & Scenario 2

* **Avik's single-shot model** (`Avik/MCS_OPTIMAL_v4_real.jl`) solves the whole day in **one**
  MILP with deterministic powers and exact terminal equalities. This code is the **MPC** version
  of the same formulation: identical variable names, objective, routing, battery, precedence
  (raw-count) and travel-pacing (`work_per_travel = 4`); the differences are the shrinking-horizon
  re-solves, per-window history seeds, the carried-in MCS start position, and the CEV terminal
  **floor** (`≥`, overcharge allowed) instead of exact equality. **Approach 0** (`run_one_shot`,
  §6.6) *is* this codebase's reproduction of Avik's single-shot model — same MILP, solved once —
  with the stochastic plant executed on top of it, so it's the direct apples-to-apples baseline
  for the closed loop — and **Approach 0 under `plant = :mean`** (§6.6) is the sharper reproduction
  still, since nothing is stochastic and realized equals planned exactly.

  **Last measured (now stale, and from Approach 1 — not yet re-measured for Approach 2):** on
  **input**, the deterministic closed loop reproduced Approach 0 (sampled) to within **~0.02 %**
  ($193.36 vs $193.40); on **synthetic** the two diverged by **~28 %** ($114.82 vs $82.50, closed
  loop higher, mostly extra MCS transit — 2.75 h vs 1.75 h). That gap is **still unexplained**.
  These figures pre-date the `work_kW` fix, the capping fix, the pacing tolerance, the `:mean`
  mode, **and the entire Approach 2 stochastic extension** — they describe Approach 1's
  certainty-equivalent closed loop, not this codebase's actual `run_mpc`. **Regenerate before
  quoting them**, and work the diagnostic order in §6.6: compare `run_mpc(plant = :mean,
  n_scenarios = 1)` against `A0(:mean)` first — if that fully certainty-equivalent closed loop
  does not reproduce the one-shot optimum exactly, the cause is a seam or carry-in bug, not
  randomness or hedging, and the hedged/sampled numbers are not yet worth interpreting. Once that
  passes, a *separate* and expected question is how much `A2(:sampled, n_scenarios = 5)` costs
  relative to `A0` — that gap is the value of hedging, not a bug signature.
* **Scenario 1 vs 2, and how that relates to Approach 2.** In Avik's original framing, Scenario 1
  (this whole codebase) is certainty-equivalent (plans on the mean `μ`) and Scenario 2 would
  sample multiple power scenarios from `N(μ, σ)` instead — the fixed `σ` was always the hook for
  it. **This "Scenario 2" hook is exactly what Approach 2 implements** (`build_window_model_stochastic`,
  §13) — but along a *different* axis than this bullet's naming suggests: "Scenario 1/2" here
  names a *dataset/problem-formulation* choice from Avik's paper, while "Approach 0/2" names a
  *control-strategy* choice (one-shot vs. re-planning) within this codebase. Approach 2 adds
  scenario-based planning to the **existing Scenario 1 dataset formulation** — it does not (yet)
  correspond to a separate "Scenario 2" dataset variant. Don't read the two numbers as the same
  axis.

---

## 12. Corrections & open items

Documented so a reviewer is not surprised by them. Full detail, with the code lines, is in
`constraints_code_vs_model.txt` (section **KNOWN DISCREPANCIES AND MODELLING CAVEATS**, D1–D10).

**Corrected in the docs (the code was always right):**
* **D1** — Eq. 12c is an **equality**, so there *is* an implicit hard no-overshoot cap on work.
  All three documents previously said the opposite (§8.10).
* **D4** — the terminal rules 8a/8b/10e are gated on the window reaching day-end, so they vanish
  under `shrinking = false` (§8.7).
* **D6** — `rest_cap` uses `round`, not `ceil` (§8.12).

**Fixed in the code:**
* **D2** — the analyst log's `work_kW` was computed from `d.true_powers` (a Fork-A hidden-truth
  curve) while the batteries drained on the Fork-B pool draw `p_true`; in `:synthetic` those
  vectors differ, so the column contradicted the batteries it described. Now computed from
  `p_true`. Figures and KPI CSVs never read the column, so only `run_log.txt` was affected.
* **D3** — the CEV SOE clamp used to silently create or destroy energy when it bit, because
  `apply_and_simulate!` credited the full realized activity duration before clamping the resulting
  SOE. Fixed: realized dig/load/travel duration is now capped by available headroom *before* it is
  credited (§6.7), so `rem_dig`/`rem_load` and the SOE trajectory stay physically honest. Events are
  counted as `n_capped` (renamed from `n_clamped`). The residual `clamp()` call is now only a
  never-should-bind safety net.
* **D11** — the two-sided travel-pacing band (Eq. 13) could become spuriously infeasible: the
  sub-interval residue D3's fix leaves in `cum_dig_e`/`cum_load_e` could land the band's floor and
  ceiling on opposite sides of an integer boundary with no whole-interval solution in between,
  permanently blocking further travel/work for a CEV even though the shortfall was physically
  meaningless (traced in detail via IIS on `run10_SOE_14.80`, §12). Fixed by adding
  `pacing_tol = 0.05` to the floor side only (§8.13) — large enough to absorb realistic capping
  residue, small enough (5% of one interval) that it cannot be mistaken for a free extra work unit.

**Open / instrumented, not resolved:**
* **D5** — only the CEV side of the plant is stochastic (§6.7).
* **D7/D8** — two never-constrained slacks and the undefined `y_trv[m,i,i,k]`; both are driven
  to zero by the objective, so harmless, but they are dead variables (§8.1).
* **D9** — precedence seeds per **site**, travel pacing seeds per **CEV**. Equivalent while
  `A[i,e]` is a one-to-one assignment, which it is in both datasets; latent if a site ever gets
  two CEVs.
* **D10** — the synthetic A0-vs-A1 gap (above).

---

## 13. Approach 2 — the stochastic extension

Everything above (§1–12) documents the window MILP exactly as Approach 1 built it — sets,
variables, constraints, data, outputs. Approach 2 does not change any of that; it changes
**what the controller plans against**. This section is the delta.

### 13.1 The one-line idea

Approach 1 solves the window MILP **once per re-solve**, on the posterior mean `μ`
(certainty-equivalent). Approach 2 solves it as **`S` linked copies**, one per sampled
scenario, coupled by **non-anticipativity**: the action chosen for *right now* (the window's
first interval) must be identical across every copy, because the controller does not yet know
which scenario is real. Later intervals may differ per scenario (recourse). See
`Understanding_Stochastic_MPC.md` for the five-level explanation and a fully worked example;
this section only covers the implementation.

### 13.2 New files, unchanged files

| File | Status |
|---|---|
| `2b_ScenarioSampler.jl` | **New.** Standalone module. `sample_scenarios(mu, sd, n_scenarios; rng)` draws `n_scenarios` independent power vectors from `Normal(mu[a], sd[a])` clipped at 0 — the same generative model as the plant pool (`draw_activity_power_pool` in `1_Common.jl`), but resampled fresh every re-solve and never touching the plant's own RNG stream. `equal_weights(n_scenarios)` returns `1/S` weights. `DEFAULT_N_SCENARIOS = 5`. |
| `3_MCSModel.jl` | `build_window_model` (§8, unchanged) is kept for Approach 0's baseline. **New:** `build_window_model_stochastic`, described below. |
| `4_MPCLoop.jl` | `run_one_shot` (Approach 0) is **unchanged** — still certainty-equivalent, single solve at 08:00. `run_mpc` (Approach 2's controller) now samples scenarios and calls `build_window_model_stochastic` every step. **New:** a `_ScenarioOneView` wrapper and `apply_and_simulate_stochastic!`, described below. |
| `1_Common.jl`, `2_DataLoader.jl`, `5_Output.jl`, `0_Regression.jl` | **Unchanged.** The estimator, the data schema, and every output artefact are identical to Approach 1. |

### 13.3 `build_window_model_stochastic` — what actually changes in the MILP

Every variable family gains a trailing scenario index `s in 1:S`. The physics, objective
structure, and every constraint from §8 apply **per scenario, unchanged** — same equations,
just with an `s` on each variable and, where the equation references the planned power, that
power is now `scenarios[s][a]` instead of the single `pvec[a]`. The objective becomes a
probability-weighted average across scenarios (`sum(w[s] * cost_s for s in 1:S)`, equal
weights `1/S` by default); every hard constraint (battery floors, plug counts, precedence,
routing, …) still applies **individually in every scenario** — nothing is relaxed on average.

**Non-anticipativity** (the one genuinely new mechanism) is a small block of equality
constraints added after the model is otherwise built, tying every *decision* variable at the
window's first interval `k0` across scenarios (`X[..., k0, s] == X[..., k0, 1]` for `s = 2..S`):

* **Tied at `k0`** — everything the controller *chooses*: `u`, `mu`, `rho`, `z`, `x`,
  `y_trv`, `beta_arr`, `beta_dep` (every binary), and `P_ch_MCS`, `P_MCS_CEV`, `P_dch_MCS`,
  `P_ch_tot`, `P_dch_tot` (the controlled power flows).
* **NOT tied at `k0`** — `P_work` and `SOE_CEV`. This is the whole mechanism: `P_work[i,e,k0,s]
  = sum_a scenarios[s][a] * u[e,i,a,k0,s]`, and `u` is tied but `scenarios[s][a]` is not, so the
  *action* ("Dig") is shared while the *energy it costs* is scenario-dependent — exactly the
  worked example in `Understanding_Stochastic_MPC.md` (hard-soil Dig costs more than
  average-soil Dig even though both are "the same decision"). `SOE_CEV` inherits this at
  `k0+1` through the recursion. `SOE_MCS` is **not** explicitly tied either, but since every
  quantity feeding its recursion at `k0` (`P_ch_tot`, `P_dch_tot`) *is* tied, it comes out equal
  across scenarios anyway without needing a constraint for it — consistent with "only the CEV
  side of the plant is stochastic" (§6.7, §12 D5). `s_miss_work`, `P_peak_NC`, `P_peak_OP` are
  window-wide aggregates over the full (scenario-dependent) trajectory and are deliberately left
  free, entering the objective as a probability-weighted average.

### 13.4 How the closed loop reads a solved stochastic model

Non-anticipativity guarantees scenario 1's slice at `k0` equals every other scenario's, so
"the shared decision" is just "scenario 1's `k0` slice". `4_MPCLoop.jl` implements this with a
tiny read-only wrapper, `_ScenarioOneView`/`_ScenarioOneVar`: indexing it `[:varname][idx...]`
forwards to the underlying variable at `[idx..., s_ref]`. Every existing single-scenario helper
(`realized_activity_durations`, `advance_mcs_state`, `applied_act_index`, `activity_label`,
`mcs_status_label`) works **unchanged** against a stochastic model through this wrapper — no
helper had to be duplicated for the scenario-indexed case. `apply_and_simulate_stochastic!` is
a thin wrapper around the existing `apply_and_simulate!` that builds this view (`s_ref = 1`)
and hands off to it, so the plant physics (activity split, pool draw, energy capping, history
bookkeeping) are byte-identical to Approach 0/1 — only how the plan for `k0` was *arrived at*
differs. The replanning grids (the forward-look plots/CSVs for `k > k0`) are also read through
the scenario-1 view — a representative single scenario's forward plan, a diagnostic display
only; it plays no role in what gets applied or in any KPI.

### 13.5 The two RNG streams

`run_mpc` seeds a **dedicated** `MersenneTwister(seed + 1_000_000)` for scenario sampling,
kept strictly separate from the plant's own `rng`/`cursor` (which walk the shared
`ActivityPowerPool` — the same pool Approach 0 draws from). Scenario draws are the planner's
internal "what ifs" for one optimisation and are thrown away the moment the window is solved;
they must never leak into, or correlate with, what the plant actually realizes. This mirrors
the existing pool-vs-estimator separation (§6.7) one level further.

### 13.6 Options added to `run_scenario_1` / `run_mpc`

| Option | Default | Meaning |
|---|---|---|
| `n_scenarios` | `5` (`ScenarioSampler.DEFAULT_N_SCENARIOS`) | Scenarios sampled fresh at every re-solve. Larger `S` hedges against a wider range of outcomes at the cost of a roughly linearly larger MILP (`S` linked copies) — see `run_soe_sweep.jl` / `RUN_ALL.jl`'s `N_SCENARIOS` to sweep or batch-set it. |

Everything else (`shrinking`, `H`, `time_limit_sec`, `approach0_plant`, `mcmc_samples`, `seed`,
…) is unchanged from §5.

### 13.7 What did *not* change

The output artefacts (§9), the input schema (§7), the estimator (§6.3–6.4, `0_Regression.jl`),
Approach 0's baseline, the shared plant pool, and every constraint's physics are all identical
to Approach 1. The only thing Approach 2 changes is *what the MILP plans against* at each
re-solve — one mean vector, or a hedged set of samples — exactly as
`Understanding_Stochastic_MPC.md`'s master table lays out.

## Changes 2, 3, 4, 5 (this session)

* **Change 2 — earlier-charging tie-break (Issue 2).** Same `1e-6`-weighted `idx * mu` term as
  Approach 1 (see that sibling doc for the `g_ch` → `mu` correction — the term originally targeted
  the wrong variable), added ONLY to `build_window_model`'s (deterministic) objective in
  `3_MCSModel.jl` — NOT to `build_window_model_stochastic`'s. Approach 2's scenario-based hedge
  already addresses margin-for-error directly (Change 4), so the tie-break stays scoped to the
  deterministic no-margin failure mode this issue actually describes.
* **Change 3 — terminal SOE_CEV shortfall penalty (Issue 1).** Same design as Approach 1 — see that
  sibling doc. Applies uniformly to all five approaches.
* **Change 4 — stratified scenario sampling (Issue 3).** `2b_ScenarioSampler.jl`'s
  `sample_scenarios` previously drew all 5 scenarios i.i.d. from `N(mu, sd)`, with no guarantee any
  landed in the risky tail. Now draws 5 fixed, evenly spread bins (extreme-low / slightly-low /
  near-mean / extreme-high / mild-high), each still genuinely random *within* its bin every
  re-solve, so a tail scenario is structurally guaranteed instead of left to chance. Applies when
  `n_scenarios == 5` (the default); other counts fall back to the original i.i.d. draws.
* **Change 5 — `n_day_run` (currently 1), internal day loop.** `run_mpc`/`run_one_shot` in
  `4_MPCLoop.jl` now contain their own internal day loop (identical design to the Approach 1
  sibling — see that doc for the full rationale). The scenario-sampling machinery (`est`,
  `rng_scenarios`, `_ScenarioOneView`) is unaffected by day boundaries: `est` and `hist` persist for
  the whole run same as Approach 1, and `apply_and_simulate_stochastic!` was updated to forward the
  new `gidx` (global write-index) through to the shared `apply_and_simulate!` it wraps.
  `n_day_run = 1` is a pure passthrough. Same follow-up fix as the Approach 1 sibling: the
  replanning-grid diagnostics (`plan_grid_kW` etc.) are now saved per day into `replan_by_day`
  (mirroring `Receding_Horizon`'s pattern) instead of being silently overwritten, and every
  `5_Output.jl` function that reads them was updated to use `res.replan_by_day[1]` with `res.nKd`
  bounds — fixed, not just flagged.
