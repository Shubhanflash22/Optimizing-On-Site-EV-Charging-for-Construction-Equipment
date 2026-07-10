# Approach 1 — Deterministic Certainty-Equivalent MPC for Mobile-Charging-Station Dispatch

This folder holds **Approach 1** to the *Mobile Charging Station (MCS) dispatch* problem:
a **self-improving, certainty-equivalent Model Predictive Controller (MPC)**. It comes in
two flavours that solve the **same problem** with the **same model and estimator** but a
different planning horizon:

```
Approach 1/
├── Shrinking_Horizon/     ← single-day controller (window shrinks toward 18:00)
├── Receding_Horizon/      ← multi-day controller (fixed-width window slides forward)
└── README.md              ← this overview
```

Each subfolder is self-contained (`code/`, `data/`, `output/`, `docs/`) and has its own
comprehensive README plus a formal `math_model.tex`. **Start with a subfolder's
`docs/README.md`** for the full detail; this file explains the shared idea and how the two
horizons differ.

---

## 1. The problem

We own **one Mobile Charging Station (MCS)** — a large battery on a truck — and a small
fleet of **Construction EVs (CEVs)**, i.e. electric excavators, working at fixed sites. Over
a work day the MCS must drive around and top the excavators up so none runs flat, while:

* getting the required **digging / loading** work done,
* keeping every battery inside its safe limits,
* returning each excavator to its **start-of-day charge** by the end of the day, and
* paying the **least** for electricity — time-of-use energy price **+** carbon **+**
  peak-demand charges **+** the labour of towing the MCS around.

The catch: the **power each activity draws is uncertain**. Digging, loading/swinging,
traveling and idling each pull a different (a-priori only approximately known) number of kW.

---

## 2. The shared idea: MPC + online Bayesian learning

Both controllers close **two feedback loops at once**, every 15 minutes:

1. **State feedback (the MPC loop).** Solve a Mixed-Integer Linear Program (MILP) that
   plans the charging, routing and work over a horizon; **apply only the first 15-min
   interval**; then re-measure the real state (battery levels, MCS position, work done) and
   **re-solve**. This is standard receding/shrinking-horizon control and it is what makes
   the controller robust to disturbances.

2. **Parameter feedback (the learning loop).** After each interval, the *energy actually
   consumed* is a new measurement of the activity powers. It is fed into a **Bayesian
   regression** (a TruncatedNormal prior over the four powers, fit with NUTS/MCMC via
   Turing.jl). The MILP then plans on the **posterior mean** — hence *certainty-equivalent*:
   the uncertainty is collapsed to a best point estimate, acted on, and continuously
   refined.

So the controller literally **gets better at predicting its own fleet as the day goes on.**
The posterior *standard deviation* is also tracked (and plotted) — it is the ready hook for
a future **Scenario 2** (scenario-based stochastic MPC) that would optimise over sampled
powers instead of just the mean.

### Common building blocks (identical in both folders)

* **15-min discretisation**, daytime horizon **08:00–18:00** (inferred from the data, not
  hard-coded), plus a deterministic **overnight smart-charge** that refills the MCS in the
  cheapest hours (Phase 2).
* **Same MILP constraints**: power-flow & battery physics, MCS routing (drive time + travel
  energy), *charge-only-while-idling*, **digging-before-loading precedence**, an **operator
  rest rule** (≥ 1 break per rolling hour), **travel pacing**, peak-demand tracking, and a
  **CEV end-of-day energy-neutral** target.
* **Two correctness features** shared by both: (a) the rolling rules (rest / precedence /
  pacing) are **seeded from the applied history**, so a work-run can't "leak" across the
  every-15-min re-solves; (b) a **keep-up reserve** keeps the hard end-of-day battery
  target *recursively feasible* (no knife-edge infeasibility).
* **Same modular code layout** (files numbered in dependency order):
  `1_Common` → `2_DataLoader` → `3_BayesianEstimator` → `4_MCSModel` → `5_MPCLoop` →
  `6_Output` → `7_*_main`. A legacy all-in-one `Scenario_1.jl` is kept in each `code/` for
  reference but is **not** used by the main.

---

## 3. Shrinking vs Receding — what actually differs

Both are receding-horizon MPCs in the general sense (re-solve, apply one step, repeat). The
difference is **how long the planning window is and whether it spans more than one day.**

### Shrinking Horizon (single day)
* Plans the **entire remaining day** each step: the window is `[now … 18:00]`, so it
  **shrinks** as the day advances (40 intervals at 08:00, 1 interval at 17:45).
* **One day only.** Work is a single **lumpsum** requirement per site; the excavators must
  end **that day** at their start charge.
* No look-ahead beyond today, no buffer day.
* Simplest and fastest; the natural baseline.

### Receding Horizon (multi-day, cross-day)
* Plans a **fixed-width, sliding** window: the rest of today **plus `lookahead_days` future
  daytime blocks**, so the plan always "sees tomorrow".
* **Many days.** Work is a genuine **per-day schedule** (each day its own quota, optionally
  from a `work_by_day.csv`); unfinished work **rolls over** to the next day (penalised).
* Simulates the reported days **plus one dropped "buffer" day** purely to give the last
  reported day a full day of look-ahead.
* Adds two behaviours the single-day model doesn't need:
  * **Daily battery realignment** — the CEV energy-neutral target fires at **every** 18:00,
    so each reported day is energy-neutral (not just the last).
  * **No working ahead** — a hard cap keeps cumulative work at each day's quota, so the
    controller can't front-load a future day's digging just because power is cheap now
    (leftover can still be *caught up*, never *borrowed forward*).

### At a glance

| Aspect | Shrinking Horizon | Receding Horizon |
|--------|-------------------|------------------|
| Horizon | single day, window shrinks to 18:00 | multi-day, fixed-width window slides forward |
| Look-ahead | none beyond today | `lookahead_days` future daytime blocks |
| Work demand | one lumpsum per site | per-day schedule (`work_by_day.csv`), rolls over |
| Buffer day | none | one extra day simulated & dropped |
| CEV energy-neutral | end of the day | **end of every day** (daily realignment) |
| Working ahead | naturally none (one day) | **hard cap forbids it** |
| Entry point | `7_Shrinking_Horizon_main.jl` | `7_Receding_Horizon_main.jl` |

**Which to use?** Use **Shrinking** for a clean single-day study or a fast baseline. Use
**Receding** when work spans multiple days, when you want the schedule to anticipate
tomorrow, or when day-to-day carry-over of work/energy matters.

---

## 4. Running either one

Install once: `Pkg.add(["JuMP","HiGHS","Plots","DataFrames","CSV","Turing"])`.
Then from the chosen `code/` folder (synthetic mode auto-runs):

```bash
# Shrinking
julia Shrinking_Horizon/code/7_Shrinking_Horizon_main.jl
# Receding
julia Receding_Horizon/code/7_Receding_Horizon_main.jl
```

Input (real-CSV) mode and all options are documented in each subfolder's
`docs/README.md`. Results (figures, KPI/cost CSVs, worker schedule, replanning grids) land
in `<subfolder>/output/<synthetic|input>/`.

---

## 5. Where to read next

* **`Shrinking_Horizon/docs/README.md`** — full single-day controller doc.
* **`Receding_Horizon/docs/README.md`** — full multi-day controller doc.
* **`<subfolder>/docs/math_model.tex`** — the formal MILP (sets, variables, every equation).
* **`<subfolder>/docs/constraints_code_vs_model.txt`** — line-by-line code-vs-model audit.
