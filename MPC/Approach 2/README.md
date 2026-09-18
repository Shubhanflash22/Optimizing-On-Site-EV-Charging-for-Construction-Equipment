# Approach 2 — Stochastic (Scenario-Based) Shrinking-Horizon MPC for Mobile-Charging-Station Dispatch

This folder is Approach 2 **on its own** — a standalone, closed-loop,
scenario-based stochastic MPC controller. It no longer contains Approach 0
(see `../Approach 0/`) or a Receding-Horizon sibling; both used to live
alongside this controller and have been removed. For a side-by-side
comparison across all three approaches, use `../Comparison_A0_A1_A2/Code/` —
that is also where every cross-approach figure and comparison CSV now lives.

```
Approach 2/
├── README.md              ← this file
├── code/
│   ├── 0_Regression.jl     STEP 0: fits the Bayesian power model from soil .xlsx files
│   ├── 1_Common.jl         shared helpers + detailed-output log structs (CEV + MCS)
│   ├── 2_DataLoader.jl     loads :synthetic / :input data
│   ├── 2b_ScenarioSampler.jl  draws the S sampled power vectors each re-solve consumes
│   ├── 3_MCSModel.jl       the window MILP (`build_window_model_stochastic`, plus the
│   │                       deterministic `build_window_model` the physics share with A0/A1)
│   ├── 4_MPCLoop.jl        the closed loop (module MPCLoop, `run_mpc`)
│   ├── 5_Output.jl         figures, KPI/cost CSVs, worker schedule, replan grids
│   └── 6_Shrinking_Horizon_main.jl   entry point (`run_scenario_1`)
├── data/
│   ├── input_data/         the 8 real CSVs, read by dataset = :input
│   └── synthetic_data/     human-readable mirror of the hardcoded :synthetic scenario
├── docs/
│   ├── README.md           full write-up: the stochastic model, the scenario MPC loop
│   ├── math_model.tex      formal equations
│   └── constraints_code_vs_model.txt   line-by-line audit of code vs equations
└── output/                 created on first run
```

## 1. What this is, in plain words

Same problem as Approach 1 (one MCS, a small fleet of excavators, minimize
cost while getting the work done), but instead of planning against a single
best-guess power forecast, every 15 minutes this controller **samples a
handful of scenarios** from the Bayesian posterior and solves **one MILP that
hedges across all of them at once** — a shared, non-anticipative decision for
right now that stays feasible no matter which scenario turns out to be true,
plus scenario-specific recourse for later intervals. Only that shared first
interval is applied to the real (stochastic) plant, then it re-samples and
re-solves. Approach 1 is the deterministic (certainty-equivalent) version of
the same idea.

## 2. Running it

```julia
julia --project=.
include("code/6_Shrinking_Horizon_main.jl")
```

or, without auto-run:

```julia
SCENARIO1_NO_AUTORUN = true
include("code/6_Shrinking_Horizon_main.jl")
res = run_scenario_1(; dataset = :input, mode = :normal, n_scenarios = 5)
```

Key arguments to `run_scenario_1` (same conventions as Approach 1 — see that
project's `README.md` §2 for the full table):

| Argument | Meaning |
|---|---|
| `dataset` | `:input` or `:synthetic` — **which data to load** |
| `mode` | `:normal`/`:high`/`:low`/`:near_mean`/`:live_data` — **how realized power is drawn**, independent of `dataset` |
| `n_scenarios` | how many scenarios are sampled and hedged against at every re-solve (default 5) |
| `detailed_output` | `true` writes the full per-15-min plan/realized CSVs (CEV + MCS) |

As with Approach 1, `dataset` and `mode` are deliberately two separate
arguments — a prior version of this file gave both the name `mode`, which
Julia does not allow; this has been fixed.

## 3. Detailed output

Pass `detailed_output = true` to get, per 15-minute interval and per
scenario, the full remaining-horizon **plan** made at every re-solve
(`DetailedPlanLog`, with a `scenario_id` column) and what actually got
**realized** (`RealizedTupleLog`, the shared, non-anticipative outcome) — for
the CEV(s) and, separately, the MCS itself (`MCSPlanLog`/`MCSRealizedLog`).
See `1_Common.jl` for the exact struct/column definitions.

## 4. Further reading

- **`docs/README.md`** — the full write-up: the stochastic model, the
  scenario-based MPC loop mechanics, and every material change made to this
  controller over its development.
- **`docs/math_model.tex`** — the formal equations.
- **`docs/constraints_code_vs_model.txt`** — a line-by-line audit confirming
  every constraint in `3_MCSModel.jl` matches its equation.

Both deep docs still occasionally compare against "Approach 0" or "Approach 1"
by name where that comparison is genuinely informative — that's a substantive
comparison, not a leftover reference to either living in this folder, which
neither does.
