# Approach 1 — Certainty-Equivalent Shrinking-Horizon MPC for Mobile-Charging-Station Dispatch

This folder is Approach 1 **on its own** — a standalone, closed-loop shrinking-horizon
MPC controller. It no longer contains Approach 0 (see `../Approach 0/`) or a
Receding-Horizon sibling; both used to live alongside this controller and have
been removed. For a side-by-side comparison across all three approaches, use
`../Comparison_A0_A1_A2/Code/` — that is also where every cross-approach figure
and comparison CSV now lives.

```
Approach 1/
├── README.md              ← this file
├── code/
│   ├── 0_Regression.jl     STEP 0: fits the Bayesian power model from soil .xlsx files
│   ├── 1_Common.jl         shared helpers + detailed-output log structs (CEV + MCS)
│   ├── 2_DataLoader.jl     loads :synthetic / :input data
│   ├── 3_MCSModel.jl       the single 24h window MILP (Eq. 1-13)
│   ├── 4_MPCLoop.jl        the closed loop (module MPCLoop, `run_mpc`)
│   ├── 5_Output.jl         figures, KPI/cost CSVs, worker schedule, replan grids
│   └── 6_Shrinking_Horizon_main.jl   entry point (`run_scenario_1`)
├── data/
│   ├── input_data/         the 8 real CSVs, read by dataset = :input
│   └── synthetic_data/     human-readable mirror of the hardcoded :synthetic scenario
├── docs/
│   ├── README.md           full write-up: the model, the MPC loop, all the "CHANGE N" notes
│   ├── math_model.tex      formal equations
│   └── constraints_code_vs_model.txt   line-by-line audit of code vs equations
└── output/                 created on first run
```

## 1. What this is, in plain words

We own ONE Mobile Charging Station (MCS) — a battery on wheels — and a small
fleet of electric excavators (Construction EVs, "CEVs"). Over a work day the
MCS drives around and tops the excavators up so none runs flat, while paying
the least for electricity (time-of-use price + demand charges + carbon) and
getting all the digging/loading work done. We don't know each activity's exact
power draw, so we fit a Bayesian power model **once**, and then every 15
minutes we (1) **optimise** a MILP over the remaining day using that fixed
model and (2) **apply** only the first interval's decisions to the plant,
observe what actually happened, and re-solve — classic MPC, one interval at a
time, with a **shrinking** horizon (the lookahead window shrinks toward the
day boundary rather than staying a fixed width).

## 2. Running it

```julia
julia --project=.
include("code/6_Shrinking_Horizon_main.jl")
```

or, without auto-run:

```julia
SCENARIO1_NO_AUTORUN = true
include("code/6_Shrinking_Horizon_main.jl")
res = run_scenario_1(; dataset = :input, mode = :normal)
```

Key arguments to `run_scenario_1`:

| Argument | Meaning |
|---|---|
| `dataset` | `:input` (real CSVs) or `:synthetic` (built-in hardcoded scenario) — **which data to load** |
| `mode` | `:normal` (default, unbiased draws), `:high`/`:low`/`:near_mean` (biased sensitivity sweeps), or `:live_data` (draw from recorded `live_powers.csv`) — **how the simulated plant's realized power is drawn**. Independent of `dataset`. |
| `shrinking` | `true` (default) for the shrinking window; `false` for a fixed-width `H`-length window (experimental — terminal rules drop out, see `docs/README.md` §8.7) |
| `detailed_output` | `true` writes the full per-15-min plan/realized CSVs (CEV + MCS); `false` (default) skips them |
| `n_day_run`, `seed`, `time_limit_sec` | as elsewhere in this project |

`dataset` and `mode` are deliberately two separate arguments now — a prior
version of this file gave both the name `mode`, which Julia does not allow
(a function cannot declare the same keyword argument twice); this has been
fixed.

## 3. Sampled vs mean plant

- **`:sampled`** (used throughout closed-loop runs) — realized power is drawn
  from the shared random pool at each interval; the MPC loop observes the
  actual outcome and re-solves next interval, so drift gets corrected.
- **`:mean`** — realized power pinned to the same mean the MILP planned on
  (only meaningful for Approach 0's one-shot replay, in `../Approach 0/`,
  where there's no re-solve to correct drift; Approach 1 always closes the
  loop, so this distinction matters far less here).

## 4. Detailed output

Pass `detailed_output = true` to get, per 15-minute interval, both the full
remaining-horizon **plan** made at every re-solve (`DetailedPlanLog`) and what
actually got **realized** (`RealizedTupleLog`) — for the CEV(s) and,
separately, the MCS itself (`MCSPlanLog`/`MCSRealizedLog`). See
`1_Common.jl` for the exact struct/column definitions. This is off by default
since it costs extra memory/time roughly proportional to (window length)²
per re-solve.

## 5. Further reading

- **`docs/README.md`** — the full write-up: the mathematical model in plain
  language, the MPC loop mechanics, and a running "CHANGE N" log of every
  material change made to this controller over its development.
- **`docs/math_model.tex`** — the formal equations.
- **`docs/constraints_code_vs_model.txt`** — a line-by-line audit confirming
  every constraint in `3_MCSModel.jl` matches its equation in `math_model.tex`.

Both deep docs still occasionally compare against "Approach 0" by name where
that comparison is genuinely informative (e.g., explaining why Approach 1
needs a rule Approach 0 doesn't) — that's a substantive comparison, not a
leftover reference to Approach 0 living in this folder, which it no longer
does.
