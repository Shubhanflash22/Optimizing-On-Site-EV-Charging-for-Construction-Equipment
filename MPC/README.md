# MPC for Mobile Charging Station Dispatch (Construction EVs)

One mobile charging station (MCS) has to keep construction electric vehicles (CEVs) charged and working through the day, without knowing exactly how much power each activity will draw. Every 15 minutes the controller decides where the MCS charges, how much it delivers to which CEV, when it drives between sites, and what each CEV is doing. The goal is to minimize electricity cost, carbon, demand charges, missed work, and towing labour.

This repo holds three controllers for that same problem, written in Julia (JuMP + HiGHS). They share the same MILP (Ghosh et al., arXiv:2608.18494), the same input data, and the same stochastic simulated plant. Only the way the plan is built and used changes.

## The three approaches

| | Approach 0 | Approach 1 | Approach 2 |
|---|---|---|---|
| Name | One-shot baseline | Closed-loop MPC | Scenario-based stochastic MPC |
| Planning | Solves the whole day once | Re-solves every 15 min from the real state | Re-solves every 15 min from the real state |
| Power assumption | Mean power | Mean power (certainty-equivalent) | Several sampled power scenarios at once |
| Reacts to surprises | No, runs open-loop | Yes, after the fact | Yes, and hedges before they happen |
| Solves per day | 1 | ~96 | ~96 (each one larger) |
| Entry function | `run_scenario_0` | `run_scenario_1` | `run_scenario_1` |
| Driver | `6_OneShot_main.jl` | `6_Shrinking_Horizon_main.jl` | `6_Shrinking_Horizon_main.jl` |
| Output prefix | `A0_` | `A1_` | `A2_` |

- **A0** commits to a full-day plan and hopes reality matches it. It is the floor the other two are measured against.
- **A1** plans against the average power and fixes mistakes once it re-measures.
- **A2** plans against a handful of possible power draws and only commits to a next move that is feasible in all of them.

## Layout

```
Approach 0/          code/  data/input_data/  README.md
Approach 1/          code/  data/input_data/  README.md
Approach 2/          code/  data/input_data/  README.md
Bayesian Regression/ field task files and scripts for the optional parameter fit (step 0)
Project.toml, Manifest.toml
```

Each approach is self-contained: `code/`, `data/input_data/` (the same 8 CSVs in all three) and `output/` sit side by side.

## How to run

1. Install Julia from julialang.org/downloads.
2. From the repo root, install the pinned packages once:
   ```julia
   using Pkg; Pkg.activate("."); Pkg.instantiate()
   ```
3. Run an approach with its defaults (about the same steps for all three):
   ```julia
   include("Approach 1/code/6_Shrinking_Horizon_main.jl")   # or Approach 0 / 2 and their driver
   ```
   Results are written to that approach's `output/normal/`.
4. To run one approach on all six sampling modes, include its `code/run_all_modes.jl`.

By default every run reads the committed `parameters.csv`. Approaches 1 and 2 can refit it first from `Bayesian Regression/` with `run_regression = true` (needs XLSX.jl). That fit currently overwrites `parameters.csv`, so use it deliberately.

Each approach's README has the input and output file formats, every run option, and the list of departures from the paper.

## Comparing the approaches

- Use the same `seed` and `plant` setting across approaches, or the realized power draws will differ.
- `plant = :mean` is a sanity check, not a result. A0 and A1 should give a terminal shortfall of exactly 0. A2 should give 0 or very close to it, with no infeasible windows.
- A2 runs the solver on 8 threads, while A0 and A1 run single-threaded, so solve times are not directly comparable.
- A2 uses `n_scenarios = 5` by default. Results for other scenario counts are not directly comparable.
- All three add a terminal shortfall penalty to total cost (not part of the paper's objective), so compare approaches on the reported total cost.

## Citation

The optimization model and power estimation follow:

> A. Ghosh, A. Taşcıkaraoğlu, et al., "Power Estimation and Optimal Work-Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations," arXiv:2608.18494.

Code authors: <ADD NAMES>
