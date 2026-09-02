# MPC — Scenario 1 (Original Baseline)

Avik's original single-scenario MILP for jointly scheduling Mobile Charging Station
(MCS) routing/charging and Construction Equipment Vehicle (CEV) work — the baseline
that `MPC_Shubhan/Approach 1` and `Approach 2` extend into shrinking/receding-horizon
MPC and stochastic variants. This version solves **one full-day MILP, once**, with no
re-planning loop.

## Files

| File | Role |
|---|---|
| `mcs_optimization_main_v4_real.jl` | Entry point — loads data, builds the time grid, solves, exports plots/CSVs |
| `helper functions/MCS_OPTIMAL_v4_real.jl` | The `MCSOptimizer` module — full JuMP model (variables, constraints, objective) and all plotting/export helpers |
| `helper functions/DataLoader_v4_real.jl` | The `DataLoader` module — reads the 7 input CSVs into the vectors/matrices the model needs |
| `simple_dataset/csv_files/*.csv` | Example single-CEV, single-MCS, single-site input dataset |

## Running it

```julia
julia mcs_optimization_main_v4_real.jl                    # defaults to simple_dataset/
julia mcs_optimization_main_v4_real.jl <dataset_folder>    # any folder with a csv_files/ subdir
julia mcs_optimization_main_v4_real.jl --all               # run every dataset folder in the cwd
```

`optimizer_choice` at the top of the main script selects which model variant to
include (`"OPTIMAL"` = the full model in this folder; `"B1"`–`"B4"` reference other
`MCS_B*_v4_real.jl` variants not included in this upload). Solver time limit defaults
to 1 hour (`time_limit_sec`); `Inf` for unlimited.

## Model summary

Single JuMP/HiGHS MILP over the whole scheduling horizon (`T`, boundary indices;
`K`, interval indices; `delta_T`-hour intervals, default 15 min). Decision variables
cover: MCS charge/discharge power and location/routing (`P_ch_MCS`, `P_dch_MCS`,
`z`, `x`, `y_trv`, `beta_arr`/`beta_dep`), MCS↔CEV power transfer and plug assignment
(`P_MCS_CEV`, `rho`), CEV activity selection (`u[E,N,B,K]`, binary — digging/
loading+swinging/travelling), CEV work power (`P_work`), and SOE trajectories for
both MCS and CEV. Objective minimizes grid energy cost + carbon cost + non-coincident
and on-peak demand charges + missed-work penalty (`rho_miss`) + MCS travel labor cost.

**Constraints present:** charging/discharging capacity and plug limits, CEV work-power
capacity, SOE dynamics and bounds, **exact** terminal SOE equality (`SOE[·,last] ==
SOE_ini`, not a floor — later extended to a floor in `MPC_Shubhan`), MCS routing/
presence/arrival-departure bookkeeping, cumulative digging→loading **precedence**
(`scale=2`), and the two-sided **travel-pacing** band (`work_per_travel=4`: `4V ≤ W ≤
4V+4` on cumulative travel `V` vs. cumulative work `W`).

**Not present in this baseline** (added later in `MPC_Shubhan`): the operator
rest-rule constraint, the `pacing_tol` numerical tolerance on the travel-pacing floor,
and the closed-loop MPC re-solve loop itself — this version plans the entire day in
one shot and does not react to realized deviations mid-day.

## Inputs (`simple_dataset/csv_files/`)

| File | Contents |
|---|---|
| `ev_data.csv` | Per-CEV `SOE_min/max/ini`, charge rate, work capacity |
| `mcs_data.csv` | Per-MCS `SOE_min/max/ini`, charge/discharge rates, plug count, efficiency |
| `place.csv` | Per-site CEV assignment and required digging/loading hours |
| `parameters.csv` | Scalar model parameters (`k_trv`, `rho_miss`, `rho_labor`, `scale`, etc.) |
| `time_data.csv` | Per-interval carbon intensity and electricity price |
| `travel_time.csv` | Node-to-node MCS travel time matrix |
| `work_flexible.csv` | Per-interval CEV work-availability mask (0/1 per 15-min slot) |

## Outputs

Per-run results directory (`<dataset>/results/`) with: cost/energy/CO₂/KPI summary
plots, per-MCS power profile plots + CSVs, cumulative cost/CO₂ timeseries, and a
parsed HiGHS MIP-progress log (`parse_highs_mip_log`) for solver convergence
diagnostics.

## Relationship to the rest of the repo

This is the reference implementation `MPC_Shubhan/Approach 1`'s
`3_MCSModel.jl`/`4_MPCLoop.jl` were built from — constraint numbering and naming
conventions in that codebase's docs (e.g. "Eq. 13", "`work_per_travel = 4 (Avik)`")
refer back to this file. See `MPC_Shubhan/Optimization.pdf` for the paper-level
formulation these constraints implement.