# Bayesian Regression — CEV Activity Power Fitting

Fits per-activity power draw (kW) for the excavator (CEV) from field task logs, by
regressing observed battery State-of-Charge (SoC) drop against time spent on each
activity. The fitted values (`z` = Digging, Loading+Swinging, Travelling, Idling, etc.,
in kW) are the `mu` values consumed by the MPC/MILP scheduling model elsewhere in this
repo.

## Files

| File | Method |
|---|---|
| `Tasks_energy_loading_swinging.py` | Weighted least-squares (point estimate), evaluated via repeated random splits and repeated k-fold CV |
| `Tasks_energy_loading_swinging_bayesian.py` | Full Bayesian regression (PyMC, MCMC) — same equation-building pipeline, produces a posterior distribution over each activity's power instead of a single point estimate |

## The core idea (shared by both scripts)

For each task-log Excel sheet (one sheet per recording day/site), the script:

1. **Reads** `Start time`, `End time`, `Activity`, and `SoC` (%) columns.
2. **Buckets rows** by accumulating activity time until cumulative `|ΔSoC|` reaches
   `MIN_DELTA_SOC` (default 3%), then emits one linear equation per bucket:
   ```
   (hours on Digging)·z_dig + (hours on Loading+Swinging)·z_load + ... = ΔSoC · battery_cap / 100
   ```
3. **Stacks** all equations from all days into `A z = b` (`A`: hours per activity per
   equation, `b`: energy drop in kWh per equation, `z`: unknown per-activity power in kW).
4. **Solves** for `z` — as a constrained weighted least-squares problem (point-estimate
   script) or as a Bayesian posterior over `z` (Bayesian script) — with idling power
   fixed to 0 in both cases.

## Point-estimate script (`Tasks_energy_loading_swinging.py`)

- Solves `min ||√W (Az − b)||² + reg·||z||²` via `cvxpy` (MOSEK solver), `z ≥ 0`.
- `W` = per-equation weights, selectable via `WEIGHT_SCHEME` (`uniform`, `linear`,
  `bounded_median`, `quadratic` — see `compute_weights()`).
- Reports uncertainty by **re-fitting** across 200 random 80/20 splits and a repeated
  5-fold CV (1000 fits), giving mean ± SD and a 95% empirical interval per activity,
  plus MAE/RMSE/MAPE/NMAE test-set metrics.
- Headline coefficients come from one final fit on **all** equations; the repeated
  splits are for uncertainty estimation only.
- Also supports an optional `grading = "True"` mode that keeps "Grading 1"/"Grading 2"
  as separate activities instead of folding them into Digging/Loading.

## Bayesian script (`Tasks_energy_loading_swinging_bayesian.py`)

- Same equation-building pipeline, but fits `z` (and the residual noise `sigma`) as a
  full posterior via PyMC/MCMC (`pm.sample`, NUTS, 4 chains × 2000 draws, 2000 tuning
  steps, `target_accept=0.9`).
- Configurable priors per activity (`x_prior_config`) — supports Normal, truncated
  Normal, LogNormal, Gamma, and Exponential priors on each `z_i`, plus a configurable
  prior on the noise scale `sigma` (`build_prior_sigma`).
- Reports posterior means/intervals (via `arviz`) instead of split-based point
  estimates, and computes predictive-interval coverage on held-out equations.
- This is the version whose output (posterior mean + std per activity) feeds the
  MPC's stochastic `mu`/`sigma` inputs.

## Inputs

Both scripts currently hardcode local file paths (`/Users/avikghosh/Desktop/CEV-Analysis/Analysis/...xlsx`)
to the per-day task Excel files (`Oct_21_Tasks_1.xlsx`, `Feb_02_Tasks_1.xlsx`, etc. —
23 files spanning Oct 2025–Feb 2026, across soil/concrete/sand sites). **These paths
need to be updated/parameterized before running on another machine.**

## Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `cvxpy` (+ MOSEK license for the solver),
`scikit-learn`, `scipy`. The Bayesian script additionally needs `pymc`, `arviz`, `xarray`.

## Output

Both scripts print fitted per-activity power (kW) with uncertainty to stdout, plus
diagnostic plots (observed-vs-predicted, per-activity coefficient distributions,
activity time-share, and a correlation matrix of activity co-occurrence).