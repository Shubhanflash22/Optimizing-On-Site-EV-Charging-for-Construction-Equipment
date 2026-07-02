# Scenario 1 output — what each file is

Results are written under **`output/<mode>/`** — i.e. **`output/input/`** for the
real 7-CSV dataset and **`output/synthetic/`** for the built-in synthetic scenario.
Both folders contain the exact same set of files described below; they are
regenerated on every `run_scenario_1(...)`.

Each run is a closed-loop MPC in **15-min steps** over the daytime
(08:00–18:00 = 40 intervals), followed by a deterministic overnight MCS recharge
(Phase 2). Two batteries are tracked: the **MCS** (mobile charging station) and each
**CEV** (excavator).

## The two schedule CSVs (by audience)

### `worker_schedule.csv` — simple, for the site crew
One row per 15-min slot with only what a worker needs:

- **`time`** — slot start (clock).
- **`CEV<e>_activity`** — what excavator `e` should do: `Digging` / `Loading/Swinging` / `Traveling` / `Idle`.
- **`CEV<e>_plug_in_charge`** — `Yes/No`: plug this CEV into the MCS this slot?
- **`MCS_charge_from_grid`** — `Yes/No`: should the MCS draw from the grid this slot?

### `closed_loop_trajectory.csv` — detailed, for analysts
One row per applied 15-min interval (the decision actually sent to the plant):

- **`k`, `clock`** — interval index and clock.
- **`price`** — electricity price (`lambda_buy`, $/kWh); **`co2`** — grid carbon intensity.
- **`grid_kW`** — MCS power drawn *from the grid* this interval.
- **`dch_kW`** — MCS power *discharged to the CEV(s)* this interval.
- **`work_kW`** — realized CEV work power (dig/load/travel).
- **`soe_mcs`, `soe_cev1`, `soe_cev2`** — end-of-interval battery energy (kWh). `NaN` = that unit doesn't exist in this dataset (this input has only 1 CEV, so `soe_cev2` is `NaN`).
- **`mcs_node`** — where the MCS is (node index; `0` = in transit).
- **`est_dig/est_load/est_trv/est_idle`** — the online **Bayesian estimate** of each activity's power (kW).
- **`unc_dig/unc_load/unc_trv/unc_idle`** — posterior std-dev (uncertainty) of those estimates.
- **`n_obs`** — telematics observations absorbed so far.

> An interval that was **infeasible under the hard constraints (no fallback)** appears
> here with `grid_kW=dch_kW=work_kW=0` and activity `Idle` — the plant *held state*.

## Overnight (Phase 2)

### `overnight_mcs_charge.csv`
Deterministic cheapest-hours refill that returns the MCS to its start level after 18:00:

- **`k`, `clock`** — overnight interval / clock.
- **`price`** — electricity price that slot.
- **`MCS<m>_charge_kW`** — grid power used to recharge MCS `m`.
- **`MCS<m>_soe_kWh`** — MCS energy after that slot.
- **`MCS<m>_charging`** — `Yes/No`.

## `replan_grids/` — how the plan evolves (analyst detail)

Each file is a **square grid**: **rows = the re-plan step** (`replan_at` clock, i.e. `k0`),
**columns = each interval's clock**. A cell is the value the optimizer **planned for that
interval at that re-plan step**.

- **Diagonal** (`k == k0`): the decision actually applied to the plant that step.
- **Right of the diagonal** (`k > k0`): the fresh forward plan made at that step for future intervals.
- **Left of the diagonal** (`k < k0`, the past): the decision already applied at that earlier interval — now **fixed** and copied from the diagonal.

So each row reads left-to-right as **already-fixed past + newly re-planned future**: at `08:00` the whole day is planned; at `08:15` the `08:00` slot is fixed and `08:15`→ is re-planned; at `08:30` the `08:00` and `08:15` slots are fixed; and so on. Reading **down a column** shows how one interval's plan was revised each step until it became fixed on the diagonal.

Each grid is written twice: a plain **`.csv`** (open in Excel) and a **colored `.html`**
(open in any browser) where **green = complete (fixed/past)** and **yellow = pending
(current + planned)**.

Files:

- **`plan_grid_kW.csv` / `.html`** — planned MCS grid draw (kW).
- **`plan_mcs_soe.csv` / `.html`** — planned MCS energy at interval end (kWh).
- **`plan_cev<e>_soe.csv` / `.html`** — planned CEV `e` energy at interval end (kWh).
- **`plan_cev<e>_activity.csv` / `.html`** — planned CEV `e` activity name.

> **Infeasible steps:** there is **no fallback**. If a re-plan step cannot satisfy the hard
> constraints from its realized state, that window produces **no plan** and the plant holds
> state; the row's future cells stay blank for that step. With the default terminal margin
> (`term_tol = 0.1` kWh) all 40 windows are feasible here, so every row is fully populated.

## Plots (PNG)

- **`01_grid_draw_vs_price.png`** — MCS grid draw (kW) vs electricity price over the day.
- **`02_state_of_energy.png`** — MCS and CEV battery SOE trajectories.
- **`03_work_power.png`** — realized CEV work power per interval.
- **`04_power_estimate_convergence.png`** — the online Bayesian power estimates (with uncertainty ribbons) converging to the hidden true activity powers.
