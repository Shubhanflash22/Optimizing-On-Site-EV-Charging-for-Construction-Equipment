"""
Independent hand-derivation for the CORRECTED tiny Approach 0 verification
scenario (6 intervals, not 4). See the explanation accompanying this file for
why the original 4-interval version was infeasible: it didn't leave the MCS
enough time to travel to the site, serve the CEV, travel BACK, and recharge
ITSELF -- which it must do exactly (not approximately) by the end of the day.

This script does NOT read Julia's output. It re-derives the expected
schedule from the input numbers alone.

Run with: python verify_toy_v2.py
"""

delta_T = 1.0
n_int = 6

SOE_CEV_min, SOE_CEV_max, SOE_CEV_ini = 0.0, 10.0, 5.0
CH_CEV = 5.0
eta_cev = 1.0

SOE_MCS_min, SOE_MCS_max, SOE_MCS_ini = 0.0, 100.0, 100.0
eta_mcs = 1.0

p_digging = 5.0
hours_digging_required = 2.0
tau_trv = 1

rho_labor = 10.0
lambda_buy = 0.20

print("=" * 72)
print("STEP 1: The MCS's OWN hard constraint, which the first attempt missed")
print("=" * 72)
print(f"""
The CEV's terminal SOE rule is lenient: it must end the day with AT LEAST
its starting energy ({SOE_CEV_ini} kWh) -- ending higher is fine.

The MCS's terminal SOE rule is strict: it must end the day with EXACTLY
its starting energy ({SOE_MCS_ini} kWh) -- not approximately, exactly equal.
That means every kWh the MCS gives away to the CEV, it must get back from
the grid before the day ends. To do that it needs to physically:
  1. travel TO the site                 ({tau_trv} interval)
  2. stay there long enough to deliver what the CEV needs
  3. travel BACK to the grid             ({tau_trv} interval)
  4. stay there long enough to recharge itself back to full

The CEV needs {hours_digging_required} h of digging at {p_digging} kW = {hours_digging_required*p_digging} kWh of work,
but can only accept charge at {CH_CEV} kW, so delivering that much energy
takes at least {hours_digging_required*p_digging/CH_CEV:.0f} full hour(s) of the MCS parked at the site.

Minimum intervals the MCS needs: {tau_trv} (out) + {hours_digging_required*p_digging/CH_CEV:.0f} (serve) + {tau_trv} (back) + 1 (recharge) = {tau_trv + int(hours_digging_required*p_digging/CH_CEV) + tau_trv + 1}
This is why the day needs {n_int} intervals, not 4 -- 4 was one short.
""")

print("=" * 72)
print("STEP 2: Hand-tracing the schedule this leaves room for")
print("=" * 72)

soe_cev = SOE_CEV_ini
soe_mcs = SOE_MCS_ini
schedule = ["dig", "charge", "dig", "charge", "travel_back", "recharge_mcs"]
mcs_location = ["Transit (i1->i2)", "i2", "i2", "i2", "Transit (i2->i1)", "i1"]
travel_intervals = 0
grid_charge_kWh = 0.0

rows = []
for k, act in enumerate(schedule, start=1):
    cs, ms = soe_cev, soe_mcs
    if act == "dig":
        soe_cev -= p_digging * delta_T
    elif act == "charge":
        d_e = CH_CEV * delta_T * eta_cev
        soe_cev += d_e
        soe_mcs -= d_e / eta_mcs
    elif act == "recharge_mcs":
        needed = SOE_MCS_ini - soe_mcs          # exactly enough to return to full
        soe_mcs += needed
        grid_charge_kWh = needed
    if mcs_location[k - 1].startswith("Transit"):
        travel_intervals += 1
    rows.append((k, act, mcs_location[k - 1], cs, soe_cev, ms, soe_mcs))

print(f"{'k':<14}{'CEV activity':<10}{'MCS location':<18}{'CEV SOE':<18}{'MCS SOE'}")
for k, act, loc, cs, ce, ms, me in rows:
    print(f"{k:<14}{act:<10}{loc:<18}{cs:>5.1f} -> {ce:<9.1f}{ms:>6.1f} -> {me:.1f}")

print(f"""
Final CEV SOE = {soe_cev} kWh   (required: >= {SOE_CEV_ini} -> {"OK" if soe_cev >= SOE_CEV_ini else "VIOLATED"})
Final MCS SOE = {soe_mcs} kWh   (required: == {SOE_MCS_ini} -> {"OK, exact" if abs(soe_mcs-SOE_MCS_ini) < 1e-9 else "VIOLATED"})
Digging completed = {schedule.count('dig')*delta_T} h   (required: {hours_digging_required} h -> {"OK" if schedule.count('dig')*delta_T==hours_digging_required else "MISMATCH"})
Travel intervals = {travel_intervals} (one leg out, one leg back)
Grid energy drawn to recharge the MCS = {grid_charge_kWh} kWh
""")

print("=" * 72)
print("STEP 3: Expected cost breakdown")
print("=" * 72)

energy_cost = grid_charge_kWh * lambda_buy
travel_cost = rho_labor * delta_T * travel_intervals
total_cost = energy_cost + travel_cost   # carbon/NCD/OPD/missed/shortfall all $0 here

print(f"""
Grid energy cost        : ${energy_cost:.2f}   ({grid_charge_kWh} kWh x ${lambda_buy}/kWh)
Carbon cost              : $0.00
NC demand charge         : $0.00
OP demand charge         : $0.00
Missed-work penalty      : $0.00
Travel labour cost       : ${travel_cost:.2f}   ({travel_intervals} interval(s) x {delta_T}h x ${rho_labor}/h)
Terminal shortfall cost  : $0.00
--------------------------------------------
TOTAL EXPECTED COST      : ${total_cost:.2f}
""")

print("=" * 72)
print("WHAT TO COMPARE THIS AGAINST")
print("=" * 72)
print("""
    res = run_scenario_0(input_dir = "toy_input_data_v2", out_dir = "toy_output_v2",
                          plant = :mean, n_day_run = 1, detailed_output = true)

Check:
  - Total_Cost_USD in A0_kpi_summary.csv matches TOTAL EXPECTED COST above
  - Missed_Work_hour = 0, Terminal_SOE_Shortfall_kWh = 0
  - CEV SOE at horizon = 5.0, MCS SOE at horizon = 100.0 (exact)
  - A0_realized_tuple.csv / A0_MCS_realized_tuple.csv match the table in
    STEP 2 row for row
  - solve_log.objective also equals the total above
""")
