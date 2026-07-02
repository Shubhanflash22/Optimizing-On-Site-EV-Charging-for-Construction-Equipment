# =============================================================================
# EDGE-CASE TEST HARNESS for Scenario_1.jl
# =============================================================================
# Calls the REAL model builder (`build_window_model`) and the closed-loop driver
# (`run_scenario_1`) from Scenario_1.jl on a battery of edge cases, and checks
# the design rules we agreed on:
#
#   * Energy neutrality (8b): MCS & CEV end the cycle at their START level.
#     - CEV: ends the daytime horizon (18:00) back at SOE_CEV_ini. In the shipped
#       closed loop this is a HARD inequality SOE_end >= SOE_ini - term_tol; here
#       `solve_day` uses soft_term=true (a penalised terminal) so a SINGLE full-day
#       solve is feasible from any realized start (precedence 12d + pacing 13 stay HARD).
#     - MCS: parked at a grid node at 18:00, then restored to SOE_MCS_ini OVERNIGHT
#       by the deterministic Phase-2 smart-charge (cheapest hours).
#   * Rest rule (12e, t_limit=1h): >=1 idle break interval per rolling hour.
#   * Travel pacing (13): CEV repositioning tied to productive work.
#   * 20% SOE floors for both MCS and CEV; daytime MCS grid charging only as needed.
#   * The window MILP is feasible from these start states (in-transit start, heavy
#     demand, drained start) using the soft-terminal solve_day.
#   * Labour is a per-hour MCS TOWING cost (objective has the y_trv transit term).
#
# Run from this folder:   julia test_scenario_1.jl
# (Defining SCENARIO1_NO_AUTORUN stops the included file from launching the full
#  35-min closed-loop run on include.)
# =============================================================================

SCENARIO1_NO_AUTORUN = true
include(joinpath(@__DIR__, "Scenario_1.jl"))

using JuMP

# ---- tiny assertion framework ----------------------------------------------
const _PASS = Ref(0)
const _FAIL = Ref(0)
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        _PASS[] += 1
        println("  [PASS] ", name)
    else
        _FAIL[] += 1
        println("  [FAIL] ", name, isempty(detail) ? "" : "  -- $detail")
    end
    return cond
end

# ---- helpers ----------------------------------------------------------------
is_work(d, k) = any(d.R_work[i, e, k] > 0 for i in d.N_c, e in d.E)

# Build + solve a full-day (terminal) window from an arbitrary realized state.
# Uses soft_term=true (a penalised terminal) so a single full-day solve is feasible from
# ANY realized start state: precedence (12d) and pacing (13) stay HARD, only the CEV
# energy-neutral terminal (8b) is relaxed to a penalised slack. The shipped closed loop
# instead keeps the terminal HARD with a small tolerance (term_tol); this soft form is a
# test convenience so the edge cases below always have a solution to inspect.
function solve_day(d;
                   soe_mcs = copy(float.(d.SOE_MCS_ini)),
                   soe_cev = copy(float.(d.SOE_CEV_ini)),
                   node    = [first(d.N_g) for _ in d.M],
                   transit = Any[nothing for _ in d.M],
                   rem_dig  = copy(float.(d.hours_digging)),
                   rem_load = copy(float.(d.hours_loading_swinging)),
                   tl::Float64 = 60.0)
    nz = length(d.E)
    return build_window_model(d, 1:length(d.K), soe_mcs, soe_cev, node, transit,
                              rem_dig, rem_load, zeros(nz), zeros(nz), zeros(nz),
                              0.0, 0.0, d.prior_mu;
                              time_limit_sec = tl, silent = true, soft_term = true)
end

# Replace fields in the (immutable) data NamedTuple.
modify(d; kw...) = merge(d, (; kw...))

feasible(model) = has_values(model)

# ---- shared structural checks on a solved full-day model --------------------
# Rest rule (12e): over any rolling window of (t_limit/dt + 1) intervals, a CEV
# does at most t_limit/dt intervals of construction work (dig/load/travel).
function check_rest_rule(tag, d, model)
    cap = Int(round(d.t_limit_rest / d.delta_T))
    win = cap + 1
    ok = true
    for e in d.E, i in d.N_c
        d.A[i, e] == 1 || continue
        for k0 in first(d.K):(last(d.K) - win + 1)
            w = sum(value(model[:u][e, i, a, k]) for a in (d.B[1], d.B[2], d.B[3]), k in k0:(k0 + win - 1))
            w > cap + 1e-4 && (ok = false)
        end
    end
    check("$tag: rest rule holds (<=$cap work per $win-interval window)", ok)
end

# MCS must be parked at a grid node at the final daytime boundary (10e).
function check_mcs_home(tag, d, model)
    ok = all(sum(value(model[:z][m, i, last(d.K)]) for i in d.N_g) > 0.5 for m in d.M)
    check("$tag: MCS parked at a grid node at 18:00 (10e)", ok)
end

function check_bounds(tag, d, model)
    okm = all(d.SOE_MCS_min[m] - 1e-4 <= value(model[:SOE_MCS][m, t]) <= d.SOE_MCS_max[m] + 1e-4
              for m in d.M, t in 1:(length(d.K) + 1))
    okc = all(d.SOE_CEV_min[e] - 1e-4 <= value(model[:SOE_CEV][e, t]) <= d.SOE_CEV_max[e] + 1e-4
              for e in d.E, t in 1:(length(d.K) + 1))
    check("$tag: MCS SOE within [min,max] at every boundary", okm)
    check("$tag: CEV SOE within [min,max] at every boundary", okc)
end

# In `solve_day` the CEV terminal is soft (soft_term=true), so the incumbent may settle a
# little short of the exact start level. We verify the MECHANISM: each CEV ends the day
# within ~1.0 kWh of its START level (SOE_CEV_ini), i.e. recharged back to where it began.
cev_end_neutral(d, model) = all(value(model[:SOE_CEV][e, last(d.K) + 1]) >=
                                d.SOE_CEV_ini[e] - 1.0 for e in d.E)

# Total CEV repositioning (travel) intervals in the solution (pacing should induce
# some travel when there is substantial productive work).
cev_travel_intervals(d, model) =
    sum(value(model[:u][e, i, d.B[3], k]) for e in d.E, i in d.N_c, k in d.K)

total_charge_kWh(d, model) =
    sum(value(model[:P_ch_tot][m, k]) * d.delta_T for m in d.M, k in d.K)

# Phase-2 overnight check: restore MCS to SOE_MCS_ini using cheapest hours only.
function overnight_summary(d, model)
    soe_end = [value(model[:SOE_MCS][m, last(d.K) + 1]) for m in d.M]
    ov_df, P_ov, ov_k = phase2_overnight_charge(d, soe_end)
    final = [ov_df[end, Symbol("MCS$(m)_soe_kWh")] for m in d.M]
    onpeak_kWh = sum(in_peak(ov_k[j], d.delta_T, d.t_start) ? P_ov[m, j] * d.delta_T : 0.0
                     for m in 1:length(d.M), j in 1:length(ov_k); init = 0.0)
    total_kWh  = sum(P_ov) * d.delta_T
    return final, onpeak_kWh, total_kWh
end

# =============================================================================
println("="^70)
println("Scenario 1 edge-case tests")
println("="^70)

d0 = load_data(:synthetic)

# ---- Test 1: MAIN day (both batteries start FULL) ---------------------------
println("\n[1] Main day: MCS=250 (full), CEVs start at 80% of max -> energy-neutral cycle")
let model = solve_day(d0; tl = 120.0)
    if check("T1: feasible", feasible(model))
        check_bounds("T1", d0, model)
        check_mcs_home("T1", d0, model)
        check_rest_rule("T1", d0, model)
        check("T1: CEVs end ~full (neutral to start)", cev_end_neutral(d0, model),
              detail = "soe=$(round.([value(model[:SOE_CEV][e, last(d0.K)+1]) for e in d0.E], digits=2))")
        check("T1: CEVs do some repositioning (travel pacing active)",
              cev_travel_intervals(d0, model) > 0.5,
              detail = "travel intervals=$(round(cev_travel_intervals(d0, model), digits=1))")
        final, onpeak, tot = overnight_summary(d0, model)
        check("T1: overnight restores MCS to SOE_ini (energy-neutral)",
              all(abs(final[m] - d0.SOE_MCS_ini[m]) < 1e-3 for m in 1:length(d0.M)),
              detail = "final=$(round.(final, digits=2)) target=$(round.(d0.SOE_MCS_ini, digits=2))")
        check("T1: overnight uses cheapest hours (no on-peak draw)", onpeak < 1e-6,
              detail = "onpeak=$(round(onpeak, digits=3)) / total=$(round(tot, digits=1)) kWh")
    end
end

# ---- Test 2: drained MCS realized start -> holds 20% floor, refills overnight 
println("\n[2] MCS realized start low (80 kWh) -> holds >=20% by day; refills full overnight")
let d = d0, model = solve_day(d; soe_mcs = [80.0], tl = 120.0)
    if check("T2: feasible", feasible(model))
        check_bounds("T2", d, model)   # includes the 20% (50 kWh) MCS floor
        check_mcs_home("T2", d, model)
        final, onpeak, tot = overnight_summary(d, model)
        check("T2: overnight refills MCS to full (SOE_ini=250)",
              abs(final[1] - d.SOE_MCS_ini[1]) < 1e-3,
              detail = "final=$(round(final[1], digits=2))")
        check("T2: overnight uses cheapest hours (no on-peak draw)", onpeak < 1e-6,
              detail = "onpeak=$(round(onpeak, digits=3)) / total=$(round(tot, digits=1)) kWh")
    end
end

# ---- Test 3: MCS in transit at the start boundary ---------------------------
println("\n[3] MCS in transit at start (arc 2->1, 1 interval left) -> feasible, ends home")
let d = d0, model = solve_day(d; node = [0], transit = Any[(2, 1, 1)], tl = 90.0)
    if check("T3: feasible despite mid-trip carry-in", feasible(model))
        check_bounds("T3", d, model)
        check_mcs_home("T3", d, model)
    end
end

# ---- Test 4: CEVs start depleted -> topped back to full by 18:00 ------------
println("\n[4] CEVs realized start low (near 20% floor) -> recharged toward start level by 18:00")
let d = d0, model = solve_day(d; soe_cev = [20.0, 14.0], tl = 120.0)
    if check("T4: feasible", feasible(model))
        check_bounds("T4", d, model)
        check("T4: CEVs end ~full by 18:00", cev_end_neutral(d, model),
              detail = "soe=$(round.([value(model[:SOE_CEV][e, last(d.K)+1]) for e in d.E], digits=2))")
    end
end

# ---- Test 5: heavy work demand (still feasible via soft missed-work) ---------
println("\n[5] Heavy demand (2x digging/loading) -> feasible, missed-work is soft")
let d = modify(d0; hours_digging = d0.hours_digging .* 2,
                   hours_loading_swinging = d0.hours_loading_swinging .* 2),
    model = solve_day(d; tl = 90.0)
    check("T5: feasible under heavy demand", feasible(model))
end

# ---- Test 6: energy-neutral to a NON-FULL start level (edge) ----------------
println("\n[6] Edge start levels: MCS 200, CEVs 40/30 -> return to THOSE levels (not full)")
let d = modify(d0; SOE_MCS_ini = [200.0], SOE_CEV_ini = [40.0, 30.0]),
    model = solve_day(d; tl = 120.0)
    if check("T6: feasible", feasible(model))
        check_bounds("T6", d, model)
        endsoe = [value(model[:SOE_CEV][e, last(d.K) + 1]) for e in d.E]
        check("T6: CEVs return to their START level (~40/30, NOT full)",
              all(abs(endsoe[e] - d.SOE_CEV_ini[e]) < 1.0 for e in d.E),
              detail = "soe=$(round.(endsoe, digits=2)) target=$(round.(d.SOE_CEV_ini, digits=2))")
        final, _, _ = overnight_summary(d, model)
        check("T6: overnight restores MCS to its START level (~200, not max)",
              abs(final[1] - d.SOE_MCS_ini[1]) < 1e-3,
              detail = "final=$(round(final[1], digits=2))")
    end
end

# ---- Test 7: structural alignment ------------------------------------------
println("\n[7] Model structure: rest rule + travel pacing present; recharge/MCS-terminal vars gone")
let d = d0, model = solve_day(d; tl = 60.0)
    od = object_dictionary(model)
    check("T7: travel-pacing slacks present (s_pace_hi/lo)",
          haskey(od, :s_pace_hi) && haskey(od, :s_pace_lo))
    check("T7: precedence slack present (s_prec)", haskey(od, :s_prec))
    check("T7: CEV energy-neutral terminal slack present (s_term_cev)", haskey(od, :s_term_cev))
    check("T7: recharge-when-low binary REMOVED", !haskey(od, :recharge))
    check("T7: MCS terminal-energy slack REMOVED", !haskey(od, :s_term_mcs))
    if feasible(model)
        final, _, _ = overnight_summary(d, model)
        check("T7: Phase-2 overnight reaches MCS SOE_ini exactly",
              abs(final[1] - d.SOE_MCS_ini[1]) < 1e-3,
              detail = "final=$(round(final[1], digits=2))")
    end
end

# =============================================================================
println("\n", "="^70)
println("RESULTS: $(_PASS[]) passed, $(_FAIL[]) failed")
println("="^70)
_FAIL[] == 0 || error("Edge-case tests FAILED ($(_FAIL[]) failures)")
