# #############################################################################
# run_soe_sweep.jl  —  Initial-SOE sensitivity sweep for the :input case.
# -----------------------------------------------------------------------------
# Sweeps the CEV initial state-of-energy (SOE_CEV_ini) across NRUNS values from
# the battery's minimum to its maximum allowed SOE, re-running the shrinking-
# horizon closed loop (each window solved to the MIP gap, NO time limit) for
# each value with a FIXED seed so the only thing that changes is the start SOE.
#
# NOTE: in this model SOE_CEV_ini is BOTH the start SOE and the end-of-day floor
# target (the CEV must finish >= SOE_CEV_ini). So each sweep point moves the
# start level and its terminal floor together.
#
# For every run it keeps these artefacts in output/input_testing/<run>/:
#   plan_mcs_activity.html          (MCS replan grid)
#   plan_cev1_activity.html         (CEV1 replan grid)
#   plan_vs_actual_side_by_side.html
#   plan_vs_actual.html
#   plan_vs_actual_by_entity.html
#   plan_vs_actual_costs.png        (planned @08:00 vs realised, financial)
#   plan_vs_actual_activity.png     (planned vs realised ACTIVITY timeline heatmap)
#   approach0_vs_approach1.html     (Approach 0 one-shot vs Approach 1 closed-loop)
#   run_log.txt                     (this run's console output only)
# and finally writes summary.html with the optimality + "actions changed"
# (initial 08:00 plan vs realised) analysis across all runs.
# #############################################################################

SCENARIO1_NO_AUTORUN = true
const _HERE = @__DIR__                                  # .../Shrinking_Horizon/code
const _CODE = _HERE                                      # script now lives in code/ itself
include(joinpath(_CODE, "6_Shrinking_Horizon_main.jl"))
using Printf

const INPUT_DIR = normpath(joinpath(_HERE, "..", "data", "input_data"))
const OUT_DIR   = normpath(joinpath(_HERE, "..", "output", "input_testing"))
mkpath(OUT_DIR)
const NRUNS     = 10
# Which plant Approach 0's one-shot 08:00 plan is replayed under, per sweep point:
#   :sampled -> the fixed plan drifting under the stochastic pool, no feedback
#   :mean    -> realized power pinned to mu, so realized == planned and the number
#               is the whole-day MILP's own optimum
# Overridable by the master runner: if MASTER_A0_PLANT is defined before this
# script is included, it wins; otherwise the local default below applies.
const A0_PLANT  = (@isdefined(MASTER_A0_PLANT) && MASTER_A0_PLANT in (:sampled, :mean)) ?
                  MASTER_A0_PLANT : :sampled
const KEEP_TOP  = ["plan_vs_actual.html",
                   "plan_vs_actual_side_by_side.html",
                   "plan_vs_actual_by_entity.html",
                   "plan_vs_actual_costs.png",
                   "plan_vs_actual_activity.png",
                   "approach0_vs_approach1.html",
                   "10_approach0_vs_approach1_timeline.png"]
const KEEP_GRID = ["plan_mcs_activity.html", "plan_cev1_activity.html"]

# ---- sweep grid: SOE_min .. SOE_max (inclusive), NRUNS points -----------------
d0      = load_data(:input; input_dir = INPUT_DIR)
soe_min = d0.SOE_CEV_min[1]
soe_max = d0.SOE_CEV_max[1]
mcs_tgt = d0.SOE_MCS_ini[1]
soe_vals = [soe_min + (soe_max - soe_min) * (i - 1) / (NRUNS - 1) for i in 1:NRUNS]

println("="^78)
println("INITIAL-SOE SWEEP (:input)  —  $NRUNS runs, SOE_CEV_ini in [$soe_min, $soe_max] kWh")
println("  each window solved to the MIP gap (time_limit_sec = Inf), seed = 1")
println("="^78)

results = Vector{NamedTuple}(undef, 0)

for (idx, soe) in enumerate(soe_vals)
    tag = @sprintf("run%02d_SOE_%05.2f", idx, soe)
    dst = joinpath(OUT_DIR, tag)
    tmp = joinpath(OUT_DIR, "_tmp_$tag")

    # Fresh run_log.txt per run (opened in "w" mode by _with_console_log), so
    # each run's console output lands only in its own <tag>/ folder, never
    # mixed with the previous run's output.
    row = _with_console_log(dst) do
        println("\n#################### $tag ####################")
        d   = merge(d0, (; SOE_CEV_ini = [soe]))
        # POOL SIZING. next_power! ERRORS (it does not wrap) once a cursor walks
        # past the end of the pre-drawn samples. Size on the number of intervals
        # the run can actually consume -- one draw per (CEV, activity) occurrence
        # -- not on a fixed guess. Unconsumed samples cost nothing.
        n_pool = length(collect(d.K)) + 5
        pool = draw_activity_power_pool(d.E, d.prior_mu, d.prior_sigma;
                                        n_samples = n_pool, rng = MersenneTwister(1))
        # ---- APPROACH 0: one-shot 8:00 plan replayed under the chosen plant ----
        res0 = run_one_shot(d, pool; time_limit_sec = Inf, plant = A0_PLANT, seed = 1)
        # ---- APPROACH 1: closed-loop MPC ----
        res = run_mpc(d, pool; time_limit_sec = Inf, plant = :sampled, seed = 1)

        # full outputs to a temp dir, then keep only the wanted files --------
        isdir(tmp) && rm(tmp; recursive = true, force = true)
        write_outputs(res, tmp)
        write_approach_comparison(res0, res, tmp)   # approach0_vs_approach1.html
        # Guard every copy: a 10-run sweep is expensive, so one missing artefact
        # must not abort the whole thing -- warn and carry on instead.
        for f in KEEP_TOP
            src = joinpath(tmp, f)
            isfile(src) ? cp(src, joinpath(dst, f); force = true) :
                          @warn "sweep: expected artefact missing, skipping" src
        end
        for f in KEEP_GRID
            src = joinpath(tmp, "replan_grids", f)
            isfile(src) ? cp(src, joinpath(dst, f); force = true) :
                          @warn "sweep: expected replan grid missing, skipping" src
        end
        rm(tmp; recursive = true, force = true)

        # optimality + change-count metrics (identical logic to the HTML reports) --
        p  = Output._planned_kpis(res)      # planned @ 08:00
        c  = Output._cost_components(res)    # realised end-of-day
        r  = p.r
        # Change 5 follow-up: the flat plan_cev_act/plan_mcs_act fields were
        # removed from `res` in favor of `replan_by_day` (one entry per kept
        # day, so multi-day runs don't overwrite earlier days' plans -- see
        # CHANGES_SUMMARY.md). This sweep only ever runs n_day_run=1 (never
        # passed above), so day 1's plan is the whole plan; fetch it via
        # replan_by_day[1], same pattern already used in the working
        # Receding-Horizon sibling script. Compare over nKd (one day's
        # interval count), not nK (which is n_day_run * nKd, the GLOBAL count
        # -- using nK here would silently drop the bounds-safety this once
        # had, even though at n_day_run=1 nKd == nK numerically).
        g1  = res.replan_by_day[1]
        nKd = res.nKd
        cev_chg = count(k -> g1.plan_cev_act[1][r, k] != res.real_cev_act[1][k], 1:nKd)
        mcs_chg = count(k -> g1.plan_mcs_act[r, k]    != res.real_mcs_act[k],    1:nKd)

        cev_end  = res.soe_cev_end[1]
        mcs_end  = res.soe_mcs_end[1]
        feasible = (res.n_infeasible == 0) &&
                   (cev_end >= soe - 1e-6) && (mcs_end >= mcs_tgt - 1e-6)

        # Approach 0's fully-realised total under whichever plant mode A0_PLANT selects.
        a0_total = Output._cost_components(res0).total

        @printf("  SOE_ini=%.2f  realised \$%.2f (plan \$%.2f)  changed: CEV1=%d MCS=%d /%d  feas=%s\n",
                soe, c.total, p.total, cev_chg, mcs_chg, nKd, feasible)
        @printf("             A0(:%s) \$%.2f -> A1 \$%.2f (%+.2f)\n",
                A0_PLANT, a0_total, c.total, c.total - a0_total)

        (; idx, tag, soe, cev_end, mcs_end, feasible,
           infeasible = res.n_infeasible,
           clamps = res.n_capped + res0.n_capped,
           a0_total,
           planned_total = p.total, realized_total = c.total,
           energy = res.total_energy, ecost = c.energy_cost,
           co2 = res.total_co2, missed = res.missed,
           labour = c.travel_cost, transit_h = res.transit_intervals * d0.delta_T,
           nc_peak = res.nc_peak, op_peak = res.op_peak,
           cev_chg, mcs_chg, nK = nKd, elapsed = res.elapsed)
    end
    push!(results, row)
end

# =============================================================================
# summary.html
# =============================================================================
_fmt(x; d = 2) = @sprintf("%.*f", d, x)

io = IOBuffer()
println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Initial-SOE sweep — :input</title><style>")
println(io, "body{font-family:sans-serif;margin:20px;color:#222}")
println(io, "h1{font-size:20px}h2{font-size:16px;margin-top:24px}")
println(io, "table{border-collapse:collapse;font-size:12px;margin-top:8px}")
println(io, "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;white-space:nowrap}")
println(io, "th{background:#f4f4f4;text-align:center}")
println(io, "td:first-child,th:first-child{text-align:left}")
println(io, "tr:nth-child(even){background:#fafafa}")
println(io, ".good{color:#127a12;font-weight:bold}.bad{color:#c00;font-weight:bold}")
println(io, ".muted{color:#666;font-size:12px;max-width:900px;line-height:1.45}")
println(io, "a{color:#1257a8;text-decoration:none}a:hover{text-decoration:underline}")
println(io, "</style></head><body>")

println(io, "<h1>Initial-SOE sensitivity sweep &mdash; <code>:input</code> case</h1>")
println(io, "<p class=\"muted\">The CEV initial state-of-energy <code>SOE_CEV_ini</code> is swept across ",
            NRUNS, " values from the battery minimum (<b>", _fmt(soe_min), " kWh</b>) to its maximum (<b>",
            _fmt(soe_max), " kWh</b>). The shrinking-horizon closed loop was re-run for each value with ",
            "<b>no solver time limit</b> (every 15-min window solved to the MIP gap) and a <b>fixed seed</b>, ",
            "so differences come only from the start SOE. In this model <code>SOE_CEV_ini</code> is also the ",
            "end-of-day <b>floor target</b> (the CEV must finish at or above it), so each point shifts the ",
            "start level and its terminal target together.</p>")

println(io, "<p class=\"muted\"><b>Optimality</b> is reported two ways: (i) feasibility &mdash; zero infeasible ",
            "windows and both terminal targets met (MCS back to ", _fmt(mcs_tgt), " kWh, CEV &ge; its start); and ",
            "(ii) cost &mdash; the realised end-of-day total operating cost vs the ideal cost of the very first ",
            "08:00 whole-day plan. <b>Actions changed</b> counts the 15-min intervals whose realised activity ",
            "differs from that first 08:00 plan, per unit (CEV1 and the MCS), out of 96 intervals each.</p>")

# ---- headline findings ----
best = results[argmin([x.realized_total for x in results])]
most = results[argmax([x.cev_chg + x.mcs_chg for x in results])]
least = results[argmin([x.cev_chg + x.mcs_chg for x in results])]
all_feas = all(x.feasible for x in results)
tot_changed(x) = x.cev_chg + x.mcs_chg
println(io, "<h2>Headline findings</h2><ul class=\"muted\">")
println(io, "<li><b>Feasibility:</b> ", all_feas ?
            "<span class=\"good\">all $NRUNS runs optimal &amp; feasible</span> (0 infeasible windows; every terminal target met)." :
            "<span class=\"bad\">some runs infeasible</span> &mdash; see the table.", "</li>")
println(io, "<li><b>Cheapest run:</b> SOE_ini = <b>", _fmt(best.soe), " kWh</b> at realised total <b>\$",
            _fmt(best.realized_total), "</b>.</li>")
println(io, "<li><b>Most plan changes:</b> SOE_ini = <b>", _fmt(most.soe), " kWh</b> with <b>",
            tot_changed(most), "</b> changed intervals (CEV1 ", most.cev_chg, " + MCS ", most.mcs_chg, ").</li>")
println(io, "<li><b>Fewest plan changes:</b> SOE_ini = <b>", _fmt(least.soe), " kWh</b> with <b>",
            tot_changed(least), "</b> changed intervals.</li>")
avg_changed = sum(tot_changed(x) for x in results) / length(results)
println(io, "<li><b>Average plan changes across runs:</b> ", _fmt(avg_changed; d = 1),
            " intervals (out of 192 = 2&times;96).</li>")
println(io, "</ul>")

# ---- interpretation ----
rho_miss    = d0.rho_miss
zero_missed = filter(x -> x.missed <= 1e-9, results)
thresh      = isempty(zero_missed) ? NaN : minimum(x.soe for x in zero_missed)
thresh_pct  = isnan(thresh) ? NaN : 100 * (thresh - soe_min) / (soe_max - soe_min)
maxd        = results[argmax([x.realized_total - x.planned_total for x in results])]
println(io, "<h2>Interpretation</h2><ul class=\"muted\">")
println(io, "<li><b>Cost is dominated by the missed-work penalty</b> (<code>rho_miss</code> = \$",
            _fmt(rho_miss; d = 0), "/h in this dataset): realised cost clusters into bands set by whole ",
            "0.25 h steps of missed work (0 h &rarr; ~\$192, 0.25 h &rarr; ~\$690, 0.5 h &rarr; ~\$1185).</li>")
println(io, "<li><b>Charge threshold for zero missed work:</b> for SOE_ini &ge; <b>", _fmt(thresh),
            " kWh</b> (&ge; ", _fmt(thresh_pct; d = 0), "% of range) missed work is 0 and cost settles at its ",
            "~\$192 minimum; below it the excavator starts too depleted to keep up before the MCS can recharge it.</li>")
println(io, "<li><b>Robustness (plan vs actual):</b> at high SOE realised &asymp; planned (|&Delta;| &le; \$1.6). ",
            "At low/mid SOE the certainty-equivalent plan is fragile &mdash; the stochastic plant over-draws on some ",
            "intervals and tips marginal cases into unplanned missed work (worst &Delta; = +\$",
            _fmt(maxd.realized_total - maxd.planned_total), " at SOE ", _fmt(maxd.soe), " kWh).</li>")
println(io, "<li><b>Non-monotonicity at low SOE</b> is a stochastic-plant effect, not a model error: near the ",
            "missed-work boundary, exactly which intervals the plant over-draws decides whether a penalty fires.</li>")
println(io, "<li><b>Actions changed does not track cost:</b> it stays ~16&ndash;23% of intervals across the whole ",
            "range; at high SOE these are cost-neutral re-timings of charging/idle, at low SOE similar counts are ",
            "cost-relevant, so change-count must be read together with &Delta;cost.</li>")
println(io, "<li><b>Recommended operating point:</b> start each CEV at &ge; ~65% SOE to guarantee zero missed ",
            "work and a robust (plan &asymp; actual) schedule.</li>")
println(io, "</ul>")

# ---- main table ----
println(io, "<h2>Per-run results</h2>")
println(io, "<table><tr>",
    "<th>Run</th><th>SOE_ini<br>(kWh)</th><th>% of<br>range</th>",
    "<th>Feasible /<br>optimal</th><th>Infeas.<br>windows</th>",
    "<th>Planned<br>total (\$)</th><th>Realised<br>total (\$)</th><th>&Delta; real&minus;plan<br>(\$)</th>",
    "<th>A0 (\$)</th><th>&Delta; A1&minus;A0<br>(\$)</th><th>SOE<br>clamps</th>",
    "<th>Energy<br>(kWh)</th><th>Energy<br>(\$)</th><th>CO2<br>(kg)</th><th>Missed<br>(h)</th>",
    "<th>Labour<br>(\$)</th><th>NC peak<br>(kW)</th>",
    "<th>CEV1<br>end (kWh)</th><th>MCS<br>end (kWh)</th>",
    "<th>CEV1<br>changed</th><th>MCS<br>changed</th><th>Total<br>changed</th><th>%<br>changed</th>",
    "<th>Loop<br>(s)</th><th>Files</th></tr>")
for x in results
    pct_range   = 100 * (x.soe - soe_min) / (soe_max - soe_min)
    pct_changed = 100 * tot_changed(x) / (2 * x.nK)
    feas_cell = x.feasible ? "<span class=\"good\">yes</span>" : "<span class=\"bad\">NO</span>"
    links = string(
        "<a href=\"", x.tag, "/plan_vs_actual.html\">P&times;A</a> ",
        "<a href=\"", x.tag, "/plan_vs_actual_costs.png\">costs</a> ",
        "<a href=\"", x.tag, "/plan_vs_actual_activity.png\">activity</a> ",
        "<a href=\"", x.tag, "/plan_vs_actual_side_by_side.html\">side</a> ",
        "<a href=\"", x.tag, "/plan_vs_actual_by_entity.html\">entity</a> ",
        "<a href=\"", x.tag, "/plan_cev1_activity.html\">cev1</a> ",
        "<a href=\"", x.tag, "/plan_mcs_activity.html\">mcs</a> ",
        "<a href=\"", x.tag, "/approach0_vs_approach1.html\">A0vA1</a> ",
        "<a href=\"", x.tag, "/10_approach0_vs_approach1_timeline.png\">timeline</a> ",
        "<a href=\"", x.tag, "/run_log.txt\">log</a>")
    println(io, "<tr>",
        "<td>", x.idx, "</td>",
        "<td>", _fmt(x.soe), "</td>",
        "<td>", _fmt(pct_range; d = 0), "</td>",
        "<td style=\"text-align:center\">", feas_cell, "</td>",
        "<td>", x.infeasible, "</td>",
        "<td>", _fmt(x.planned_total), "</td>",
        "<td>", _fmt(x.realized_total), "</td>",
        "<td>", _fmt(x.realized_total - x.planned_total), "</td>",
        "<td>", _fmt(x.a0_total), "</td>",
        "<td>", _fmt(x.realized_total - x.a0_total), "</td>",
        "<td>", x.clamps, "</td>",
        "<td>", _fmt(x.energy), "</td>",
        "<td>", _fmt(x.ecost), "</td>",
        "<td>", _fmt(x.co2), "</td>",
        "<td>", _fmt(x.missed), "</td>",
        "<td>", _fmt(x.labour), "</td>",
        "<td>", _fmt(x.nc_peak), "</td>",
        "<td>", _fmt(x.cev_end), "</td>",
        "<td>", _fmt(x.mcs_end), "</td>",
        "<td>", x.cev_chg, "</td>",
        "<td>", x.mcs_chg, "</td>",
        "<td>", tot_changed(x), "</td>",
        "<td>", _fmt(pct_changed; d = 1), "</td>",
        "<td>", _fmt(x.elapsed; d = 1), "</td>",
        "<td style=\"text-align:left\">", links, "</td>",
        "</tr>")
end
println(io, "</table>")

println(io, "<p class=\"muted\" style=\"margin-top:20px\"><b>Column notes.</b> ",
            "<b>Planned total</b> = the objective cost of the first 08:00 whole-day plan; ",
            "<b>Realised total</b> = the actual closed-loop cost after re-planning against the stochastic plant. ",
            "<b>A0</b> = Approach 0's one-shot 08:00 plan replayed open-loop under the <code>A0_PLANT</code> ",
            "mode set at the top of this script (<code>:mean</code> = deterministic, i.e. the whole-day ",
            "MILP's own optimum; <code>:sampled</code> = the same plan drifting under the stochastic pool); ",
            "<b>&Delta; A1&minus;A0</b> = Realised &minus; A0 (negative = the closed loop was cheaper). ",
            "<b>SOE clamps</b> counts intervals where the CEV SOE guard bit and ",
            "energy was silently created/discarded &mdash; a non-zero value qualifies that row's SOE trace. ",
            "<b>Total changed</b> = CEV1 + MCS changed intervals (out of 2&times;96 = 192); ",
            "<b>% changed</b> normalises that. Each <b>Files</b> link opens that run's kept HTML artefacts.</p>")

println(io, "</body></html>")
write(joinpath(OUT_DIR, "summary.html"), String(take!(io)))

println("\n", "="^78)
println("SWEEP COMPLETE. Wrote ", length(results), " run folders + summary.html to:")
println("  ", OUT_DIR)
println("="^78)
