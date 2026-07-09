# #############################################################################
# _validate_legacy.jl  —  per-row constraint checker for the LEGACY Scenario_1.jl
# output (which writes only worker_schedule.csv + closed_loop_trajectory.csv).
# Works for BOTH horizons (Receding labels "D1 08:00"; Shrinking labels "08:00").
#
# Usage:  julia --project=. _validate_legacy.jl <code_dir> <synthetic|input> <out_dir>
# Checks per row: rest rule (12e), precedence (12d), travel pacing (13), CEV SOE
# bounds, MCS end-at-grid (10e), and per-day work achieved vs the schedule.
# #############################################################################
using CSV, DataFrames, Printf

code_dir = ARGS[1]; mode = Symbol(ARGS[2]); out = ARGS[3]
include(joinpath(code_dir, "Common.jl"))
include(joinpath(code_dir, "DataLoader.jl"))
# Some modules export load_data; the legacy monolith defines it at top level. Use the
# DataLoader module's loader (identical parameters) for a clean, dependency-free check.
using .DataLoader: load_data
d = load_data(mode; input_dir = joinpath(dirname(code_dir), "data", "input_data"))

is_work(a) = a in ("Digging", "Loading/Swinging", "Traveling")
is_dig(a)  = a == "Digging"
is_load(a) = a == "Loading/Swinging"
is_trv(a)  = a == "Traveling"

ws  = CSV.read(joinpath(out, "worker_schedule.csv"), DataFrame)
tj  = CSV.read(joinpath(out, "closed_loop_trajectory.csv"), DataFrame)
# day index: "D<k> HH:MM" (receding) or plain "HH:MM" (shrinking single day)
function day_of(lbl)
    m = match(r"D(\d+)", string(lbl))
    m === nothing ? 1 : parse(Int, m.captures[1])
end
days = day_of.(ws.time)
nE   = length(d.E)
acts = [String.(ws[!, Symbol("CEV$(e)_activity")]) for e in 1:nE]

dt       = d.delta_T
rest_cap = Int(round(d.t_limit_rest / dt))
rest_win = rest_cap + 1
scale    = d.scale
kappa    = d.kappa_wt
uniq_days = sort(unique(days))

fails = String[]
chk(c, m) = c ? nothing : push!(fails, m)

# REST RULE (per CEV, within each day)
for e in 1:nE, dy in uniq_days
    seq = acts[e][findall(==(dy), days)]
    for s in 1:(length(seq) - rest_win + 1)
        w = sum(is_work.(seq[s:s+rest_win-1]))
        chk(w <= rest_cap, "REST: CEV$e day$dy win@$s has $w work in $rest_win (cap $rest_cap)")
    end
end

# PRECEDENCE (per CEV, within each day)
for e in 1:nE, dy in uniq_days
    seq = acts[e][findall(==(dy), days)]; cdig = 0.0; cload = 0.0
    for (t, a) in enumerate(seq)
        is_dig(a) && (cdig += dt); is_load(a) && (cload += dt)
        chk(cload <= scale * cdig + 1e-6, "PREC: CEV$e day$dy t$t load $(round(cload,digits=2)) > $scale*dig $(round(scale*cdig,digits=2))")
    end
end

# TRAVEL PACING (two-sided band, per CEV, within each day)
for e in 1:nE, dy in uniq_days
    seq = acts[e][findall(==(dy), days)]; ctrv = 0; cwork = 0
    for (t, a) in enumerate(seq)
        is_trv(a) && (ctrv += 1); (is_dig(a) || is_load(a)) && (cwork += 1)
        chk(kappa * ctrv <= cwork + kappa && cwork <= kappa * ctrv + kappa,
            "PACE: CEV$e day$dy t$t kappa*trv=$(kappa*ctrv) work=$cwork out of band")
    end
end

# CEV SOE bounds (from trajectory soe_cev1/soe_cev2)
for e in 1:nE
    col = tj[!, Symbol("soe_cev$e")]
    lo = d.SOE_CEV_min[e]; hi = d.SOE_CEV_max[e]
    for (t, v) in enumerate(col)
        chk(lo - 1e-6 <= v <= hi + 1e-6, "SOE: CEV$e row$t soe=$(round(v,digits=2)) not in [$lo,$hi]")
    end
end

# MCS end-at-grid: last applied interval of each day is at a grid node (mcs_node in N_g)
tdays = day_of.(tj.clock)
for dy in uniq_days
    idx = findall(==(dy), tdays)
    node = tj.mcs_node[idx[end]]
    chk(node in d.N_g || node == 0, "MCS HOME: day$dy final mcs_node=$node not grid $(d.N_g)")
end

# PER-DAY work achieved vs schedule
has_perday = hasproperty(d, :dig_by_day)
site_of = Dict(e => findfirst(i -> d.A[i, e] == 1, d.N) for e in 1:nE)
println("\n---- PER-DAY WORK (achieved dig/load hours per site) ----")
for dy in uniq_days, e in 1:nE
    seq = acts[e][findall(==(dy), days)]
    adig = sum(is_dig.(seq)) * dt; aload = sum(is_load.(seq)) * dt
    i = site_of[e]
    tdig  = has_perday && dy <= length(d.dig_by_day)  ? d.dig_by_day[dy][i]  : d.hours_digging[i]
    tload = has_perday && dy <= length(d.load_by_day) ? d.load_by_day[dy][i] : d.hours_loading_swinging[i]
    @printf("  day%d CEV%d (site %d): dig %.2f/%.2f h, load %.2f/%.2f h\n", dy, e, i, adig, tdig, aload, tload)
end

println("\n================ LEGACY VALIDATION ($mode @ $out) ================")
if isempty(fails)
    println("ALL PER-ROW CONSTRAINTS PASSED  (rows: $(nrow(ws)), days: $(length(uniq_days)))")
else
    println("FAILURES: $(length(fails))")
    for m in first(fails, 30); println("  - ", m); end
end
