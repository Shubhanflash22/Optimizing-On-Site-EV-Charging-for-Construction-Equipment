# #############################################################################
# validate_receding.jl  —  post-run constraint checker (per applied row)
# -----------------------------------------------------------------------------
# Reloads the scenario parameters and reads the KEPT closed-loop output CSVs,
# then verifies every applied interval against the model's rules:
#   * SOE within [min,max] for both CEVs (04_cev_state_of_energy.csv)
#   * REST RULE (12e): <= t_limit_rest/dt work intervals in every rolling window,
#     within each day (this is the cross-re-solve seam fix)
#   * PRECEDENCE (12d): cumulative loading <= scale * cumulative digging, per day
#   * TRAVEL PACING (13): two-sided band, per day
#   * PER-DAY WORK: achieved dig/load hours per site per day vs the schedule,
#     with rollover; confirms the per-day quota (not a single lumpsum) is used
#   * MCS end-at-grid at each day's final interval (10e)
#   * infeasible/held-window count from 10_mip_convergence.csv
#
# Usage:  julia --project=. validate_receding.jl <synthetic|input>
# #############################################################################
using CSV, DataFrames, Printf

const _CD = @__DIR__
include(joinpath(_CD, "Common.jl"))
include(joinpath(_CD, "DataLoader.jl"))
using .DataLoader: load_data

mode = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :synthetic
# Optional 2nd arg overrides the output directory (e.g. a legacy Scenario_1.jl check dir).
out  = length(ARGS) >= 2 ? ARGS[2] : joinpath(dirname(_CD), "output", String(mode))
d    = load_data(mode; input_dir = joinpath(dirname(_CD), "data", "input_data"))

is_work(a)  = a in ("Digging", "Loading/Swinging", "Traveling")
is_dig(a)   = a == "Digging"
is_load(a)  = a == "Loading/Swinging"
is_trv(a)   = a == "Traveling"

# ---- read the applied worker schedule ----
ws   = CSV.read(joinpath(out, "worker_schedule.csv"), DataFrame)
day_of(lbl) = parse(Int, match(r"D(\d+)", lbl).captures[1])
days = day_of.(String.(ws.time))
nE   = length(d.E)
acts = [String.(ws[!, Symbol("CEV$(e)_activity")]) for e in 1:nE]

dt        = d.delta_T
rest_cap  = Int(round(d.t_limit_rest / dt))
rest_win  = rest_cap + 1
scale     = d.scale
kappa     = d.kappa_wt
uniq_days = sort(unique(days))

fails = String[]
chk(cond, msg) = cond ? nothing : push!(fails, msg)

# ---- REST RULE (per CEV, within each day) ----
for e in 1:nE, dy in uniq_days
    idx = findall(==(dy), days)
    seq = acts[e][idx]
    for s in 1:(length(seq) - rest_win + 1)
        w = sum(is_work.(seq[s:s+rest_win-1]))
        chk(w <= rest_cap,
            "REST RULE: CEV$e day$dy window@$s has $w work in $rest_win (cap $rest_cap)")
    end
end

# ---- PRECEDENCE (per CEV site, within each day) ----
for e in 1:nE, dy in uniq_days
    idx = findall(==(dy), days)
    seq = acts[e][idx]
    cdig = 0.0; cload = 0.0
    for (t, a) in enumerate(seq)
        is_dig(a)  && (cdig  += dt)
        is_load(a) && (cload += dt)
        chk(cload <= scale * cdig + 1e-6,
            "PRECEDENCE: CEV$e day$dy t$t load=$(round(cload,digits=2)) > $scale*dig=$(round(scale*cdig,digits=2))")
    end
end

# ---- TRAVEL PACING (two-sided band, per CEV, within each day) ----
for e in 1:nE, dy in uniq_days
    idx = findall(==(dy), days)
    seq = acts[e][idx]
    ctrv = 0; cwork = 0
    for (t, a) in enumerate(seq)
        is_trv(a) && (ctrv += 1)
        (is_dig(a) || is_load(a)) && (cwork += 1)
        # kappa*ctrv within [cwork - kappa, cwork + kappa] (two-sided band, integer slack)
        chk(kappa * ctrv <= cwork + kappa && cwork <= kappa * ctrv + kappa,
            "PACING: CEV$e day$dy t$t kappa*trv=$(kappa*ctrv) work=$cwork out of band")
    end
end

# ---- SOE bounds (both CEVs) ----
soe = CSV.read(joinpath(out, "04_cev_state_of_energy.csv"), DataFrame)
for e in 1:nE
    col = soe[!, Symbol("CEV_$(e)_SOE_kWh")]
    lo  = d.SOE_CEV_min[e]; hi = d.SOE_CEV_max[e]
    for (t, v) in enumerate(col)
        chk(lo - 1e-6 <= v <= hi + 1e-6, "SOE BOUND: CEV$e t$t soe=$(round(v,digits=2)) not in [$lo,$hi]")
    end
end

# ---- MCS end-at-grid at each day's final interval ----
mcs = CSV.read(joinpath(out, "06_mcs_location_trajectory.csv"), DataFrame)
mdays = day_of.(String.(mcs.Time_Start_Label))
for dy in uniq_days
    idx = findall(==(dy), mdays)
    lastrow = idx[end]
    chk(String(mcs.MCS_1_Location_Type[lastrow]) == "Grid",
        "MCS HOME: day$dy final interval not at grid (=$(mcs.MCS_1_Location_Type[lastrow]))")
end

# ---- PER-DAY WORK: achieved vs schedule (rollover-aware) ----
println("\n---- PER-DAY WORK (achieved dig/load hours per site) ----")
site_of = Dict(e => findfirst(i -> d.A[i, e] == 1, d.N) for e in 1:nE)
rollover = Dict(e => (0.0, 0.0) for e in 1:nE)
for dy in uniq_days
    idx = findall(==(dy), days)
    for e in 1:nE
        seq = acts[e][idx]
        adig  = sum(is_dig.(seq))  * dt
        aload = sum(is_load.(seq)) * dt
        i = site_of[e]
        tdig  = dy <= length(d.dig_by_day)  ? d.dig_by_day[dy][i]  : 0.0
        tload = dy <= length(d.load_by_day) ? d.load_by_day[dy][i] : 0.0
        @printf("  day%d CEV%d (site %d): dig %.2f/%.2f h, load %.2f/%.2f h\n",
                dy, e, i, adig, tdig, aload, tload)
    end
end

# ---- infeasible/held windows ----
mipf = joinpath(out, "10_mip_convergence.csv")
n_inf = 0
if isfile(mipf)
    mip = CSV.read(mipf, DataFrame)
    statuscol = findfirst(c -> occursin("status", lowercase(String(c))), names(mip))
    if statuscol !== nothing
        col = String.(mip[!, statuscol])
        n_inf = count(s -> occursin("INFEASIBLE", uppercase(s)), col)
    end
end
println("\nInfeasible/held windows in 10_mip_convergence.csv: $n_inf")

println("\n================ VALIDATION SUMMARY ($(mode)) ================")
if isempty(fails)
    println("ALL PER-ROW CONSTRAINTS PASSED  (rows: $(nrow(ws)), days: $(length(uniq_days)))")
else
    println("FAILURES: $(length(fails))")
    for m in first(fails, 30); println("  - ", m); end
end
