# #############################################################################
# Dummy stress-test harness for the (multi-day) RECEDING-Horizon MPC.
# Builds N dummy :input datasets (1 MCS, any number of CEVs), runs the closed
# loop on each (n_days = 1 kept day + 1 dropped buffer day), writes per-case
# input+output under Dummy/<case>/, and compiles one comprehensive comparison
# table -> Dummy/summary_table.html.
# #############################################################################

SCENARIO1_NO_AUTORUN = true
const _HERE = @__DIR__
const _CODE = normpath(joinpath(_HERE, "..", "code"))
include(joinpath(_CODE, "6_Receding_Horizon_main.jl"))
using DataFrames, CSV, Printf

# ---------------------------------------------------------------------------
# helpers to emit the 7 CSVs
# ---------------------------------------------------------------------------
clocklbl(hdec) = (h = mod(hdec, 24);
    m = Int(round(h * 60)) % (24 * 60);
    @sprintf("%d:%02d", div(m, 60), m % 60))

# per-interval label at the END boundary (row 1 = 8:15 -> t_start = 8.0)
labels(n) = [clocklbl(8 + 0.25 * k) for k in 1:n]

function price_series(profile, n)
    ps = Float64[]; co2 = Float64[]
    for k in 1:n
        hour = mod(8 + 0.25 * k, 24)
        p = profile === :flat  ? 0.20 :
            profile === :cheap ? 0.08 :
            profile === :spike ? (16 <= hour < 20 ? 1.50 : (7 <= hour < 16 ? 0.15 : 0.08)) :
                                 (16 <= hour < 21 ? 0.45 : (7 <= hour < 16 ? 0.18 : 0.10))  # :tou
        push!(ps, p); push!(co2, 0.30)
    end
    ps, co2
end

avail_windows(a) = a === :allday ? [(8.0, 18.0)] :
                   a === :late   ? [(13.0, 18.0)] :
                   a === :early  ? [(8.0, 13.0)] :
                                   [(8.0, 12.0), (14.0, 17.0)]   # :shift
function avail_cap(a, n, cap)
    W = avail_windows(a)
    [any(lo <= mod(8 + 0.25 * (k - 1), 24) < hi for (lo, hi) in W) ? cap : 0 for k in 1:n]
end

# write one dummy dataset directory from a case spec
function write_case(indir, c)
    mkpath(indir)
    n     = c.n_int
    lab   = labels(n)
    nsite = length(c.sites)
    ncev  = length(c.cevs)
    nodes = ["i$(i)" for i in 1:(nsite + 1)]      # i1 = grid, i2.. = sites
    evids = ["e$(e)" for e in 1:ncev]

    # parameters.csv
    open(joinpath(indir, "parameters.csv"), "w") do io
        println(io, "Parameter,Value")
        for (k, v) in (("k_trv", 10), ("rho_miss", 500), ("delta_T", 0.25),
                       ("p_digging", c.pow[1]), ("p_loading_swinging", c.pow[2]),
                       ("p_traveling", c.pow[3]), ("lambda_demand_NC", c.dnc),
                       ("lambda_demand_OP", c.dop), ("carbon_price_per_ton", 50),
                       ("rho_labor", 250), ("p_idling", c.pidle), ("scale", 2),
                       ("t_limit_rest", 1), ("prior_sigma_frac", 0.2),
                       ("obs_noise_std", 0.05), ("co2_unit_scale", 1))
            println(io, "$k,$v")
        end
    end

    # ev_data.csv
    open(joinpath(indir, "ev_data.csv"), "w") do io
        println(io, "id,SOE_min,SOE_max,SOE_ini,ch_rate")
        for (e, ev) in enumerate(c.cevs)
            println(io, "$(evids[e]),$(ev.smin),$(ev.smax),$(ev.sini),$(ev.ch)")
        end
    end

    # mcs_data.csv
    open(joinpath(indir, "mcs_data.csv"), "w") do io
        println(io, "id,SOE_min,SOE_max,SOE_ini,CH_MCS,DCH_MCS,C_MCS_plug,DCH_MCS_plug,eta_ch_dch")
        println(io, "m1,$(c.mcs.smin),$(c.mcs.smax),$(c.mcs.sini),$(c.mcs.ch),$(c.mcs.dch),$(c.mcs.plugs),$(c.mcs.dchplug),0.95")
    end

    # place.csv  (grid row all-zero; each site row has 1 for its CEVs)
    open(joinpath(indir, "place.csv"), "w") do io
        println(io, "site," * join(evids, ",") * ",hours_digging,hours_loading_swinging")
        println(io, nodes[1] * "," * join(fill(0, ncev), ",") * ",0,0")   # grid
        for (s, st) in enumerate(c.sites)
            row = [ (c.cevs[e].site == s ? 1 : 0) for e in 1:ncev ]
            println(io, nodes[s + 1] * "," * join(row, ",") * ",$(st.dig),$(st.load)")
        end
    end

    # travel_time.csv (uniform tau between distinct nodes)
    open(joinpath(indir, "travel_time.csv"), "w") do io
        println(io, "Node," * join(nodes, ","))
        for i in 1:length(nodes)
            row = [ i == j ? 0 : c.tau for j in 1:length(nodes) ]
            println(io, nodes[i] * "," * join(row, ","))
        end
    end

    # time_data.csv
    ps, co2 = price_series(c.price, n)
    open(joinpath(indir, "time_data.csv"), "w") do io
        println(io, "time,lambda_buy,intensity_tons_emissions")
        for k in 1:n
            println(io, "$(lab[k]),$(ps[k]),$(co2[k])")
        end
    end

    # work_flexible.csv  (one row per assigned site/CEV)
    open(joinpath(indir, "work_flexible.csv"), "w") do io
        println(io, "Location,EV," * join(lab, ","))
        for e in 1:ncev
            s   = c.cevs[e].site
            cap = avail_cap(get(c, :avail, :shift), n, 7)
            println(io, nodes[s + 1] * ",$(evids[e])," * join(cap, ","))
        end
    end
end

# ---------------------------------------------------------------------------
# case specs  (1 MCS everywhere; CEV count / sites / stress vary)
# ---------------------------------------------------------------------------
mcs_default   = (smin = 30, smax = 250, sini = 250, ch = 31.25, dch = 80, plugs = 2, dchplug = 52)
ev(; site = 1, smin = 5, smax = 100, sini = 50, ch = 30) = (; site, smin, smax, sini, ch)
site(dig, load) = (; dig, load)
defpow = (4.79, 3.16, 4.71)

cases = [
 (name = "C01_baseline_1cev", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5)], cevs = [ev(site = 1)]),

 (name = "C02_two_cev_two_sites", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5), site(2.5, 1)], cevs = [ev(site = 1), ev(site = 2)]),

 (name = "C03_three_cev_three_sites", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5), site(2, 1), site(2.5, 2)],
  cevs = [ev(site = 1), ev(site = 2), ev(site = 3)]),

 (name = "C04_two_cev_shared_site", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(4, 2)], cevs = [ev(site = 1), ev(site = 1)]),

 (name = "C05_heavy_work_low_batt", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(6, 4)], cevs = [ev(site = 1, smax = 25, sini = 20, ch = 20)]),

 (name = "C06_tiny_battery_shuttle", n_int = 96, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5), site(3, 1.5)],
  cevs = [ev(site = 1, smax = 20, sini = 15, ch = 15), ev(site = 2, smax = 20, sini = 15, ch = 15)]),

 (name = "C07_long_travel", n_int = 96, price = :tou, avail = :shift, tau = 4, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5), site(2.5, 1)], cevs = [ev(site = 1), ev(site = 2)]),

 (name = "C08_price_spike_peak", n_int = 96, price = :spike, avail = :late, tau = 1, pidle = 0.0,
  dnc = 20, dop = 60, pow = defpow, mcs = mcs_default,
  sites = [site(3, 1.5), site(2.5, 1)], cevs = [ev(site = 1), ev(site = 2)]),

 (name = "C09_allday_work_reststress", n_int = 96, price = :tou, avail = :allday, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(5, 3), site(5, 3)], cevs = [ev(site = 1), ev(site = 2)]),

 (name = "C10_short_day_12h", n_int = 48, price = :tou, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 20, dop = 20, pow = defpow, mcs = mcs_default,
  sites = [site(2, 1), site(2, 1)], cevs = [ev(site = 1), ev(site = 2)]),

 (name = "C11_cheap_flat_easy", n_int = 96, price = :cheap, avail = :shift, tau = 1, pidle = 0.0,
  dnc = 10, dop = 10, pow = defpow, mcs = mcs_default,
  sites = [site(2, 1), site(2, 1), site(2, 1)],
  cevs = [ev(site = 1), ev(site = 2), ev(site = 3)]),
]

# ---------------------------------------------------------------------------
# HTML-only output: the replan-grid HTML files EXACTLY as the pipeline writes
# them (Output._write_replan_grid_html), no PNGs and no CSVs.
# ---------------------------------------------------------------------------
function html_outputs(res, outdir)
    d = res.d; nKd = res.nK_day
    for day in 1:res.n_days_keep
        g = res.replan_by_day[day]
        gdir = joinpath(outdir, "replan_grids", "day$(day)"); mkpath(gdir)
        Output._write_replan_grid_html(joinpath(gdir, "plan_grid_kW.html"), g.plan_grid_kW, res, nKd)
        Output._write_replan_grid_html(joinpath(gdir, "plan_mcs_soe.html"), g.plan_mcs_soe, res, nKd)
        Output._write_replan_grid_html(joinpath(gdir, "plan_mcs_activity.html"), g.plan_mcs_act, res, nKd)
        for e in d.E
            Output._write_replan_grid_html(joinpath(gdir, "plan_cev$(e)_soe.html"),      g.plan_cev_soe[e], res, nKd)
            Output._write_replan_grid_html(joinpath(gdir, "plan_cev$(e)_activity.html"), g.plan_cev_act[e], res, nKd)
        end
    end
end

# Per-case interval-by-interval comparison of the APPLIED plan (grid diagonal):
# every CEV's activity next to the MCS status, so "Charging" (CEV) always lines up
# with "Serving CEV" (MCS). Writes comparison.html per case and appends a compact
# grouped view (consecutive identical rows collapsed) to cmp_io.
function write_comparison(res, outdir, name, cmp_io)
    d = res.d; nK = res.nK; nKd = res.nK_day; labels = res.time_labels
    clock(k)    = k <= length(labels) ? labels[k] : string(k)
    # Map a concatenated kept-interval k to its (day, within-day) diagonal cell.
    cevlab(e,k) = (day = div(k-1, nKd)+1; kl = mod1(k, nKd);
                   v = res.replan_by_day[day].plan_cev_act[e][kl,kl]; isempty(v) ? "-" : v)
    mcslab(k)   = (day = div(k-1, nKd)+1; kl = mod1(k, nKd);
                   v = res.replan_by_day[day].plan_mcs_act[kl,kl];    isempty(v) ? "-" : v)

    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>",
        "body{font-family:sans-serif;margin:16px}",
        "table{border-collapse:collapse;font-size:12px}",
        "th,td{border:1px solid #ccc;padding:3px 8px;text-align:center;white-space:nowrap}",
        "th{background:#f4f4f4}td:first-child{color:#888}",
        ".chg{background:#fff3cd;font-weight:bold}.grid{background:#d1e7dd}",
        "</style></head><body>")
    println(io, "<h2>", name, " \u2014 per-interval activity: CEV(s) vs MCS (applied plan)</h2>")
    print(io, "<table><tr><th>#</th><th>Clock</th>")
    for e in d.E; print(io, "<th>CEV", e, "</th>"); end
    println(io, "<th>MCS</th></tr>")
    for k in 1:nK
        print(io, "<tr><td>", k, "</td><td>", clock(k), "</td>")
        for e in d.E
            v = cevlab(e,k); cls = v == "Charging" ? " class=\"chg\"" : ""
            print(io, "<td", cls, ">", v, "</td>")
        end
        mv  = mcslab(k)
        cls = mv == "Serving CEV" ? " class=\"chg\"" : (mv == "Charging (grid)" ? " class=\"grid\"" : "")
        println(io, "<td", cls, ">", mv, "</td></tr>")
    end
    println(io, "</table></body></html>")
    write(joinpath(outdir, "comparison.html"), String(take!(io)))

    # ---- grouped text (collapse consecutive identical rows) ----
    println(cmp_io, "#################### ", name, " ####################")
    println(cmp_io, "  #      Clock  ", join(["CEV$e" for e in d.E], " | "), " || MCS")
    emit(k1,k2,cevs,mcs) = begin
        rng = k1 == k2 ? @sprintf("%3d    ", k1) : @sprintf("%3d-%-3d", k1, k2)
        println(cmp_io, @sprintf("%-8s %-6s %s || %s", rng, clock(k1), join(cevs, " | "), mcs))
    end
    prevkey = nothing; startk = 1
    for k in 1:nK
        key = ([cevlab(e,k) for e in d.E], mcslab(k))
        if prevkey === nothing
            prevkey = key; startk = k
        elseif key != prevkey
            emit(startk, k-1, prevkey[1], prevkey[2]); prevkey = key; startk = k
        end
    end
    prevkey !== nothing && emit(startk, nK, prevkey[1], prevkey[2])
    println(cmp_io, "")
end

# ---------------------------------------------------------------------------
# PHASE 1: generate ALL 11 input datasets first
# ---------------------------------------------------------------------------
println("==================== PHASE 1: writing 11 input datasets ====================")
for c in cases
    write_case(joinpath(_HERE, c.name, "input"), c)
    println(@sprintf("  input ready: %-28s  %d CEV(s), %d site(s), %d intervals",
                     c.name, length(c.cevs), length(c.sites), c.n_int))
end

# ---------------------------------------------------------------------------
# PHASE 2: run each case one by one, collect the comprehensive row
# ---------------------------------------------------------------------------
println("\n==================== PHASE 2: running 11 cases ====================")
rows = DataFrame()
cmp_io = IOBuffer()
for c in cases
    indir  = joinpath(_HERE, c.name, "input")
    outdir = joinpath(_HERE, c.name, "output")
    println("\n#################### $(c.name) ####################")
    d   = load_data(:input; input_dir = indir)
    res = run_mpc(d; n_days = 1, time_limit_sec = 8.0, seed = 1)
    html_outputs(res, outdir)
    write_comparison(res, outdir, c.name, cmp_io)

    dt      = d.delta_T
    eta     = d.eta_ch_dch[1]
    mcs_ini = d.SOE_MCS_ini[1];  mcs_end = res.soe_mcs_end[1]
    cev_ini = sum(d.SOE_CEV_ini); cev_end = sum(res.soe_cev_end)
    overchg = sum(max.(res.soe_cev_end .- d.SOE_CEV_ini, 0.0))
    grid    = res.total_energy
    dch     = sum(res.log.dch_kW) * dt
    consumed = dch - overchg

    e_cost   = res.total_cost
    c_cost   = d.carbon_price_per_ton / 1000 * res.total_co2
    nc_cost  = d.lambda_demand_NC * res.nc_peak
    op_cost  = d.lambda_demand_OP * res.op_peak
    miss_cost = d.rho_miss * res.missed
    lab_cost = res.labour_cost
    total    = e_cost + c_cost + nc_cost + op_cost + miss_cost + lab_cost

    avg_price   = grid > 1e-9 ? e_cost / grid : 0.0
    oc_grideq   = overchg / eta^2
    oc_cost     = oc_grideq * avg_price
    exact_total = total - oc_cost

    push!(rows, (
        Case              = c.name,
        CEVs              = length(d.E),
        Sites             = length(d.N_c),
        Horizon_h         = round(d.n_day * dt, digits = 1),
        MCS_start_kWh     = round(mcs_ini, digits = 1),
        MCS_end_kWh       = round(mcs_end, digits = 1),
        CEV_start_kWh     = round(cev_ini, digits = 1),
        CEV_target_kWh    = round(cev_ini, digits = 1),
        CEV_end_kWh       = round(cev_end, digits = 1),
        Overcharge_kWh    = round(overchg, digits = 2),
        Grid_energy_kWh   = round(grid, digits = 2),
        Delivered_kWh     = round(dch, digits = 2),
        Consumed_kWh      = round(consumed, digits = 2),
        Energy_cost       = round(e_cost, digits = 2),
        Carbon_cost       = round(c_cost, digits = 2),
        NC_charge         = round(nc_cost, digits = 2),
        OP_charge         = round(op_cost, digits = 2),
        Missed_penalty    = round(miss_cost, digits = 2),
        Labour_cost       = round(lab_cost, digits = 2),
        Total_cost        = round(total, digits = 2),
        Overcharge_cost   = round(oc_cost, digits = 2),
        Cost_if_exact     = round(exact_total, digits = 2),
        Missed_work_h     = round(res.missed, digits = 2),
        Transit_h         = round(res.transit_intervals * dt, digits = 2),
        Infeasible_wins   = res.n_infeasible,
        Loop_time_s       = round(res.elapsed, digits = 1),
    ); promote = true)
end

# ---------------------------------------------------------------------------
# write the one BIG comparison table as HTML (no CSV)
# ---------------------------------------------------------------------------
function write_summary_html(path, df)
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>")
    println(io, "body{font-family:sans-serif;margin:16px}")
    println(io, "h2{margin:4px 0}")
    println(io, "table{border-collapse:collapse;font-size:12px}")
    println(io, "th,td{border:1px solid #ccc;padding:3px 7px;text-align:right;white-space:nowrap}")
    println(io, "th{background:#f4f4f4;text-align:center}")
    println(io, "td:first-child,th:first-child{text-align:left}")
    println(io, "tr:nth-child(even){background:#fafafa}")
    println(io, "</style></head><body>")
    println(io, "<h2>Receding-Horizon MPC \u2014 dummy stress cases (1 MCS, varying CEVs; 1 kept day)</h2>")
    println(io, "<p style=\"font-size:12px;color:#555\">Start\u2192end charge, energy flow, cost breakdown, and the ",
                "overcharge (\u201ccharged extra\u201d) accounting for every case. ",
                "<b>Cost_if_exact</b> = total minus the estimated cost of the energy left in CEVs above target.</p>")
    println(io, "<table><tr>")
    for n in names(df); print(io, "<th>", n, "</th>"); end
    println(io, "</tr>")
    for r in 1:nrow(df)
        println(io, "<tr>")
        for n in names(df); print(io, "<td>", df[r, n], "</td>"); end
        println(io, "</tr>")
    end
    println(io, "</table></body></html>")
    write(path, String(take!(io)))
end

write(joinpath(_HERE, "comparisons_grouped.txt"), String(take!(cmp_io)))
write_summary_html(joinpath(_HERE, "summary_table.html"), rows)
println("\n==================== SUMMARY ====================")
show(rows, allrows = true, allcols = true, truncate = 0)
println("\n\nWrote: ", joinpath(_HERE, "summary_table.html"))
