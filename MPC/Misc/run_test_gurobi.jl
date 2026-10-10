# =============================================================================
# run_test_gurobi.jl
# -----------------------------------------------------------------------------
# Copy of run_test.jl in which all three approaches solve with GUROBI.
# Put this file in:   C:\Users\shubh\Desktop\MPC\Test\
#
# Needs, in each approach's code folder:
#   * mcs_model_garobi.jl                    (Gurobi version of 3_MCSModel.jl)
#   * the MCS_MODEL_FILE line in the main script (6_OneShot_main.jl for
#     Approach 0, 6_Shrinking_Horizon_main.jl for Approaches 1 and 2)
#
# What it does
#   1. Runs Approach 0, Approach 1 and Approach 2 once each, with the same
#      seed, on the recorded live data (mode = :live_data), with a solver time
#      limit of 600 s per MILP solve. Each approach runs in its own Julia
#      process, because the three code bases use the same module names.
#   2. Writes each approach's outputs into this Test folder:
#         Test\A0_gurobi_output\   Test\A1_gurobi_output\   Test\A2_gurobi_output\
#      These are NEW folders, so the HiGHS results of run_test.jl
#      (A0_output, A1_output, A2_output) are not touched.
#   3. Reads those outputs and writes one comparison page:
#         Test\comparison_gurobi.html
#
# How to run (Julia REPL or VS Code Run button):
#   include("C:/Users/shubh/Desktop/MPC/Test/run_test_gurobi.jl")
#
# Notes
#   * Each approach reads its own  Approach N\data\input_data  folder. The page
#     checks that the input files are identical across the three approaches.
#   * TIME_LIMIT_SEC is the limit for EACH MILP solve. Approach 0 makes one
#     solve per day; Approaches 1 and 2 make one solve per 15-minute interval.
#   * Set RUN_SIMULATIONS = false to rebuild comparison_gurobi.html from the
#     existing output folders without re-running anything.
#   * Each run deletes and rewrites its own A#_gurobi_output folder only.
# =============================================================================

using Dates
using Printf
using SHA
using Statistics
using CSV
using DataFrames

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
const TEST_DIR = @__DIR__
const ROOT     = dirname(TEST_DIR)          # ...\MPC

const MODE            = :live_data
const SEED            = 1
const TIME_LIMIT_SEC  = 600.0               # per MILP solve, in seconds
const N_SCENARIOS     = 5                   # Approach 2 only
const N_DAY_RUN       = 1
const RUN_SIMULATIONS = true

const APPROACHES = [
    (tag = "A0", name = "Approach 0 (one-shot)", dir = joinpath(ROOT, "Approach 0"),
     main = "6_OneShot_main.jl", fn = "run_scenario_0", flag = "SCENARIO0_NO_AUTORUN"),
    (tag = "A1", name = "Approach 1 (CE-MPC)", dir = joinpath(ROOT, "Approach 1"),
     main = "6_Shrinking_Horizon_main.jl", fn = "run_scenario_1", flag = "SCENARIO1_NO_AUTORUN"),
    (tag = "A2", name = "Approach 2 (SB-MPC)", dir = joinpath(ROOT, "Approach 2"),
     main = "6_Shrinking_Horizon_main.jl", fn = "run_scenario_1", flag = "SCENARIO1_NO_AUTORUN"),
]

const SERIES_COLORS = Dict("A0" => "#8b95a5", "A1" => "#2563eb", "A2" => "#e5484d")

const COST_COMPONENTS = [
    ("Total_Energy_Cost_USD",          "Energy",            "#2563eb"),
    ("Total_CO2_Cost_USD",             "CO2",               "#14a38b"),
    ("NC_demand_charge_USD",           "NC demand charge",  "#8b5cf6"),
    ("OP_demand_charge_USD",           "OP demand charge",  "#d97706"),
    ("Missed_Work_Penalty_USD",        "Missed work",       "#e5484d"),
    ("Travel_Labour_USD",              "Travel labour",     "#64748b"),
    ("Terminal_Shortfall_Penalty_USD", "Terminal shortfall", "#a16207"),
]

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
_esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

out_dir(a) = joinpath(TEST_DIR, a.tag * "_gurobi_output")

function fmtnum(x)
    isfinite(x) || return "n/a"
    if x == round(x) && abs(x) < 1e9
        return @sprintf("%d", round(Int, x))
    end
    return abs(x) >= 100 ? @sprintf("%.1f", x) : @sprintf("%.2f", x)
end

# Returns the column `name` of df as a Float64 vector (missing becomes NaN), or an empty vector if the column is absent.
function colvec(df, name)
    df === nothing && return Float64[]
    (string(name) in names(df)) || return Float64[]
    return Float64[ismissing(x) ? NaN : Float64(x) for x in df[!, name]]
end

# ----------------------------------------------------------------------------
# Step 1: run the three approaches, each in its own Julia process
# ----------------------------------------------------------------------------
function runner_source(a)
    main = joinpath(a.dir, "code", a.main)
    out  = out_dir(a)
    extra = a.tag == "A0" ? "detailed_output = true" :
            a.tag == "A1" ? "detailed_output = true,\n    important_only = false" :
                            "detailed_output = true,\n    important_only = false,\n    n_scenarios = $(N_SCENARIOS)"
    return """
redirect_stderr(stdout)
$(a.flag) = true
MCS_MODEL_FILE = "mcs_model_garobi.jl"
include($(repr(main)))
isdefined(MCSModel, :GRB_ENV) || error("Gurobi model NOT loaded: add the MCS_MODEL_FILE line to $(a.main) and check that mcs_model_garobi.jl is in the code folder. Stopping so a HiGHS run is not mistaken for a Gurobi run.")
$(a.fn)(
    mode = $(repr(MODE)),
    time_limit_sec = $(TIME_LIMIT_SEC),
    seed = $(SEED),
    n_day_run = $(N_DAY_RUN),
    out_dir = $(repr(out)),
    $extra,
)
"""
end

function run_one(a)
    runner = joinpath(TEST_DIR, "_runner_" * a.tag * ".jl")
    write(runner, runner_source(a))
    out = out_dir(a)
    rm(out; force = true, recursive = true)
    mkpath(out)

    jl   = Base.julia_cmd()
    proj = Base.active_project()
    cmd  = proj === nothing ? `$jl $runner` : `$jl --project=$(dirname(proj)) $runner`

    t0 = time()
    ok = true
    try
        open(joinpath(out, a.tag * "_console.txt"), "w") do lf
            open(cmd) do p
                for line in eachline(p)
                    println(line)
                    println(lf, line)
                    flush(lf)
                end
            end
        end
    catch err
        ok = false
        @warn "$(a.name) failed. The comparison page will show it as failed." exception = err
    finally
        rm(runner; force = true)
    end
    return (ok = ok, secs = time() - t0)
end

# ----------------------------------------------------------------------------
# Step 2: read the outputs
# ----------------------------------------------------------------------------
function load_df(a, suffix)
    p = joinpath(out_dir(a), a.tag * "_" * suffix * ".csv")
    isfile(p) || return nothing
    try
        return CSV.read(p, DataFrame)
    catch err
        @warn "Could not read $p" exception = err
        return nothing
    end
end

function load_kpis(a)
    df = load_df(a, "kpi_summary")
    df === nothing && return nothing
    cols = names(df)
    ("Metric" in cols && length(cols) >= 2) || return nothing
    vcol = "Value" in cols ? "Value" : first(filter(c -> c != "Metric", cols))
    vals  = Dict{String,Float64}()
    order = String[]
    for r in eachrow(df)
        k = String(r.Metric)
        push!(order, k)
        v = r[vcol]
        vals[k] = ismissing(v) ? NaN :
                  (v isa Number ? Float64(v) : something(tryparse(Float64, string(v)), NaN))
    end
    return (vals = vals, order = order)
end

function input_hashes(a)
    d = joinpath(a.dir, "data", "input_data")
    h = Dict{String,String}()
    isdir(d) || return h
    for f in sort(readdir(d))
        p = joinpath(d, f)
        # posterior_draws.csv exists only in Approach 2 and is not an input of the others, so it is left out of the check
        if isfile(p) && endswith(lowercase(f), ".csv") && f != "posterior_draws.csv"
            h[f] = bytes2hex(sha256(read(p)))[1:8]
        end
    end
    return h
end

# ----------------------------------------------------------------------------
# Step 3: SVG charts (no external libraries)
# ----------------------------------------------------------------------------
function nice_axis(lo, hi)
    if !(hi > lo)
        hi = lo + 1.0
    end
    raw  = (hi - lo) / 5
    mag  = 10.0^floor(log10(raw))
    r    = raw / mag
    mult = r <= 1 ? 1.0 : (r <= 2 ? 2.0 : (r <= 5 ? 5.0 : 10.0))
    step = mult * mag
    lo2  = floor(lo / step) * step
    hi2  = ceil(hi / step) * step
    if hi2 <= lo2
        hi2 = lo2 + step
    end
    return lo2, hi2, step
end

# One chart with several lines. series is a vector of (name, color, values).
function svg_lines(title, ylabel, labels, series)
    w, h = 760, 300
    ml, mr, mt, mb = 58, 16, 40, 38
    pw, ph = w - ml - mr, h - mt - mb

    allv = Float64[]
    n = 0
    for s in series
        n = max(n, length(s[3]))
        for v in s[3]
            isfinite(v) && push!(allv, v)
        end
    end
    if isempty(allv) || n == 0
        return "<p class=\"muted\">No data for " * _esc(title) * ".</p>"
    end
    lo, hi = minimum(allv), maximum(allv)
    lo >= 0 && (lo = 0.0)
    lo2, hi2, step = nice_axis(lo, hi)

    xpos(i) = ml + (n == 1 ? 0.0 : (i - 1) / (n - 1) * pw)
    ypos(v) = mt + ph - (v - lo2) / (hi2 - lo2) * ph

    io = IOBuffer()
    println(io, "<svg viewBox=\"0 0 $w $h\" role=\"img\" aria-label=\"", _esc(title), "\">")
    println(io, "<text x=\"$ml\" y=\"16\" class=\"ctitle\">", _esc(title), "</text>")

    t = lo2
    while t <= hi2 + step * 1e-6
        y = ypos(t)
        println(io, @sprintf("<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" class=\"grid\"/>", ml, y, w - mr, y))
        println(io, @sprintf("<text x=\"%d\" y=\"%.1f\" class=\"tick\" text-anchor=\"end\">%s</text>", ml - 6, y + 4, fmtnum(t)))
        t += step
    end

    stride = max(1, round(Int, n / 12))
    for i in 1:stride:n
        lab = (!isempty(labels) && i <= length(labels)) ? labels[i] : string(i)
        println(io, @sprintf("<text x=\"%.1f\" y=\"%d\" class=\"tick\" text-anchor=\"middle\">%s</text>", xpos(i), h - mb + 16, _esc(lab)))
    end
    println(io, @sprintf("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/>", ml, mt + ph, w - mr, mt + ph))
    println(io, @sprintf("<text transform=\"rotate(-90 14,%.1f)\" x=\"14\" y=\"%.1f\" class=\"tick\" text-anchor=\"middle\">%s</text>", mt + ph / 2, mt + ph / 2, _esc(ylabel)))

    for s in series
        d = IOBuffer()
        pen = false
        for (i, v) in enumerate(s[3])
            if isfinite(v)
                print(d, pen ? "L" : "M", @sprintf("%.1f,%.1f ", xpos(i), ypos(v)))
                pen = true
            else
                pen = false
            end
        end
        println(io, "<path d=\"", String(take!(d)), "\" fill=\"none\" stroke=\"", s[2], "\" stroke-width=\"1.8\" stroke-linejoin=\"round\"/>")
    end

    lx = w - mr
    for s in reverse(series)
        println(io, @sprintf("<text x=\"%d\" y=\"16\" class=\"tick\" text-anchor=\"end\">%s</text>", lx, _esc(s[1])))
        lx -= 8 * length(s[1]) + 34
        println(io, @sprintf("<line x1=\"%d\" y1=\"12\" x2=\"%d\" y2=\"12\" stroke=\"%s\" stroke-width=\"3\"/>", lx + 8 * length(s[1]) + 6, lx + 8 * length(s[1]) + 22, s[2]))
    end
    println(io, "</svg>")
    return String(take!(io))
end

# Stacked bars of the cost components, one bar per approach.
function svg_cost_bars(avail)
    w, h = 760, 340
    ml, mr, mt, mb = 64, 170, 40, 40
    pw, ph = w - ml - mr, h - mt - mb
    if isempty(avail)
        return "<p class=\"muted\">No KPI data to chart.</p>"
    end
    fv(d, k) = (x = get(d, k, 0.0); isfinite(x) ? max(x, 0.0) : 0.0)
    totals = [sum(fv(a.vals, c[1]) for c in COST_COMPONENTS) for a in avail]
    ymax = maximum(totals)
    ymax > 0 || (ymax = 1.0)
    lo2, hi2, step = nice_axis(0.0, ymax)
    ypos(v) = mt + ph - (v - lo2) / (hi2 - lo2) * ph

    io = IOBuffer()
    println(io, "<svg viewBox=\"0 0 $w $h\" role=\"img\" aria-label=\"Cost breakdown\">")
    println(io, "<text x=\"$ml\" y=\"16\" class=\"ctitle\">Cost breakdown (USD)</text>")
    t = lo2
    while t <= hi2 + step * 1e-6
        y = ypos(t)
        println(io, @sprintf("<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" class=\"grid\"/>", ml, y, ml + pw, y))
        println(io, @sprintf("<text x=\"%d\" y=\"%.1f\" class=\"tick\" text-anchor=\"end\">%s</text>", ml - 6, y + 4, fmtnum(t)))
        t += step
    end
    n = length(avail)
    slot = pw / n
    bw = min(110.0, slot * 0.55)
    for (j, a) in enumerate(avail)
        x = ml + (j - 1) * slot + (slot - bw) / 2
        acc = 0.0
        for c in COST_COMPONENTS
            v = fv(a.vals, c[1])
            v > 0 || continue
            y1 = ypos(acc + v)
            y0 = ypos(acc)
            println(io, @sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\"><title>%s: %s</title></rect>", x, y1, bw, y0 - y1, c[3], _esc(c[2]), fmtnum(v)))
            acc += v
        end
        println(io, @sprintf("<text x=\"%.1f\" y=\"%.1f\" class=\"tick\" text-anchor=\"middle\">%s</text>", x + bw / 2, ypos(acc) - 6, fmtnum(totals[j])))
        println(io, @sprintf("<text x=\"%.1f\" y=\"%d\" class=\"tick\" text-anchor=\"middle\">%s</text>", x + bw / 2, h - mb + 18, _esc(a.tag)))
    end
    println(io, @sprintf("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/>", ml, mt + ph, ml + pw, mt + ph))
    ly = mt
    for c in COST_COMPONENTS
        println(io, @sprintf("<rect x=\"%d\" y=\"%d\" width=\"12\" height=\"12\" fill=\"%s\"/>", ml + pw + 16, ly, c[3]))
        println(io, @sprintf("<text x=\"%d\" y=\"%d\" class=\"tick\">%s</text>", ml + pw + 34, ly + 10, _esc(c[2])))
        ly += 20
    end
    println(io, "</svg>")
    return String(take!(io))
end

# ----------------------------------------------------------------------------
# Step 4: build comparison.html
# ----------------------------------------------------------------------------
const CSS = """
:root { --bg:#ffffff; --fg:#1b2430; --muted:#5b6675; --line:#d9dee5; --card:#f6f8fa; --best:#e3f5e8; --warn:#fdecea; --ok:#1a7f4b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f141a; --fg:#e6ebf1; --muted:#9aa6b5; --line:#2a3340; --card:#161d26; --best:#173a27; --warn:#3a1b1b; --ok:#4cc38a; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width:980px; margin:0 auto; padding:24px 16px 56px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:18px; margin:36px 0 10px; }
.muted { color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:6px 10px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:var(--card); position:sticky; top:0; }
td.best { background:var(--best); font-weight:600; }
td.bad { background:var(--warn); }
.d { color:var(--muted); font-size:12px; font-weight:400; }
.scroll { overflow-x:auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:10px 0; }
svg { width:100%; height:auto; display:block; margin:6px 0 14px; }
svg .grid { stroke:var(--line); stroke-width:1; }
svg .axis { stroke:var(--muted); stroke-width:1; }
svg .tick { fill:var(--muted); font-size:11px; }
svg .ctitle { fill:var(--fg); font-size:13px; font-weight:600; }
.ok { color:var(--ok); font-weight:600; }
.fail { color:#e5484d; font-weight:600; }
"""

function build_html(results)
    kp = Dict{String,Any}()
    logs = Dict{String,Any}()
    solves = Dict{String,Any}()
    for a in APPROACHES
        kp[a.tag]     = load_kpis(a)
        logs[a.tag]   = load_df(a, "interval_log")
        solves[a.tag] = load_df(a, "solve_log")
    end

    io = IOBuffer()
    println(io, "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">")
    println(io, "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    println(io, "<title>Approach comparison (Gurobi)</title><style>", CSS, "</style></head><body><main>")
    println(io, "<h1>Approach 0, 1 and 2: comparison, all solved with Gurobi</h1>")
    println(io, "<p class=\"muted\">Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Mode ", _esc(MODE), ", seed ", SEED, ", ", N_DAY_RUN, " day, solver time limit ",
            fmtnum(TIME_LIMIT_SEC), " s per solve, ", N_SCENARIOS, " scenarios for Approach 2.</p>")

    # --- run summary
    println(io, "<h2>Runs</h2><div class=\"scroll\"><table><thead><tr><th>Approach</th><th>Status</th><th>Wall time (min)</th><th>Output folder</th></tr></thead><tbody>")
    for a in APPROACHES
        r = get(results, a.tag, (ok = false, secs = NaN))
        has_out = kp[a.tag] !== nothing
        status = r.ok && has_out ? "<span class=\"ok\">finished</span>" :
                 (has_out ? "<span class=\"fail\">error, partial output</span>" : "<span class=\"fail\">no output</span>")
        mins = isfinite(r.secs) ? @sprintf("%.1f", r.secs / 60) : "n/a"
        println(io, "<tr><td>", _esc(a.name), "</td><td>", status, "</td><td>", mins, "</td><td>", _esc(a.tag * "_gurobi_output"), "</td></tr>")
    end
    println(io, "</tbody></table></div>")

    # --- input consistency
    hashes = Dict(a.tag => input_hashes(a) for a in APPROACHES)
    allfiles = sort(collect(union([Set(keys(hashes[a.tag])) for a in APPROACHES]...)))
    differing = String[]
    for f in allfiles
        hs = [get(hashes[a.tag], f, "missing") for a in APPROACHES]
        length(unique(hs)) > 1 && push!(differing, f)
    end
    println(io, "<h2>Input check</h2><div class=\"card\">")
    if isempty(allfiles)
        println(io, "Could not find the input_data folders.")
    elseif isempty(differing)
        println(io, "<span class=\"ok\">All ", length(allfiles), " input files are identical across the three approaches.</span>")
    else
        println(io, "<span class=\"fail\">These input files differ between approaches:</span> ", join(_esc.(differing), ", "),
                ". Differences in these files can explain differences in the results.")
    end
    println(io, "</div>")

    # --- KPI table
    first_kpi = nothing
    for a in APPROACHES
        k = kp[a.tag]
        k === nothing && continue
        (first_kpi === nothing || length(k.order) > length(first_kpi.order)) && (first_kpi = k)
    end
    println(io, "<h2>KPI summary</h2>")
    if first_kpi === nothing
        println(io, "<p class=\"muted\">No KPI files were found.</p>")
    else
        println(io, "<p class=\"muted\">Green cell: lowest value in the row. Percentages are relative to Approach 0.</p>")
        println(io, "<div class=\"scroll\"><table><thead><tr><th>Metric</th>")
        for a in APPROACHES
            println(io, "<th>", _esc(a.name), "</th>")
        end
        println(io, "</tr></thead><tbody>")
        for m in first_kpi.order
            vs = Float64[kp[a.tag] === nothing ? NaN : get(kp[a.tag].vals, m, NaN) for a in APPROACHES]
            fin = filter(isfinite, vs)
            best = (length(fin) > 1 && minimum(fin) != maximum(fin)) ? minimum(fin) : NaN
            print(io, "<tr><td>", _esc(m), "</td>")
            for (j, a) in enumerate(APPROACHES)
                v = vs[j]
                cls = (isfinite(best) && isfinite(v) && v == best) ? " class=\"best\"" : ""
                print(io, "<td", cls, ">", fmtnum(v))
                if j > 1 && isfinite(v) && isfinite(vs[1]) && vs[1] != 0
                    print(io, " <span class=\"d\">", @sprintf("%+.1f%%", (v - vs[1]) / abs(vs[1]) * 100), "</span>")
                end
                print(io, "</td>")
            end
            println(io, "</tr>")
        end
        println(io, "</tbody></table></div>")
    end

    # --- cost bars
    avail = [(tag = a.tag, vals = kp[a.tag].vals) for a in APPROACHES if kp[a.tag] !== nothing]
    println(io, "<h2>Cost breakdown</h2>", svg_cost_bars(avail))

    # --- time series
    labels = String[]
    for a in APPROACHES
        lg = logs[a.tag]
        if lg !== nothing && ("clock" in names(lg))
            labels = String[string(x) for x in lg.clock]
            break
        end
    end
    function make_series(colname)
        out = Any[]
        for a in APPROACHES
            v = colvec(logs[a.tag], colname)
            isempty(v) || push!(out, (a.tag, SERIES_COLORS[a.tag], v))
        end
        return out
    end
    println(io, "<h2>Time series</h2>")
    println(io, svg_lines("Grid charging power (all MCSs)", "kW", labels, make_series("grid_kW")))
    println(io, svg_lines("MCS discharge power to CEVs", "kW", labels, make_series("dch_kW")))
    println(io, svg_lines("CEV work power", "kW", labels, make_series("work_kW")))
    soe_cols = String[]
    for a in APPROACHES
        lg = logs[a.tag]
        lg === nothing && continue
        for n in names(lg)
            if occursin(r"^soe_(mcs|cev)\d+$", n) && !(n in soe_cols)
                push!(soe_cols, n)
            end
        end
    end
    sort!(soe_cols)
    for c in soe_cols
        kind = startswith(c, "soe_mcs") ? "MCS " : "CEV "
        idx  = replace(c, r"^soe_(mcs|cev)" => "")
        println(io, svg_lines("State of energy, " * kind * idx, "kWh", labels, make_series(c)))
    end

    # --- solver statistics
    println(io, "<h2>Solver statistics</h2><div class=\"scroll\"><table><thead><tr><th>Approach</th><th>Solves</th><th>Total solve time (s)</th><th>Mean (s)</th><th>Max (s)</th><th>Max MIP gap (%)</th><th>Non-optimal solves</th></tr></thead><tbody>")
    for a in APPROACHES
        sl = solves[a.tag]
        if sl === nothing
            println(io, "<tr><td>", _esc(a.name), "</td><td colspan=\"6\" class=\"muted\">no solve log</td></tr>")
            continue
        end
        st = filter(isfinite, colvec(sl, "solve_time_s"))
        gp = filter(isfinite, colvec(sl, "gap_percent"))
        nonopt = "status" in names(sl) ? count(x -> !ismissing(x) && string(x) != "OPTIMAL", sl.status) : 0
        println(io, "<tr><td>", _esc(a.name), "</td><td>", nrow(sl), "</td><td>", isempty(st) ? "n/a" : fmtnum(sum(st)),
                "</td><td>", isempty(st) ? "n/a" : fmtnum(mean(st)), "</td><td>", isempty(st) ? "n/a" : fmtnum(maximum(st)),
                "</td><td>", isempty(gp) ? "n/a" : fmtnum(maximum(gp)), "</td><td>", nonopt, "</td></tr>")
    end
    println(io, "</tbody></table></div>")
    println(io, "<p class=\"muted\">Solve counts: Approach 0 solves once per day. Approaches 1 and 2 solve once per interval. Approach 2 solves all five scenarios together in each solve. Runs are single-seed, so differences between approaches are one sample, not an average.</p>")

    println(io, "<h2>Output files</h2>")
    for a in APPROACHES
        od = out_dir(a)
        println(io, "<h3>", _esc(a.name), "</h3>")
        if !isdir(od)
            println(io, "<p class=\"muted\">No output folder.</p>")
            continue
        end
        println(io, "<div class=\"card\">")
        for (root, _, files) in walkdir(od)
            for f in sort(files)
                full = joinpath(root, f)
                rel  = replace(relpath(full, TEST_DIR), "\\" => "/")
                println(io, "<a href=\"", _esc(rel), "\">", _esc(rel), "</a> <span class=\"d\">", @sprintf("%.0f KB", filesize(full) / 1024), "</span><br>")
            end
        end
        println(io, "</div>")
    end

    println(io, "</main></body></html>")
    return String(take!(io))
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
function main()
    results = Dict{String,Any}()
    if RUN_SIMULATIONS
        for a in APPROACHES
            println("\n", "="^72)
            println("Running ", a.name, "   (output -> ", out_dir(a), ")")
            println("="^72)
            results[a.tag] = run_one(a)
        end
    else
        for a in APPROACHES
            results[a.tag] = (ok = isdir(out_dir(a)), secs = NaN)
        end
    end

    html_path = joinpath(TEST_DIR, "comparison_gurobi.html")
    try
        write(html_path, build_html(results))
        println("\nComparison page written to: ", html_path)
    catch err
        @error "Could not build comparison_gurobi.html" exception = (err, catch_backtrace())
    end
    return results
end

main()
