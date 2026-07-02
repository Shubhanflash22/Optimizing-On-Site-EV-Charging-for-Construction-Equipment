# =============================================================================
# export_synthetic_data.jl
# -----------------------------------------------------------------------------
# Dumps the built-in SYNTHETIC dataset (build_default_data) into the SAME 7-CSV
# schema as `data/input_data/`, so the synthetic and real datasets can be diffed /
# swapped 1:1. CSVs -> `data/synthetic_data/`; its guide -> `docs/synthetic_data_explained.md`.
#
# Run:  julia export_synthetic_data.jl
# (loads Scenario_1.jl with autorun disabled, builds the data, writes the CSVs)
# =============================================================================

SCENARIO1_NO_AUTORUN = true
include(joinpath(@__DIR__, "Scenario_1.jl"))

using CSV, DataFrames, Printf

d = build_default_data()

out = joinpath(dirname(@__DIR__), "data", "synthetic_data")   # <root>/data/synthetic_data
mkpath(out)

nint = length(d.lambda_whl_elec)                       # full price horizon (incl. overnight)
node_name(i) = "i$(i)"
ev_name(e)   = "e$(e)"

# interval END-time labels (interval k ends at t_start + k*delta_T)
function end_label(k; secs::Bool = false)
    mins = mod(Int(round((d.t_start + k * d.delta_T) * 60)), 24 * 60)
    h = div(mins, 60); m = mins % 60
    return secs ? @sprintf("%d:%02d:00", h, m) : @sprintf("%d:%02d", h, m)
end

# ---- parameters.csv ---------------------------------------------------------
prior_frac = d.prior_mu[1] > 0 ? round(d.prior_sigma[1] / d.prior_mu[1], digits = 3) : 0.2
params = [
    ("k_trv",                d.k_trv,                "kWh/hour", "MCS average energy consumption per hour"),
    ("rho_miss",             d.rho_miss,             "\$/h",     "Missed-work penalty"),
    ("delta_T",              d.delta_T,              "h",        "Time step duration"),
    ("p_digging",            d.p_digging,            "kW",       "Power of digging"),
    ("p_loading_swinging",   d.p_loading_swinging,   "kW",       "Power of loading+swinging"),
    ("p_traveling",          d.p_traveling,          "kW",       "Power of traveling"),
    ("lambda_demand_NC",     d.lambda_demand_NC,     "\$/kW",    "NCDC rate"),
    ("lambda_demand_OP",     d.lambda_demand_OP,     "\$/kW",    "OPDC rate"),
    ("carbon_price_per_ton", d.carbon_price_per_ton, "\$/tonneCO2", "Carbon price"),
    ("rho_labor",            d.rho_labor,            "\$/hr",    "MCS towing labour per hour"),
    ("p_idling",             d.p_idling,             "kW",       "Idle power (idling kept as a 4th activity)"),
    ("scale",                d.scale,                "-",        "Precedence multiple kappa_seq (loading<=scale*digging)"),
    ("t_limit_rest",         d.t_limit_rest,         "h",        "Rest rule: max work per (t_limit_rest+delta_T) window"),
    ("kappa_wt",             d.kappa_wt,             "-",        "Travel-pacing productive intervals per travel"),
    ("day_end_hour",         d.day_end_hour,         "h",        "Daytime horizon end (two-phase split)"),
    ("prior_sigma_frac",     prior_frac,             "-",        "Bayesian prior std as fraction of power"),
    ("obs_noise_std",        d.obs_noise_std,        "kWh",      "Bayesian telemetry noise std (simulation)"),
    ("co2_unit_scale",       1,                      "-",        "Raw intensity_tons_emissions scale"),
]
CSV.write(joinpath(out, "parameters.csv"),
          DataFrame(Parameter = first.(params),
                    Value = getindex.(params, 2),
                    Unit = getindex.(params, 3),
                    Description = getindex.(params, 4)))

# ---- ev_data.csv ------------------------------------------------------------
work_cap = maximum(d.R_work; init = 0.0)               # nominal per-interval work cap (kW)
ev = DataFrame("Unnamed: 0" => [ev_name(e) for e in d.E],
               "SOE_min" => [d.SOE_CEV_min[e] for e in d.E],
               "SOE_max" => [d.SOE_CEV_max[e] for e in d.E],
               "SOE_ini" => [d.SOE_CEV_ini[e] for e in d.E],
               "ch_rate" => [d.CH_CEV[e] for e in d.E],
               "work_cap" => fill(work_cap, length(d.E)))
CSV.write(joinpath(out, "ev_data.csv"), ev)

# ---- mcs_data.csv -----------------------------------------------------------
mcs = DataFrame("Unnamed: 0" => [ "m$(m)" for m in d.M],
                "SOE_min" => [d.SOE_MCS_min[m] for m in d.M],
                "SOE_max" => [d.SOE_MCS_max[m] for m in d.M],
                "SOE_ini" => [d.SOE_MCS_ini[m] for m in d.M],
                "CH_MCS" => [d.CH_MCS[m] for m in d.M],
                "DCH_MCS" => [d.DCH_MCS[m] for m in d.M],
                "C_MCS_plug" => [d.C_MCS_plug[m] for m in d.M],
                "DCH_MCS_plug" => [d.DCH_MCS_plug[m] for m in d.M],
                "eta_ch_dch" => [d.eta_ch_dch[m] for m in d.M])
CSV.write(joinpath(out, "mcs_data.csv"), mcs)

# ---- place.csv --------------------------------------------------------------
place = DataFrame(site = [node_name(i) for i in d.N])
for e in d.E
    place[!, ev_name(e)] = [Int(round(d.A[i, e])) for i in d.N]
end
place[!, "hours_digging"]          = [d.hours_digging[i] for i in d.N]
place[!, "hours_loading_swinging"] = [d.hours_loading_swinging[i] for i in d.N]
CSV.write(joinpath(out, "place.csv"), place)

# ---- travel_time.csv (matrix; interval counts) ------------------------------
tt = DataFrame(Node = [node_name(i) for i in d.N])
for j in d.N
    tt[!, node_name(j)] = [d.tau_trv[i, j] for i in d.N]
end
CSV.write(joinpath(out, "travel_time.csv"), tt)

# ---- time_data.csv ----------------------------------------------------------
td = DataFrame("Unnamed: 0" => [end_label(k; secs = true) for k in 1:nint],
               "Unnamed: 1" => ["t$(k)" for k in 1:nint],
               "lambda_CO2" => d.lambda_CO2[1:nint],
               "lambda_buy" => d.lambda_whl_elec[1:nint],
               "intensity_tons_emissions" => d.lambda_CO2[1:nint])
CSV.write(joinpath(out, "time_data.csv"), td)

# ---- work_flexible.csv (per site-CEV pair; per-interval work cap R_work) -----
rows = NamedTuple[]
labels = [end_label(k) for k in 1:nint]
for e in d.E, i in d.N_c
    d.A[i, e] == 1 || continue
    vals = [k <= d.n_day ? d.R_work[i, e, k] : 0.0 for k in 1:nint]
    push!(rows, (; Location = node_name(i), EV = ev_name(e),
                 (Symbol(labels[k]) => vals[k] for k in 1:nint)...))
end
CSV.write(joinpath(out, "work_flexible.csv"), DataFrame(rows))

# ---- README.md (plain-language description, derived from d) ------------------
site_of(e) = findfirst(i -> d.A[i, e] == 1, d.N)
mins(iv)   = Int(round(iv * d.delta_T * 60))
io = IOBuffer()
println(io, "# Synthetic dataset — the scenario in plain words\n")
println(io, "Describes the **built-in synthetic scenario** — the 7 CSVs in `../data/synthetic_data/`,")
println(io, "written in the exact same schema as `../data/input_data/` so the two can be compared or")
println(io, "swapped 1:1. Generated by `../code/export_synthetic_data.jl` (re-run it to regenerate the")
println(io, "CSVs and this README).\n")
println(io, "Run it like the real data (from the `code/` folder):\n")
println(io, "```julia\nrun_scenario_1(mode = :input, input_dir = \"../data/synthetic_data\")\n```\n")
println(io, "## What's in this scenario\n")
println(io, "- **$(length(d.M)) mobile charging station(s) (MCS)** — drive around the site and charge")
println(io, "  the excavators. Battery $(d.SOE_MCS_min[1])–$(d.SOE_MCS_max[1]) kWh (starts at ",
            "$(d.SOE_MCS_ini[1])), charges from the grid at up to $(d.CH_MCS[1]) kW, with ",
            "$(d.C_MCS_plug[1]) plug(s) at up to $(d.DCH_MCS_plug[1]) kW each.")
println(io, "- **$(length(d.E)) construction electric vehicle(s) (CEVs / excavators)** — battery ",
            "$(d.SOE_CEV_min[1])–$(d.SOE_CEV_max[1]) kWh (starts at $(d.SOE_CEV_ini[1])), ",
            "receive up to $(d.CH_CEV[1]) kW.")
println(io, "- **$(length(d.N)) locations (nodes)**:")
for i in d.N_g
    println(io, "  - `$(node_name(i))` = the **grid connection** (where the MCS recharges). No excavator here.")
end
for i in d.N_c
    e = findfirst(ev -> d.A[i, ev] == 1, d.E)
    println(io, "  - `$(node_name(i))` = a **work site**", e === nothing ? "." : ", home of excavator `$(ev_name(e))`.")
end
println(io, "\n## The work each site must get done (per day)\n")
println(io, "| Site | Excavator | Digging | Loading / swinging |")
println(io, "|------|-----------|---------|--------------------|")
for i in d.N_c
    e = findfirst(ev -> d.A[i, ev] == 1, d.E)
    println(io, "| `$(node_name(i))` | ", e === nothing ? "—" : "`$(ev_name(e))`",
            " | $(d.hours_digging[i]) h | $(d.hours_loading_swinging[i]) h |")
end
println(io, "\n## How far apart things are (MCS travel time, in $(Int(round(d.delta_T*60)))-min steps)\n")
print(io, "| From \\ To |"); for j in d.N; print(io, " `$(node_name(j))` |"); end; println(io)
print(io, "|---|"); for _ in d.N; print(io, "---|"); end; println(io)
for i in d.N
    print(io, "| `$(node_name(i))` |")
    for j in d.N
        iv = d.tau_trv[i, j]
        print(io, i == j ? " – |" : " $(iv) ($(mins(iv)) min) |")
    end
    println(io)
end
println(io, "\n## The day\n")
println(io, "- **Daytime ($(Int(d.t_start)):00–$(Int(d.day_end_hour)):00, $(d.n_day) × ",
            "$(Int(round(d.delta_T*60)))-min steps):** the excavators dig / load / travel / idle; the")
println(io, "  MCS drives between the grid and the sites to keep them charged. Each excavator must")
println(io, "  end the day back at its start-of-day charge (energy-neutral), and the MCS must be")
println(io, "  parked at the grid by $(Int(d.day_end_hour)):00.")
println(io, "- **Overnight:** a cheapest-hours refill tops the MCS back up, closing its 24-hour")
println(io, "  energy-neutral cycle.")
println(io, "- Electricity price and grid-carbon intensity vary across all $nint quarter-hours of")
println(io, "  the day (see `time_data.csv`).\n")
println(io, "## Note on the numbers\n")
println(io, "The **activity power draws are treated as unknown** in the model — the controller learns")
println(io, "them online from telematics. The per-interval work caps in `work_flexible.csv` are a")
println(io, "large, non-binding value (they only mark *when* each excavator is allowed to work).")
readme_path = joinpath(dirname(@__DIR__), "docs", "synthetic_data_explained.md")
write(readme_path, String(take!(io)))

println("Synthetic dataset written to: ", abspath(out))
for f in ["parameters.csv","ev_data.csv","mcs_data.csv","place.csv",
          "travel_time.csv","time_data.csv","work_flexible.csv"]
    println("  - ", f)
end
println("README written to: ", abspath(readme_path))
