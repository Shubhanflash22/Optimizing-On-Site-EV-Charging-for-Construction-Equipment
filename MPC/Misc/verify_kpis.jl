# To run - verify_kpis(raw"C:\Users\shubh\Desktop\MPC\Approach 0\output\normal"; prefix = "A0")
# #############################################################################
# verify_kpis.jl  -  standalone sanity checker, not part of the pipeline
# -----------------------------------------------------------------------------
# Reads a run's own output CSVs and independently recomputes several numbers
# from the raw per-interval log, rather than trusting the same code that
# produced the KPI file to also verify it. Every check prints PASS/FAIL with
# the two numbers it compared. Works on any approach's output folder as long
# as the file-naming prefix matches (A0_..., A1_..., A2_...).
#
# Run it from the REPL after a run:
#   include("verify_kpis.jl")
#   verify_kpis("output/normal"; prefix = "A0")
# #############################################################################

using CSV, DataFrames, Printf, Dates

# Both bounds of the paper's on-peak demand window (16:00-21:00 / 4-9pm).
const ON_PEAK_START = Dates.Time(16, 0)
const ON_PEAK_END   = Dates.Time(21, 0)

# CSV.jl auto-detects an "HH:MM"-looking column as Dates.Time; a multi-day
# run's "Dn HH:MM" labels stay a String instead. Handles either so the on-peak
# filter below doesn't break on the column type CSV.jl happened to infer.
function _as_time(c)
    c isa Dates.Time && return c
    s = string(c)
    m = match(r"(\d{1,2}):(\d{2})", s)
    m === nothing && error("verify_kpis: couldn't parse a clock time out of '$s'")
    return Dates.Time(parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function _close(a, b; rtol = 0.02, atol = 1e-3)
    abs(a - b) <= atol + rtol * abs(b)
end

function _check(name, a, b; rtol = 0.02, atol = 1e-3)
    ok = _close(a, b; rtol, atol)
    @printf("%-4s %-42s  got=%-12.4f  expected=%-12.4f  diff=%.4f\n",
            ok ? "PASS" : "FAIL", name, a, b, a - b)
    return ok
end

function verify_kpis(out_dir::AbstractString; prefix::AbstractString = "A0", delta_T::Float64 = 0.25)
    kpi   = CSV.read(joinpath(out_dir, "$(prefix)_kpi_summary.csv"), DataFrame)
    solve = CSV.read(joinpath(out_dir, "$(prefix)_solve_log.csv"), DataFrame)
    ilog  = CSV.read(joinpath(out_dir, "$(prefix)_interval_log.csv"), DataFrame)

    # Column 1 is always "Metric"; column 2 is the KPI value column, whatever
    # that approach's build_overall_kpi_table happens to call it (A0, A1, A2, ...).
    val(name) = only(kpi[kpi.Metric .== name, 2])
    gap_frac  = maximum(solve.gap_percent) / 100

    println("==== Independent KPI verification: $(prefix), $(out_dir) ====")
    results = Bool[]

    # 1. Every grid-side KPI, recomputed straight from the interval log rather
    #    than trusted from the KPI file that itself was built from it.
    #    atol = 0.01 rather than a tight rtol, since the KPI file's own values
    #    are rounded to 2 decimals (see _kpi_column_values) -- a few thousandths
    #    of difference here is expected rounding, not a real mismatch.
    push!(results, _check("Total grid energy (kWh)",
        sum(ilog.grid_kW) * delta_T,
        val("Total_Grid_Energy_kWh"); atol = 0.01))

    push!(results, _check("Total energy cost (\$)",
        sum(ilog.grid_kW .* ilog.price) * delta_T,
        val("Total_Energy_Cost_USD"); atol = 0.01))

    push!(results, _check("Total CO2 (kg)",
        sum(ilog.grid_kW .* ilog.co2) * delta_T,
        val("Total_CO2_Emissions_kg"); atol = 0.01))

    push!(results, _check("NC peak (kW)",
        maximum(ilog.grid_kW),
        val("NCD_Peak_kW"); atol = 0.01))

    op_rows = [ON_PEAK_START <= _as_time(c) < ON_PEAK_END for c in ilog.clock]
    op_peak = any(op_rows) ? maximum(ilog.grid_kW[op_rows]) : 0.0
    push!(results, _check("On-peak peak (kW)", op_peak, val("OPD_Peak_kW"); atol = 0.01))

    # 2. The whole-run cost total, recomputed by summing the 7 KPI cost rows,
    #    checked against the solver's own reported objective. Tolerance scales
    #    with the achieved MIP gap since the two need not match exactly above 0% gap.
    cost_rows = ["Total_Energy_Cost_USD", "Total_CO2_Cost_USD", "NC_demand_charge_USD",
                 "OP_demand_charge_USD", "Missed_Work_Penalty_USD", "Travel_Labour_USD",
                 "Terminal_Shortfall_Penalty_USD"]
    kpi_total = sum(val(r) for r in cost_rows)
    push!(results, _check("KPI-summed total vs solver objective",
        kpi_total, sum(solve.objective); rtol = max(gap_frac, 1e-4)))

    push!(results, _check("KPI Total_Cost_USD vs KPI-summed total",
        val("Total_Cost_USD"), kpi_total; atol = 0.01))

    # 3. MCS transit/labour cross-check, recomputed from the MCS realized log
    #    if it exists, independent of the interval log entirely.
    mcs_path = joinpath(out_dir, "$(prefix)_MCS_realized_tuple.csv")
    if isfile(mcs_path)
        mcs = CSV.read(mcs_path, DataFrame)
        transit_h = count(==("Transit"), mcs.mcs_node_realized) * delta_T
        push!(results, _check("MCS transit hours",
            transit_h, val("MCS_Transit_hour"); atol = 0.01))
    end

    # 4. Plan-vs-realized equality, only meaningful for a deterministic run
    #    (plant = :mean). Every realized value should exactly equal the
    #    planned one, and n_capped should be 0.
    realized_path = joinpath(out_dir, "$(prefix)_realized_tuple.csv")
    if isfile(realized_path)
        r = CSV.read(realized_path, DataFrame)
        if "activity_planned" in names(r)
            mismatches = count(r.activity_executed .!= r.activity_planned)
            if mismatches == 0
                println("INFO one activity_planned/activity_executed row-by-row check ran; 0 mismatches (consistent with a :mean run or a quiet :sampled run)")
            else
                println("INFO $mismatches / $(nrow(r)) rows have activity_executed != activity_planned -- expected under plant = :sampled, should be 0 under plant = :mean")
            end
        end
    end

    n_pass = count(results); n_all = length(results)
    println("---- $n_pass / $n_all automated checks passed ----")
    return n_pass == n_all
end