# #############################################################################
# outputs_a0.jl  —  included inside `module A0App`
# -----------------------------------------------------------------------------
# Writes exactly what was asked for and nothing else: the detailed plan +
# realized CSVs (CEV and MCS), the solve log, and the KPI summary table(s).
# No plots, no per-interval grid-power/dispatch-trace CSVs, no HTML.
# #############################################################################

function write_a0_detailed_outputs(resd, out_dir::AbstractString)
    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "A0_CEV_plan_full.csv"),   resd.detailed_plan_df_cev)
    CSV.write(joinpath(out_dir, "A0_CEV_realized.csv"),    resd.detailed_realized_df_cev)
    CSV.write(joinpath(out_dir, "A0_MCS_plan_full.csv"),   resd.mcs_plan_df)
    CSV.write(joinpath(out_dir, "A0_MCS_realized.csv"),    resd.mcs_realized_df)
    CSV.write(joinpath(out_dir, "A0_solve_log.csv"),       resd.solve_log)
    CSV.write(joinpath(out_dir, "A0_day_snapshots.csv"),   resd.day_snapshots_df)
    return nothing
end

function write_a0_kpi_outputs(resd, d, out_dir::AbstractString)
    mkpath(out_dir)
    overall_df = build_overall_kpi_table(resd.res, d)
    CSV.write(joinpath(out_dir, "A0_kpi_summary.csv"), overall_df)
    if resd.res.n_day_run > 1
        daily_df = build_daily_kpi_table(resd, d)
        CSV.write(joinpath(out_dir, "A0_kpi_summary_by_day.csv"), daily_df)
    end
    return nothing
end
