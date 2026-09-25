# Porting this into your original `Comparison_A0_A1_A2` codebase

This standalone `A0/` folder works on its own, but if you'd rather have this
logging live inside your real repo (so it stays next to `Approach 1/`,
`Approach 2/`, and the rest of `Comparison_A0_A1_A2/`), here's exactly what to
add. **Nothing existing needs to be deleted or rewritten** — this is purely
additive, the same way A1S/A2S's own `detailed_output` flag was added without
touching their MILP code.

## 1. Copy 3 files in, unchanged

Copy these three files from this `A0/Code/` folder straight into your real
`Comparison_A0_A1_A2/Code/` folder:

- `run_one_shot_detailed.jl`
- `kpi_tables.jl`
- `outputs_a0.jl`

They don't need any edits — they call `MPCLoop.apply_and_simulate!`,
`MPCLoop.activity_label`, `MPCLoop.mcs_status_label`, `MPCLoop._terminal_soe_shortfall`,
`build_window_model`, and Common's `DetailedPlanLog`/`RealizedTupleLog`, all of
which already exist unmodified in your `Approach 1/Shrinking_Horizon/code/`.

## 2. Include them where `A1ShrinkingApp` is defined

In `7_Comparison_main_ShrinkingOnlyVersion.jl`, inside `module A1ShrinkingApp`,
add the 3 new files to the include list (after `4_MPCLoop.jl`, since
`run_one_shot_detailed.jl` calls into `MPCLoop`):

```julia
module A1ShrinkingApp
    const _DIR = normpath(joinpath(@__DIR__, "..", "..", "Approach 1", "Shrinking_Horizon", "code"))
    include(joinpath(_DIR, "1_Common.jl"))
    include(joinpath(_DIR, "0_Regression.jl"))
    include(joinpath(_DIR, "2_DataLoader.jl"))
    include(joinpath(_DIR, "3_MCSModel.jl"))
    include(joinpath(_DIR, "4_MPCLoop.jl"))
    include(joinpath(_DIR, "5_Output.jl"))

    # NEW -- Approach 0's detailed logging + A0-only KPI tables.
    include(joinpath(@__DIR__, "run_one_shot_detailed.jl"))
    include(joinpath(@__DIR__, "kpi_tables.jl"))
    include(joinpath(@__DIR__, "outputs_a0.jl"))
end
```

`@__DIR__` inside the `module A1ShrinkingApp` block is the directory of
`7_Comparison_main_ShrinkingOnlyVersion.jl` itself (`Comparison_A0_A1_A2/Code`),
so put the 3 new files directly in that folder (step 1 above already does this).

## 3. Call it from the run scripts, alongside the existing A0 solve

Wherever a script currently calls Approach 0 like this (e.g.
`RUN_ALL_25_RUNS.jl` via `run_comparison_sweep`, or your own new A0-only
script):

```julia
res0 = A1ShrinkingApp.MPCLoop.run_one_shot(dA1S, pool; plant = approach0_plant,
                                           time_limit_sec, multi_activity,
                                           require_site_visit, single_visit_per_site,
                                           n_day_run, seed)
```

add a second call, right next to it, that ALSO produces the detailed version
(the two are independent — you can keep both, or replace the first with the
second since `resd.res` already contains everything `res0` does):

```julia
resd = A1ShrinkingApp.run_one_shot_detailed(dA1S, pool; plant = approach0_plant,
                                            time_limit_sec, multi_activity,
                                            require_site_visit, single_visit_per_site,
                                            n_day_run, seed)
res0 = resd.res   # same fields run_one_shot returns -- drop-in replacement
```

Then write the outputs for that run, next to wherever `_write_detailed_output`
already writes A1S's/A2S's CSVs:

```julia
A1ShrinkingApp.write_a0_detailed_outputs(resd, joinpath(detailed_out_dir, "A0"))
A1ShrinkingApp.write_a0_kpi_outputs(resd, dA1S, joinpath(detailed_out_dir, "A0"))
```

## 4. If you want it wired into `run_comparison_sweep` itself

Inside `7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl`'s `run_comparison_sweep`,
the existing `detailed_output` block near the bottom of the mode loop
(the one calling `_write_detailed_output(resA1S, "A1", ...)` and
`_write_detailed_output(resA2S, "A2", ...)`) is the natural place to add a
third call:

```julia
if detailed_output
    resd0 = A1ShrinkingApp.run_one_shot_detailed(dA1S, pool; plant = approach0_plant,
                                                 time_limit_sec, multi_activity,
                                                 require_site_visit, single_visit_per_site,
                                                 n_day_run, seed)
    a0_dir = joinpath(dirname(joinpath(dirname(out_dir), "Detailed_" * basename(dirname(out_dir)))),
                       basename(out_dir), String(mode), "A0")
    # (or, simpler: reuse the same `detailed_root`/`mode_dir` pattern
    # `_write_detailed_output` already builds, just with an "A0" subfolder)
    A1ShrinkingApp.write_a0_detailed_outputs(resd0, a0_dir)
    A1ShrinkingApp.write_a0_kpi_outputs(resd0, dA1S, a0_dir)
    d1 = _write_detailed_output(resA1S, "A1", out_dir, mode)
    d2 = _write_detailed_output(resA2S, "A2", out_dir, mode)
end
```

Since this doubles the A0 solve (once for the plain KPI comparison in
`all_apps["A0"]`, once here for the detailed logs), if solve time matters you
can instead just reuse `resd0.res` as `all_apps["A0"].res` directly and drop
the separate plain `run_one_shot` call entirely — they produce identical
numbers, `run_one_shot_detailed` just also captures the extra logging.

## Notes

- `run_one_shot_detailed` has no `detailed_output` on/off switch — it always
  logs, since that's the only thing it's for. If you fold it into
  `run_one_shot` itself instead of keeping it separate, add a boolean flag the
  same way `run_mpc` already has one, and skip the `log_plan_row!` /
  `push!(mcs_plan_rows, ...)` calls when it's false.
- The 2 new price/CO2 columns are joined on **after** the loop finishes
  (`_add_price_co2`), not inside it — if you fold this into `run_mpc`'s own
  detailed-output block instead, you can do the same join on `resA1S`'s and
  `resA2S`'s existing `detailed_plan_df` / `detailed_realized_df` for free,
  giving A1/A2 the same price/CO2 explainability columns A0 gets here.
