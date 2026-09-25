# #############################################################################
# A0App.jl  —  self-contained Approach-0-only application module
# -----------------------------------------------------------------------------
# Mirrors the `module A1ShrinkingApp` wrapper pattern used by the original
# Comparison_A0_A1_A2/Code/7_Comparison_main_ShrinkingOnlyVersion.jl: each of
# Approach 1's own source files becomes a SUBMODULE of A0App when include()-d
# from here, so `..Common` inside 3_MCSModel.jl / 4_MPCLoop.jl resolves to
# A0App.Common exactly the way it resolves to A1ShrinkingApp.Common in the
# original driver. Nothing in src/ has been modified.
# #############################################################################
module A0App

using DataFrames
using JuMP
using Random
using Printf
using LinearAlgebra
using CSV
using Dates

const _DIR = @__DIR__

include(joinpath(_DIR, "src", "1_Common.jl"))
include(joinpath(_DIR, "src", "2_DataLoader.jl"))
include(joinpath(_DIR, "src", "3_MCSModel.jl"))
include(joinpath(_DIR, "src", "4_MPCLoop.jl"))

using .Common
using .DataLoader
using .MCSModel
using .MPCLoop

include(joinpath(_DIR, "run_one_shot_detailed.jl"))
include(joinpath(_DIR, "kpi_tables.jl"))
include(joinpath(_DIR, "outputs_a0.jl"))

export run_one_shot_detailed, write_a0_detailed_outputs, write_a0_kpi_outputs,
       build_overall_kpi_table, build_daily_kpi_table

end # module A0App
