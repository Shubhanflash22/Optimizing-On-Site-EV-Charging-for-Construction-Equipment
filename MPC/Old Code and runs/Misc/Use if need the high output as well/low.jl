COMPARISON_NO_AUTORUN = true
include(raw"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Code\7_Comparison_main_ShrinkingOnlyVersion.jl")

build_comparison_input()

dA1S = A1ShrinkingApp.DataLoader.load_data(:input; input_dir = _COMPARISON_INPUT)
dA2S = A2ShrinkingApp.DataLoader.load_data(:input; input_dir = _COMPARISON_INPUT)

pool = A1ShrinkingApp.Common.draw_activity_power_pool(dA1S.E, dA1S.prior_mu, dA1S.prior_sigma;
                                                       n_samples = 101, rng = MersenneTwister(1), mode = :low)

res0 = A1ShrinkingApp.MPCLoop.run_one_shot(dA1S, pool; plant = :sampled, time_limit_sec = 1200.0,
                                            multi_activity = false, require_site_visit = false,
                                            single_visit_per_site = false, n_day_run = 1, seed = 1)

resA1S = A1ShrinkingApp.MPCLoop.run_mpc(dA1S, pool; shrinking = true, H = 16, time_limit_sec = 1200.0,
                                         multi_activity = false, require_site_visit = false,
                                         single_visit_per_site = false, mcmc_samples = 500,
                                         plant = :sampled, n_day_run = 1, seed = 1)

resA2S = A2ShrinkingApp.MPCLoop.run_mpc(dA2S, pool; shrinking = true, H = 16, time_limit_sec = 1200.0,
                                         multi_activity = false, require_site_visit = false,
                                         single_visit_per_site = false, mcmc_samples = 500,
                                         plant = :sampled, n_day_run = 1, seed = 1)

all_apps = Dict(
    "A0"  => Approach("A0",  "Approach 0 (one-shot, :sampled)", res0,   :gray40),
    "A1S" => Approach("A1S", "Approach 1 - Shrinking",          resA1S, :firebrick),
    "A2S" => Approach("A2S", "Approach 2 - Shrinking (stochastic)", resA2S, :darkorange),
)

for (folder, keys) in _ALL_COMBOS
    apps = [all_apps[k] for k in keys]
    sub_out = joinpath(_COMPARISON_OUT, "low", folder)
    Base.invokelatest(write_comparison_outputs, apps, sub_out)
    println("wrote: ", sub_out)
end
println("ALL DONE — low mode")
