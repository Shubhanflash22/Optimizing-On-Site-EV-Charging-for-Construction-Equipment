# check_priors.jl  (temporary check, delete after use)
#
# Tests Daniela's point that the step 0 priors are already the least-squares answer,
# so the data is counted twice. It rebuilds the same equations step 0 uses, then compares:
#   1. plain least squares (estimate and standard error)
#   2. the Bayesian fit with the current priors (two seeds, to show run-to-run noise)
#   3. the Bayesian fit with weak priors, same center
#   4. the Bayesian fit with weak priors, different center
# It never touches parameters.csv.
#
# From the MPC root folder:
#   julia --project=. --threads=auto check_priors.jl

using Printf, Statistics, LinearAlgebra, Random

const ROOT     = dirname(@__DIR__)
const CODE     = joinpath(ROOT, "Approach 1", "code")
const DATA_DIR = joinpath(ROOT, "Bayesian Regression")

include(joinpath(CODE, "1_Common.jl"))
include(joinpath(CODE, "0_Regression.jl"))
using .Common: BayesianActivityEstimator, observe!, refit!

# Same equation building as run_regression in 0_Regression.jl
A_rows = Vector{Vector{Float64}}(); b_rows = Float64[]
nfiles = 0
for f in Regression.SOIL_FILES
    path = joinpath(DATA_DIR, f)
    if !isfile(path); println("missing file: ", f); continue; end
    cols = Regression._read_task_file(path)
    cols === nothing && continue
    Regression._equations_from_file!(A_rows, b_rows, cols...)
    global nfiles += 1
end
A = reduce(vcat, (reshape(r, 1, :) for r in A_rows))
b = b_rows
n = length(b)
@printf("\nbuilt %d equations from %d files\n", n, nfiles)

# 1. Plain least squares on the three real activities (idle is pinned to 0 kW in step 0)
X     = A[:, 1:3]
x_ols = X \ b
res   = b .- X * x_ols
s_res = sqrt(sum(abs2, res) / (n - 3))
se    = s_res .* sqrt.(diag(inv(X' * X)))

# 2 to 4. Bayesian fits with the same estimator step 0 uses
function bayes_fit(mu0, sd0, seed)
    Random.seed!(seed)
    est = BayesianActivityEstimator(mu0, sd0; mcmc_samples = 2000)
    for i in 1:n
        observe!(est, A[i, :], b[i])
    end
    refit!(est; nchains = 4)
    return est.mu[1:3], est.sd[1:3]
end

cases = [
    ("current priors, seed 1",       Regression.PRIOR_MU, Regression.PRIOR_SIGMA, 1),
    ("current priors, seed 2",       Regression.PRIOR_MU, Regression.PRIOR_SIGMA, 2),
    ("weak priors, same center",     Regression.PRIOR_MU, [5.0, 5.0, 5.0, 0.0],   1),
    ("weak priors, other center",    [3.0, 3.0, 3.0, 0.0], [5.0, 5.0, 5.0, 0.0],   1),
]

println("\nkW                              dig      load     travel")
@printf("%-28s %8.4f %8.4f %8.4f\n", "least squares (estimate)", x_ols[1], x_ols[2], x_ols[3])
@printf("%-28s %8.4f %8.4f %8.4f\n", "least squares (std error)", se[1], se[2], se[3])
@printf("residual std of the equations: %.4f kWh\n\n", s_res)

for (label, mu0, sd0, seed) in cases
    m, s = bayes_fit(mu0, sd0, seed)
    @printf("%-28s %8.4f %8.4f %8.4f   (posterior mean)\n", label, m[1], m[2], m[3])
    @printf("%-28s %8.4f %8.4f %8.4f   (posterior sd)\n", "", s[1], s[2], s[3])
end

println("""

How to read it:
- If the two weak-prior means are close to least squares (and to each other), while the
  current-prior sds are much smaller than the least-squares std errors, the priors are
  tightening the fit beyond what the data supports, as Daniela suspects.
- The gap between the two current-prior seeds is the run-to-run noise from the unseeded sampler.
""")
