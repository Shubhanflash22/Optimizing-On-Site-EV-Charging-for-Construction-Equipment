# README: Approach 2 on Gurobi

Only `3_MCSModel.jl` needs editing. Approaches 0 and 1 stay on HiGHS.

## 1. Get a license (one time)
- Register at gurobi.com with your ucsd.edu email and request the free academic license. It gives you a key.
- Activate it from the campus network or the UCSD VPN.

## 2. Install (one time, in the Julia REPL)
```julia
import Pkg
Pkg.add("Gurobi")
Pkg.add("Gurobi_jll")
using Gurobi_jll
run(`$(Gurobi_jll.grbgetkey()) YOUR-KEY-HERE`)   # saves gurobi.lic in your home folder
```
Keep HiGHS installed, because Approaches 0 and 1 still use it.

If `grbgetkey` fails, install Gurobi from gurobi.com, set `GUROBI_HOME` and `GUROBI_JL_USE_GUROBI_JLL=false`, then run `Pkg.build("Gurobi")`.

Check it works:
```julia
using Gurobi
Gurobi.Env()
```
It should print the license info with no error.

## 3. Code changes in `3_MCSModel.jl`

Find:
```julia
using HiGHS
```
Replace with:
```julia
using Gurobi
const GRB_ENV = Gurobi.Env()
```

Find:
```julia
model = Model(HiGHS.Optimizer)
```
Replace with:
```julia
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
```

Find:
```julia
set_attribute(model, "threads", 8)
set_attribute(model, "parallel", "on")
set_attribute(model, "mip_heuristic_effort", 0.2)
set_attribute(model, "mip_detect_symmetry", true)
set_attribute(model, "mip_rel_gap", 1.0e-3)
```
Replace with:
```julia
set_attribute(model, "Threads", 8)
set_attribute(model, "Heuristics", 0.0)
set_attribute(model, "Symmetry", 0)
set_attribute(model, "MIPGap", 1.0e-3)
```

- `parallel` is dropped because Gurobi has no such switch and runs in parallel on its own.
- Symmetry detection is left out because Gurobi's default (automatic) is the equivalent of "on".
- Optional: update the two comments that still say HiGHS (line 7 and line 91).

## 4. Test before the full sweep
- Run `run_scenario_1(time_limit_sec = 120.0)` and stop it after a few windows.
- Open the first rows of `A2_solve_log.csv` and check `objective`, `gap_percent` and `solve_time_s` for the 08:00 window.
- Compare with your HiGHS run. The 08:00 objective should be near the expected ~$70, not $1176.

## 5. Paper note
- Say that Approach 2 uses Gurobi and Approaches 0 and 1 use HiGHS.
- Do not make speed comparisons across approaches.

## Files
| File | Change |
|---|---|
| `3_MCSModel.jl` | Yes, the three edits above |
| All other files | None |
