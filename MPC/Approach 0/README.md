# Approach 0 — One-Shot Baseline

```
Approach 0/
├── README.md              ← this file
├── code/                  1_Common … 6_OneShot_main (see docs/README.md §3)
├── data/
│   ├── input_data/        the 8 real CSVs, read by mode = :input (or :normal/:high/...)
│   └── synthetic_data/    human-readable mirror of the hardcoded :synthetic scenario
└── docs/
    └── README.md          full write-up: what this is, why it's a separate folder now,
                            the multi-day day-reset fix (with a worked example), and the
                            detailed-output CSV schema
```

**Quick start:**
```julia
julia --project=.
include("code/6_OneShot_main.jl")
```

This folder is Approach 0 on its own — no longer borrowed from Approach 1 or
Approach 2's codebase. It shares the same MILP/data-loading code (identical
copies of `1_Common.jl`, `2_DataLoader.jl`, `3_MCSModel.jl`) since the
underlying vehicle/MCS model doesn't depend on which control approach is
driving it, but the executor (`4_OneShot.jl`) and its detailed CEV+MCS output
logging are entirely its own.

Used standalone (`code/6_OneShot_main.jl`) or from
`../Comparison_A0_A1_A2/Code/`, which now calls
`A0App.OneShot.run_one_shot(...)` directly instead of reaching into
whichever of Approach 1/2 happened to be picked.

See `docs/README.md` for everything else, especially §5 if you're running
multi-day (`n_day_run > 1`) and seeing demand charges climb day over day —
that section explains why, with a worked example, and what the fix does.
