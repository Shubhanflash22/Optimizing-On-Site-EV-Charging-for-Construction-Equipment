# MPC — Mobile Charging Station Dispatch for Construction EVs

Top-level overview of everything under `C:\Users\shubh\Desktop\MPC`.

```
MPC/
├── Approach 1/                 Certainty-Equivalent (deterministic) MPC
├── Approach 2/                 Stochastic (scenario-based) MPC
├── Comparison_A0_A1_A2/        5-way comparison spanning BOTH approaches
├── Approaches 1 and 2.pptx
├── Optimization.pdf
└── README.md                   this file
```

## The problem, in one paragraph

One **MCS** (Mobile Charging Station — a big battery on a towed truck) has to keep a fleet
of **CEVs** (electric excavators, fixed at construction sites) charged and working, all day,
without knowing in advance exactly how much power each activity (digging, loading, etc.)
will actually draw. The controller decides, every 15 minutes: where the MCS charges, how
much it delivers to which CEV, when it drives between sites, and what each CEV is doing —
subject to battery physics, work quotas, precedence and pacing rules — while minimizing
time-of-use electricity cost, carbon, demand charges, missed-work penalties and towing
labour. Both approaches below solve this same problem; they differ in how they handle the
fact that the power estimate is a guess, not a certainty.

## `Approach 1/` — Certainty-Equivalent MPC

The baseline controller: plans against the **posterior mean** power draw (`μ`) as if it
were exactly correct, and relies on **re-solving every 15 minutes from the real measured
state** to correct for the fact that it wasn't. Two horizon variants live here
(`Shrinking_Horizon/` — single day, window shrinks to the day boundary; `Receding_Horizon/`
— multi-day, fixed-width window slides into tomorrow), plus its own 3-way comparison driver
(`Comparison/`, — Approach 0 one-shot baseline vs this tree's Shrinking vs Receding) and a
one-click batch runner (`RUN_ALL.jl`).

Start at **`Approach 1/README.md`** — it's the master overview for that whole tree: the
plain-language explanation, the sampled-vs-mean plant switch, Shrinking vs Receding,
the file-by-file map, and a five-level appendix on planning vs reality.

## `Approach 2/` — Stochastic (Scenario-Based) MPC

Same folder structure, same data, same underlying MILP physics as Approach 1 — the
difference is entirely in *how* it hedges against the uncertain power draw. Instead of
planning against one mean value, at every re-solve it samples a small set of **scenarios**
from the fitted posterior and solves one MILP that must stay feasible under **all of them
at once**, coupled by a **non-anticipativity** constraint (the very next action must be
identical across every scenario, since you don't yet know which one is real). That's what
lets it hedge *before* a bad draw happens, rather than only reacting after re-measuring —
see the worked example at the bottom of this file for exactly what that buys you.

Start at **`Approach 2/README.md`** — same master-overview structure as Approach 1's, with
§13 covering the full stochastic extension.

## `Comparison_A0_A1_A2/` — the 5-way comparison

Approach 1 and Approach 2 each ship their own 3-way comparison (Approach 0 vs that tree's
own Shrinking vs Receding). This folder is the one that spans **both** trees at once: it
solves Approach 0, Approach 1 – Shrinking, Approach 2 – Shrinking, Approach 1 – Receding,
and Approach 2 – Receding **once each**, from one shared random power pool, and slices the
results into 11 output folders — the full 5-way comparison plus 10 requested subsets (every
A0-vs-one pairing, both A0-vs-shrinking-pair and A0-vs-receding-pair triples, shrinking vs
shrinking, receding vs receding, and each approach's shrinking vs its own receding).

Start at **`Comparison_A0_A1_A2/README.md`** for the full folder layout, the run
instructions, the parameter table, and the compatibility notes on how four codebases safely
share one power pool.

## Where the diagrams and slides live

- **`Approaches 1 and 2.pptx`** — the slide deck.
- **`Optimization.pdf`** — the source optimization writeup.

---

# Understanding Deterministic MPC vs Stochastic MPC

The next part explains **Deterministic (Certainty-Equivalent) MPC** vs **Stochastic
(Scenario-Based) MPC** at five levels of depth — from a simple analogy to a formal
treatment — and finishes with one worked numerical example that runs **an open-loop
baseline, Deterministic MPC, and Stochastic MPC side by side** on the exact same hidden
reality, so you can see the logical difference, not just the definition.

It assumes you already understand the general idea of **Model Predictive Control (MPC)**:
every interval, solve an optimization over a future window, apply only the first step,
re-measure, repeat.

---

## The five levels

### Level 1 — Explain it to a kid

Imagine you're carrying a cup of water across a room, and you don't know if someone might
bump into you.

- **Deterministic MPC** fills the cup as full as looks fine *if nobody bumps you*, and only
  slows down *after* someone actually bumps you and water spills.
- **Stochastic MPC** imagines a few different "what ifs" *before* you even start walking —
  what if nobody bumps you, what if someone bumps you a little, what if someone bumps you
  hard — and it fills the cup a little less full **right now**, so that even in the "bumped
  hard" case, nothing spills. It picks one safe way to walk *right now* that works out okay
  no matter which of those things happens next.

### Level 2 — Explain it to a teenager

Both approaches are trying to control a robot truck that charges digging-machine batteries,
15 minutes at a time, without knowing exactly how much power the diggers will use.

- **Deterministic MPC** makes its best guess (the *average* expected power draw), plans as
  if that guess is exactly correct, and only finds out it was wrong *after* the fact — once
  the battery is already lower than expected. Then it fixes course next round. It's
  reactive: it fixes mistakes, it doesn't prevent them.
- **Stochastic MPC** doesn't trust one guess. It imagines several different possible power
  draws — optimistic, average, pessimistic — **at the same time**, and only commits to a
  next move that would still be *safe* under all of them, even the pessimistic one. It's
  proactive: it avoids the mistake in the first place, at the cost of sometimes being more
  cautious (and slower to solve) than it strictly needed to be.

### Level 3 — Explain it to an engineering student

Both are **Model Predictive Control**: every interval, solve an optimization over a future
window, apply only the first step, re-measure, repeat. The difference is entirely in what
happens *inside* the optimization at each step.

- **Deterministic MPC (certainty-equivalent):** the Bayesian estimator produces a posterior
  mean `μ` for each activity's power draw. The MILP is solved **once**, treating `μ` as if
  it were the true, deterministic value. One future, one plan.
- **Stochastic MPC (scenario-based):** instead of collapsing the posterior to `μ`, you draw
  a small set of `S` sample vectors from the fitted posterior — call them scenarios
  `s = 1…S`. You then solve **one** MILP that contains `S` parallel copies of the future,
  one per scenario, **coupled together** by a rule called **non-anticipativity**: the
  decision variables for the very next interval (the one about to be applied for real) must
  be *identical* across all `S` scenario copies, because you don't yet know which scenario
  is real. Only the decisions for *later* intervals are allowed to differ per scenario,
  since by the time you reach them you'll have new measurements.

The optimizer is therefore forced to pick a next action that is feasible (and good) in
**every** scenario simultaneously — not just the average one. That's what buys the
robustness, and it's also exactly why the problem gets `S` times bigger.

### Level 4 — Explain it to a controls / optimization practitioner

Formally, Deterministic MPC solves, at each re-plan step `k0`, the deterministic program

```
min   f(x, μ)
s.t.  g(x, μ) ≤ 0        (battery, routing, precedence, etc., all HARD)
      x = (x_{k0}, x_{k0+1}, …, x_{n})
```

where `μ` is the posterior mean pulled from `parameters.csv`. `x_{k0}` is applied; the rest
is discarded and recomputed next step.

Stochastic MPC replaces the single deterministic parameter `μ` with a discrete distribution
`{(μ_s, π_s)}_{s=1}^{S}` (samples drawn from — or built to approximate — the fitted
posterior `N(μ, σ)`), and solves a **two-stage stochastic program with recourse**:

```
min   Σ_s π_s · f(x_{k0}, x^{(s)}, μ_s)
s.t.  g(x_{k0}, x^{(s)}, μ_s) ≤ 0         for every s = 1…S   (HARD, per scenario)
      x_{k0}                              — first-stage / here-and-now, shared across s
      x^{(s)} = (x^{(s)}_{k0+1}, …, x^{(s)}_{n})   — second-stage / wait-and-see, one per s
```

The only structural difference from the deterministic model is:
1. every constraint and cost term that involves the **uncertain future** gets an `s`
   index and is duplicated `S` times,
2. the **first-stage variables** (this interval only) are explicitly *not* duplicated —
   this is the non-anticipativity constraint, and it's the entire mechanism that makes the
   plan hedge instead of average,
3. the objective becomes a probability-weighted (or, for safety-critical constraints, a
   worst-case / chance-constrained) aggregate over scenarios rather than a single value.

Two standard refinements you'll eventually want: **scenario reduction** (cluster/prune the
sampled `μ_s` down to a smaller representative set that preserves the tail behavior, since
problem size scales roughly linearly in `S`), and choosing between **worst-case** hard
constraints (safest, most conservative) versus **chance constraints** (e.g., "battery floor
holds in ≥95% of scenarios," less conservative, needs mixed-integer indicator variables or
CVaR-style reformulations).

### Level 5 — Explain it to a PhD / stochastic-programming formalist

Deterministic MPC is the certainty-equivalent (CE) controller: at each stage it plugs the
posterior mean into the nominal MPC problem, which is optimal only under the (generally
false) assumption of *no epistemic or aleatoric variance propagating into feasibility* —
i.e., it is a separation-principle heuristic that is exact only for unconstrained LQG-type
problems, and here is provably suboptimal (and can be constraint-violating) whenever hard
state constraints (the SOE floor) are active, because `E[g(x,ω)] ≤ 0` does **not** imply
`g(x,μ) ≤ 0` is a valid proxy for `P(g(x,ω) ≤ 0) = 1` under a nonlinear/binary feasible
region.

Stochastic MPC is a receding-horizon instantiation of **two-stage stochastic MPC with
recourse**, where the uncertain parameter vector `ω` (activity powers) is represented by an
empirical/sample-based approximation of its posterior, `ω ∈ {ω_1,…,ω_S}` with weights `π_s`
(here derived from the same `TruncatedNormal` NUTS posterior already fit in
`1_Common.jl`/`0_Regression.jl` — no new estimator, only a new consumption pattern of the
existing one). The **non-anticipativity constraint** `x_{k0}^{(s)} = x_{k0}^{(s')} ∀ s,s'`
is the discrete-scenario encoding of the filtration constraint `x_{k0}` must be
`𝓕_{k0}`-measurable, i.e., a function of information available at `k0` only — this is
precisely what separates a stochastic program from `S` independent deterministic ones
solved in parallel. Under a shrinking/receding re-solve, this collapses to a certainty
about the *executed* trajectory (only `x_{k0}` is ever realized) while retaining decision
quality gains from anticipating recourse — the value of this is quantifiable as the **Value
of the Stochastic Solution (VSS)**, i.e., the gap between Stochastic MPC's realized cost and
Deterministic MPC's realized cost under identical realized disturbances (directly analogous
to your existing `approach0_vs_approach1.html` comparison, extended with a third column).
The practical cost is that the MILP's constraint count and, for the binary
activity-selection variables, its combinatorial branching factor, both scale with `S`,
motivating scenario reduction (Wasserstein-based clustering, or moment-matching) to keep
`S` small while preserving tail coverage of the constraint-relevant region — since it is
specifically the tail scenarios (not the mean-adjacent ones) that drive the
non-anticipativity decision, as the worked example below shows directly.

---

## Full worked example — Open-Loop vs Deterministic MPC vs Stochastic MPC, same reality, same numbers

### The toy world

- **4 intervals ahead**: `k0, k0+1, k0+2, k0+3` (15 min each)
- **CEV battery**: starts at **SOE = 10 kWh**. Hard floor **6 kWh**. Cap **12 kWh**.
- **Work requirement**: must complete **2 digging intervals** somewhere in this 4-interval
  window (a mini version of your lumpsum work quota).
- **Three possible actions each interval** (pick exactly one): **Dig**, **Travel**,
  **Charge** (MCS delivers power to the CEV).
- **Energy per interval:**
  - Travel: **always 0.5 kWh** drawn (no uncertainty).
  - Charge: **always +4.0 kWh** delivered (no uncertainty — it's supplied, not consumed).
  - Dig: **uncertain.** Posterior mean `μ → 2.0 kWh`/interval. Three sampled scenarios for
    what it could really be: **s1 = 1.5 kWh** (easy soil), **s2 = 2.0 kWh** (average),
    **s3 = 5.0 kWh** (hard soil, tail case).

**The hidden truth (unknown to every controller in advance, fixed for a fair comparison):**
the **1st time** any controller actually digs, the real draw turns out to be the hard-soil
case, **5.0 kWh**. The **2nd** time any controller digs, the real draw is the average case,
**2.0 kWh**. (This mirrors your real `ActivityPowerPool`: the true draw is keyed to *which
occurrence* it is, not which interval it happens to fall on.)

---

### Baseline — one-shot open-loop plan, solved once at `k0`, executed blindly

Solved once, using only `μ = 2.0 kWh` for digging, for the whole window, then **never
re-solved** no matter what actually happens. (This isn't MPC at all — no re-planning — but
it's included as a floor to show why re-planning matters before comparing the two real MPC
variants.)

**The plan (computed once, looks perfectly safe under `μ`):**

| Interval | Action (planned) | Planned SOE after |
|---|---|---|
| k0 | Dig | 10 − 2.0 = **8.0** |
| k0+1 | Charge | 8.0 + 4.0 = **12.0** |
| k0+2 | Dig | 12.0 − 2.0 = **10.0** ✔ 2nd dig done, quota met |
| k0+3 | Travel | 10.0 − 0.5 = **9.5** |

Looks completely fine on paper — never dips below the floor, quota met, ends at 9.5 kWh.

**Now execute it blindly against the hidden truth:**

| Interval | Action (fixed, unchanged) | Real energy | Real SOE after |
|---|---|---|---|
| k0 | Dig | **5.0** (1st dig = hard soil) | 10 − 5.0 = **5.0** ⚠️ **below the 6 kWh floor** |
| k0+1 | Charge (per fixed plan) | 4.0 | 5.0 + 4.0 = **9.0** |
| k0+2 | Dig | **2.0** (2nd dig = average) | 9.0 − 2.0 = **7.0** |
| k0+3 | Travel | 0.5 | 7.0 − 0.5 = **6.5** |

**What happened:** the plan never once looked at reality after `k0`. It got lucky — the
very next fixed step happened to be a Charge, which pulled it back above the floor purely
by coincidence of scheduling, not by design. If the fixed plan's next step had been another
Dig instead of a Charge, this approach would have driven the battery deeply negative and
had no mechanism to notice or correct it. **A real safety violation occurred and the
open-loop baseline never even measured it.**

---

### Deterministic MPC — certainty-equivalent, re-solves every interval, reacts after the fact

Same `μ = 2.0 kWh` guess each time, but now it **re-measures and re-solves** every 15 min.

**k0 — optimize using `μ`:** Digging looks safe (`10 − 2.0 = 8.0 ≥ 6`), nothing in the model
suggests otherwise. **Apply: Dig.**
Real draw (1st dig, hard soil): **5.0 kWh**. **Real SOE = 10 − 5.0 = 5.0 kWh — below the
floor.** This is *measured* this time (unlike the open-loop baseline).

**k0+1 — re-optimize from the real state, SOE = 5.0 (already violating the floor):** the
controller now sees it's in trouble and reacts. Best move to recover: **Charge.**
Real (certain): SOE = 5.0 + 4.0 = **9.0**.

**k0+2 — re-optimize from SOE = 9.0:** 1 of 2 dig intervals is already done (the k0 dig
counted as work even though it cost more energy than planned). Still safe-looking to dig
again under `μ`: `9.0 − 2.0 = 7.0 ≥ 6`. **Apply: Dig.**
Real draw (2nd dig, average): **2.0 kWh**. Real SOE = 9.0 − 2.0 = **7.0**. Quota now met.

**k0+3 — re-optimize from SOE = 7.0:** nothing left to do but finish safely. **Apply:
Travel.** Real SOE = 7.0 − 0.5 = **6.5**.

**What happened:** Deterministic MPC hit the exact same real violation as the open-loop
baseline at `k0` (SOE = 5.0, below floor) — it had no way to see the hard-soil draw coming
either, since it only ever plans against a single point estimate `μ`. The difference is
**it noticed immediately** (because it re-measures) and **spent the very next interval
recovering** instead of blindly continuing a stale plan. Reactive, not proactive — but at
least self-correcting.

---

### Stochastic MPC — scenario-based, hedges before acting

**k0 — sample 3 scenarios for digging** (s1 = 1.5, s2 = 2.0, s3 = 5.0 kWh), and test each
candidate first-move against **all three at once**, since the first move must be one shared
decision that survives every scenario:

| Candidate action at k0 | SOE under s1 | SOE under s2 | SOE under s3 | Safe in *every* scenario? |
|---|---|---|---|---|
| Dig | 10−1.5=8.5 | 10−2.0=8.0 | 10−5.0=**5.0** ❌ | **No — rejected** |
| Travel | 9.5 | 9.5 | 9.5 | Yes |
| Charge | 12.0 (cap) | 12.0 | 12.0 | Yes |

Digging is thrown out immediately — not because it's usually bad (it's fine in 2 of 3
scenarios), but because the one scenario where it fails (s3) is a hard-constraint breach,
and hard constraints must hold in *every* scenario, not on average. Between the two
survivors, charging first buys the most safety buffer for the risky digging still to come.
**Apply: Charge.** Real (certain): SOE = 10 + 4.0 = **12.0**.

**k0+1 — re-sample scenarios, test Dig now from SOE = 12.0:**

| Candidate | s1 | s2 | s3 | Safe in every scenario? |
|---|---|---|---|---|
| Dig | 12−1.5=10.5 | 12−2.0=10.0 | 12−5.0=**7.0** ✅ | **Yes — now safe even in the worst case** |

Because it built a buffer first, even the hard-soil scenario (5.0 kWh) can't push it below
the floor anymore. **Apply: Dig.**
Real draw (1st dig, hard soil, exactly as before): **5.0 kWh**. Real SOE = 12.0 − 5.0 =
**7.0 kWh — the floor is never touched.**

**k0+2 — re-optimize from real SOE = 7.0. Test Dig for the 2nd (final) required dig
interval:**

| Candidate | s1 | s2 | s3 | Safe in every scenario? |
|---|---|---|---|---|
| Dig | 7−1.5=5.5 | 7−2.0=5.0 | 7−5.0=**2.0** ❌ | **No — rejected, not enough buffer yet** |

Not enough buffer to risk digging yet — needs to recharge first. **Apply: Charge.**
Real: SOE = 7.0 + 4.0 = **11.0**.

**k0+3 (last interval, quota deadline) — test Dig from SOE = 11.0:**

| Candidate | s1 | s2 | s3 | Safe in every scenario? |
|---|---|---|---|---|
| Dig | 11−1.5=9.5 | 11−2.0=9.0 | 11−5.0=**6.0** ✅ (exactly at the floor) | **Yes** |

**Apply: Dig.** Real draw (2nd dig, average, exactly as before): **2.0 kWh**. Real SOE =
11.0 − 2.0 = **9.0 kWh**. Quota met, window ends.

**What happened:** Stochastic MPC hit the **exact same hidden truth** as the other two —
hard soil on the first dig, average on the second. But because it refused to dig until it
had built enough buffer to survive the worst sampled case, **the floor was never breached,
not even once** — the lowest the real SOE ever reached was 7.0 kWh, comfortably above 6.

---

### Side-by-side comparison

| | Open-loop baseline (one-shot) | Deterministic MPC (certainty-equivalent) | Stochastic MPC (scenario-based) |
|---|---|---|---|
| k0 decision | Dig (fixed in advance) | Dig | **Charge** (hedge first) |
| Floor breached? | **Yes, SOE=5.0** (never noticed) | **Yes, SOE=5.0** (noticed, fixed next step) | **No — never below 7.0** |
| Recovered how? | Pure luck (next fixed step happened to be Charge) | Reactive re-solve after the fact | Never needed recovery — avoided it |
| Final SOE | 6.5 | 6.5 | 9.0 |
| Re-solves? | 0 (open-loop) | 4 (every interval) | 4 (every interval) |
| Optimizer size | 1 deterministic MILP, once | 1 deterministic MILP, ×4 | 1 MILP with 3 linked scenario-copies, ×4 |

### The takeaway this example is meant to show

The open-loop baseline and Deterministic MPC made the **identical first mistake** — both
trusted `μ = 2.0` and dug immediately, both got blindsided by the same hard-soil draw, both
ended up below the safety floor in reality. Deterministic MPC's only advantage over the
open-loop baseline is that it *noticed* and *fixed* the problem one interval later — because
it re-plans, but re-plans against the same single point estimate every time, so it can only
react, never anticipate.

Stochastic MPC made a **different first decision** — it gave up one interval of digging
progress to charge first — specifically because it refused to commit to any action that
could fail under a scenario it had already sampled and could see coming. It paid for that
safety with one interval of delayed work (a real, quantifiable cost: less digging progress
early, more grid energy drawn for charging) — which is exactly your slide's trade-off:
*"proactively prevents battery deaths"* against *"computationally expensive, more
conservative."* This tiny example only has 3 scenarios and 4 intervals; your real MILP
would carry this same logic across every CEV, every activity, and a full-day horizon,
which is why scenario count and scenario reduction become the practical engineering
problem once the concept itself is settled.

## Changes 2–5, all four solver codebases (this session)

Following a diagnosis run that identified three real, data-confirmed issues (A0's realized
day-end battery shortfall going unreflected in reported KPIs; the deterministic MPC having no
tie-break preference between cost-tied "charge now" vs "wait" schedules; Approach 2's 5 i.i.d.
scenario draws having no guarantee of tail coverage), four changes were applied:

* **Change 2** — a small (`1e-6`) deliberately-scaled objective term breaking ties toward earlier
  grid-charging, in all four `3_MCSModel.jl` files (Approach 2's stochastic objective is
  deliberately excluded — its scenario hedge already covers that failure mode).
* **Change 3** — a terminal `SOE_CEV` shortfall penalty, computed after every run from the realized
  trajectory (pure end-of-day check, unrelated to the physical-floor capping that already existed),
  applied uniformly across all five approaches (A0, A1S, A1R, A2S, A2R) via each codebase's
  `4_MPCLoop.jl`, and surfaced through every output file's `TOTAL cost`.
* **Change 4** — Approach 2's `2b_ScenarioSampler.jl` now draws 5 fixed, stratified bins instead of
  5 i.i.d. draws, guaranteeing a tail scenario every re-solve instead of leaving it to chance.
* **Change 5** — a new `n_day_run` knob (currently 1) for chaining multiple days with real-state
  carry-over and a constant work requirement, in all four codebases.

(Change 1 — the fourth item originally proposed — turned out to need no code changes: the physical
floor check it was meant to add already existed, unmodified, in `apply_and_simulate!`.)

See `CHANGES_SUMMARY.md` at the repo root for the complete file-by-file list, and each
`Approach N/{Shrinking,Receding}_Horizon/docs/README.md` for the full rationale and worked
numeric examples.
