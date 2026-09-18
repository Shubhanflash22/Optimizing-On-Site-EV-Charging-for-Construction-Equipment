# Peak-Ordering Analysis, By Mode

---

## Section 1: `:high` mode

**Expected order: OS (best/lowest peak) < CE < SB (worst/highest peak), and here's why.**

Confirmed directly from the code (`_draw_mode_z` in `1_Common.jl`): `:high` mode's z-score is `2.0 + 0.5*|randn()|` — **always** at least 2 standard deviations above the planning mean, every single draw, all day. This means reality *always* needs more energy than the fixed assumption expects — a systematic, one-directional surprise that never corrects itself.

OS commits to one full-day plan and executes it blind, never checking or reacting to anything — so it's never *punished* for discovering, mid-day, that things are worse than planned, because it never discovers anything. CE re-solves every 15 minutes and can only ever plan for what's left of the day; a correction discovered late (since the shortfall keeps compounding, always in the same direction) can only be sharper and more disruptive than the same fix would have been if built in from the first resolve. SB has that same "can't revisit the past" limitation as CE, plus an extra one: its 5 scenarios are freshly, independently re-randomized every single resolve, adding more instability on top of the same underlying systematic gap.

**Expected: OS < CE < SB.**

### Data — all 5 `:high` runs

| Run | NCD Peak order | Total cost (OS / CE / SB) | Missed work h (OS / CE / SB) |
|---|---|---|---|
| **04** (seed 7, 1200s) | OS(1.95) < SB(2.10) < CE(2.15) | 532.8 / 572.4 / 571.6 | 0.23 / 0.25 / 0.25 |
| **05** (seed 18, 1200s) | OS(1.95) < CE(2.12) < SB(2.33) | 462.9 / 572.3 / 76.7 | 0.20 / 0.25 / 0.00 |
| **06** (seed 55, 1200s) | OS(1.95) < CE(2.12) < SB(2.36) | 516.2 / 572.2 / 77.3 | 0.22 / 0.25 / 0.00 |
| **17** (seed 7, 5-day) | OS(2.53) < SB(2.85) < CE(2.86) | **2800.6** / 762.8 / 182.3 | **1.28** / 0.29 / 0.00 |
| **22** (seed 7, 3600s — same scenario as 04, 3x solve time) | OS(1.95) < SB(2.10) < CE(2.15) | 532.8 / 572.4 / 571.6 | 0.23 / 0.25 / 0.25 |

### Conforms?

**OS lowest in all 5 — clean, complete confirmation, no exceptions.**

**CE vs. SB splits into two groups, both already explained:**
- **05, 06** (CE < SB, matching the raw theory exactly): SB avoids the missed-work penalty entirely (0.00h), so its peak reflects only the base "extra 5-scenario instability" disadvantage — smaller than CE's, which is *additionally* paying for a forced digging-interval displacement.
- **04, 17, 22** (SB < CE, theory's predicted order reversed for this pair): here SB *also* gets caught by the same missed-work squeeze as CE (04: both miss exactly 0.25h). Once both are cornered by the same shortage, CE's single, exact, no-longer-flexible plan runs out of options more decisively than SB's marginally more adaptable one — so CE ends up worse specifically in the cases where the squeeze catches both of them, not just CE alone.

**Run 17 flagged separately**: OS's total cost is a severe **$2800.6**, driven by a 1.28-hour missed-work penalty compounding across 5 days of blind execution. "Best peak" and "best overall strategy" are different claims — worth remembering this run exists.

---

## Section 2: `:low` mode

**Expected order: CE ≈ SB (best/lowest peak, roughly tied) < OS (worst/highest peak), and here's why.**

`:low` mode's z-score is `-2.0 - 0.5*|randn()|` — the exact mirror of `:high`, always at least 2 standard deviations *below* the mean, every draw. Reality *always* needs less energy than planned — the same systematic, one-directional surprise, just in the pleasant direction.

CE and SB both discover this live and correctly scale back their charging. **OS cannot do this** — it committed to one plan built around the (here, unnecessarily generous) assumption and executes it blindly regardless of what reality actually needs, so it keeps charging as much as its now-excessive original plan called for, with no mechanism to ever correct it.

**Expected: CE ≈ SB < OS.**

### Data — all 5 `:low` runs

| Run | NCD Peak order | Total cost (OS / CE / SB) | Missed work h (all zero) |
|---|---|---|---|
| **07** (seed 7, 1200s) | SB(1.60) < CE(1.78) < OS(1.95) | 68.5 / 63.5 / 59.5 | 0 / 0 / 0 |
| **08** (seed 18, 1200s) | SB(1.61) < CE(1.80) < OS(1.95) | 68.5 / 63.6 / 59.8 | 0 / 0 / 0 |
| **09** (seed 55, 1200s) | SB(1.61) < **OS(1.95) < CE(2.10)** | 68.5 / **60.0** / 59.8 | 0 / 0 / 0 |
| **18** (seed 7, 5-day) | SB(2.21) < CE(2.21) < OS(2.65) | 180.2 / 133.6 / 152.7 | 0 / 0 / 0 |
| **23** (seed 7, 3600s — same scenario as 07, 3x solve time) | SB(1.60) < CE(1.78) < OS(1.95) | 68.5 / 63.5 / 59.5 | 0 / 0 / 0 |

### Conforms?

**SB lowest in all 5, cleanly — actually stronger than the theory predicted.** I'd expected CE and SB to be roughly tied for best; instead SB is individually, clearly best in 4 of 5. This suggests SB's fresh-every-window resampling doesn't just "keep pace" with CE under a downward-biased mode — it may let it react and scale back *faster* than CE's single, still relatively conservative fixed number can, turning what was a liability under `:high` into a genuine (if narrow) advantage here.

**OS worst in 4 of 5, exactly as expected.**

**Run 09 is the one clean exception, unresolved**: CE ends up worst there (2.10), not OS, and total cost shows CE actually *winning* overall (60.0, beating OS's 68.5) despite the worse peak — meaning something else is compensating. This doesn't have a proven mechanism yet the way the `:high`-mode CE/SB split does; it would need its own direct trace (CE's plan data, SOE trajectory) to pin down, the same way we resolved the `:high`-mode anomalies earlier. Flagging it honestly rather than folding it into the "clean match" count.

**07/23 nearly identical despite a 3x solver time difference** — same confirmation as `:high`: this is a structural effect of who gets to revise their plan, not a matter of search depth or solve quality.

---

## Section 3: `:near_mean` mode

**Expected order: no consistent winner — close to a coin flip, and here's why.**

`:near_mean`'s z-score is `0.5*randn()` — centered exactly on the assumed planning mean, with small spread. There is **no systematic direction** here for CE or SB's real-time correction to reliably exploit, the way there is under `:high`/`:low`.

The key resolution (worked through in discussion, not just assumed): OS and CE/SB each carry a genuine, opposing advantage that has nothing to do with bias existing or not.
- **OS's advantage**: it solves the whole day as one coordinated, joint decision — structurally better at finding a smooth spread than 96 disconnected 15-minute pieces ever can be, *but only for the assumed mean trajectory*, not the real one.
- **CE/SB's advantage**: they react to the *real*, currently-observed state at every resolve — but only one irreversible piece at a time, never able to revisit an earlier decision.

Under `:high`/`:low`, one of these two advantages dominates completely because the real trajectory diverges from the assumption *the same way, every interval, all day* — a strong, consistent signal to react to (or fail to). Under `:near_mean`, the real trajectory only wobbles randomly around the assumption with no persistent direction, so there's nothing systematic for CE/SB's reaction to reliably win against OS's coordination advantage. **Whichever wins on a given day depends entirely on which way that day's specific noise happened to lean — not a hidden, exploitable pattern.**

**Expected: OS ≈ CE ≈ SB, no consistent ordering.** SB carries one additional, separate source of its own volatility on top of this (its 5 scenarios are freshly re-randomized every resolve, independent of what reality is doing), so it may show *somewhat* more scatter than CE — but not a reliable "always worst."

### Data — all 5 `:near_mean` runs

| Run | NCD Peak order | Total cost (OS / CE / SB) | Missed work (all zero) |
|---|---|---|---|
| **10** (seed 7, 1200s) | OS(1.95) < CE(1.98) < SB(2.03) | 68.5 / 68.2 / 69.2 | 0 / 0 / 0 |
| **11** (seed 18, 1200s) | CE(1.91) ≈ SB(1.91) < OS(1.95) | 68.5 / 67.1 / 66.8 | 0 / 0 / 0 |
| **12** (seed 55, 1200s) | CE(1.90) < OS(1.95) < SB(2.05) | 68.5 / 67.1 / 69.7 | 0 / 0 / 0 |
| **19** (seed 7, 5-day) | CE(2.52) < SB(2.52) < OS(2.65) | 180.2 / 174.8 / 175.6 | 0 / 0 / 0 |
| **24** (seed 7, 3600s) | OS(1.95) < CE(1.98) < SB(2.06) | 68.5 / 68.2 / 70.0 | 0 / 0 / 0 |

### Conforms?

**Yes — the "no consistent winner" prediction holds up cleanly.** Across 5 seeds: SB worst in 3 (10, 12, 24), OS worst in 2 (11, 19) — indistinguishable from a coin flip, not the clean "SB always worst" pattern the raw hedging-only theory would have predicted. This is the corrected version of the theory actually matching the data, rather than the version that only looked right because it was checked against a selection-biased sample.

---

## Section 4: `:normal` mode

**Expected order: same as `:near_mean` — no consistent winner, but potentially more scatter, and here's why.**

`:normal`'s z-score is `randn()` — same centering on the mean as `:near_mean`, but the *full*, wider spread rather than `:near_mean`'s tighter clustering. The underlying mechanism is identical to Section 3 (no systematic direction, two opposing advantages with no reason to favor either), but with more raw variance in play, the day-to-day swings between "OS wins" and "CE/SB win" should plausibly be larger in magnitude, even without a consistent direction.

**Expected: OS ≈ CE ≈ SB, no consistent ordering, but a wider spread of outcomes than `:near_mean`.**

### Data — all 5 `:normal` runs

| Run | NCD Peak order | Total cost (OS / CE / SB) | Missed work (all zero) |
|---|---|---|---|
| **13** (seed 7, 1200s) | CE(1.91) < OS(1.95) < SB(2.02) | 68.5 / 67.2 / 69.3 | 0 / 0 / 0 |
| **14** (seed 18, 1200s) | SB(1.89) < CE(1.91) < OS(1.95) | 68.5 / 67.2 / 66.4 | 0 / 0 / 0 |
| **15** (seed 55, 1200s) | SB(1.92) < OS(1.95) < CE(1.96) | 68.5 / 67.9 / 67.2 | 0 / 0 / 0 |
| **20** (seed 7, 5-day) | CE(2.53) < SB(2.54) < OS(2.65) | 180.2 / 175.8 / 155.6 | 0 / 0 / 0 |
| **25** (seed 7, 3600s) | CE(1.91) < OS(1.95) < SB(2.02) | 68.5 / 67.2 / 69.3 | 0 / 0 / 0 |

### Conforms?

**Yes — again a genuine mix, not a systematic pattern**: SB worst in 2 (13, 25), OS worst in 2 (14, 20), CE worst in 1 (15). Notably, **SB is actually the *best* (lowest peak) in 2 of 5 seeds here (14, 15)** — directly contradicting "SB should be higher" as a blanket rule, even though the extra-volatility mechanism behind that intuition is real. This confirms the theory correctly predicts "no reliable winner," rather than the theory being wrong to abandon a directional claim.

---

## Section 5: `:live_data` mode

**Expected order: same reasoning as `:near_mean`/`:normal` — no consistent winner, and here's why.**

`:live_data` draws from real recorded field values rather than a constructed `Normal(mu,sd)` z-score at all — but the same underlying logic applies: nothing about how these values were recorded constructs them to sit consistently above or below the planning assumption the way `:high`/`:low` are deliberately built to. Without that systematic direction, the same two-competing-advantages argument from Sections 3–4 applies unchanged.

**Expected: OS ≈ CE ≈ SB, no consistent ordering.**

### Data — all 5 `:live_data` runs

| Run | NCD Peak order | Total cost (OS / CE / SB) | Missed work (all zero) |
|---|---|---|---|
| **01** (seed 7, 1200s) | SB(1.95) < OS(1.98) < CE(2.03) | 69.3 / 69.4 / 67.8 | 0 / 0 / 0 |
| **02** (seed 18, 1200s) | CE(1.91) < SB(1.95) < OS(1.98) | 69.3 / 67.1 / 67.7 | 0 / 0 / 0 |
| **03** (seed 55, 1200s) | SB(1.94) < OS(1.98) < CE(2.09) | 69.3 / 71.5 / 68.1 | 0 / 0 / 0 |
| **16** (seed 7, 5-day) | CE(2.51) < SB(2.66) < OS(2.68) | 171.7 / 154.7 / 158.7 | 0 / 0 / 0 |
| **21** (seed 7, 3600s) | OS(1.98) < CE(2.03) < SB(2.11) | 69.3 / 69.4 / 71.3 | 0 / 0 / 0 |

### Conforms?

**Yes, consistent with Sections 3–4**: SB worst in 2 (16, 21), CE worst in 2 (01, 03), OS worst in 1 (02) — again a genuine mix. This is a useful independent check, since `:live_data`'s draws come from a completely different generation process (real telemetry, not a constructed z-score) than `:near_mean`/`:normal`, yet produces the same "no systematic winner" signature — supporting that the *presence or absence of directional bias*, not the specific mechanism generating the numbers, is what actually determines whether a consistent ordering shows up.
