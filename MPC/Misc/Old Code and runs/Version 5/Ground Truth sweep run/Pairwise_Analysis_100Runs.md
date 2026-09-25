# Pairwise Analysis: Why Each Approach Beats the Other, 100-Run GroundTruth Sweep

---

## A note before the three comparisons: why "the other one blew up" is not, by itself, a real answer

Before splitting wins into categories, it's worth addressing the question
directly: if all three approaches are supposedly facing the same real-world
digger, why would one of them randomly have a catastrophic day that the
others don't? If digging really needs 5 kW instead of the assumed 3 kW, all
three should see that same 5 kW, so why does only CE (or only SB) get
punished for it?

**Two separate things are going on, and both matter.**

**First -- they do not actually draw the identical sequence of real values.**
Confirmed directly from the simulation code: the live-recorded-power pool
each seed uses is shared, but OS, CE, and SB draw from it *sequentially* out
of one continuously-advancing random stream (OS draws first, then CE, then
SB). Each approach calls "give me the next real digging value" a different
number of times, in a different pattern, depending on its own schedule of
decisions. Because of that, by the time CE starts drawing, it has landed on a
different position in that shared stream than OS did -- and SB lands on a
still-different position after CE. Same underlying real-world dataset for
everyone, same seed for reproducibility, but not the identical specific
sequence of numbers. So it is not automatically true that "seed 1 gives
everyone the same hard day."

**Second, and more important -- even on days where the real demand genuinely
is high for everyone, the three approaches are not equally equipped to
absorb it.** This is the mechanism already proven in depth for `:high` mode,
and it applies identically here: the working day has a fixed budget of 28
fifteen-minute slots. If the real demand that day requires more charging
breaks than the plan assumed, CE and SB (which react in real time) each
discover the shortfall and defensively swap a work interval for a charging
interval -- a full 0.25h, costing a flat $500 penalty, every single time this
happens. OS never performs this swap because it never checks anything --
instead it runs its fixed plan through and ends the day with the battery
genuinely short of its terminal target, paying a different, usually smaller
penalty (~$471.7/kWh of shortfall) instead. **The blowup isn't random bad
luck landing unfairly on one approach -- it's a structural consequence of
each approach's specific reaction (or non-reaction) to a demanding day, and
that reaction is what this whole analysis is actually about.**

With that established, here are the three comparisons.

================================================================================
## 1. Why A0 (OS) beats A1 (CE) -- 44 of 100 seeds
================================================================================

### Easy explanation

Most of the time, OS only wins because CE has a bad day, not because OS made
a better call. CE's real-time re-planning sometimes runs into a day where the
digger needs more charging breaks than expected, and CE has to skip one work
turn to fit them in -- that skipped turn costs a flat $500 fine. OS never
skips anything (it doesn't react to the day at all), so it dodges that
specific fine. But on the rare occasion where CE doesn't have a bad day, OS
still edges it out a little, some of the time, just from getting a slightly
smoother charging schedule by luck.

### Technical explanation

The 28-slot working-day budget is the same for both. CE's real-time
re-solves, discovering a demanding day's actual power draws, defensively
convert a productive interval into a charging interval when the alternative
would risk the CEV's terminal SOE floor -- this is a discrete, all-or-nothing
swap (0.25h, $500), not a gradual cost. OS never performs this swap because
it has no feedback loop to trigger it; instead, on an equally demanding day,
OS risks a terminal SOE shortfall instead, which is a smaller, more gradual
penalty (~$471.7/kWh). This asymmetry in failure mode -- not decision
quality -- is what drives most of A0's wins here. On non-demanding days,
OS's single, uninterrupted full-day optimization can still occasionally
out-schedule CE's fragmented, 96-times-re-solved plan for the same reason
established in the earlier `:high`/`:low` mode analysis: A0 never has to
compromise a decision it's already made.

### Data

| Category | Count |
|---|---|
| A1 hit the interval-squeeze failure, A0 didn't | 31 |
| Both healthy, A0 still cheaper (real scheduling edge) | 11 |
| Both hit a failure, A0's was smaller | 2 |
| Total A0 wins | 44 |

Of A1's 33 total squeeze failures, 31 result in an A0 win here (the other 2
are cases where A0 also failed that seed, just less severely). The 11
"both healthy" wins are the only ones reflecting an actual scheduling
advantage -- about 16% of the 67 seeds where A1 stayed healthy.

================================================================================
## 2. Why A0 (OS) beats A2 (SB) -- 60 of 100 seeds
================================================================================

### Easy explanation

This one is different from the CE comparison. SB almost never has the
catastrophic $500 bad day (it's much better protected against that than CE
is) -- so most of OS's wins here are NOT because SB failed. Instead, on an
ordinary, healthy day, OS's schedule is just a little smoother than SB's,
more often than not. SB pays a small, regular cost for constantly hedging
against 5 different guesses instead of committing to one -- and that small
cost shows up as OS winning the "fair fight" more than half the time.

### Technical explanation

SB's current-interval charging decision has to remain reasonable across 5
independently-resampled scenarios at every single resolve, rather than one
confidently-known trajectory. This does not, by itself, cause the
catastrophic interval-squeeze failure (SB is well-protected against that,
with only an 8% rate) -- but it does mean SB's charging profile is smoothed
less precisely than OS's single, complete, whole-day optimization, even on
days with nothing structurally wrong. That precision gap is small per
instance but shows up consistently across most healthy-day comparisons.

### Data

| Category | Count |
|---|---|
| A2 hit the interval-squeeze failure, A0 didn't | 8 |
| Both healthy, A0 still cheaper (real scheduling edge) | 52 |
| Total A0 wins | 60 |

Unlike the A0-vs-A1 result, the large majority of these wins (52 of 60) are
genuine, both-healthy wins -- roughly 57% of all 92 seeds where A2 stayed
healthy. This is the clearest evidence in the sweep that SB's 5-scenario
hedging carries a real, repeatable cost on ordinary days, separate from its
much lower catastrophic-failure rate.

================================================================================
## 3. Why A1 (CE) beats A2 (SB) -- 50 of 100 seeds (needs the most care)
================================================================================

### Easy explanation

The raw score is a dead-even tie, 50 to 50 -- but that number hides two very
different stories. When CE beats SB, it's almost always because CE genuinely
scheduled things better that day. When SB beats CE, it's almost always
because CE had one of its bad ($500) days and SB didn't. So on a fair day
where nothing goes wrong for either one, CE is actually the better performer,
by a real margin -- SB's habit of hedging against 5 guesses instead of one
costs it a little bit, regularly, on ordinary days. What SB gets in return
is much stronger protection against CE's kind of catastrophic bad day.
That's a normal trade-off, like paying a small amount regularly for
insurance against a much bigger, rarer cost -- but it does mean SB is not
simply "better" than CE; it is trading small everyday losses for large
occasional savings.

### Technical explanation

Splitting the 50/50 by cause:

- SB's wins are driven almost entirely by CE's 33% interval-squeeze failure
  rate -- when CE fails and SB doesn't, SB wins by default, the same
  structural asymmetry established in Section 1.
- CE's wins, on days where neither approach fails, come from the same
  mechanism established in Section 2: CE's single, exact planning assumption
  lets it commit precisely to a smooth charging schedule, while SB's
  non-anticipative, 5-scenario-averaged decision is structurally less able
  to do the same, even with no bias in the underlying draws to exploit.

This is not a contradiction of SB's hedging being worthwhile -- it's the
expected shape of an insurance trade-off, and the two halves of that trade
are both directly measurable in this data: SB avoids roughly 25 extra
catastrophic days per 100 (33% minus 8%) at the cost of a real, smaller
scheduling disadvantage on the 44 fair-fight days it loses. Whether that
trade is a good one in aggregate is a real question, not an assumption --
the Data section below gives the actual arithmetic.

### Data

| | A1 (CE) wins | A2 (SB) wins |
|---|---|---|
| Driven by the other one's interval-squeeze failure | 6 | 31 |
| Both healthy, this one still cheaper | 44 | 17 |
| Total | 50 | 50 |

On the 61 seeds where both stayed healthy, CE wins 44 and SB wins 17 -- a
2.6-to-1 ratio in CE's favor when neither has a bad day.

**Checking whether the insurance trade actually pays off, in dollars, not
just win-count**: SB avoids roughly 25 extra $500 penalties per 100 seeds
that CE is exposed to (~$12,500 saved). Against that, SB loses 44
fair-fight comparisons to CE, each by a typically small margin (a few
dollars, based on the near-identical baseline costs seen throughout this
sweep) -- on the order of $100-200 total across all 44 losses. This is
consistent with, and explains, the summary statistic already established:
SB's mean cost ($100.52) sits far below CE's ($232.24) across the full
100-seed sweep, despite CE winning more head-to-head genuine matchups.

**What would undercut this framing, and hasn't been checked yet**: whether
CE's losing margin on those 44 fair-fight days is actually as small as
assumed here, or whether SB's 8% failure rate holds up or drifts upward
with more seeds. Both are worth verifying directly before treating this
trade-off as fully settled.
