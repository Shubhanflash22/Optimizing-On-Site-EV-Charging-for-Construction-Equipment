# Why A1's NCD Peak Comes Out Higher Than A0's — All 5 Modes

## Quick version — one glance, no numbers

**`:high` mode** — A0 makes one careful plan at the start of the day and
sticks to it. A1 keeps checking in and adjusting, but since its underlying
assumption about the day never actually improves, checking in repeatedly
just means getting caught off guard by the same kind of bad news, over and
over. Each time it gets caught off guard, it pushes a little harder to
compensate, and those small pushes stack up into a noticeably higher peak
by the end of the morning.

**`:low` mode** — Both approaches have to sit out a mandatory
no-charging evening window, and both do so fairly. A0 spends the free
afternoon beforehand quietly topping itself up. A1, sensing it genuinely
doesn't need to charge yet, correctly does nothing during that same
afternoon — but that means once the mandatory quiet period ends, A1 has a
much bigger job left to finish in the same amount of remaining time, and has
to push harder to get it done.

**`:near_mean` mode** — Nothing is systematically off in either
direction here, so the small difference that shows up comes down to a
scheduling habit. A0 tends to its vehicle's needs early and often, which
frees it up for a long, relaxed charging session before the evening
restrictions kick in. A1 tends to its vehicle less often, so the vehicle
needs attention again sooner — cutting A1's charging session short and
forcing a slightly brisker pace to fit the same job into less time.

**`:normal` mode** — The same story as `:near_mean` mode, just even
milder. A0's habit of servicing its vehicle generously and early leaves it a
long, calm stretch to charge in. A1's more sparing approach means its
vehicle needs help again a little sooner, trimming down A1's available
charging time and nudging its pace up just slightly to compensate.

**`:live_data` mode** — A0 and A1 both start the day with the exact same
good plan — genuinely identical, down to the details. A0 just carries that
plan out and never looks back. A1 gets a chance to double-check its plan
partway through the morning, even though nothing has actually gone wrong
yet — and this time, checking again leads it to a different answer than
before, one that happens to work out slightly worse. Everything that
follows is just the natural consequence of that new answer playing out.
There's no real mistake here and nothing forced it — it comes down to
A1 getting a second chance to answer the same question, and this time,
landing on a different one.

---

Each section below stands on its own and explains that mode's mechanism
using only what happens within that mode, without relying on comparisons
to any other mode.

---

## `:high` mode

Under `:high` mode, the digger genuinely uses more power than the plan
assumes, every single interval, all day, with no exceptions. A0 calculates
its charging rate exactly once, before the day even starts, using that same
wrong assumption — but because it only ever calculates it that one time, it
locks in a number (1.951 kW) and simply runs with it, unaffected by anything
that happens afterward. A1 does something that sounds smarter but backfires
here: it recalculates its charging rate fresh every fifteen minutes, reacting
to the real state of the battery each time. The problem is that A1's
assumption about the future never actually changes — it's the same wrong,
too-low number every single resolve — so every fifteen minutes, A1 discovers
its battery is a little worse off than its own last calculation expected, and
revises its charging rate upward to compensate. This happens eight times in a
row between 10:15am and noon, each nudge small on its own (2.03, 2.04, 2.05
kW, and so on), but by noon they've stacked into a rate of 2.10 kW that then
holds for the rest of the day and becomes the number the demand charge is
billed on. A0 never goes through this ratcheting, not because its assumption
is any more accurate, but because it only gets asked the question once,
before reality has had any chance to disagree with it even a single time —
let alone eight times over. In short: reacting to reality sounds like it
should help, but if the model reacting can't actually fix what's wrong with
its own assumption, reacting repeatedly just means getting surprised
repeatedly, and each surprise pushes the number a little higher than a
single, unhurried calculation ever would have needed to go.

---

## `:low` mode

Under `:low` mode, real power draws run consistently below what the plan
assumes, so A1's real-time monitoring correctly recognizes there's no urgent
need to keep the battery topped up — it can afford to travel less and defer
charging without risk. Both A0 and A1 are additionally bound by a hard,
shared constraint that has nothing to do with either strategy's
intelligence: the on-peak demand-charge window runs from 4pm to 9pm exactly,
and drawing any power at all during that window triggers a separate, costly
on-peak demand charge. Both approaches correctly avoid it completely,
confirmed directly by their reported on-peak peak sitting at exactly zero for
each — neither one draws a single watt between 4pm and 9pm, and both resume
at precisely 9:00pm, the first moment that's genuinely safe. The two pricing
rates involved don't move in lockstep either: the raw per-kWh energy price
actually drops back to its normal rate at 8pm, an hour before the
demand-charge window itself ends, which is what can make an 8pm restart look
plausible at first glance — but the model is correctly responding to the
separate, longer demand-charge boundary, not the energy price alone. What
actually separates A0 from A1 here is what each one did before that shared
blackout began: A0 had already been charging steadily since 12:45pm and had
banked a substantial reserve by 4pm, while A1 did no charging at all during
that same legal, cheap afternoon window, since its real-time view of the
battery genuinely didn't call for it yet. Both are released from the
blackout at the identical moment, 9pm, but A1 arrives there with a much
larger shortfall still to fill in the same remaining hours — and that
compressed, larger catch-up is what pushes its peak above A0's, not any
difference in how well each one respects the on-peak rule itself.

---

## `:near_mean` mode

Under `:near_mean` mode, the digger's real power draws land close to the
plan's assumption on average, with no systematic bias in either direction,
so there's no consistent force pushing one approach's charging rate above
the other's. The gap that does appear (a small 0.016 kW) traces back to
something more specific than either strategy's charging philosophy: it's
about how each one schedules the CEV's own morning top-ups. A0 happens to
service its CEV generously and early, finishing that obligation by 12:15 and
freeing its MCS for a long, uninterrupted, low-rate charging session lasting
until 15:45, right up to the on-peak blackout. A1 schedules its CEV's
top-ups more sparsely, letting its battery run further down, which means the
CEV needs attention again by 13:30 — forcing the MCS to leave the grid node
earlier and squeezing its own charging into a shorter 10:15–13:00 window.
Both approaches then face the identical mandatory 4pm–9pm blackout and both
need real reserve to serve their CEVs through part of it, so pre-blackout
charging is genuinely necessary for either one — this isn't a case of one
strategy doing something wasteful the other avoids. The difference is simply
that a shorter available window, driven by an earlier CEV service
obligation, mechanically requires a slightly higher rate to bank the same
energy in less time — a scheduling coincidence rather than any systematic
bias, which is why the resulting gap stays small.

---

## `:normal` mode

Under `:normal` mode, the digger's real power draws land close to the plan's
assumption with no systematic bias, so there's no consistent force pushing
one approach's charging rate above the other's. The gap that does appear is
tiny — the reported KPI figures round to 1.95 kW for A0 and 1.96 kW for A1,
though the precise underlying values (1.951 vs 1.957 kW) put the real gap
at only about 0.006 kW. It traces back to how each approach schedules its
CEV's morning top-ups. A0 services its CEV generously and early — five
separate charging events by noon, keeping its battery comfortably above 8
kWh by late morning — finishing that obligation early enough to free its
MCS for a long, uninterrupted, low-rate charging session lasting all the way
to 15:45, right up against the on-peak blackout. A1 schedules its CEV's
top-ups more sparsely (only three by noon), lets its battery run down
further (to 5.94 kWh), and consequently needs the CEV serviced again at
13:30 — forcing its MCS to leave the grid node by 13:15 and compressing its
own charging into a shorter 10:15–13:00 window, during which it charges hard
enough to fill completely to 250 kWh. Both approaches then face the
identical mandatory 4pm–9pm blackout, and both need genuine reserve to serve
their CEVs through part of it — this is not a case of one approach doing
something wasteful the other avoids. The difference is simply that a shorter
available window, driven by an earlier CEV service obligation, mechanically
requires a slightly higher rate to bank the same energy in less time, and
because the scheduling gap between the two approaches is smaller here than
in other cases, the resulting peak difference stays essentially negligible.

---

## `:live_data` mode

Under `:live_data` mode, the digger's real recorded power draws mostly track
what the plan assumes, so the interesting part of this mode isn't a
mismatch between plan and reality at all — it's what happens when a
correct, sensible plan gets recalculated for no real reason and comes back
different.

A1's very first plan, computed at 8:00am with the whole day ahead of it, is
essentially identical to A0's actual schedule: it plans to start grid
charging at 12:45pm, at exactly 1.951 kW, running continuously through
15:45, pausing for the mandatory 4pm–9pm blackout, and resuming at 9:00pm.
Not just similar — the same start time and the same rate, down to the exact
decimal. Since A0 and A1's very first plan begin from the identical starting
state and the identical assumptions, this makes sense: given the same
problem, they arrive at the same answer.

A0 then simply executes that plan. It never gets asked to reconsider, so
whatever it decided at 8:00am is exactly what happens all day.

A1, being closed-loop, re-solves again at 10:15am. In the couple of hours
before that, the MCS's own battery had briefly run about 2 kWh behind what
the original plan expected — but that gap had already fully closed by
10:00am, back to an exact match. So by the time the 10:15 re-solve happens,
there is nothing left to react to; the real world is sitting right where
the original plan expected it to be. And yet the 10:15 re-solve comes back
with a different answer: rather than waiting until 12:45 the way the
original plan intended, it decides to start charging immediately, right
then, at a higher rate than before.

This is the actual heart of it: solving the same kind of problem twice does
not guarantee the same answer twice. The first solve, at 8:00am, landed on
one valid schedule. The second solve, at 10:15am, facing a nearly identical
situation, happened to land on a different valid schedule instead — not a
worse one by any obvious measure at that moment, just a different one that
the solver arrived at this particular time. Once that new schedule is
adopted, everything that follows is a mechanical consequence of it: the MCS
commits to an uninterrupted charging block from 10:15 to 13:00, filling
close to capacity; the CEV, still following its own separate schedule,
happens to need servicing again at 13:30, so the MCS has to stop and leave
for it; the block that's left over before the 9pm blackout goes unused; and
the overnight recovery has to run at a correspondingly higher rate to make
up for the shorter, earlier block — which is what ultimately sets the day's
peak about 0.055 kW above where A0's stayed.

None of this was forced by anything going wrong. The CEV's real consumption
did drift ahead of plan later in the morning, but only after the 10:15
decision had already been made — it isn't the cause of that decision, just
something that happened afterward and got folded into how the story played
out. The actual cause is simpler and, in a sense, less satisfying: A1 got a
second chance to answer the same question the 8am plan had already answered
well, and this time, the answer it landed on happened to be a little more
expensive. A0 never got that second chance, so it never had the opportunity
to do worse — or better — than its first try. The whole gap, in the end,
comes down to chance: re-solving offers no guarantee of reproducing a good
answer, only the opportunity to find one again, and this time it found a
different one instead.

---

## Corrections and additions made from the original notes

- **`:normal` mode's gap figure**: the original notes stated "0.01 kW,"
  which is the KPI CSV's rounded value (1.95 vs 1.96). The precise
  underlying charging-rate values are 1.951 kW (A0) and 1.957 kW (A1), so
  the actual gap is closer to 0.006 kW. Both figures are now given, with the
  more precise one flagged as such.
- **Cross-references removed**: the `:near_mean` and `:normal` sections
  originally referenced `:low` mode's gap ("the same general principle
  behind `:low` mode's bigger gap") and referenced each other directly
  ("exactly as with `:near_mean` mode," "even smaller than Near Mean's
  0.016 kW"). Per the requirement that each mode's explanation stand
  independently, these cross-references have been removed; the underlying
  mechanism explanation for each mode is otherwise unchanged.
- **`:live_data` mode — completely rewritten.** Earlier drafts of this
  section explained the gap as the MCS deliberately catching the day's
  cheapest price window, and separately as the MCS needing to top up
  itself after using more than expected. Both were checked directly and
  found incomplete: A1's very first plan (8:00am) turned out to be
  essentially identical to A0's actual schedule — same start time (12:45),
  same exact rate (1.951 kW) — meaning both approaches arrive at the same
  answer when solving the same problem. The MCS's brief early depletion had
  also already fully closed by 10:00am, before the 10:15 re-solve that
  actually changed the plan. The real mechanism is that A1's 10:15 re-solve
  simply landed on a different, valid schedule than its own first plan had,
  for no forced reason — a consequence of re-solving being repeated rather
  than fixed, not a response to anything going wrong. The section and the
  plain-language summary above have both been rewritten to reflect this.
- Everything else in the original notes — every specific number, timestamp,
  and mechanism described — was checked directly against the corresponding
  mode's actual KPI, realized, and plan CSVs and confirmed accurate; nothing
  else was altered.
