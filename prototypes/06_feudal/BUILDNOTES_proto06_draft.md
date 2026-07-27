# Proto 06 — feudal decomposition + honest eval

Not a hypothesis ledger. The reframe, the changes and why, and the one question Proto 06
exists to answer. Numbers are oracle-stack, smell 3, curriculum band, radius 20, band (9,11),
decay 0.7, h_fill 1.6 / s_fill 1.3, h_crit=s_crit 0.7, 8 seeds, unless stated.
solveScore = eval timeouts / (timeouts + deaths) on a nondoomed eval.

## The reframe

**Route→rule is settled; the wall is exploration, so stop testing survival and test finding.**
Proto 04 isolated the failure to one mechanism: the agent cannot self-commit to a directed
crossing of the signal-free middle of the commute. Every fix on the fixed-map framing left it
untouched. Proto 06 stops asking "can one end-to-end policy survive a cold spawn on a resampled
map" and asks the isolated question directly: *given survival is not sabotaged by luck or by
weak exploitation, can a legal agent FIND the second resource across the dead band?*

**Decompose the policy.** Orchestrator / explorer / pathfinder / consumer, each with an oracle
and a learned impl sharing one signature; memory is earned state, not a trained module. This
lets the explorer carry its own objective (reach food) instead of being poisoned by comfort,
and makes the oracle→learned swap testable per module.

**Demote the metrics.** solveScore is the metric. Comfort is a monitoring readout, not a tuning
target (below).

## Changes, and why

**Oracle is no longer god.** Proto 05's oracle knew the coords and navigated perfectly — its
"survives every seed" proves the world is *survivable by a perfect-knowledge agent*, nothing
more. Proto 06's oracle is a perfect pathfinder + consumer with a *legal reactive* explorer.
The oracle is a valid ceiling for exploitation and only a baseline for exploration. The legal
optimum is bracketed, not pinned — god 1.00 (illegal) > legal best > reactive 0.48.

**Orchestrator critical-drive interrupt (h_crit/s_crit=0.7).** The orchestrator explored
whenever *either* resource was unknown, so a thirsty agent hunting food never returned to
*known* water and died of thirst next to remembered water — abandoning, not camping. Fix:
detour to a known drive when it goes critical. Hydration eval deaths collapse (random 43→10,
smell_momentum 13→1). It is the minimal interleave a learned orchestrator would have to find.

**Comfort: asymmetric peak → tolerance band (free buffer to ideal+1).** The old OVER_W=0.02
still taxed the survival buffer — comfort fell monotonically as fills rose (1.3→1.9:
0.881→0.723), so the highest-comfort fill cell was the deadliest. The flat safe-zone (OVER_TOL=1.0)
is honest indifference: safe is safe, no lean-running pull. A smooth soft-knee (quartic) was
tried to kill the flat-zone kink and reverted — it re-taxed the buffer to buy a learnability
that is unwanted inside the safe zone.

**Fill economics.** h_fill is the survival lever — sweet spot 1.6 (1.3 too thin → thirst on
the away-trip; 1.9 wasteful). s_fill is nearly flat, slightly worse high; the pre-registered
call that s_fill would dominate backfired. decay_mult 0.7 is robust at h_fill 1.6; 0.9 bites
thin buffers hard — smell_momentum drops 0.48 → 0.35 (35 → 60 eval deaths) between the two.

**Honest eval: nondoomed spawn filter.** Reject any spawn where even perfect water-first play
cannot secure both resources within the expected-decay survival clocks + leeway. Raw deaths
conflate doomed spawns (luck) with real failure; the filter guarantees every eval spawn is
survivable by construction, so a death is a policy failure, not a death trap. The filter's slack
is the single largest lever on the number: smell_momentum scores 0.46 / 0.48 / 0.57 at leeway
0 / 10 / 25 (38 / 35 / 24 eval deaths). Leeway 10 is the operating point; solveScore is only
meaningful with the leeway stated.

**Smell held at 3.** Radius 5–12 span the commute and delete the dead band — the whole
phenomenon. 3 keeps the scentless middle. The cost of holding at 3 depends on the eval, and the
split is itself diagnostic. When water is handed over (food-isolation), smell 3 and smell 5 tie
at 0.86 — a wider scent buys nothing once the only open problem is the last approach to food.
On the nondoomed eval, where water must also be found from cold, smell 3 scores 0.48 against
smell 5 at 0.71 — a 0.23 gap. The cost is entirely on the water-finding leg: radius-5 scent
shortens *both* legs of the commute, so it partially deletes the dead band on the way out to
water, not just on the way to food. Holding at 3 preserves the band on both legs; the 0.23 is
the phenomenon, not an artifact, and paying it is the point.

## Standing results (nondoomed, smell 3, dm 0.7, h1.6/s1.3, 8 seeds)

| policy | eval deaths | solveScore | eval causes |
|---|---|---|---|
| GOD navigation (`eval_god_memory`, Proto-05 oracle power) | 0 | 1.00 | — |
| legal reactive (`smell_momentum`) | 35 | 0.48 | hydration 29, satiation 6 |
| momentum (no scent) | 62 | 0.33 | hydration 55, satiation 7 |
| random | 121 | 0.17 | hydration 85, satiation 35, both 1 |

GOD takes zero eval deaths → the world is fair, survivable by construction, and the nondoomed
filter holds. Random at 0.17 → the task is not trivial. The reactive explorer's deaths split
across hydration and satiation (29 / 6) → genuine two-resource navigation, not a food-only
artifact. The explorer ladder 0.17 → 0.33 → 0.48 (random → momentum → smell_momentum) is
monotone: chemotaxis is a real signal above blind momentum, which is above random, and 0.48 is
the floor a learned explorer must clear.

Food-isolation lineage (water handed over, beside-water eval): smell_momentum 0.86 at both
smell 3 and smell 5.

## Comfort and death_rate are not tuning targets

Mean comfort is uninformative as an objective. On the fill axis under the old asymmetric-peak
surface it was actively anti-correlated with survival (comfort fell 0.881 → 0.723 as fills rose
1.3 → 1.9, so tuning on it selects the deadliest cell). Under the current tolerance-band surface
that pull is gone and comfort instead tracks survival incidentally — 0.826 → 0.919 → 0.943 across
the explorer ladder as deaths fall 121 → 62 → 35, and 0.942 → 0.956 across leeway as deaths fall
38 → 24. Either way it carries no signal the death-vs-timeout score does not carry more directly.
Monitor only. death_rate is diluted by healthy ticks; per-life solve or per-life
death-vs-timeout is the honest cut.

## The question Proto 06 attacks

With survival decoupled from spawn luck (nondoomed) and from exploitation skill (oracle
pathfinder + consumer), and the dead band intact (smell 3):

> **Does a learned, memory-based explorer beat reactive chemotaxis (solveScore 0.48) and move
> up toward the god ceiling (1.00)?**

This is the non-redundant reason for memory that a fixed map could never provide: Proto 04 H4
falsified memory because latching a static route needs none. Here memory has a real job —
directed coverage of the scentless middle to *find* the second resource. The oracle hands over
the corridor (reactive floor 0.48, god cap 1.00); the learned explorer's whole task is to move
inside it. Beat 0.48 at smell 3 and memory has earned its keep; fail to, and the verdict is
"reactive chemotaxis suffices, the dead band is uncrossable by a local learner."
