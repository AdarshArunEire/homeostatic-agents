# Prototype 04 — generalisation: route or rule?

Prototype 03b showed that the fixed radius-5 commute can be learned consistently-ish: the best configuration learns a clean water→food cycle in 52/100 seeds and survives it in 38/100. But a policy that solves one fixed map has not been shown to have learned anything transferable. It may have learned *the route* — this water, that food, this corridor — rather than *a rule* for staying alive between two resources wherever they happen to be.

Prototype 04 asks the separating question:

> does the agent learn a survival rule that holds across layouts, or did 3b just memorise one route?

Current answer: it learned the route, not the rule — and the reason it cannot learn the rule is now isolated. Across four hypotheses the failure narrows to a single mechanism: the agent cannot **self-commit to a directed crossing through the signal-free middle of the commute**. It is not a representation problem, not a reaching problem, and not an architecture problem. It is a directed-exploration problem, and no fix tried on the fixed-map framing moves it.

## One-line result

Carrying the 3b headliner onto a procedurally generated radius-20 world, the survival solve-rate is **0%** once the one thing propping it up — a boundary artifact — is removed. Senses don't help, a shorter approach doesn't help, and a memory channel doesn't help. The verdict is a reframe, not a number: stop chasing a config robust across seeds on a fixed map, and switch the success criterion to one weight set that generalises across resampled maps.

The four hypotheses, in order:

| H | question | verdict |
|---|---|---|
| H1 | does the 3b config transfer to a procedural map? | falsified on count (8%), survival was a rim crutch |
| H1.1 | is the 8% a boundary crutch? | confirmed — interior-only placement → 0/100 |
| H2 | do local senses help? | falsified — senses can't span the dead band |
| H3 | is the wall *reaching* the crossing or *valuing* it? | valuing — a shorter approach changes nothing |
| H4 | does a memory channel (GRU) help commit? | falsified — bottleneck is upstream of architecture |

## What changed from 3b

The substrate (comfort-v3 surface, death-penalty scaling $k/(1-\gamma)$, masked Bellman target, n-step returns, NoisyNet exploration, count-based novelty, annulus respawn) is carried over unchanged from 03b. Only the world changes.

| | 3b | 04 |
|---|---|---|
| world | fixed radius-5, water `(-5,0)`, food `(0,5)` | procedural radius-20, ~1261 hexes |
| layout | one map, all seeds | resampled bush/lake scatter |
| commute | fixed ~10-move trip | held at 9–11 by construction |
| evaluation | one route | split across map seeds — memorising one route cannot count |

The carried configuration is the 3b headliner: Noisy DQN + count-based novelty ($\beta = 0.1$) + 50k replay + 10-step returns + $\gamma = 0.99$.

### Solve gates (carried from 3b)

The strict, survival-aware gate is used throughout:

$$
\text{path efficiency} \ge 0.9
\quad\wedge\quad
\text{perfectish trip rate} > 0
\quad\wedge\quad
\text{eval deaths} \le k
$$

`clean-solve` (crossed the valley at least once, no death cap) is reported alongside as the looser companion — the gap between clean-solve and solved is the survival cost of the crossing. Per-seed solve-rate is the metric, not mean comfort: the camp-vs-cycle bimodality swamps any aggregate.

## The hypothesis arc

### H1 — does the 3b config transfer?

**Bet.** The oracle survives every seed, so the procedural world is fair. Can the 3b headliner learn it? Commute held at 9–11 so only the scatter and radius differ.

**Prediction.** None solve — deaths reset into fresh regions, the buffer fills with uncorrelated fragments, no commute is sampled densely enough.

**Result.** 8% solve [4.1–15.0% Wilson]. Falsified on count, but the failure structure is not the predicted one. Survival is bimodal (8 survive ≤5 deaths, 88 die ~125 median, gap empty), there is no safe camp (non-commuters die ~113× as much as commuters), and the cause axis is thirst↔hunger (corr(hyd_frac, deaths)=0.71). Crucially, **every one of the 8 winners is rim-localised** — `edge_proximity` unanimous at 1.0, dominant hub at radius for all 8.

**Verdict.** Partial — falsified on count, mechanism informative. The agent *learns* the commute but can't *sustain* it; winners survive only where the rim sits. Masking is one ring deep (only `d=radius` has clipped moves; `d ≤ 19` keeps the full 6), so the wall mechanically prevents thrashing and funnels the agent into a cycle it cannot self-generate. → **H1.1**.

### H1.1 — is the 8% a boundary crutch?

**Bet.** The 8% isn't generalisation. The rim is the only ring with clipped action masks, and the wall is doing the agent's commitment for it. Remove rim hubs and the wins should collapse.

**Prediction.** Ban rim hubs (interior-only candidates, MARGIN=1, which clips only `d=radius` and leaves commute length unchanged) → solve rate collapses toward ~0.

**Result.** 0/100 pass the strict gate [0.0–3.7% Wilson]. The collapse is camp structure, not absence of resource knowledge: all 100 seeds touch food, 83/100 have a nonzero trip-success rate, and at-resource affordances are intact (drink-at-water 96.6%, eat-at-food 75.0%). But those contacts never become a stable loop — water:food visit ratio ~25:1, 86/100 seeds visit food <3%.

**Verdict.** Confirmed for the survival claim; partial for the representation claim. The original 8% was a boundary artifact — remove the clipped rim and survival drops 8/100 → 0/100. What the agent lacks is not affordance recognition but **self-generated cycle stability away from the edge**.

> **Generator band bug (retires pre-fix procedural numbers).** The generator placed food in each water's 9–11 band but never checked food against *other* hubs. At `WATER_R_MIN=14` the triangle floor was 14−11=3, so cross-hub pairs undercut the band — 28/30 maps had a sub-band commute (min-pair median 5). Every procedural number above was measured on commutes shorter than 3b's 9–11. The generator now affirms `min(food, water) ≥ band_min` on every build (`WATER_R_MIN=20` → floor = band_min, parent-tracking, hard crash-on-violation). H1/H1.1 verdicts stand with greater reason; only the senses sweep is re-run on enforced-band maps.

### H2 — do local senses help?

**Bet.** H1.1 removed the crutch but left partial resource understanding. Giving the agent vision (r=1 tile features) and a scalar food-smell (r=3 cutoff) should make the corridor recognisable before the agent is on top of it. If the bottleneck is weak local representation, senses should reduce camping.

> Design stance: senses are added as **observation**, not reward. Scent-as-observation is policy-invariant and biologically honest chemotaxis; proximity-as-reward is rejected because it would encode the navigation knowledge that is the claimed contribution.

**Prediction.** Strict pass-rate rises above 0, or the failure visibly shifts — higher food visit %, lower water:food ratio, fewer deaths, more repeated trips.

**Result.** Null. Three arms (no-sense / vision / smell+vision), 40 seeds each, enforced-band maps. Solve **0/40 every arm**, Wilson CIs all [−0, 8.8%], fully overlapping. None of the predicted shifts appear (food visit is *highest* in no-sense). The one signal in the predicted direction — smell+vision perfectish-trip 0.250 vs 0.111 on W→F — is consistent with scent helping the last ≤3 hexes, but is computed over a handful of trips/seed and is noise-dominated.

**Verdict.** Falsified. The sensorium is local; the 9–11 commute carries a ~3–5 hex **dead band** with no resource signal. Senses sharpen the terminal approach but cannot touch the mid-commute navigation that does the killing. → H3: is the wall *reaching* the crossing, or *valuing* it?

### H3 — reaching vs valuing (midpoint probe)

**Bet.** Clean-solve ~22–27% with solved 0% means the agent crosses sometimes but never sustains. Two causes fit: it rarely *reaches* a crossing start, or it can't *commit* to one when positioned. An eval-only midpoint respawn — spawn ~4 hexes from both resources, on the line between a food and its nearest water — deletes the reaching half.

**Prediction.** If reaching is the wall, halving the approach lets it finish and survive (death-rate drops below ~0.027). If valuing is the wall, the shorter approach changes nothing.

**Result.** Confirmed null. 0/21 solved [0.0–15.5% Wilson], death-rate median 0.0268 — unchanged from the senses baseline. Some partial crossings (5/21 hit clean-crossing, median 4 successful trips) but no stable rhythm (W→F success 0.043, two-way floor 0.009). Occupancy argues against simple camping (food 0.64%, water 0.19%, dominant-cell 2.34%): the agent moves, occasionally crosses, then fails to close the loop. Water occupancy stays ~0.2% even when spawned adjacent.

**Verdict.** Confirmed-valuing. The failure is not reaching — placed near the middle, survival does not improve. The bottleneck is sustaining a directed crossing through the signal-free middle, where the observation is **aliased**: the same local state requires opposite actions depending on whether the agent is committed to food or to water. → H4: does the agent need persistent route intention?

### H4 — does a memory channel (GRU) help commit?

**Bet.** Position is fully observed via coordinates, so DRQN is *not* motivated by spatial partial observability. The bet is architectural: the MLP has no channel to carry route-commitment across the aliased dead band; a GRU hidden state does. Tested matched — `noisy_drqn_DQN` vs `noisy_DQN` control, **batch 64 in both arms** (BPTT cost forbids 512; matched-batch FF is the honest control), n-step 10, 40 seeds each. Buffers asymmetric by necessity (DRQN 1k-episode sequence replay; FF 50k transitions — batch, not buffer, is the matched lever).

**Prediction.** DRQN ≥ FF on crossing quality and solve rate; a non-trivial DRQN solve rate where FF sits at zero.

**Result.** Falsified, in the informative direction — the memory arm is no better and is *worse on crossing quality*.

| median metric | DRQN | FF |
|---|---|---|
| solved (gate) | 0/40 | 0/40 |
| clean-solve (no death cap) | 25.0% (10/40) | 32.5% (13/40) |
| wf success rate | 0.043 | 0.028 |
| wf path efficiency | 0.778 | **0.932** |
| wf perfect-ish trip rate | 0.250 | **0.550** |
| eval deaths | 530.5 | 532.0 |
| comfort | 0.409 | 0.406 |

Both arms 0/40 solved (Wilson UB 8.8%). The GRU makes *more* crossings but *scrappier* ones — roughly half the path efficiency and half the perfect-ish rate of the memoryless control. Paired by seed, FF produces the cleaner crossing on 12 seeds to DRQN's 7.

**Verdict.** Falsified. A memory channel did not buy route-commitment — it slightly degraded crossing quality and moved solve rate not at all. Memory is redundant on a fixed map because the task rewards latching a static route, which neither needs nor benefits from carried intention. The bottleneck sits **upstream of the architecture**: the policy must deliberately *seek* the crossing before any channel can carry intention through it, and it does not. A hidden state cannot remember a route it never takes.

## What this isolates

The four verdicts compose into one mechanism. The killing region is the dead band — the ~3–5 hex middle of the commute with no resource signal — and the agent fails there for a single reason that survives every intervention:

- It is **not representation** (H2): no local sense can reach into the band.
- It is **not reaching** (H3): placed in the band, it still doesn't cross.
- It is **not architecture** (H4): a memory channel doesn't help, and slightly hurts.

What's left is directed exploration. Credit cannot assign to a journey that never enters replay, and the policy never *commits* to sampling the journey — it crosses by accident at an exploration-determined rate, not because it has learned the crossing is worth taking. This is the same wall 3b's H2/H3 named ("cannot assign credit to a trajectory that almost never enters the training distribution"), now sharpened: on the procedural map there is no rim to substitute for the missing commitment, so it is exposed cleanly.

### Side-probe — Go-Explore (corroboration, not a solve attempt)

A uniform Go-Explore variant was wired in as a quick check on the exploration reading: archive visited cells, return to the frontier on respawn (weighted toward less-visited tiles, built only from where the agent went — no resource coordinates, so no knowledge leak). A training-crossing counter (`train_wf_trips`) was added to measure whether crossings actually enter replay, distinguishing "memory didn't help" from "exploration never sampled the loop."

Two findings, both consistent with the exploration verdict:

1. The crossing counter rises **flat-linearly** (R²≈0.99, no acceleration) — crossings accumulate at a constant exploration-rate cadence and never compound into deliberate seeking. A valuing agent would hockey-stick; this doesn't.
2. Forcing broader sampling did not raise crossings at 1M ticks and slightly *suppressed* the baseline's own late lock-in (uniform return scatters the agent across a full-map archive, diluting the corridor consolidation a locking-in seed needs).

So more *uniform* sampling is not the lever. What helps is letting a seed that stumbles into the loop *concentrate* on it — which points at credit/consolidation, not coverage.

## Design stances held throughout

- **Senses as observation, never as reward.** Proximity-as-reward would encode the navigation knowledge that is the contribution.
- **Per-seed solve-rate over mean comfort.** The camp-vs-cycle bimodality makes aggregate comfort meaningless.
- **One variable per sweep, matched controls.** H4's matched-batch FF control is the honest comparison even though BPTT forbids matching the buffer.
- **Written prediction before every sweep.** The entries above are the ledger; the contradicted predictions (H1's structure, H4's sign) are the most load-bearing.

## Stopping point and Proto 05

Across every config tried, the strict solve-rate is 0 and the bimodal camp-vs-limit-cycle failure is intact. The binding constraint is directed-exploration scarcity, and no credit-assignment intervention (n-step, DDQN, death penalty, $\gamma$, memory) has moved it. Further tuning on the fixed-map framing has no upside.

The reframe: stop optimising for a config robust across inits, and change the success criterion to **one weight set that generalises across maps** — each sim instance a freshly resampled world, evaluated on held-out maps with frozen weights. This is the route-vs-rule thesis stated directly: map resampling makes route-memorisation impossible by construction (it kills the H1 boundary/route crutch) and forces the policy onto the invariant — cross the band by sense.

**Proto 05 candidate:** curriculum over resampled maps, with band width as a non-leaking curriculum variable (it doesn't reveal where water is, only how far), gradient RL retained. 3b's coordinate curriculum failed (≈6/10 → 1/10); the likely culprit is the widening handoff / forgetting at each step-up, not the principle — logged as a known risk.

**Open tension, flagged not resolved:** resampling strips the coordinate crutch and leans on the local senses H2 showed are too short-range to span the band. So Proto 05 may re-motivate the memory channel H4 just retired — now for a non-redundant reason: carrying the sensory-gradient direction across the band when absolute position is useless. Same architecture, different and cleaner justification than the one it was first given.