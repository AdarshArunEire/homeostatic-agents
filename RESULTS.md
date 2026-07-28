# Results

Every headline number, with seed counts. No narrative — see
[`README.md`](README.md) for the argument and [`BUILDNOTES.md`](BUILDNOTES.md) for the
predictions that preceded each result.

Conventions: `solveScore = timeouts / (timeouts + deaths)` on the evaluation segment. Wilson
intervals where n is small. "Seeds" means independent map/init seeds unless stated.

---

## Prototype 3b — fixed radius-5 commute

Best configuration: Noisy DQN + count-based novelty (β=0.1) + 50k replay + 10-step returns +
comfort-v3 + γ=0.99. **100 seeds.**

| metric | value |
|---|---|
| learn a clean water→food limit cycle | **52%** |
| also survive it (≤5 eval-death gate) | **38%** [29–48% Wilson] |
| median comfort among solved seeds | 0.93 |
| seeds finishing greedy eval with zero deaths | 49/100 |

**Replay buffer.** 5k → 50k–100k improves solve-rate; a near-non-evicting 520k buffer collapses
to **0/10**. Rules out simple FIFO forgetting — the failure mode is distributional staleness
under non-stationarity.

---

## Prototype 04 — procedural r=20 maps: route or rule?

| H | question | verdict |
|---|---|---|
| H1 | does the 3b config transfer? | falsified on count (8%); survival was a rim artifact |
| H1.1 | is the 8% a boundary crutch? | **confirmed** — interior-only placement → **0/100** [0–3.7%] |
| H2 | do local senses help? | falsified — 0/40 every arm; senses cannot span the dead band |
| H3 | is the wall *reaching* or *valuing* the crossing? | valuing — a shorter approach changes nothing (0/21) |
| H4 | does a memory channel (GRU) help commit? | falsified — 0/40 both arms; FF beat DRQN on crossing quality |

H4 detail, 40 seeds each: clean-solve DRQN 25.0% vs FF 32.5%; path efficiency 0.778 vs **0.932**;
perfect-ish trip rate 0.250 vs **0.550**. The GRU makes *more* crossings but scrappier ones.

**Isolated mechanism:** directed exploration. Credit cannot be assigned to a journey that never
enters replay, and the policy never commits to sampling it. Corroborated by a training-crossing
counter rising flat-linearly (R²≈0.99 — no acceleration, so crossings are accidental rather than
sought).

---

## Prototype 06 — feudal decomposition

Standing config throughout: smell 3, band (9,11), nondoomed eval (leeway 10), decay 0.7,
h_fill 1.6 / s_fill 1.3, crit 0.7. **8 seeds** unless stated.

### Explorer ladder — the reference frame

| policy | eval deaths | solveScore |
|---|---|---|
| GOD navigation (perfect-knowledge oracle) | 0 | **1.0000** |
| legal reactive (`smell_momentum`) | 35 | **0.4776** |
| momentum, no scent | 62 | 0.3333 |
| random | 121 | 0.1712 |

Reactive deaths split hydration 29 / satiation 6 — genuine two-resource navigation, not a
food-only artifact.

### Pathfinder — solved

Trained out-of-sim on bare hex geometry; sparse arrival reward + HER; no cloning, no shaping.

| seed | approach (1–5) | commute (6–11) | long (12–20) | far (21–40) | gate 0.99 |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.9954 | 0.8943 | PASS |
| 1 | 1.0000 | 1.0000 | 0.9850 | 0.9511 | FAIL |
| 2 (shipped) | **1.0000** | **1.0000** | **1.0000** | 0.9880 | **PASS** |

**2/3 clear the pre-registered gate.** `excess_steps_per_move` = 0.0000 on every gated band for
the shipped module. In-sim: 34,859 invocations, solveScore delta **+0.000** (35 deaths both arms).
Oracle agreement only 0.56 — independently optimal, not a copy.

**Checkpoint selection matters.** Same seed, same run, different checkpoint:

| seed 2 | commute | long | far | gate |
|---|---|---|---|---|
| final weights (ep 4000) | 0.9771 | 0.7685 | 0.3273 | FAIL |
| best checkpoint (ep 2500) | 1.0000 | 1.0000 | 0.9880 | PASS |

### Consumer — learnable, unmeasurable

Drink consumer, held-out: **37 eval deaths vs oracle 35**; `drink_rate_at_water` deviation 1.2%
against a 4.4% detection floor → certified.

Calibration sweep found **`solveScore` non-monotone** under controlled degradation (a knowably
worse consumer scored higher), and only `drink_rate_at_water` qualifies as a detector, spanning
0.044. Cause: `OracleOrchestrator` re-issues CONSUME every tick while below target and
`DRINK_AMOUNT = 0.15`, so fill fraction is a **rate, not a level** — the retry loop absorbs a
timid consumer.

Eat consumer: **0 dispatches in 1200 nondoomed ticks** (279 drinks in the same run). Untrained by
design; finding food is the phenomenon under study.

### Orchestrator — learns, and learns the water-cult attractor

**Calibration (P4.0).** `solveScore` monotone on the epsilon axis, dynamic range **0.478**;
`p05_satiation` range **0.940**. The most measurable module in the prototype.

| epsilon | solveScore | deaths | | crit_scale | solveScore | deaths |
|---|---|---|---|---|---|---|
| 0 | 0.4776 | 35 | | 0.0 | 0.3721 | 54 |
| 0.05 | 0.4366 | 40 | | 0.5 | 0.3444 | 59 |
| 0.15 | 0.4267 | 43 | | **1.0 (baseline)** | 0.4776 | 35 |
| 0.35 | 0.2066 | 96 | | **1.5** | 0.5926 | 22 |
| 0.70 | **0.0000** | 382 | | 3.0 | 0.2347 | 75 |

`crit_scale` is a clean inverted U with an interior optimum. Both predicted failure modes
reproduce: at 0 the critical-drive interrupt never fires and hydration deaths rise 29 → 45; at
3.0 it fires constantly and `water_visit_pct` reaches **80.9%**. Only `crit_scale=3.0`
[0.162, 0.328] separates from the baseline [0.363, 0.595].

**Learned result (P4.1).** 4-way abstract head (GO_WATER / GO_FOOD / CONSUME / EXPLORE),
exploration delegated, survival reward (+0.01/tick, −1.0 on death).

| | solveScore | CI | deaths |
|---|---|---|---|
| oracle | 0.4776 | [0.363, 0.595] | 35 |
| learned `o0` | **0.1622** | [0.105, 0.242] | 93 |

CI-separated below. Action shares over the eval segment, 8 seeds:

| action | oracle | learned |
|---|---|---|
| GO_WATER | 0.342 | **0.571** |
| GO_FOOD | 0.342 | **0.075** |
| CONSUME | 0.192 | 0.235 |
| EXPLORE | 0.123 | 0.119 |
| **GO_WATER : GO_FOOD** | **1.0 : 1** | **7.6 : 1** |

Deaths hydration 62 / satiation 31, against the oracle's 29 / 6.

**That 7.6 : 1 is measured on the wrong denominator — see P4.2.** GO_FOOD is masked illegal
unless `food_known`, and the learned module holds `food_known` on 32.1% of eval ticks
against the oracle's 92.8%. Conditioning:

| denominator | oracle | learned |
|---|---|---|
| raw, as originally published | 1.00 : 1 | 7.63 : 1 |
| ticks where both GO_* were legal | 1.00 : 1 | **3.37 : 1** |
| ...and standing on neither resource | 1.00 : 1 | **24.22 : 1** |

Half the headline was an availability artefact. The last row is the honest arbitration
number and it is far worse, because most GO_FOOD is emitted while the agent is still
standing on water and abandoned one tick later. Median GO_FOOD run length is **1.0** against
the oracle's **10.0**.

### Orchestrator — commitment recovers a third of the gap; a switching cost does not

Enforcing an option horizon at inference on the same weights, 24 seeds:

| arm | solveScore | Wilson | deaths | w→f per life | food_known |
|---|---|---|---|---|---|
| oracle | 0.5053 | [0.43, 0.58] | 94 | 17.20 | 92.1% |
| hold=1 (as trained) | 0.1683 | [0.13, 0.21] | 262 | 0.09 | 32.9% |
| **hold=15** | **0.2689** | **[0.22, 0.33]** | 174 | 1.13 | 45.4% |
| hold=20 | 0.2624 | [0.21, 0.32] | 194 | 1.31 | 53.3% |
| hold=25 | 0.2259 | [0.18, 0.28] | 209 | 0.95 | 43.2% |
| hold=40 | 0.2194 | [0.17, 0.27] | 217 | 1.31 | 50.8% |

hold=15 separates from hold=1 and the curve is an inverted U with an interior optimum near
commute length. **Control:** the same horizon applied to a uniformly random legal option
scores **0.0000** at both hold=1 and hold=15 — persistence helps this policy and harms an
arbitrary one, so it is commitment to the *learned* choices that does the work.

Recovered: (0.2689 − 0.1683)/(0.5053 − 0.1683) ≈ **30% of the gap**. The residual is
acquisition, which the horizon does not touch.

Replacing the enforced horizon with a learned one — a deliberation cost per abstract-action
switch — fails monotonically:

| delib_cost | CONSUME share | CONSUME dwell | best solveScore |
|---|---|---|---|
| 0 | 0.235 | — | 0.168 |
| 0.002 | 0.573 | 30.4 | 0.149 |
| 0.02 | **0.789** | **307.2** | 0.174 |

Every arm collapses onto CONSUME and dies of thirst standing on food. A switching penalty
cannot distinguish following through on a plan from standing still — both have zero
switches — which is precisely what the enforced horizon can express and the cost cannot.

The policy is not collapsed — it uses all four actions, and its exploration share is within 0.004
of the oracle's. It simply prefers water **7.6 to one** and dies of thirst at twice the rate.
That is the water-cult attractor from Prototype 3b, reproduced in a module whose only job is
arbitration, with a non-aliased observation and oracle execution beneath it. **The attractor
survives isolation, so it is a property of the task rather than of end-to-end learning.**

### Explorer — not learnable on this observation contract

| attempt | held-out solveScore | mechanism of failure |
|---|---|---|
| DQN, 3 seeds | 0.1130 / 0.1092 / 0.0553 | reward diluted to 0.06 rewarded samples per batch |
| DQN + stratified replay | 0.1059 / 0.1030 / 0.0591 | discovery bootstrapped into an unrelated post-travel state |
| DQN + terminal fix | 0.1037 | fix correct, not binding |
| DQN + softmax eval (T sweep) | peak ≈ random | best T was the *most uniform*; best use of Q was to ignore it |
| policy gradient, entropy 0.02 | 0.1589 | policy stayed at 96% of uniform entropy |
| policy gradient, entropy 0.003 | collapsed | entropy → 0, deterministic looping |

All at or below the random floor of 0.1712.

**Reactive ceiling.** Coordinate sweep over the five baseline parameters: **no configuration's
Wilson interval separates from the baseline's** [0.36–0.60]. The apparent best (`persist_p=0.7`
→ 0.5424) sits at [0.42–0.66]. The spread is sampling noise, not a tuning gradient.

**Sharpest single statement:** `momentum` scores 0.333 with *no smell at all* — persistence is
the largest single jump in the ladder. The baseline persists ~75% of the time; the learned policy
persists ~1 time in 6, despite the last action being in its observation as a one-hot.

### Dead-band geometry

Blind width `d = max(0, L − 2r)` for commute length `L`, smell radius `r`:

| | L=9 | L=10 | L=11 |
|---|---|---|---|
| r=3 | 3 | 4 | 5 |
| r=5 | 0 | 0 | 1 |

Detection area on a hex grid is `N(r) = 1 + 3r(r+1)` (quadratic); blind traverse is diffusive so
cost scales as `d²`. Compound difficulty ≈ `(L−2r)²/r²` — a factor of ~70 between r=3 and r=5 at
L=11, and exactly zero for L≤10. **r=5 does not ease the problem, it deletes it.** Confirmed by
the split already on record: smell 3 and smell 5 tie at 0.86 on the food-isolation eval where
water is handed over, and diverge to 0.48 vs 0.71 only on the cold-start eval.

---

## Figures

All generated by [`prototypes/06_feudal/figures/make_figures_v1.py`](prototypes/06_feudal/figures/make_figures_v1.py).

| figure | shows |
|---|---|
| `explorer_ladder__headline` | the ladder and all five failed learned attempts |
| `module_gap` | learned performance as a fraction of each oracle |
| `orchestrator_action_mix` | what it chose vs what killed it — the water-cult attractor |
| `dead_band_geometry` | `L−2r` and the difficulty curve |
| `pathfinder_checkpoint_selection` | the late collapse and what selection recovered |
| `calibration_detection_floor` | orchestrator / explorer / consumer metric sensitivity |
| `reactive_ceiling_sweep` | the parameter surface inside one Wilson band |
| `arbitration_denominator` | the published ratio, and what legality does to it |
| `commitment` | imposed horizon works, learned switching cost degenerates |
| `agent_alive` (gif) | the world itself — demo, hand-picked seed, not a measurement |

**Measurement bases differ in `module_gap`.** Drink, orchestrator and explorer are
`solveScore` ratios against the same 0.4776 oracle. The pathfinder has no solveScore of its own;
its 1.000 records a **+0.000 in-sim delta** over 34,859 invocations. Same conclusion, different
measurement — stated rather than blended.
