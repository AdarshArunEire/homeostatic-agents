# Homeostatic Agents

Reinforcement-learning agents that regulate internal state under spatial constraints — and a
six-prototype investigation into **where value-based RL stops working** as the environment
becomes less stationary and less forgiving.

**Results index:** [`RESULTS.md`](RESULTS.md) · **Hypothesis ledger:** [`BUILDNOTES.md`](BUILDNOTES.md) · **Re-run it:** [`REPRODUCE.md`](REPRODUCE.md) · **Latest work:** [`06_feudal`](prototypes/06_feudal)

---

## In three sentences

An agent must keep hydration and satiation near their setpoints, but water and food are
*places*, not buttons — so regulation becomes a spatial credit-assignment problem. On a fixed
map, DQN solves it: 52% of seeds learn the water→food cycle and 38% survive it. On procedurally
resampled maps that same approach collapses to 0%, and six prototypes of work isolate the cause
to a single mechanism — the agent cannot commit to crossing a region where no sensory signal
exists.

---

## The arc

**It worked.** Prototype 3b, fixed radius-5 map: **52%** of 100 seeds learn a clean water→food
limit cycle, **38%** also survive it under a strict evaluation gate. Getting there required
repairing the reward geometry, then discovering that exploration — not credit assignment — was
the binding constraint.

**Then it stopped working.** Prototype 04 asked whether that policy learned *a route* or *a
rule*. Carried onto procedurally generated maps, the survival rate is **0%** once a boundary
artifact propping it up is removed. Four hypotheses (senses, reachability, memory, a shorter
approach) were each falsified in turn, narrowing the failure to one mechanism: the agent cannot
self-commit to a directed crossing of the **dead band** — a 3–5 hex stretch mid-commute where
neither resource is detectable.

**Then we decomposed it.** Prototype 06 splits the agent into four modules — orchestrator,
pathfinder, consumer, explorer — and learns each separately. Navigation becomes provably optimal.
Exploration turns out not to be learnable at all on its observation contract. And arbitration —
the module that chooses between the two drives — *does* learn a coherent policy, then converges
on the same **water-cult attractor** Prototype 3b identified three prototypes earlier. The
pathology survives isolation, which means it belongs to the task rather than to end-to-end
learning.

<p>
  <img src="prototypes/06_feudal/results/best_figures/module_gap.png" width="640">
  <br>
  <sub><em>Each learned module as a fraction of the hand-written policy it replaced. The
  pathfinder matches it exactly; the explorer never left random.</em></sub>
</p>

---

## What stopped it working

The explorer is the module that must cross the dead band. Five methods were tried; none reached
even the random-walk floor.

<p>
  <img src="prototypes/06_feudal/results/best_figures/explorer_ladder__headline.png" width="880">
  <br>
  <sub><em>Hand-written policies span 0.17–1.00. Every learned policy — three DQN seeds, stratified
  replay, a terminal-condition fix, stochastic evaluation, and policy gradient — sits at ~0.10.</em></sub>
</p>

**The mechanism.** The explorer's observation contract carries legality, recent actions, smell
readings and need flags — but **no positional memory**. Coverage, revisit-avoidance and
frontier-seeking are therefore not representable. That confines the learnable policy class to
*reactive correlated walks with chemotaxis*: a five-parameter family that a hand-tuned heuristic
already occupies at or near its optimum. A parameter sweep found no setting whose confidence
interval separates from the baseline's.

**Learning has nothing to add where a five-parameter search suffices.**

This is a bounded claim, and the boundary matters: the explorer is not learnable *on this
observation contract*. Prototype 07's specification follows directly — give it visit counts, a
decaying coverage trace, or episodic novelty, so the policy class contains something a tuned
heuristic cannot reach.

---

## What did work: the pathfinder

Worth stating alongside the negative result, because it is the same algorithm class.

The pathfinder was trained **entirely outside the simulation**, on bare hex geometry with no
drives, decay or resources, using sparse arrival rewards and hindsight experience replay. Dropped
into the full agent it was invoked 34,859 times for a **+0.000** change in survival, at
**1.0000 optimality** across every operational distance band.

It is not a copy of the reference implementation — agreement is only 0.56, because ties are
broken differently. It is *independently optimal*, learned from arrival signals alone.

The claim this supports: the algorithm class that solved **0/40** end-to-end in Prototype 04
recovers optimal navigation once decomposition hands it an observed goal. **The decomposition is
the finding**, not the pathfinder.

---

## Why the sensory radius is not a free parameter

The dead band has a width, and it is not linear in the sensing radius.

For commute length $L$ and smell radius $r$, the agent is guided while within $r$ of water (an
anti-gradient — it knows where it came from) and within $r$ of food. The scentless middle is

$$d(r) = \max(0,\; L - 2r)$$

Radius is subtracted from **both ends**, so $+2$ on $r$ removes $4$ from $d$. At band $(9,11)$:
$r=3$ gives $d \in \{3,4,5\}$; $r=5$ gives $d \in \{0,0,1\}$.

Two further effects compound. Detection area on a hex grid is $N(r) = 1 + 3r(r+1)$, quadratic in
$r$, and mean hitting time for a random walk scales as domain area over target area. And in the
blind region the walk is unbiased, so expected crossing time for a gap of width $d$ scales as
$d^2$, not $d$. Together:

$$\mathcal{D} \;\propto\; \frac{(L-2r)^2}{r^2}$$

At $L=11$ that is $25/9 \approx 2.78$ for $r=3$ against $1/25 = 0.04$ for $r=5$ — a factor of ~70.
At $L \le 10$ with $r=5$ the numerator is zero.

**So $r=5$ does not make the task easier; it removes the task.** The phenomenon is parameterised by
$L - 2r$, not by $r$, which means smell 5 with band $(13,15)$ would be the *same* problem. The
project holds $r=3$ deliberately, and the cost of doing so is itself diagnostic: smell 3 and smell 5
tie at 0.86 on an evaluation where water is handed over, and diverge to 0.48 vs 0.71 only when
water must be found from cold.

## What counts as cheating

The word has a precise meaning here, and the distinction drove most of the design decisions.

**An intervention is cheating if it changes what the agent *knows*, rather than how it *learns*.**

| intervention | verdict | why |
|---|---|---|
| scent as an **observation** | legal | changes the observation space — a different POMDP, but the agent still has to act on the signal |
| scent as a **reward** | **cheating** | the navigation is then performed by the reward function's gradient, not by the agent. The thing being claimed is exactly the thing being supplied |
| oracle with pre-populated coordinates | cheating (except as a plumbing smoke test) | the agent never earned the knowledge; the privilege does not survive the oracle→learned swap |
| a learned module importing the true physics | **cheating** | it reads the consume dynamics instead of estimating them. `world_v1` is the greppable boundary and a probe enforces it |
| stratified replay sampling | legal | changes which existing transitions the optimiser sees. No new information enters |
| raising the sensory radius | legal but **changes the problem** | see above — it deletes the dead band rather than easing it, so results are not comparable across $r$ |

The operational test: *what privilege evaporates when an oracle is replaced by a learned module?*
Anything that would be impossible to replace without handing the agent privileged information fails.

Biological plausibility is not the standard here — but it happens to coincide with it at one point
worth noting. Real chemotaxis is scent-as-observation: an organism senses a gradient and acts on it;
it is not paid for proximity. The honest engineering boundary and the biologically real mechanism
turn out to be the same line.

## Engineering

Roughly 5,000 lines across six prototypes. The parts that took the most work are the parts that
made the results trustworthy rather than the parts that produce them.

**Module contract.** A frozen interface (`contract_v1.py`) defines four modules — orchestrator,
pathfinder, consumer, explorer — each with an oracle implementation and a learned implementation
sharing one signature. Swapping any single module is a one-string change to a `ModuleSpec`, which
is what makes per-module attribution possible at all.

**Checkpointing.** Weights carry architecture, observation-field list, training config, seed, git
SHA and achieved metric — never a bare `state_dict`. Modules receive a *string tag* and load inside
the worker, because sweeps spawn processes on Windows and live torch objects do not survive the
pickle. Loading asserts the observation encoding matches what the module was trained on; a stale net
fed a reshaped observation is the worst available failure mode because it fails silently.

**Sweep harness.** Process-parallel with resumable checkpointing, atomic pickle writes, segment-aware
evaluation metrics (each evaluation slice scored against its own map's coordinates before pooling),
and Wilson intervals throughout.

**Probe suite.** Nine invariants asserted before any result is read — privilege boundary, checkpoint
round-trip, registry integrity, determinism, action-scheme shim correctness, dispatch-record
correctness, and the record-to-tick join the training rigs depend on. Every one guards a failure that
would otherwise be silent.

**Calibration instruments.** Each module has a deliberately degraded twin (`noisy_oracle`) used to
establish what a metric can actually detect *before* a learned module is trained against it. This is
the piece most easily skipped and it changed two conclusions.

## Project map

| Prototype | Focus | Status |
|---|---|---|
| [`00_tabular_hydration`](prototypes/00_tabular_hydration) | Tabular Q-learning, one-axis hydration, delayed effects | Superseded |
| [`01_numpy_dqn_homeostasis`](prototypes/01_numpy_dqn_homeostasis) | From-scratch NumPy DQN with manual backprop | Superseded |
| [`01b_pytorch_dqn_port`](prototypes/01b_pytorch_dqn_port) | PyTorch port; reproduced behaviour and failure modes | Superseded |
| [`02_spatial_dqn`](prototypes/02_spatial_dqn) | Hex world, local observation, movement, action masks | Superseded |
| [`03_spatial_robust`](prototypes/03_spatial_robust) | Radius-5 exposed the old reward/metric setup as unfit | Superseded |
| [`03b_nstep_robust`](prototypes/03b_nstep_robust) | **52% / 38%** on the fixed commute; exploration beats credit assignment | Headline (fixed map) |
| [`04_generalisation`](prototypes/04_generalisation) | Procedural r=20 maps — route or rule? H1–H4 all falsified | Superseded |
| [`05_generalisation`](prototypes/05_generalisation) | Curriculum + earned-memory attempt; abandoned mid-hypothesis | Superseded |
| [`06_feudal`](prototypes/06_feudal) | **Feudal decomposition — which modules can RL learn?** | Current |

---

## The central question

> When does value-based reinforcement learning stop working as the environment becomes less
> stationary, less forgiving, and less tied to one fixed layout?

Each prototype keeps the same control problem and makes the environment less forgiving. Early
versions were abstract homeostatic control. Later versions move corrective actions into physical
space: water and food become locations, and the agent must learn *when* to act, *where* the
action is available, and how to survive the delay between needing a resource and reaching it.

### Prototype 3b, briefly

On the fixed radius-5 commute, DQN is bimodal. Some seeds discover the water→food limit cycle;
others fall into the **water-cult attractor** — camping near water, protecting hydration, never
crossing to food. Mean comfort hides this, because one internal variable can stay well-controlled
while the policy is behaviourally wrong. Seed-level solve-rate is the honest metric.

Three findings, in the order they were established:

1. **Reward geometry first.** Deficit and surplus are not symmetric errors — an agent crossing a
   map needs to carry a surplus. Separating them was a precondition for anything else working.
2. **Credit assignment was not the bottleneck.** Double DQN, longer n-step returns, tuned death
   penalties and larger γ all changed how value propagated, and none reliably broke the attractor.
   *A value function can only assign credit to trajectories that enter the training distribution.*
3. **Exploration was.** NoisyNets and count-based novelty worked only in combination — novelty
   alone reinforced the comfortable water region; with NoisyNets it moved the replay distribution
   into the food corridor. The mechanism is not "curiosity solves the task" but "induced
   exploration changes the replay distribution enough for the useful trajectory to become
   learnable."

A near-non-evicting 520k replay buffer collapsing to 0/10 ruled out simple FIFO forgetting: the
buffer is a **sampling distribution**, not just memory, and its optimal size is a trade-off
against staleness under a non-stationary policy.

---

## Method notes

The experimental apparatus is a deliberate part of this project, and several conclusions came
from it rather than from any single run.

- **Calibrate before training.** A null result is uninterpretable unless the metric has been shown
  capable of producing a non-null one. Every module has a deliberately degraded control
  (`noisy_oracle`) used to establish a detection floor *first*. The explorer's metric spans 0.362;
  the consumer's spans 0.044 — which is why the consumer's certification means less than it looks.
- **A non-monotone metric cannot rank policies.** Under controlled degradation, `solveScore`
  scored a knowably worse consumer *higher*. Any detection floor computed from it is an artifact.
- **Error character dominates error rate.** Two pathfinders differing by 0.02 in optimality
  differed **90×** in expected excess steps, because one's mistakes were sideways and the other's
  were reversals.
- **A module's episode boundary is not the agent's.** It ends when the sub-task ends;
  bootstrapping past it corrupts exactly the transitions the module should learn from.
- **Never evaluate on a resampled world.** Fixed evaluation seeds, disjoint from certification
  seeds — otherwise the curve measures the world, not the policy.
- **Never relax a pre-registered threshold.** The pathfinder's 0.99 gate failed three runs before
  a properly selected checkpoint scored 1.0000. The gate was right; the training was not.

Predictions are written into [`BUILDNOTES.md`](BUILDNOTES.md) **before** each sweep runs, in
Bet → Prediction → Result → Verdict form. The entries where the result contradicts the prediction
are the load-bearing ones and are left exactly as written.

---

## Repository layout

```
prototypes/<NN>_<name>/     one prototype: code, README (hypothesis ledger), results/
  06_feudal/
    model_modules/          contract, oracle impls, learned impls, checkpointing
    training/               rigs and per-module trainers
    tests/                  probes, calibration harnesses, acceptance gates
    figures/                figure generation
BUILDNOTES.md               the lab notebook — chronological, predictions before results
RESULTS.md                  every headline number in one place
```

**Stack:** Python, NumPy, PyTorch. DQN family (vanilla / double / n-step / NoisyNet / DRQN),
hindsight experience replay, policy gradient with a value baseline, count-based novelty,
process-parallel sweep harness with resumable checkpointing.
