# Build Notes

## What is this?

BUILDNOTES.md serves as a compressed project arc with hypothesis ledgers where relevant.

The front page states the current result, while this file records the build path: what each prototype changed, what broke, what got superseded, and why the next prototype existed.

## Prototype 00 — tabular hydration

### Why it existed

The first version used tabular Q-learning because the state space was tiny: hydration bucket, incoming drink bucket, and action.

This made it easy to test whether delayed drink effects were learnable before adding neural networks.

### Why it was superseded

The state space needed to grow beyond one hydration axis. Later prototypes needed satiation, brightness, delayed action memory, and eventually physical position.

In a tabular Q-table, each extra discretised state axis multiplies the number of values that must be stored and explored.

If each state axis has $b$ bins and there are $d$ state dimensions, then the table grows exponentially:

$$
N_Q = |\mathcal{A}| b^d
$$

where $N_Q$ is the number of stored Q-values, $|\mathcal{A}|$ is the number of actions, $b$ is the number of bins per state axis, and $d$ is the number of state dimensions.

This becomes impractical quickly. It also means neighbouring states do not naturally share information: the agent has to separately explore similar bins instead of learning a smoother relationship across the state space.

That motivated moving to a neural Q-function.

Full folder: [`00_tabular_hydration`](prototypes/00_tabular_hydration)

## Prototype 01 — NumPy DQN

### Why the next prototype used a DQN

Moving from tabular Q-learning to a neural Q-function allowed the model to take continuous state variables directly, instead of forcing every variable into coarse bins.

This made it much easier to add extra state axes such as hydration, satiation, brightness, delayed action effects, and eventually spatial observations.

The output was still a discrete set of action values, but the action set could now be expanded more realistically. Later versions could include finer choices such as full, half, and quarter drinking/eating, plus combined drink/eat actions. In the tabular version, every new action would multiply the Q-table and require more exploration in every discretised state. In the DQN version, the model can share learned state features across all action outputs.

### Why it was written in NumPy first

The NumPy DQN had a learning purpose.

I wanted to understand the matrix shapes, dot products, manual backpropagation, replay sampling, target networks, terminal states, state representation, and reward surface design before relying on a library.

That made the fundamentals much clearer, but the implementation became too slow and awkward for the next stage. Physical embedding adds larger state vectors, action masking, local observations, and eventually multiple agents. For that, a tighter debugging and testing loop matters more than hand-written backpropagation.

So the next prototype moved to PyTorch.

Full folder: [`01_numpy_dqn_homeostasis`](prototypes/01_numpy_dqn_homeostasis)

## Prototype 01b — PyTorch port

### Why it existed

The PyTorch port was not meant to make the agent more advanced immediately.

The goal was to check whether the original NumPy DQN behaviour could be reproduced in a cleaner learning framework before using that framework for larger spatial experiments.

### What it showed

The port broadly mirrored Prototype 1’s learning pattern and final hydration/satiation control. It also preserved the same short-gap death-clustering failure mode.

That suggested the weakness belonged to the learned DQN policy rather than the hand-written NumPy implementation.

### Why it mattered

PyTorch became the better base for the next stage. Larger state vectors, action masking, local observations, memory modules, and eventually multiple agents would make hand-written NumPy backpropagation tedious and fragile.

PyTorch kept the learning code flexible while still allowing the simulation logic itself to remain explicit and inspectable.

Full folder: [`01b_pytorch_dqn_port`](prototypes/01b_pytorch_dqn_port)

## Prototype 02 — physical embedding

### Why it existed

Prototype 2 moved the agent into an actual hex world.

The agent no longer chooses drink/eat/wait directly from internal state. It has to physically move between resources while managing hydration and satiation.

Before this, the main difficulty was delayed body effects: drinking changes hydration over the next few ticks, and eating changes satiation over the next few ticks. In the physical version, the agent also has to learn timing and movement. It has to know when to leave water for food, and when to leave food for water.

The state now includes physical position, hydration, satiation, brightness, and short delayed-action memory. The action set includes drinking, eating, waiting, and movement actions.

So the task is no longer just homeostatic correction.

It is spatial homeostatic control.

### What physical space exposed

Once resources exist as locations, distance becomes part of the body-control problem.

Even before the map becomes large, the agent has to survive the gap between needing a resource and reaching it. A food → water → food cycle is no longer an instant correction loop. It contains travel time, failed movement, bad choices, waiting, drinking, eating, and exploration.

That means body decay and map distance are coupled. If decay is too fast relative to the travel cycle, the task stops testing control and starts testing whether the agent can survive an unfair commute.

Prototype 2 exposed this relationship, but did not fully solve it. The later radius-5 work in Prototype 3 made the decay-scaling problem unavoidable.

### What Prototype 2 revealed

Prototype 2 showed that the DQN was not completely useless in the physical world. It began to learn visible paths/corridors between the water and food locations.

But it still struggled to maintain the full food → water → food cycle. More training time did not automatically fix it. Different epsilon values also did not fully solve it.

The likely problems were:

* long-horizon credit assignment
* fixed exploration randomly damaging learned paths
* vanilla DQN value instability
* limited short-term memory
* no real map memory
* no proper generalisation test yet

The agent could begin to learn a route, but it did not yet reliably learn the full control loop.

Full folder: [`02_spatial_dqn`](prototypes/02_spatial_dqn)

## Prototype 03 — radius-5 world and reward geometry

### Environment change

Prototype 3 pushed the spatial task from the smaller radius-3 world into a radius-5 world.

Water moved to $(-5,0)$ and food moved to $(0,5)$. The trip between them was now long enough that any surviving policy had to overfill before travelling: top up hydration before walking to food, and top up satiation before walking back.

Travel buffers became part of the task.

### Decay normalised based on world size

When the world radius increased, the original decay rates became too harsh because the agent had to survive longer food → water → food cycles.

The scaling function was chosen to roughly match:

* radius 1 → 1.5
* radius 3 → 0.7
* radius 5 → 0.5

with a floor so decay never disappears completely:

$$
g(R) = 0.05 + \frac{1.45}{(1 + 1.0426(R - 1))^{0.7122}}
$$

Implemented as:

```python
hydration_decay_scaling = 0.05 + 1.45 / ((1 + 1.0426 * (world_size - 1)) ** 0.7122)
satiation_decay_scaling = 0.9 * hydration_decay_scaling
```

### Comfort surface repair

The old comfort surface was isotropic around the ideal point:

$$
d^2 = (h - h^\star)^2 + (s - s^\star)^2,
\qquad
C(h,s) = 2e^{-kd^2} - 1
$$

That became wrong in the radius-5 world.

A useful buffer cost the same as a dangerous deficit. On the bigger map, that taxes exactly what the agent needs to survive the trip. The old-surface sweep showed the damage: vanilla and Double DQN collapsed the same way, one need pinned while the other fell toward death.

That shared failure was a reward problem, not a model problem.

So I split the squared distance and discounted only the over-fill component:

$$
\begin{aligned}
d^2 &=
h_{\text{under}}^2

* \lambda h_{\text{over}}^2
* s_{\text{under}}^2
* \lambda s_{\text{over}}^2, \
  \lambda &= 0.3
  \end{aligned}
  $$

keeping the same exponential mapping.

Underfill keeps full curvature, moderate overfill is cheap, and extreme overfill still decays. A travel buffer becomes affordable, but hoarding cannot be optimal. The internal-state cap widened from 2 to 3 so over-buffered states get charged by the surface instead of hiding at the clip boundary.

### Oracle check

`oracle.py` — a hand-coded controller that overfills to 1.7 and services the weaker need — confirmed the world was physically survivable:

* 0 deaths over 500k ticks
* comfort −0.608
* the surface correctly charged it for living in the over-buffer band

Under the new surface, vanilla reached comfort 0.501 where the old medians sat near zero. So the geometry fix recovered real capability.

But it stayed high-variance across seeds: median −0.03, spanning −0.69 to 0.57.

So this was a capability demonstration, not a solved benchmark.

> Open question: does the new geometry remove the one-variable collapse across all seeds without making the task trivial?

Full folder: [`03_spatial_robust`](prototypes/03_spatial_robust)

## Prototype 03b — consistency and the water-cult attractor

Full detailed record: [`03b_nstep_robust`](prototypes/03b_nstep_robust)

Prototype 3b turns the best-seed radius-5 result into a consistency question.

The problem is not:

> can one lucky seed learn the commute?

The problem is:

> how often does the training process actually find the water→food control cycle?

### Failure mode

The radius-5 vanilla agent is bimodal.

Some seeds discover the water→food limit cycle. Others fall into the water-cult attractor: camp near water, protect hydration, and never cross the comfort valley to food.

Mean comfort hides this failure. A clean cycler and a confident water-camper can look too similar in aggregate comfort, because the water-camper controls one variable well while the other drifts toward death.

That pushed the evaluation away from mean comfort alone and toward per-seed solve-rate, food-reaching gates, death counts, and path diagnostics.

### Hypothesis 1 — comfort metric

**Bet:** comfort is the objective, so optimise and read mean comfort directly.

**What happened:** the bimodal gap between cyclers and campers swamped the meaning of the mean. Mean comfort was not false, but it was too blunt. It could not reliably distinguish behavioural competence from one-variable collapse.

**Verdict:** dissolved into a measurement fix.

Use solve-rate, food-reaching gates, death counts, and path diagnostics. Mean comfort is still useful, but it cannot be the whole story.

### Hypothesis 2 — credit assignment

**Bet:** the food reward is too far away, so better credit assignment should help the agent understand the value of the full water→food cycle.

The first 3b attempt tried to reduce learning noise directly:

* Double DQN for value stability
* n-step returns for long-horizon credit assignment
* tuned death penalties
* larger $\gamma$

This did help in one sense. Some settings reduced seed spread dramatically. For example, vanilla 1-step varied from −0.69 to 0.57 comfort, while Double DQN with 10-step returns compressed into a much tighter band: 0.22 to 0.25 comfort.

So n-step and DDQN were changing the training dynamics. They made the agent less seed-chaotic.

But the thing they stabilised was not always the full control loop.

A lot of the stable runs were stable because they found a degenerate local maximum: camp at water, manage hydration perfectly, and eventually die just to repeat. They never really solved the food → water → food loop.

So the result was not:

> credit assignment is useless.

It was:

> credit assignment helps once useful trajectories exist, but it cannot manufacture trajectories that never enter replay.

The key issue became:

> a value function can only assign credit to trajectories that enter the training distribution.

**Verdict:** partly right, incomplete. Better value propagation was not the bottleneck by itself.

### Hypothesis 3 — exploration

**Bet:** the policy needs to experience the useful loop often enough for the value function to learn it.

So the next knob to turn was exploration.

Cranking epsilon higher was too atomic. It may make the agent discover the good loop, but it also fills the replay buffer with garbage random transitions. The training distribution changes, but not in a controlled way.

So Prototype 3b moved toward exploration mechanisms that could change the replay distribution more usefully:

* NoisyNets for state-dependent exploration
* count-based novelty for pressure away from overused regions

NoisyNets and count-based novelty worked only together.

NoisyNets gave state-dependent exploration. Novelty created pressure away from overused regions. Alone, neither was enough. On vanilla DQN, novelty was spent reinforcing the comfortable water region. With NoisyNets, the same bonus helped move the replay distribution into the food corridor.

So the mechanism was *not*:

> curiosity solves the task.

It was:

> induced exploration changes the replay distribution enough for the useful trajectory to become learnable.

**Verdict:** confirmed in combination.

### Hypothesis 4 — NoisyNet sigma collapse

**Bet:** novelty might help because NoisyNet $\sigma$ collapses toward zero, stopping discovery.

**What happened:** logged $\sigma$ rose rather than collapsed.

The likely reason is that the target remained non-stationary under drive cycles, local reward changes, and changing replay distribution. The agent was not becoming calmly certain and then freezing exploration. The instability was more complicated than that.

**Verdict:** dissolved. The premise was wrong, even if NoisyNets still helped.

### Hypothesis 5 — curriculum

**Bet:** make the good trajectory reachable often enough that the buffer contains useful examples, then slowly push the task back toward the real radius-5 problem.

This was the natural next idea after epsilon exploration looked too noisy.

But coordinate curriculum backfired. It made early resource habits easier, but those habits did not necessarily become the final radius-5 loop. In practice, curriculum often entrenched camping rather than breaking it.

Targeted respawn was more useful because it injected informative internal states without changing the final evaluation task.

**Verdict:** curriculum abandoned; targeted starts partially salvaged.

### Hypothesis 6 — replay buffer

**Bet:** food transitions are rare, so perhaps a small FIFO buffer evicts them before the agent can learn from them. A larger buffer should retain useful journeys and improve solve-rate.

The small-to-medium buffer results supported the retention hypothesis at first.

Increasing the buffer from 5k to 50k–100k improved solve-rate and reduced deaths. But a near-non-evicting 520k buffer collapsed to 0/10.

That ruled out simple FIFO eviction as the whole explanation.

If the mechanism were only eviction, more retention should keep helping. It did not.

The issue is not only that useful transitions disappear. Old transitions can become stale under a changing policy and reward distribution. The replay buffer is therefore not just memory; it is a sampling distribution, and its optimal size is a tradeoff.

**Verdict:** falsified and reframed.

50k is chosen as a survival-useful point in the tradeoff, not as a pure retention fix.

### Headline result

The best current configuration uses:

* Noisy DQN
* count-based novelty, $\beta = 0.1$
* 50k replay buffer
* 10-step returns
* comfort-v3 surface
* $\gamma = 0.99$
* 100 seeds

Headline:

* **52%** learn a clean water→food limit cycle.
* **38%** also survive that cycle under the stricter survival-aware gate.
* Median comfort among solved seeds is about **0.93**.
* 49/100 seeds finish greedy evaluation with zero deaths.

The gap between 52% and 38% separates route discovery from reliable survival: some agents learn the water→food cycle, but still execute it with enough instability to die during evaluation.

### Stopping point

At this point, more tuning on the fixed radius-5 commute has diminishing returns. It *could* improve the benchmark, but it risks turning the project into narrow, map-specific fitting.

That is where Prototype 3b stops: the fixed commute is no longer the highest-value test of the idea.

## Prototype 04 — generalisation

### Question

Can the agent learn a survival rule across layouts?

### Planned build

Prototype 4 should separate generalisation from regime shift.

The next environment is a procedurally generated static radius-20 map with different bush and lake layouts. Training and evaluation should be split across map seeds, so success cannot come from memorising one route.

### Why this comes before regime shift

The fixed radius-5 benchmark asks:

> can the agent learn this commute?

Prototype 4 asks:

> can the agent learn a survival rule across layouts?

Only after that should the project move to true regime shift: seasonal brightness, scarce food, moving resources, and reward distributions that change under the value function.
