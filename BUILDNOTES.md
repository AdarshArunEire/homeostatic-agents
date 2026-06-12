# Build Notes

## Prototype 0: Why the first prototype was tabular

The first version used tabular Q-learning because the state space was tiny: hydration bucket, incoming drink bucket, and action. This made it easy to test whether delayed drink effects were learnable before adding neural networks.

## Why tabular was superseded

The state space needed to grow beyond one hydration axis. Later prototypes needed to include satiation, brightness, delayed action memory, and eventually physical position. In a tabular Q-table, each extra discretised state axis multiplies the number of values that must be stored and explored.

If each state axis has $b$ bins and there are $d$ state dimensions, the table grows like $\text{number of Q-values} = \text{number of actions} \times b^d$.

This becomes impractical quickly. It also means neighbouring states do not naturally share information: the agent must separately explore similar bins instead of learning a smoother relationship across the state space. This motivated moving to a neural Q-function.

## Prototype 1: Why the next prototype used a DQN

Moving from tabular Q-learning to a neural Q-function allowed the model to take continuous state variables directly, rather than forcing every variable into coarse bins. This made it much easier to add extra state axes such as hydration, satiation, brightness, delayed action effects, and eventually spatial observations.

The output is still a discrete set of action values, but the action set can be expanded more realistically. Later versions could include finer choices such as full, half, and quarter drinking/eating, plus combined drink/eat actions. In the tabular version, every new action would multiply the Q-table and require more exploration in every discretised state. In the DQN version, the model can share learned state features across all action outputs.

## Why the NumPy DQN was superseded

The NumPy DQN had a learning purpose: I wanted to understand the matrix shapes, dot products, manual backpropagation, replay sampling, target networks, terminal states, state representation, and reward surface design before relying on a library.

That made the fundamentals much clearer, but the implementation became too slow and awkward for the next stage. Physical embedding adds larger state vectors, action masking, local observations, and eventually multiple agents. For that, a tighter debugging and testing loop is more important, so the next prototype moves to PyTorch.

## Prototype 1b: Why port to PyTorch

The goal of the PyTorch port was not to make the agent more advanced immediately. It was to check whether the original NumPy DQN behaviour could be reproduced in a cleaner learning framework before using that framework to scale to larger spatial experiments.

The port broadly mirrors Prototype 1’s learning pattern and final hydration/satiation control, which suggests that the core learned behaviour survived the backend change. It also preserves the same short-gap death-clustering failure mode, suggesting that this weakness belongs to the learned DQN policy rather than the hand-written NumPy implementation.

This makes PyTorch a better base for the next stage. Larger state vectors, action masking, local observations, memory modules, and eventually multiple agents would make hand-written NumPy backpropagation tedious and fragile. PyTorch keeps the code flexible while still allowing the simulation logic itself to remain explicit and inspectable.

## Prototype 2: Physical embedding

Prototype 2 moved the agent into an actual hex world. The agent no longer just chooses drink/eat/wait from internal state. It now has to physically move between resources while managing hydration and satiation.

This changed the problem a lot. Before, the main thing was delayed body effects: drinking now changes hydration over the next few ticks, and eating now changes satiation over the next few ticks. In the physical version, the agent also has to learn timing and movement. It has to know when to leave water for food, and when to leave food for water.

The state now includes physical position, hydration, satiation, brightness, and short delayed-action memory. The action set includes drinking, eating, waiting, and movement actions.

So the task is no longer just homeostatic correction. It is now spatial homeostatic control.

## Why bigger worlds exposed a decay problem

When I made the world bigger, the same decay settings became much harsher. This was not just because the map was bigger visually. It was because the resource distance became bigger.

If the best path from water to food is around $10$ moves, then a food > water > food cycle is already around $20$ moves minimum, before failed movement, bad choices, waiting, drinking, eating, or exploration.

So the old decay was not really testing the same task anymore. It was making the agent survive a much longer travel cycle with the same body decay speed.

That means the decay needed to be normalised based on worldsize. This is not meant to make the world randomly easier. It is meant to keep the relationship between body decay and world distance more fair.

## What Prototype 2 revealed

Prototype 2 showed that the DQN is not completely useless in the physical world. In the bigger world, it started to learn a visible path/corridor between the water and food locations.

But it still struggled to maintain the full food > water > food cycle. More training time did not automatically fix it. Different epsilon values also did not fully solve it. This suggests that the issue is not just one bad environment parameter.

The likely problems are:

- long-horizon credit assignment
- fixed exploration randomly damaging learned paths
- vanilla DQN value instability
- limited short-term memory
- no real map memory
- no proper generalisation test yet

The agent can begin to learn a route, but it does not yet reliably learn the full control loop.

## Start of Prototype 3: Environment changes, generalisation and memory

### Decay normalised based on world size

When the world radius increased, the original decay rates became too harsh because the agent had to survive longer food > water > food cycles. So Prototype 3 began by normalising decay based on world size.

The scaling function was chosen to roughly match:

- radius 1 -> 1.5
- radius 3 -> 0.7
- radius 5 -> 0.5

with a floor so decay never disappears completely:

$g(R) = 0.05 + \dfrac{1.45} {(1 + 1.0426(R - 1))^{0.7122}}$

Implemented as:

```python
hydration_decay_scaling = 0.05 + 1.45 / ((1 + 1.0426 * (world_size - 1)) ** 0.7122)
satiation_decay_scaling = 0.9 * hydration_decay_scaling
```

### Generalisation and memory question

> Can the model learn a survival rule, or is it just learning one world?

A trained model should eventually be saved, frozen, and dropped into new worlds without continuing to train. If it only survives the original training world, then it probably learned a map-specific habit. If it survives new starts, mirrored maps, changed resource positions, or new worlds with similar rules, then it has learned something more general.

The future testing ladder is:

1. same map, new start positions
2. same map, different random seeds
3. mirrored or rotated map
4. same radius, different food and water locations
5. different radius with decay normalised based on worldsize
6. many training worlds and held-out test worlds

This also makes memory a real experiment instead of just an upgrade.

A short-history DQN can react to what it currently sees, but it cannot really build a map of a new world. So Prototype 3 should compare:

- short action-memory DQN
- explicit resource-memory DQN
- recurrent/RNN memory
- eventually external map-memory

The goal is not just to make the agent survive one bigger world. The goal is to test when memory becomes necessary for generalisation in spatial homeostatic control.
