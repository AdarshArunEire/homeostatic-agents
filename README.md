# Homeostatic Agents

A reinforcement-learning simulation project about agents learning to regulate internal needs.

The project is built around a simple idea: create a small survival environment, then train an agent to solve the control problem inside it.

The current environment contains hydration, satiation, daylight-driven decay, delayed action effects, comfort, and death.

The current agent is a from-scratch NumPy DQN controller: a single hidden layer Q-net trained with replay sampling, target network updates, epsilon-greedy exploration, terminal-state masking, and manual vectorized backpropagation.

The current focus is to build increasingly strict prototypes and test what breaks as the environment becomes more complex: delayed dynamics, spatial constraints, local observations, action masking, and eventually parameter-shared multi-agent control, where agents use the same policy weights while maintaining separate state vectors and transition histories.

## Current state

The current environment is a two-axis homeostasis problem, with delayed action effects and cyclic daylight-driven decay:

- hydration (internal state to balance)
- satiation (internal state to balance)
- brightness-driven decay (external factor and input)
- delayed drink/eat effects (environment-level dynamics)

The best current model is a DQN controller that can effectively maintain homeostatic behavior:

- manual NumPy forward/backward passes for a small ReLU Q-network
- state augmentation with recent drink/eat action-effect history to handle delayed dynamics
- replay sampling with a warmup period
- target network updates
- terminal-state handling with masked Bellman targets
- final greedy-policy window with epsilon-greedy exploration disabled

The model learns to keep its internal state near the ideal comfort region instead of drifting into dehydration, starvation, or unstable oscillation.

### Best current result

<p>
  <img src="results/best_figures/phase_trajectory_wu500_buf5k_g95_e03_u50.gif" width="520">
  <br>
  <sub><em>Evaluation phase density. The learned policy keeps hydration and satiation clustered near the ideal point.</em></sub>
</p>
<br>

<p>
  <img src="results/best_figures/comfort_hill_3d_eval_wu500_buf5k_g95_e10_u50_rank10.gif" width="700">
  <br>
  <sub><em>Comfort surface with evaluation trajectory. Comfort peaks around the target hydration/satiation region.</em></sub>
</p>
<br>

<p>
  <img src="results/best_figures/rolling_comfort_wu500_buf5k_g95_e03_u50.png" width="700">
  <br>
  <sub><em>Rolling comfort across seeded runs. The highlighted run shows the representative evaluation behavior.</em></sub>
</p>
<br>

<p>
  <img src="results/best_figures/eval_hs_wu500_buf5k_g95_e03_u50.png" width="700">
  <br>
  <sub><em>Final greedy-policy window with exploration disabled. Hydration and satiation remain close to the target region.</em></sub>
</p>
<br>


## Prototype path

| Prototype | Focus | Status |
|---|---|---|
| `00_tabular_hydration` | Tabular Q-learning for one-axis hydration control with delayed drink effects | Superseded |
| `01_numpy_dqn_homeostasis` | Neural Q-function for hydration + satiation control | Current best non-spatial model |
| `02_pytorch_spatial_homeostasis` | Physical hex-world with food, water, local observations, and action masks | Next |

## What this project is trying to build toward

The next step is physical embedding.

In the current model, drinking and eating are abstract actions that are always available. In the next prototype, food and water will exist as locations in a small hex-grid world. The agent will have to move, observe local surroundings, and choose from valid actions.

That means:

- drinking should only be valid near water
- eating should only be valid near food
- movement depends on neighbouring cells
- action masks become necessary
- the model will move from handwritten NumPy to PyTorch for faster testing and cleaner debugging

*The goal for the next version is simple:*

> Can an agent learn a physical survival loop between food and water while still regulating its internal state?