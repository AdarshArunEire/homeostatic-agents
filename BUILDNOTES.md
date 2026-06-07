# Build Notes

## Why the first prototype was tabular

The first version used tabular Q-learning because the state space was tiny: hydration bucket, incoming drink bucket, and action. This made it easy to test whether delayed drink effects were learnable before adding neural networks.

## Why tabular was superseded

The state space needed to grow beyond one hydration axis. Later prototypes needed to include satiation, brightness, delayed action memory, and eventually physical position. In a tabular Q-table, each extra discretised state axis multiplies the number of values that must be stored and explored.

If each state axis has `b` bins and there are `d` state dimensions, the table grows like:

`number of Q-values = number of actions × b^d`

This becomes impractical quickly. It also means neighbouring states do not naturally share information: the agent must separately explore similar bins instead of learning a smoother relationship across the state space. This motivated moving to a neural Q-function.

## Why the next prototype used a DQN

Moving from tabular Q-learning to a neural Q-function allowed the model to take continuous state variables directly, rather than forcing every variable into coarse bins. This made it much easier to add extra state axes such as hydration, satiation, brightness, delayed action effects, and eventually spatial observations.

The output is still a discrete set of action values, but the action set can be expanded more realistically. Later versions could include finer choices such as full, half, and quarter drinking/eating, plus combined drink/eat actions. In the tabular version, every new action would multiply the Q-table and require more exploration in every discretised state. In the DQN version, the model can share learned state features across all action outputs.

## Why the NumPy DQN was superseded

The NumPy DQN had a learning purpose: I wanted to understand the matrix shapes, dot products, manual backpropagation, replay sampling, target networks, terminal states, state representation, and reward surface design before relying on a library.

That made the fundamentals much clearer, but the implementation became too slow and awkward for the next stage. Physical embedding adds larger state vectors, action masking, local observations, and eventually multiple agents. For that, a tighter debugging and testing loop is more important, so the next prototype moves to PyTorch.
