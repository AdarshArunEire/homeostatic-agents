from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DQN", "make_model", "make_target", "sync_target", "select_action", "learn_step"]


class DQN(nn.Module):

    def __init__(self, n_input: int, n_hidden: int, n_act: int):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def make_model(n_input: int, n_hidden: int, n_act: int, lr: float):
    model = DQN(n_input, n_hidden, n_act)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def make_target(model: DQN) -> DQN:
    target = copy.deepcopy(model)
    target.eval()
    return target


def sync_target(target: DQN, model: DQN) -> None:
    target.load_state_dict(model.state_dict())


def select_action(model: DQN, state, action_mask) -> int:
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32) 
        q = model(x)

        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        q_masked = q.masked_fill(~mask, -1e9)

        return int(torch.argmax(q_masked).item())


def learn_step(model: DQN, target: DQN, optimiser, batch, gamma: float) -> float:

    state = torch.as_tensor(np.array([tr[0] for tr in batch]), dtype=torch.float32)
    action = torch.tensor([tr[1] for tr in batch], dtype=torch.long)
    reward = torch.tensor([tr[2] for tr in batch], dtype=torch.float32)
    next_state = torch.as_tensor(np.array([tr[3] for tr in batch]), dtype=torch.float32)
    done = torch.tensor([tr[4] for tr in batch], dtype=torch.bool)

    # this was created by env.get_action_mask() after the transition
    next_mask = torch.as_tensor(np.array([tr[5] for tr in batch]), dtype=torch.bool)

    q_all = model(state)                       # (B, n_act)
    rows = torch.arange(len(action))
    q_chosen = q_all[rows, action]             # (B,)

    with torch.no_grad():
        next_q_all = target(next_state)        # (B, n_act)

        # kill impossible next actions before max
        next_q_masked = next_q_all.masked_fill(~next_mask, -1e9)

        best_next_q = next_q_masked.max(dim=1).values
        target_q = reward + (gamma * best_next_q) * (~done).float()

    loss = F.smooth_l1_loss(q_chosen, target_q)

    optimiser.zero_grad()                      # clear last step's grads
    loss.backward()                            # the whole hand-written backward, abstracted
    optimiser.step()                           # nudge weights by their blame for the loss

    return loss.item()
