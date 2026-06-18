from __future__ import annotations

import copy

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class vanilla_DQN(nn.Module):

    def __init__(self, n_input: int, n_hidden: int, n_act: int):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def vanilla_make_model(n_input: int, n_hidden: int, n_act: int, lr: float):
    model = vanilla_DQN(n_input, n_hidden, n_act)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def vanilla_make_target(model: vanilla_DQN) -> vanilla_DQN:
    target = copy.deepcopy(model)
    target.eval()
    return target


def vanilla_sync_target(target: vanilla_DQN, model: vanilla_DQN) -> None:
    target.load_state_dict(model.state_dict())


def vanilla_select_action(model: vanilla_DQN, state, action_mask) -> int:
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32) 
        q = model(x)

        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        q_masked = q.masked_fill(~mask, -1e9)

        return int(torch.argmax(q_masked).item())


def vanilla_learn_step(model: vanilla_DQN, target: vanilla_DQN, optimiser, batch, gamma: float) -> float:

    state = torch.as_tensor(np.array([tr[0] for tr in batch]), dtype=torch.float32)
    action = torch.tensor([tr[1] for tr in batch], dtype=torch.long)
    reward = torch.tensor([tr[2] for tr in batch], dtype=torch.float32)
    next_state = torch.as_tensor(np.array([tr[3] for tr in batch]), dtype=torch.float32)
    done = torch.tensor([tr[4] for tr in batch], dtype=torch.bool)
    next_mask = torch.as_tensor(np.array([tr[5] for tr in batch]), dtype=torch.bool)
    n_used = torch.tensor([tr[6] for tr in batch], dtype=torch.float32)

    q_all = model(state)                       # (B, n_act)
    rows = torch.arange(len(action))
    q_chosen = q_all[rows, action]             # (B,)

    with torch.no_grad():
        next_q_all = target(next_state)        # (B, n_act)

        # kill impossible next actions before max
        next_q_masked = next_q_all.masked_fill(~next_mask, -1e9)

        best_next_q = next_q_masked.max(dim=1).values
        target_q = reward + ((gamma**n_used) * best_next_q) * (~done).float()

    loss = F.smooth_l1_loss(q_chosen, target_q)

    optimiser.zero_grad()                      # clear last step's grads
    loss.backward()                            # the whole hand-written backward, abstracted
    optimiser.step()                           # nudge weights by their blame for the loss

    return loss.item()

def double_learn_step(model: vanilla_DQN, target: vanilla_DQN, optimiser, batch, gamma: float) -> float:

    state = torch.as_tensor(np.array([tr[0] for tr in batch]), dtype=torch.float32)
    action = torch.tensor([tr[1] for tr in batch], dtype=torch.long)
    reward = torch.tensor([tr[2] for tr in batch], dtype=torch.float32)
    next_state = torch.as_tensor(np.array([tr[3] for tr in batch]), dtype=torch.float32)
    done = torch.tensor([tr[4] for tr in batch], dtype=torch.bool)
    next_mask = torch.as_tensor(np.array([tr[5] for tr in batch]), dtype=torch.bool)
    n_used = torch.tensor([tr[6] for tr in batch], dtype=torch.float32)

    q_all = model(state)                       # (B, n_act)
    rows = torch.arange(len(action))
    q_chosen = q_all[rows, action]             # (B,)

    with torch.no_grad():
        # job 1 — ONLINE selects action (mask)
        next_q_online = model(next_state).masked_fill(~next_mask, -1e9)
        next_actions  = next_q_online.argmax(dim=1)

        # job 2 — TARGET scores action
        next_q_target = target(next_state)       
        best_next_q   = next_q_target[torch.arange(len(next_actions)), next_actions]

        target_q = reward + ((gamma**n_used) * best_next_q) * (~done).float()

    loss = F.smooth_l1_loss(q_chosen, target_q)

    optimiser.zero_grad()                      # clear last step's grads
    loss.backward()                            # the whole hand-written backward, abstracted
    optimiser.step()                           # nudge weights by their blame for the loss

    return loss.item()

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("eps_w", torch.empty(out_features, in_features))

        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_b", torch.empty(out_features))

        mu_range = 1.0 / math.sqrt(in_features)
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.mu_b.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(sigma_0 / math.sqrt(in_features))
        self.sigma_b.data.fill_(sigma_0 / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._f(torch.randn(self.in_features, device=self.mu_w.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.mu_w.device))
        self.eps_w.copy_(eps_out.outer(eps_in))
        self.eps_b.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.mu_w + self.sigma_w * self.eps_w
            b = self.mu_b + self.sigma_b * self.eps_b
        else:
            w = self.mu_w
            b = self.mu_b
        return F.linear(x, w, b)


class noisy_DQN(nn.Module):
    def __init__(self, n_input, n_hidden, n_act, sigma_0=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = NoisyLinear(n_hidden, n_act, sigma_0=sigma_0)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def reset_noise(self):
        self.fc2.reset_noise()


def noisy_make_model(n_input, n_hidden, n_act, lr, sigma_0=0.5):
    model = noisy_DQN(n_input, n_hidden, n_act, sigma_0=sigma_0)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def noisy_make_target(model):
    target = copy.deepcopy(model)
    target.train()
    return target


def noisy_sync_target(target, model):
    target.load_state_dict(model.state_dict())


def noisy_learn_step(model, target, optimiser, batch, gamma):
    model.reset_noise()
    target.reset_noise()
    return vanilla_learn_step(model, target, optimiser, batch, gamma)