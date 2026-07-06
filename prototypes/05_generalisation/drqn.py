from __future__ import annotations

import copy

import numpy as np
from collections import deque
import random
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

class drqn_DQN(nn.Module):

    def __init__(self, n_input: int, n_hidden: int, n_act: int):
        super().__init__()

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.gru = nn.GRU(
            input_size=n_hidden,
            hidden_size=n_hidden,
            batch_first=True,
        )
        self.fc2 = nn.Linear(n_hidden, n_act)

    def forward(self, x: torch.Tensor, hidden=None):
        single_step = (x.dim() == 2)

        if single_step:
            x = x.unsqueeze(1)   # (B, F) -> (B, 1, F)

        z = F.relu(self.fc1(x))  # (B, T, H)
        z, hidden = self.gru(z, hidden)
        q = self.fc2(F.relu(z))  # (B, T, A)

        if single_step:
            q = q.squeeze(1)     # (B, 1, A) -> (B, A)

        return q, hidden


def drqn_make_model(n_input: int, n_hidden: int, n_act: int, lr: float):
    model = drqn_DQN(n_input, n_hidden, n_act,)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def drqn_make_target(model: drqn_DQN) -> drqn_DQN:
    target = copy.deepcopy(model)
    target.eval()
    return target


def drqn_sync_target(target: drqn_DQN, model: drqn_DQN) -> None:
    target.load_state_dict(model.state_dict())


def drqn_select_action(model: drqn_DQN, state, action_mask, hidden):
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).view(1, 1, -1)

        q, hidden = model(x, hidden)
        q = q[0, 0]  # (A,)

        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        q_masked = q.masked_fill(~mask, -1e9)

        action = int(torch.argmax(q_masked).item())

    if hidden is not None:
        hidden = hidden.detach()

    return action, hidden


class SequenceReplay:

    def __init__(self, max_episodes: int, seq_len: int):
        self.max_episodes = max_episodes
        self.seq_len = seq_len

        self.episodes = deque(maxlen=max_episodes)
        self.cur_episode = []

    def clear(self):
        self.episodes.clear()
        self.cur_episode.clear()

    def __len__(self):
        return sum(len(ep) for ep in self.episodes) + len(self.cur_episode)

    def append(self, state, action, reward, next_state, done, next_mask):
        self.cur_episode.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            np.asarray(next_mask, dtype=bool),
        ))

        if done:
            self.end_episode()

    def end_episode(self):
        if len(self.cur_episode) > 0:
            self.episodes.append(self.cur_episode)
            self.cur_episode = []

    def _sources(self):
        src = list(self.episodes)

        # allow learning from the current unfinished life too
        if len(self.cur_episode) > 0:
            src.append(self.cur_episode)

        return [ep for ep in src if len(ep) > 0]

    def sample(self, batch_size: int):
        sources = self._sources()

        if not sources:
            raise ValueError("SequenceReplay is empty")

        batch = []

        for _ in range(batch_size):
            ep = random.choice(sources)
            start = random.randint(0, len(ep) - 1)
            chunk = ep[start:start + self.seq_len]

            batch.append(chunk)

        return self._collate(batch)

    def _collate(self, batch):
        B = len(batch)
        T = self.seq_len

        # infer sizes from first real transition
        first = batch[0][0]
        F_dim = len(first[0])
        A_dim = len(first[5])

        states = np.zeros((B, T, F_dim), dtype=np.float32)
        actions = np.zeros((B, T), dtype=np.int64)
        rewards = np.zeros((B, T), dtype=np.float32)
        next_states = np.zeros((B, T, F_dim), dtype=np.float32)
        dones = np.ones((B, T), dtype=bool)
        next_masks = np.zeros((B, T, A_dim), dtype=bool)
        valid = np.zeros((B, T), dtype=bool)

        for b, chunk in enumerate(batch):
            for t, tr in enumerate(chunk):
                s, a, r, ns, d, nm = tr

                states[b, t] = s
                actions[b, t] = a
                rewards[b, t] = r
                next_states[b, t] = ns
                dones[b, t] = d
                next_masks[b, t] = nm
                valid[b, t] = True

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "next_masks": next_masks,
            "valid": valid,
        }


def drqn_learn_step(model: drqn_DQN, target: drqn_DQN, optimiser, batch, gamma: float, burn_in: int = 5, n_step: int = 1) -> float:
    if hasattr(model, "reset_noise"):
        model.reset_noise()
        target.reset_noise()

    states = torch.as_tensor(batch["states"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32)
    dones = torch.as_tensor(batch["dones"], dtype=torch.bool)
    next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool)
    valid = torch.as_tensor(batch["valid"], dtype=torch.bool)

    B, T, F_dim = states.shape

    # Run one recurrent stream over s0, s1, ..., sT.
    # q_all[:, t] is Q(s_t), next_q_all[:, t] is Q(s_{t+1}).
    full_states = torch.cat([states, next_states[:, -1:, :]], dim=1)

    q_full, _ = model(full_states)
    q_all = q_full[:, :-1, :]              # (B, T, A)

    q_chosen = q_all.gather(
        dim=2,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)                          # (B, T)

    with torch.no_grad():
        next_q_full, _ = target(full_states)
        next_q_all = next_q_full[:, 1:, :]                      # (B,T,A) = Q(s_{t+1})
        next_q_masked = next_q_all.masked_fill(~next_masks, -1e9)
        best_next_q = next_q_masked.max(dim=2).values          # (B,T)

        B, T = rewards.shape

        # accumulate n discounted rewards forward from each t, cutting at done
        n_step_R   = torch.zeros_like(rewards)                  # (B,T)
        boot_q     = torch.zeros_like(rewards)                  # Q to bootstrap from
        boot_gamma = torch.zeros_like(rewards)                  # gamma**(steps taken)
        boot_live  = torch.zeros_like(rewards)                  # 0 if episode ended in window

        for t in range(T):
            R = torch.zeros(B, dtype=rewards.dtype)
            g = torch.ones(B, dtype=rewards.dtype)
            ended = torch.zeros(B, dtype=torch.bool)
            last = torch.full((B,), t, dtype=torch.long)       # index we bootstrap from
            for k in range(n_step):
                idx = t + k
                if idx >= T:
                    break
                step_live = (~ended).float()
                R = R + step_live * g * rewards[:, idx]
                g = g * gamma
                last = torch.where(ended, last, torch.full((B,), idx, dtype=torch.long))
                ended = ended | dones[:, idx]
            n_step_R[:, t]   = R
            boot_gamma[:, t] = g                                # gamma**(steps actually taken)
            boot_live[:, t]  = (~ended).float()                # 0 → window hit a done, no bootstrap
            boot_q[:, t]     = best_next_q[torch.arange(B), last]

        target_q = n_step_R + boot_gamma * boot_q * boot_live   # (B,T)

    loss_mask = valid.clone()

    if burn_in > 0:
        loss_mask[:, :burn_in] = False

    if not loss_mask.any():
        return 0.0

    loss = F.smooth_l1_loss(
        q_chosen[loss_mask],
        target_q[loss_mask],
    )

    optimiser.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

    optimiser.step()

    return loss.item()

class noisy_drqn_DQN(nn.Module):

    def __init__(self, n_input: int, n_hidden: int, n_act: int, sigma_0: float):
        super().__init__()

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.gru = nn.GRU(
            input_size=n_hidden,
            hidden_size=n_hidden,
            batch_first=True,
        )
        self.fc2 = NoisyLinear(n_hidden, n_act, sigma_0=sigma_0)

    def forward(self, x: torch.Tensor, hidden=None):
        single_step = (x.dim() == 2)

        if single_step:
            x = x.unsqueeze(1)   # (B, F) -> (B, 1, F)

        z = F.relu(self.fc1(x))  # (B, T, H)
        z, hidden = self.gru(z, hidden)
        q = self.fc2(F.relu(z))  # (B, T, A)

        if single_step:
            q = q.squeeze(1)     # (B, 1, A) -> (B, A)

        return q, hidden
    
    def reset_noise(self):
        self.fc2.reset_noise()


def noisy_drqn_make_model(n_input: int, n_hidden: int, n_act: int, lr: float, sigma_0: float):
    model = noisy_drqn_DQN(n_input, n_hidden, n_act, sigma_0=sigma_0)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def noisy_drqn_make_target(model: noisy_drqn_DQN) -> noisy_drqn_DQN:
    target = copy.deepcopy(model)
    target.eval()
    return target