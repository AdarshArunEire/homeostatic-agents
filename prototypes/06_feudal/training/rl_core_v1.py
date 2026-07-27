"""
rl_core_v1.py  —  shared RL machinery for all four Proto 06 module trainers.

Deliberately thin. The algorithms are the ones 03b convicted (n-step returns, masked
Bellman target, double-DQN option, NoisyNets); this file just makes them reusable across
four modules whose only real differences are the observation encoder, the reward, and
where the episode boundary sits.

Transition format matches 05_generalisation/drqn.py exactly:

    (state, action, reward, next_state, done, next_mask, n_used)

so anything written here stays diff-able against the Proto 04/05 numbers rather than
quietly becoming a different algorithm with the same name.

MASKED TARGET, always. Illegal next-actions are driven to -1e9 before the max, or the
bootstrap leaks value through moves the agent can never take — the same bug the rim
masking made visible in Proto 04.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Flat FIFO over 7-tuples. Sized in transitions, not episodes."""

    def __init__(self, capacity: int, rng: np.random.Generator | None = None):
        self.buf = deque(maxlen=int(capacity))
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buf)

    def append(self, transition) -> None:
        self.buf.append(transition)

    def extend(self, transitions) -> None:
        self.buf.extend(transitions)

    def sample(self, batch_size: int):
        n = len(self.buf)
        idx = self.rng.integers(0, n, size=min(batch_size, n))
        return [self.buf[int(i)] for i in idx]


class NStepAccumulator:
    """
    Rolls raw 1-step experience into n-step transitions.

    `n_used` is carried per transition because episodes end mid-window: the tail of an
    episode yields transitions of length < n, and the target must discount by the actual
    horizon used. drqn's learn steps already expect this field.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.q = deque()

    def clear(self) -> None:
        self.q.clear()

    def push(self, state, action, reward, next_state, done, next_mask):
        """Returns a list of completed n-step transitions (usually 0 or 1)."""
        self.q.append((state, action, reward, next_state, done, next_mask))
        out = []
        if len(self.q) >= self.n_step:
            out.append(self._collapse(self.n_step))
            self.q.popleft()
        if done:
            # flush the tail: every remaining prefix ends at this terminal
            while self.q:
                out.append(self._collapse(len(self.q)))
                self.q.popleft()
        return out

    def flush(self):
        """
        Drain the window at a TRUNCATION (step cap), as opposed to a terminal.

        These transitions keep done=False so the target still bootstraps: running out of
        clock is not the same event as arriving, and marking it terminal would teach the
        agent that value stops at the cap.
        """
        out = []
        while self.q:
            out.append(self._collapse(len(self.q)))
            self.q.popleft()
        return out

    def _collapse(self, k: int):
        state, action = self.q[0][0], self.q[0][1]
        r = 0.0
        for i in range(k):
            r += (self.gamma ** i) * self.q[i][2]
            if self.q[i][4]:  # done inside the window
                k = i + 1
                break
        last = self.q[k - 1]
        return (state, action, r, last[3], last[4], last[5], k)


def select_action(net, state, action_mask, greedy: bool = False) -> int:
    """
    Masked argmax. NoisyNets carry their own exploration, so there is no epsilon here;
    a trainer that wants epsilon should apply it around this call.
    """
    with torch.no_grad():
        q = net(torch.as_tensor(state, dtype=torch.float32))
        mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool)
        return int(torch.argmax(q.masked_fill(~mask, -1e9)).item())


def learn_step(net, target, optimiser, batch, gamma: float, double: bool = True) -> float:
    """One gradient step. Mirrors drqn.double_learn_step / vanilla_learn_step."""
    state = torch.as_tensor(np.array([tr[0] for tr in batch]), dtype=torch.float32)
    action = torch.tensor([tr[1] for tr in batch], dtype=torch.long)
    reward = torch.tensor([tr[2] for tr in batch], dtype=torch.float32)
    next_state = torch.as_tensor(np.array([tr[3] for tr in batch]), dtype=torch.float32)
    done = torch.tensor([tr[4] for tr in batch], dtype=torch.bool)
    next_mask = torch.as_tensor(np.array([tr[5] for tr in batch]), dtype=torch.bool)
    n_used = torch.tensor([tr[6] for tr in batch], dtype=torch.float32)

    q_all = net(state)
    q_chosen = q_all[torch.arange(len(action)), action]

    with torch.no_grad():
        if double:
            # online selects, target scores
            next_actions = net(next_state).masked_fill(~next_mask, -1e9).argmax(dim=1)
            best_next_q = target(next_state)[torch.arange(len(next_actions)), next_actions]
        else:
            best_next_q = target(next_state).masked_fill(~next_mask, -1e9).max(dim=1).values
        target_q = reward + ((gamma ** n_used) * best_next_q) * (~done).float()

    loss = F.smooth_l1_loss(q_chosen, target_q)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return float(loss.item())


def sync_target(target, net) -> None:
    target.load_state_dict(net.state_dict())


def seed_everything(seed: int) -> np.random.Generator:
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
