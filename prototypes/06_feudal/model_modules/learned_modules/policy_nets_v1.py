"""
policy_nets_v1.py  —  stochastic policy networks. Outputs a distribution, not values.

Added because the explorer's target policy family is IRREDUCIBLY STOCHASTIC and DQN cannot
represent it.

The reasoning, from P3.1-P3.3. The explorer's observation carries no positional memory, so
the best policy it can express is reactive: a correlated random walk with chemotaxis — which
is precisely `smell_momentum`, the 0.4776 baseline. But that baseline is defined by
PROBABILITIES (persist_p 0.75, follow_p 0.95, reverse_on_drop_p 0.75), and a correlated
random walk without the randomness is a straight line or a loop. Q-learning returns a greedy
deterministic map from an aliased observation, which in the dead band cycles — measurably
worse than random. Consistent with Singh, Jaakkola & Jordan (1994): for memoryless POMDP
policies the best deterministic policy can be arbitrarily suboptimal, and the optimum may
require stochasticity.

A policy network learns `persist with probability p` as a PARAMETER. That is the whole point;
it is not a change of optimiser preference.

Kept separate from nets_v1 because the semantics differ: these emit logits over actions
interpreted as a distribution, and a value head used only as a variance-reducing baseline —
never for action selection.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    Shared trunk, two heads: policy logits and a state-value baseline.

    The value head exists ONLY to reduce gradient variance in the advantage term. It never
    selects an action, so the deterministic-argmax failure has no route back in.
    """

    def __init__(self, n_input: int, n_hidden: int, n_act: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(n_hidden, n_act)
        self.v = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def logits(self, x):
        return self.pi(self.trunk(x))

    def reset_noise(self) -> None:
        return


def build_policy_net(arch: dict) -> PolicyNet:
    arch = dict(arch)
    kind = arch.pop("kind", "policy")
    if kind != "policy":
        raise ValueError(f"policy_nets_v1 builds kind='policy', got {kind!r}")
    arch.pop("sigma_0", None)
    return PolicyNet(**arch)


def masked_categorical(logits: torch.Tensor, legal_mask) -> torch.distributions.Categorical:
    """
    Distribution over LEGAL actions only.

    Masking in logit space rather than renormalising probabilities afterwards keeps
    log_prob() and entropy() consistent with what was actually sampled — renormalising after
    the fact silently breaks the gradient.
    """
    mask = torch.as_tensor(np.asarray(legal_mask), dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0) if logits.ndim == 2 else mask
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def discounted_returns(rewards, dones, gamma: float) -> np.ndarray:
    """
    Monte-Carlo returns, reset at terminals.

    NO BOOTSTRAP — which is why policy gradient is immune to the failure that produced an
    anti-informative Q-function in P3.2: there is no `next_state` value to be wrong about.
    Terminals still matter, because a return must not run across the end of a sub-task.
    """
    r = np.asarray(rewards, dtype=np.float64)
    d = np.asarray(dones, dtype=bool)
    out = np.zeros_like(r)
    running = 0.0
    for i in range(len(r) - 1, -1, -1):
        running = r[i] + (0.0 if d[i] else gamma * running)
        out[i] = running
    return out
