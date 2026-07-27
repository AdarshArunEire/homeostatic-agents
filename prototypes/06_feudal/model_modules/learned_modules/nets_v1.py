"""
nets_v1.py  —  the nets the learned modules are built from.

Self-contained on purpose. Every prototype in this repo carries its own copy of the
files it depends on (hex_world_cached, map_curriculum, plot_fn, sweep_fn), so Proto 06
does not reach across into `05_generalisation/drqn.py`. The architectures below mirror
that file's `vanilla_DQN` / `NoisyLinear` / `noisy_DQN` so results stay comparable;
they are not a redesign.

Everything is built through `build_net(arch)` from a plain dict. Nothing else is allowed
to instantiate a net directly, because a checkpoint has to be rebuildable from its stored
`arch` alone, without importing a class by name that may since have been renamed.

Heads:
  mlp    — plain MLP, n_act logits. Pathfinder (6 hex moves), consumer (fill bins).
  noisy  — NoisyNet MLP. For modules whose exploration is the point (explorer,
           orchestrator); carried over because 03b convicted NoisyNets as the winner.

Masking is NOT applied here. Modules apply their own legality mask at act() time
(Orchestrator.legal_mask, or the rig's legal-move mask) so the net stays a pure
state -> Q mapping and the mask stays where the contract defines it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Mirrors drqn.vanilla_DQN."""

    def __init__(self, n_input: int, n_hidden: int, n_act: int):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def reset_noise(self) -> None:
        """No-op, so callers can treat MLP and NoisyMLP identically."""
        return


class NoisyLinear(nn.Module):
    """Mirrors drqn.NoisyLinear (factorised gaussian noise)."""

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_0 / math.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma_0 / math.sqrt(in_features))

        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            # frozen at sim time: mean weights only, so eval is deterministic
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class NoisyMLP(nn.Module):
    """Mirrors drqn.noisy_DQN."""

    def __init__(self, n_input: int, n_hidden: int, n_act: int, sigma_0: float = 0.5):
        super().__init__()
        self.fc1 = NoisyLinear(n_input, n_hidden, sigma_0)
        self.fc2 = NoisyLinear(n_hidden, n_act, sigma_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def reset_noise(self) -> None:
        self.fc1.reset_noise()
        self.fc2.reset_noise()


_KINDS = {
    "mlp": MLP,
    "noisy": NoisyMLP,
}


def build_net(arch: dict):
    """
    Rebuild a net from a stored arch dict.

    arch = {"kind": "mlp"|"noisy", "n_input": int, "n_hidden": int, "n_act": int,
            "sigma_0": float (noisy only)}
    """
    arch = dict(arch)
    kind = arch.pop("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown net kind {kind!r}; expected one of {sorted(_KINDS)}")
    if kind == "mlp":
        arch.pop("sigma_0", None)
    return _KINDS[kind](**arch)


def make_trainable(arch: dict, lr: float):
    """Net + Adam + a synced target copy. The standard triple every trainer opens with."""
    import copy

    net = build_net(arch)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    target = copy.deepcopy(net)
    target.eval()
    return net, target, optimiser
