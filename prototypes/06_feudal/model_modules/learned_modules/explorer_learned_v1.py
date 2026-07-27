"""
explorer_learned_v1.py  —  the module Proto 06 exists to test.

Question: does a learned, memory-carrying explorer beat reactive chemotaxis (solveScore
0.48) and move toward the god ceiling (1.00), at smell 3 with the dead band intact?

WHY MEMORY IS NON-REDUNDANT HERE, unlike Proto 04 H4. That experiment falsified a GRU
because a fixed map rewards latching a static route, and a static route needs no carried
intention. Here absolute position is useless (maps resample), the only signal is a 3-slot
smell history, and the job is directed coverage of a region where the gradient is flat. The
architecture is the same; the justification is not.

ENCODING — the part that matters.

`ExplorerObs` gives smell as three readings, newest first. A flattened triple is nearly
useless on its own: run-and-tumble chemotaxis needs to know WHICH MOVE produced WHICH change
in smell. So the module carries its own last three actions and pairs them with the readings:

    (a_{t-1}, s_t), (a_{t-2}, s_{t-1}), (a_{t-3}, s_{t-2})

That action history is the module's OWN state — an efference copy, which is both
biologically standard and exactly what the oracle `SmellMomentumExplorer` already keeps
internally. It is not privileged information: the agent knows what it just did.

First differences (s0-s1, s1-s2) are handed over explicitly. They are algebraically
redundant but they are the quantity chemotaxis actually acts on, and making a 2-layer MLP
rediscover a subtraction before it can start learning the policy wastes capacity on
arithmetic.

DOES NOT IMPORT world_v1 — see the privilege boundary probe.
"""

from __future__ import annotations

import numpy as np
import torch

from model_modules.contract_v1 import Explorer, ExplorerObs, HexMove
from model_modules.learned_modules.recorder_v1 import Recorder

N_DIRS = 6
HIST = 3  # matches the 3-slot smell history: one action per reading

OBS_FIELDS = (
    [f"legal_{k}" for k in range(N_DIRS)]
    + [f"act{h}_{k}" for h in range(HIST) for k in range(N_DIRS)]
    + [f"act{h}_valid" for h in range(HIST)]
    + ["w0", "w1", "w2", "w_d1", "w_d2"]
    + ["f0", "f1", "f2", "f_d1", "f_d2"]
    + ["need_water", "need_food"]
)
N_INPUT = len(OBS_FIELDS)
N_ACT = N_DIRS


def encode_explorer_obs(obs: ExplorerObs, action_hist) -> np.ndarray:
    """
    ExplorerObs + own action history -> net input.

    `action_hist` is newest-first, entries are 0..5 or None for "no action yet this life".
    Alignment is deliberate: action_hist[i] is the move that produced smell reading [i].
    """
    legal = np.asarray(obs.legal_moves, dtype=np.float32).reshape(N_DIRS)

    acts = np.zeros(HIST * N_DIRS, dtype=np.float32)
    valid = np.zeros(HIST, dtype=np.float32)
    for h in range(HIST):
        a = action_hist[h] if h < len(action_hist) else None
        if a is not None:
            acts[h * N_DIRS + int(a)] = 1.0
            valid[h] = 1.0

    def smell_block(v):
        v = np.asarray(v, dtype=np.float32).reshape(3)
        # first differences: what run-and-tumble actually acts on
        return np.array([v[0], v[1], v[2], v[0] - v[1], v[1] - v[2]], dtype=np.float32)

    return np.concatenate([
        legal,
        acts,
        valid,
        smell_block(obs.water_smell),
        smell_block(obs.food_smell),
        np.array([float(obs.need_water), float(obs.need_food)], dtype=np.float32),
    ])


def default_arch(n_hidden: int = 128, kind: str = "noisy", sigma_0: float = 0.5) -> dict:
    """
    NoisyNet by default. 03b convicted NoisyNets as the winning exploration mechanism, and
    this is the one module whose whole job IS exploration — epsilon-greedy would inject
    undirected noise into a policy being trained to be directed.
    """
    arch = {"kind": kind, "n_input": N_INPUT, "n_hidden": n_hidden, "n_act": N_ACT}
    if kind == "noisy":
        arch["sigma_0"] = sigma_0
    return arch


CALLS = 0


def reset_calls() -> None:
    global CALLS
    CALLS = 0


class LearnedExplorer(Explorer, Recorder):
    """
    Built by build_modules as explorer_make(rng=..., **oracle_params["explorer"]), so it
    takes rng plus kwargs. `weights` is a string tag resolved in-worker.

    Action history resets per life (`reset()`), because a new life starts on a fresh spawn
    with no continuity — carrying the previous life's last move across a death would pair a
    move with a smell reading from a different place entirely.
    """

    def __init__(self, weights: str = None, net=None, rng=None,
                 explore_eps: float = 0.0, temperature: float = 0.0, **_ignored):
        if net is None and weights is None:
            raise ValueError(
                "LearnedExplorer needs weights='<tag>' or an explicit net= (trainer only)."
            )
        self.rng = rng if rng is not None else np.random.default_rng()
        self.explore_eps = float(explore_eps)
        # temperature > 0 samples from softmax(Q/T) over LEGAL moves instead of argmax.
        #
        # Not a convenience knob — it changes the POLICY CLASS. Q-learning yields a greedy
        # deterministic policy, and for memoryless POMDP policies the best deterministic
        # one can be arbitrarily suboptimal, with the optimum requiring stochasticity
        # (Singh, Jaakkola & Jordan 1994). In the dead band both smells read zero, so the
        # observation collapses to (legality, recent actions) and physically distinct cells
        # alias onto one input: a deterministic map from that either walks straight or
        # cycles, and nothing in the TD loss prefers the former. Every baseline that beats
        # this module — random, momentum, smell_momentum — is stochastic.
        self.temperature = float(temperature)
        # policy_mode: net emits action LOGITS (policy_nets_v1.PolicyNet) rather than
        # Q-values, and actions are sampled from the resulting distribution.
        self.policy_mode = bool(_ignored.pop("policy_mode", False))
        self.tag = weights
        self.action_hist: list = []
        self.n_calls = 0
        self._init_recorder()

        if net is not None:
            self.net = net
        elif self.policy_mode:
            from model_modules.checkpoint_v1 import load_payload
            from model_modules.learned_modules.policy_nets_v1 import build_policy_net

            payload = load_payload("explorer", weights)
            self.net = build_policy_net(payload["arch"])
            self.net.load_state_dict(payload["state_dict"])
            self.net.eval()
            for prm in self.net.parameters():
                prm.requires_grad_(False)
            if payload["obs_fields"] != OBS_FIELDS:
                raise ValueError(f"explorer '{weights}' obs_fields mismatch. Retrain.")
        else:
            from model_modules.checkpoint_v1 import load_net

            self.net, payload = load_net("explorer", weights)
            if payload["obs_fields"] != OBS_FIELDS:
                raise ValueError(
                    f"explorer '{weights}' trained on {len(payload['obs_fields'])} fields, "
                    f"this tree encodes {len(OBS_FIELDS)}. Retrain."
                )

    def reset(self) -> None:
        self.action_hist = []

    def act(self, obs: ExplorerObs) -> HexMove:
        global CALLS
        CALLS += 1
        self.n_calls += 1

        x = encode_explorer_obs(obs, self.action_hist)
        legal = np.asarray(obs.legal_moves, dtype=bool).reshape(N_DIRS)
        if not legal.any():
            legal = np.ones(N_DIRS, dtype=bool)

        if self.policy_mode:
            # Sample from pi(a|s). The policy IS the distribution — there is no argmax step
            # to collapse it, which is the entire reason this mode exists. See
            # policy_nets_v1: the target policy family (correlated random walk with
            # chemotaxis) is defined by probabilities, and a deterministic version of it is
            # a straight line or a loop.
            with torch.no_grad():
                lg = self.net.logits(torch.from_numpy(x))
            lg = lg.masked_fill(~torch.from_numpy(legal), -1e9)
            p = torch.softmax(lg, dim=-1).numpy().astype(np.float64)
            s = p.sum()
            k = (int(self.rng.choice(len(p), p=p / s)) if s > 0
                 else int(self.rng.choice(np.flatnonzero(legal))))
        elif self.explore_eps > 0.0 and self.rng.random() < self.explore_eps:
            k = int(self.rng.choice(np.flatnonzero(legal)))
        else:
            with torch.no_grad():
                q = self.net(torch.from_numpy(x)).numpy()
            q = np.where(legal, q, -np.inf)
            if self.temperature > 0.0:
                # subtract the max before exponentiating; Q here can reach ~10 (discover
                # reward) and exp overflows without it
                z = (q - np.nanmax(q[np.isfinite(q)])) / self.temperature
                p = np.exp(np.where(np.isfinite(z), z, -np.inf))
                s = p.sum()
                k = (int(self.rng.choice(len(p), p=p / s)) if s > 0
                     else int(self.rng.choice(np.flatnonzero(legal))))
            else:
                k = int(np.argmax(q))

        # record the action ACTUALLY taken, then push it into history so the next call
        # pairs it with the smell reading it produced
        self.record(x, k)
        self.action_hist.insert(0, k)
        del self.action_hist[HIST:]
        return HexMove(k)


def make_explorer_learned(weights: str = None, net=None, rng=None,
                          explore_eps: float = 0.0, temperature: float = 0.0,
                          **kw) -> LearnedExplorer:
    return LearnedExplorer(weights=weights, net=net, rng=rng,
                           explore_eps=explore_eps, temperature=temperature, **kw)
