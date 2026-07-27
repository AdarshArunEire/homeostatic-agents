"""
consumer_learned_v1.py  —  learned eat/drink consumer.

Returns a fill FRACTION, graded, because the whole skill is stopping near ideal rather
than slamming 1.0: the comfort surface gives a free buffer to ideal+OVER_TOL and bites
beyond it, so "how much" is the decision, not "whether".

DOES NOT IMPORT world_v1, AND MUST NOT. world_v1's docstring names itself the greppable
privilege boundary: an oracle may know the true physics (that is the privilege which
evaporates on swap), a learned module may not. The consume dynamics -- DRINK_AMOUNT, the
(0.8 + 0.3*s) coupling, the absorption profiles -- are exactly what this module has to
discover from reward. `tests/probe_infra_v1.py` greps for the violation.

The normalisation constants below are NOT a way round that. Knowing the range of your own
observation is not privileged information; knowing how a drink converts to hydration is.

Trained in-sim (training/rigs/insim_rig_v1) with oracles in the other three slots, because
every ConsumerObs field except brightness is sim state, and the overfill cost only shows up
over the ticks after the decision.
"""

from __future__ import annotations

import numpy as np
import torch

from model_modules.contract_v1 import Consumer, ConsumerObs
from model_modules.learned_modules.recorder_v1 import Recorder

# Observation encoding. Stored in the checkpoint as `obs_fields` and asserted on load.
OBS_FIELDS = ["h", "s", "brightness", "tile_water_lvl", "tile_food_lvl", "h_deficit", "s_deficit"]
N_INPUT = len(OBS_FIELDS)

# Normalisation only. Drives live in [0, 3] and levels/brightness in [0, 1]; dividing by
# the observed range keeps inputs ~[-1, 1]. No physics is encoded here.
DRIVE_SCALE = 3.0
IDEAL = 1.0

# 21 bins at 0.05 resolution. The contract allows anything from 5 to ~20; finer is better
# here because the free buffer is 1.0 wide and DRINK_AMOUNT is 0.15, so a coarse grid
# cannot express "top up a little" without overshooting the band.
N_BINS = 21
BINS = np.linspace(0.0, 1.0, N_BINS).astype(np.float32)


def encode_consumer_obs(obs: ConsumerObs) -> np.ndarray:
    """
    ConsumerObs -> net input.

    The two deficit terms are redundant algebraically (ideal - drive) but handed over
    explicitly: the decision is almost entirely a function of deficit, and making a 2-layer
    MLP discover a subtraction before it can start learning the interesting part wastes
    capacity on arithmetic. Distance-to-ideal is observable, not privileged.
    """
    return np.array(
        [
            obs.h / DRIVE_SCALE,
            obs.s / DRIVE_SCALE,
            obs.brightness,
            obs.tile_water_lvl,
            obs.tile_food_lvl,
            (IDEAL - obs.h) / DRIVE_SCALE,
            (IDEAL - obs.s) / DRIVE_SCALE,
        ],
        dtype=np.float32,
    )


def default_arch(n_hidden: int = 64, kind: str = "mlp", sigma_0: float = 0.5) -> dict:
    arch = {"kind": kind, "n_input": N_INPUT, "n_hidden": n_hidden, "n_act": N_BINS}
    if kind == "noisy":
        arch["sigma_0"] = sigma_0
    return arch


# instrumentation, same rationale as the pathfinder: a swap that moves nothing is
# ambiguous between "module is fine" and "module never ran"
CALLS = {"eat": 0, "drink": 0}


def reset_calls() -> None:
    CALLS["eat"] = 0
    CALLS["drink"] = 0


class LearnedConsumer(Consumer, Recorder):
    """
    Built by build_modules as eat_make(**oracle_params["eat"]) — kwargs only, no rng.
    `weights` is a string tag resolved inside the worker; see checkpoint_v1.
    """

    def __init__(self, resource: str, weights: str = None, net=None,
                 explore_eps: float = 0.0, rng=None):
        if resource not in ("water", "food"):
            raise ValueError(f"resource must be 'water' or 'food', got {resource!r}")
        if net is None and weights is None:
            raise ValueError(
                "LearnedConsumer needs weights='<tag>' or an explicit net= (trainer only)."
            )
        self.resource = resource
        self.slot = "drink" if resource == "water" else "eat"
        self.tag = weights
        # TRAINING ONLY. Defaults to 0 so anything loaded from a checkpoint is a pure
        # argmax — a module that explored at eval time would make every reported number
        # noisier than the policy it is meant to be measuring.
        self.explore_eps = float(explore_eps)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._init_recorder()

        if net is not None:
            self.net = net
        else:
            from model_modules.checkpoint_v1 import load_net

            self.net, payload = load_net(self.slot, weights)
            if payload["obs_fields"] != OBS_FIELDS:
                raise ValueError(
                    f"{self.slot} '{weights}' trained on {payload['obs_fields']}, "
                    f"this tree encodes {OBS_FIELDS}. Retrain."
                )

    def consume(self, obs: ConsumerObs) -> float:
        CALLS[self.slot] += 1
        x = encode_consumer_obs(obs)
        if self.explore_eps > 0.0 and self.rng.random() < self.explore_eps:
            k = int(self.rng.integers(N_BINS))
        else:
            with torch.no_grad():
                q = self.net(torch.from_numpy(x))
            k = int(torch.argmax(q).item())
        # record the action ACTUALLY taken, exploratory or not — recording the greedy
        # choice while the sim absorbed the exploratory one is off-policy in the broken
        # sense, and the reward joined to it would belong to a different decision
        self.record(x, k)
        return float(BINS[k])


def make_eat_learned(weights: str = None, net=None) -> LearnedConsumer:
    return LearnedConsumer("food", weights=weights, net=net)


def make_drink_learned(weights: str = None, net=None) -> LearnedConsumer:
    return LearnedConsumer("water", weights=weights, net=net)
