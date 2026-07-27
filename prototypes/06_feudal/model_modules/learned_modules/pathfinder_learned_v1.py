"""
pathfinder_learned_v1.py  —  learned goal-conditioned pathfinder.

Contract: PathfinderObs carries `to_goal` and nothing else, and the goal is ALWAYS a
real remembered coordinate. So this module is a pure function of a displacement, and it
is trained OUT OF SIM, in `training/rigs/pathfinder_rig_v1.py` — a bare hex-geometry
goal-reaching task with no drives, no decay, no resources.

That is the point, not a shortcut. If a pathfinder trained with no access to the sim
then holds up inside the full stack, the observation contract is proven closed: nothing
undeclared leaked in. If it degrades in-sim, there is a dependency PathfinderObs does not
name, and that is a contract bug worth finding. `tests/test_module_diff_v1.py` is where
that check lives.

STATELESS BY REQUIREMENT. The orchestrator may switch goals mid-commute and this module
is called fresh every tick with no episode boundary. Out-of-sim training is only valid
because there is no hidden state to carry. If a recurrent pathfinder is ever tried, this
rig stops being a legitimate way to train it.

Encoding lives here, next to inference, and the rig imports it. One definition, so train
and inference cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import torch

from model_modules.contract_v1 import Pathfinder, PathfinderObs, HexMove

# Order and meaning of the encoded vector. Stored in the checkpoint as `obs_fields`
# and asserted on load, so a reshaped encoder cannot be fed to a stale net silently.
OBS_FIELDS = ["dq", "dr", "ds", "dist", "on_goal"]
N_INPUT = len(OBS_FIELDS)
N_ACT = 6  # HexMove.D0..D5


def hex_len(q: int, r: int) -> int:
    """Hex distance from origin in axial coords. Matches hex_world_cached.HexWorld.hex_dist."""
    return max(abs(q), abs(r), abs(-q - r))


def encode_to_goal(to_goal, scale: float) -> np.ndarray:
    """
    Displacement -> net input.

    All three cube components are given rather than just (q, r): the third is redundant
    algebraically but makes the six move directions symmetric to the net, which matters
    for a 2-layer MLP that would otherwise have to learn the asymmetry.

    `scale` normalises magnitude — pass the world radius so inputs sit in ~[-1, 1] and
    the same weights behave across radii. It is a normalisation constant, not knowledge
    about the map.

    `on_goal` flags zero displacement explicitly. The oracle's behaviour there is
    arbitrary (it steps off and back) and the composer should not be calling us anyway,
    but an explicit flag stops the net inventing a direction from an all-zero input.
    """
    gq, gr = int(to_goal[0]), int(to_goal[1])
    gs = -gq - gr
    d = hex_len(gq, gr)
    s = float(scale) if scale else 1.0
    return np.array(
        [gq / s, gr / s, gs / s, d / s, 1.0 if d == 0 else 0.0],
        dtype=np.float32,
    )


# --- instrumentation ------------------------------------------------------------
# A swap that changes nothing is ambiguous: the module may be matching the oracle, or it
# may never have been called. These counters separate those, and the histogram reports the
# displacement distribution the sim ACTUALLY produces — which is what the optimality gate
# should be set against, rather than against a uniform sweep of a space the agent never
# visits. Counting is always on (an int increment is free); the histogram is opt-in.
CALLS = 0
DIST_HIST: dict[int, int] = {}
INSTRUMENT = False


def reset_instrumentation(instrument: bool = True) -> None:
    global CALLS, DIST_HIST, INSTRUMENT
    CALLS = 0
    DIST_HIST = {}
    INSTRUMENT = bool(instrument)


def instrumentation() -> dict:
    return {"calls": CALLS, "dist_hist": dict(sorted(DIST_HIST.items()))}


def default_arch(n_hidden: int = 64, kind: str = "mlp", sigma_0: float = 0.5) -> dict:
    arch = {"kind": kind, "n_input": N_INPUT, "n_hidden": n_hidden, "n_act": N_ACT}
    if kind == "noisy":
        arch["sigma_0"] = sigma_0
    return arch


class LearnedPathfinder(Pathfinder):
    """
    Constructed by sim_instance.build_modules as:

        path_make(**oracle_params.get("pathfinder", {}))

    which passes no rng and no explorer, so the signature is kwargs-only. `weights` is a
    STRING TAG, never a live net — configs cross a spawn boundary into sweep workers and
    torch objects do not survive that pickle. The net is loaded here, in the worker, on
    first construction, and cached per process by checkpoint_v1.
    """

    def __init__(self, weights: str = None, scale: float = 20.0, net=None):
        if net is None and weights is None:
            raise ValueError(
                "LearnedPathfinder needs weights='<tag>' (loaded in-worker) or an "
                "explicit net= (trainer use only)."
            )

        self.scale = float(scale)
        self.tag = weights

        if net is not None:
            # trainer path: evaluate a net mid-training without a round-trip through disk
            self.net = net
        else:
            from model_modules.checkpoint_v1 import load_net

            self.net, payload = load_net("pathfinder", weights)
            if payload["obs_fields"] != OBS_FIELDS:
                raise ValueError(
                    f"pathfinder '{weights}' was trained on obs_fields "
                    f"{payload['obs_fields']}, this tree encodes {OBS_FIELDS}. Retrain."
                )
            # scale is part of the encoding, so it must come from the checkpoint, not a default
            self.scale = float(payload["train_config"].get("scale", self.scale))

    def move(self, obs: PathfinderObs) -> HexMove:
        global CALLS
        CALLS += 1
        if INSTRUMENT:
            d = hex_len(int(obs.to_goal[0]), int(obs.to_goal[1]))
            DIST_HIST[d] = DIST_HIST.get(d, 0) + 1

        x = encode_to_goal(obs.to_goal, self.scale)
        with torch.no_grad():
            q = self.net(torch.from_numpy(x))
        return HexMove(int(torch.argmax(q).item()))


def make_pathfinder(weights: str = None, scale: float = 20.0, net=None) -> LearnedPathfinder:
    """Factory registered in sim_instance.MODULE_REGISTRY under 'learned'."""
    return LearnedPathfinder(weights=weights, scale=scale, net=net)
