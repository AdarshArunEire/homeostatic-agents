"""
pathfinder_noisy_v1.py  —  a deliberately degraded pathfinder. Measurement instrument,
not a candidate policy.

Exists to answer a question the learned/oracle comparison cannot answer on its own: when a
swap moves a metric by zero, is the module fine, or is the metric blind?

A null result is only informative if the measurement has been shown capable of producing a
non-null one. This module is the positive control. It plays the oracle's greedy descent but
takes a uniformly random legal-direction step with probability `epsilon`, so its optimality
is knowably ~(1 - 5/6 * epsilon). Sweeping epsilon and watching where a metric starts to
move gives that metric's DETECTION FLOOR — the amount of pathfinder degradation it can
actually see. A learned module whose degradation sits below that floor is certified by the
metric; one that sits above it and still shows no effect means the metric is not measuring
what it is being asked to measure.

The noise stream is independent of the sim seed on purpose: it is a controlled perturbation
applied identically across seeds, so the comparison stays paired.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Pathfinder, PathfinderObs, HexMove
from model_modules.oracle_modules.pathfinder_oracle_v1 import AXIAL_DIRS, hex_len

N_DIRS = len(AXIAL_DIRS)


class NoisyOraclePathfinder(Pathfinder):
    """
    Constructed by build_modules as path_make(**oracle_params["pathfinder"]), so it takes
    no rng and seeds its own.

    epsilon=0 reproduces the oracle exactly, including its first-wins tie-break, so the
    epsilon sweep starts from a true zero point rather than from something merely similar.
    """

    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.perturbed = 0

    def _greedy(self, to_goal) -> int:
        gq, gr = to_goal
        best_k, best_d = 0, None
        for k, (dq, dr) in enumerate(AXIAL_DIRS):
            d = hex_len(gq - dq, gr - dr)
            if best_d is None or d < best_d:
                best_d, best_k = d, k
        return best_k

    def move(self, obs: PathfinderObs) -> HexMove:
        self.calls += 1
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            self.perturbed += 1
            return HexMove(int(self.rng.integers(N_DIRS)))
        return HexMove(self._greedy(obs.to_goal))


def make_noisy_pathfinder(epsilon: float = 0.0, seed: int = 0) -> NoisyOraclePathfinder:
    """Factory registered in sim_instance.MODULE_REGISTRY under 'noisy_oracle'."""
    return NoisyOraclePathfinder(epsilon=epsilon, seed=seed)
