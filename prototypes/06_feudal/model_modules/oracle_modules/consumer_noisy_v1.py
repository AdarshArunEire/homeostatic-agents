"""
consumer_noisy_v1.py  —  deliberately degraded consumer. Measurement instrument.

Same job as pathfinder_noisy_v1: give the metrics a known-bad input so a null result means
something. P1.1 established the rule the hard way — solveScore looked like it had a
detection floor until a controlled degradation showed it was non-monotone, i.e. blind.

TWO KNOBS, not one, and that is the point.

  sigma    gaussian offset on every fill. Small errors, everywhere, always.
  epsilon  with probability epsilon, return a uniformly random fill. Rare, large errors.

The pathfinder result showed error CHARACTER dominates error RATE: a module at 0.98
optimality did ~12x less damage than 2% uniform noise, because its mistakes were the cheap
kind. A single scalar "how wrong" knob cannot express that distinction. Sweeping sigma and
epsilon separately tells you which shape of error the metric is actually sensitive to, and
therefore which one the learned module needs to be checked against.

Both default to 0, so `epsilon=0, sigma=0` reproduces the oracle exactly and the sweep
starts from a true zero rather than something merely close.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Consumer, ConsumerObs
import world_v1 as world


class NoisyOracleConsumer(Consumer):
    """
    Imports world_v1 — legitimately. This is an ORACLE-family module: knowing the true
    consume dynamics is the privilege that evaporates when the learned impl replaces it.
    The learned consumer importing world would be a leak; this one doing so is the baseline
    that leak would be hiding from.
    """

    def __init__(self, resource: str, fill_target: float = 1.7,
                 sigma: float = 0.0, epsilon: float = 0.0, seed: int = 0):
        if resource not in ("water", "food"):
            raise ValueError(f"resource must be 'water' or 'food', got {resource!r}")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.resource = resource
        self.fill_target = float(fill_target)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.perturbed = 0

    def _oracle(self, obs: ConsumerObs) -> float:
        if self.resource == "water":
            return world.drink_frac_for(obs.h, obs.s, self.fill_target)
        return world.eat_frac_for(obs.s, self.fill_target)

    def consume(self, obs: ConsumerObs) -> float:
        self.calls += 1
        base = self._oracle(obs)

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            self.perturbed += 1
            return float(self.rng.random())

        if self.sigma > 0.0:
            self.perturbed += 1
            return float(np.clip(base + self.rng.normal(0.0, self.sigma), 0.0, 1.0))

        return float(base)


def make_eat_noisy(fill_target: float = 1.7, sigma: float = 0.0,
                   epsilon: float = 0.0, seed: int = 0) -> NoisyOracleConsumer:
    return NoisyOracleConsumer("food", fill_target, sigma, epsilon, seed)


def make_drink_noisy(fill_target: float = 1.7, sigma: float = 0.0,
                     epsilon: float = 0.0, seed: int = 1) -> NoisyOracleConsumer:
    # different default seed from eat so the two slots do not share a noise stream and
    # accidentally correlate their errors
    return NoisyOracleConsumer("water", fill_target, sigma, epsilon, seed)
