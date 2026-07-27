"""
explorer_noisy_v1.py  —  deliberately degraded explorer. Measurement instrument.

Wraps `SmellMomentumExplorer` (the 0.48 reactive floor) and corrupts it by a known amount,
so the sensitivity of every candidate metric to EXPLORER quality can be measured before any
learned number is read.

Run this BEFORE the trainer. The consumer thread cost four training runs discovering
afterwards that the environment could barely resolve the module at all; the rule adopted
from it is calibrate first, train second.

Expectation here is different from the consumer's, and the difference is the point. The
explorer ladder already on record — random 0.17, momentum 0.33, smell_momentum 0.48, god
1.00 — is monotone in explorer quality across a 0.83 range. That is a metric with genuine
dynamic range, so this control should show a clean dose-response rather than the flat
plateau the consumer produced. If it does not, the verdict metric is wrong and needs finding
before the thesis run, not after.

epsilon = probability of replacing the chemotactic choice with a uniform legal move.
epsilon=1.0 degrades to (roughly) the random explorer, so the sweep spans the known ladder
and can be checked against the 0.17 floor as an anchor.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Explorer, ExplorerObs, HexMove
from model_modules.oracle_modules.explorer_oracle_v1 import SmellMomentumExplorer


class NoisyExplorer(Explorer):
    """
    Legitimately an oracle-family module: it is the reactive baseline plus calibrated noise,
    used to measure the measuring apparatus.
    """

    def __init__(self, rng: np.random.Generator | None = None, epsilon: float = 0.0,
                 persist_p: float = 0.75, avoid_reverse: bool = True,
                 trend_eps: float = 0.01, follow_p: float = 0.95,
                 reverse_on_drop_p: float = 0.75, **kwargs):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = float(epsilon)
        self.inner = SmellMomentumExplorer(
            rng=self.rng, persist_p=persist_p, avoid_reverse=avoid_reverse,
            trend_eps=trend_eps, follow_p=follow_p,
            reverse_on_drop_p=reverse_on_drop_p,
        )
        self.n_calls = 0
        self.perturbed = 0

    def reset(self) -> None:
        self.inner.reset()

    def act(self, obs: ExplorerObs) -> HexMove:
        self.n_calls += 1
        # the inner explorer is stepped either way, so its internal momentum state stays
        # consistent with the passage of time rather than freezing whenever noise fires
        base = self.inner.act(obs)

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            self.perturbed += 1
            legal = np.flatnonzero(np.asarray(obs.legal_moves, dtype=bool))
            if len(legal):
                return HexMove(int(legal[self.rng.integers(len(legal))]))
        return base


def make_explorer_noisy(rng: np.random.Generator | None = None, epsilon: float = 0.0,
                        **kwargs) -> NoisyExplorer:
    return NoisyExplorer(rng=rng, epsilon=epsilon, **kwargs)
