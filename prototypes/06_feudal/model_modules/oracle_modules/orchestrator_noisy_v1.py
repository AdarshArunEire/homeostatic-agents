"""
orchestrator_noisy_v1.py  —  deliberately degraded orchestrator. Measurement instrument.

Run BEFORE training an orchestrator. The rule earned in P2.x/P3.0: a null result is
uninterpretable unless the metric has been shown capable of a non-null one, and the consumer
thread cost four training runs learning that the hard way.

TWO AXES, and the second is the interesting one.

`epsilon`   with probability epsilon, replace the chosen action with a uniformly random LEGAL
            one. General-purpose degradation, directly comparable to the explorer's control.

`crit_scale` multiplies the critical-drive thresholds (h_crit / s_crit, baseline 0.7). This is
            the ARBITRATION knob — the decision to abandon a hunt for the missing resource and
            detour to a known one. It is two-sided and both failure modes are real:

              crit_scale = 0    interrupt never fires. The agent explores single-mindedly for
                                the missing resource and dies of thirst beside remembered
                                water. Recorded effect when this behaviour was added:
                                hydration deaths 43 -> 10 (random explorer) and 13 -> 1
                                (smell_momentum). So this ablation has a KNOWN magnitude and
                                serves as the sweep's anchor, the way eps=1.0 -> random did
                                for the explorer.
              crit_scale >> 1   interrupt fires constantly. The agent camps on whichever drive
                                it already knows and never completes exploration.

The optimum is interior, which is what makes this a genuine arbitration problem rather than a
threshold to be pushed in one direction. It is also the module carrying the homeostatic content
of the project: deciding *when* to interleave two competing drives.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Action, Orchestrator, OrchestratorObs
from model_modules.oracle_modules.orchestrator_oracle_v1 import OracleOrchestrator


class NoisyOracleOrchestrator(Orchestrator):
    """
    Wraps the oracle rather than subclassing its `act`, so the inner policy's sticky-target
    state advances exactly as it would unperturbed. Freezing that state whenever noise fires
    would degrade two things at once and confound the sweep.
    """

    def __init__(self, rng=None, explorer=None, epsilon: float = 0.0,
                 crit_scale: float = 1.0, h_crit: float = 0.7, s_crit: float = 0.7,
                 **kwargs):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if crit_scale < 0:
            raise ValueError(f"crit_scale must be >= 0, got {crit_scale}")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = float(epsilon)
        self.crit_scale = float(crit_scale)

        self.inner = OracleOrchestrator(
            rng=self.rng, explorer=explorer,
            h_crit=h_crit * self.crit_scale,
            s_crit=s_crit * self.crit_scale,
            **kwargs,
        )
        self.explorer = explorer
        self.n_calls = 0
        self.perturbed = 0

    def reset(self) -> None:
        self.inner.reset()

    def act(self, obs: OrchestratorObs) -> Action:
        self.n_calls += 1
        base = self.inner.act(obs)   # stepped either way, so sticky target stays coherent

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            self.perturbed += 1
            mask = Orchestrator.legal_mask(obs)
            legal = np.flatnonzero(mask)
            if len(legal):
                return Action(int(legal[self.rng.integers(len(legal))]))
        return base


def make_orchestrator_noisy(rng=None, explorer=None, epsilon: float = 0.0,
                            crit_scale: float = 1.0, **kwargs) -> NoisyOracleOrchestrator:
    """Factory registered in sim_instance.MODULE_REGISTRY under 'noisy_oracle'."""
    return NoisyOracleOrchestrator(rng=rng, explorer=explorer, epsilon=epsilon,
                                   crit_scale=crit_scale, **kwargs)
