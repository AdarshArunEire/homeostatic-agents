"""
recorder_v1.py  —  what a learned module writes down about its own calls.

Lives in model_modules, not training, so the dependency runs one way: a module never
imports the trainer.

A module records only what it SAW and what it DID. It is never handed a reward, a tick
index, or any sim state. That is a deliberate constraint rather than a convenience — it
makes it structurally impossible for a module to consume information its Obs contract does
not grant, which is the same privilege boundary world_v1 draws for the oracles. Rewards are
joined on afterwards, in the trainer, from the timeseries the sim already returns.
"""

from __future__ import annotations

import numpy as np


class Recorder:
    """
    Mixin. Call `_init_recorder()` from the module's __init__, then `record(obs_vec, action)`
    inside act()/move()/consume().

    Recording is OFF by default: modules are constructed inside sweep workers where the
    buffer would grow unbounded across a 7000-tick run for no reason.
    """

    def _init_recorder(self) -> None:
        self._records: list[tuple] = []
        self._recording = False

    def start_recording(self) -> None:
        self._records = []
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    @property
    def recording(self) -> bool:
        return getattr(self, "_recording", False)

    @property
    def records(self) -> list[tuple]:
        return getattr(self, "_records", [])

    def record(self, obs_vec, action_idx) -> None:
        if getattr(self, "_recording", False):
            self._records.append((np.asarray(obs_vec, dtype=np.float32), int(action_idx)))
