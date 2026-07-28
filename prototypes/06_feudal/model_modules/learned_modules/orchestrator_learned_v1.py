"""
orchestrator_learned_v1.py  —  learned arbitration between two competing drives.

This is the module that carries the homeostatic content of the project: deciding *when* to
abandon a hunt for the missing resource and detour to a known one, and when to stop consuming
and move on. Every other module executes; this one chooses.

FOUR ABSTRACT ACTIONS, NOT NINE — and this is the load-bearing design decision.

`Action` has nine members, six of which are direct EXPLORE_k moves. A learned orchestrator
emitting those directly would be doing the EXPLORER's job as well as its own — and the explorer
is separately established as not learnable on its observation contract (P3.4). Training a
nine-way head would therefore confound arbitration with a task already known to fail, and a null
result would be unattributable.

So the head is four-way:

    0 GO_WATER   legal iff water_known
    1 GO_FOOD    legal iff food_known
    2 CONSUME    legal iff the tile holds something consumable
    3 EXPLORE    always legal — DELEGATED to the explorer module, exactly as the oracle does

The module learns *when to explore*, not *how*. That keeps the question clean: can RL learn the
arbitration policy, given competent execution beneath it?

DOES NOT IMPORT world_v1 — the drive dynamics are what it must infer from reward.
"""

from __future__ import annotations

import numpy as np
import torch

from model_modules.contract_v1 import (
    Action, Explorer, ExplorerObs, Orchestrator, OrchestratorObs,
)
from model_modules.learned_modules.recorder_v1 import Recorder

OBS_FIELDS = ["h", "s", "h_deficit", "s_deficit", "min_drive", "h_minus_s",
              "water_known", "food_known", "tile_water_lvl", "tile_food_lvl"]
N_INPUT = len(OBS_FIELDS)
N_ACT = 4

GO_WATER, GO_FOOD, CONSUME, EXPLORE = range(N_ACT)
DRIVE_SCALE = 3.0
IDEAL = 1.0


def encode_orchestrator_obs(obs: OrchestratorObs) -> np.ndarray:
    """
    OrchestratorObs -> net input.

    Deficits, the minimum drive, and the h−s difference are algebraically redundant but handed
    over explicitly: arbitration is almost entirely a function of "which drive is in more
    danger", and making a small MLP rediscover subtraction and min() before it can start
    learning the interesting part wastes capacity on arithmetic. All are computable from
    observable quantities — no privilege.
    """
    h, s = float(obs.h), float(obs.s)
    return np.array([
        h / DRIVE_SCALE,
        s / DRIVE_SCALE,
        (IDEAL - h) / DRIVE_SCALE,
        (IDEAL - s) / DRIVE_SCALE,
        min(h, s) / DRIVE_SCALE,
        (h - s) / DRIVE_SCALE,
        float(obs.water_known),
        float(obs.food_known),
        float(obs.tile_water_lvl > 0),
        float(obs.tile_food_lvl > 0),
    ], dtype=np.float32)


def abstract_mask(obs: OrchestratorObs) -> np.ndarray:
    """4-way legality, mirroring Orchestrator.legal_mask over the collapsed action set."""
    m = np.ones(N_ACT, dtype=bool)
    m[GO_WATER] = bool(obs.water_known)
    m[GO_FOOD] = bool(obs.food_known)
    m[CONSUME] = (obs.tile_water_lvl > 0) or (obs.tile_food_lvl > 0)
    return m


def default_arch(n_hidden: int = 64, kind: str = "noisy", sigma_0: float = 0.5) -> dict:
    arch = {"kind": kind, "n_input": N_INPUT, "n_hidden": n_hidden, "n_act": N_ACT}
    if kind == "noisy":
        arch["sigma_0"] = sigma_0
    return arch


CALLS = 0


def reset_calls() -> None:
    global CALLS
    CALLS = 0


class LearnedOrchestrator(Orchestrator, Recorder):
    """
    Built by build_modules as orch_make(rng=..., explorer=..., **oracle_params["orchestrator"]).
    """

    def __init__(self, rng=None, explorer: Explorer = None, weights: str = None,
                 net=None, explore_eps: float = 0.0,
                 hold: int = 1, hold_natural: bool = True, hold_consume: bool = False,
                 hold_explore: bool = True, **_ignored):
        """
        `hold` is the option-commitment horizon. hold=1 is the original behaviour —
        re-decide every tick — and is the default, so nothing recorded before this
        parameter existed changes.

        WHY IT IS HERE. With hold=1 the module is a set of options with termination
        beta=1 everywhere, and P4.2 measured the textbook consequence: median GO_FOOD run
        length 1.0 against the oracle's 10.0. Options collapsed to primitive actions. The
        oracle does not collapse because `OracleOrchestrator.self.target` is sticky — it
        is running the same options with beta=0 until arrival.

        `hold_natural` terminates an option on its task condition instead of waiting out
        the counter: GO_* on arrival at the tile it named, EXPLORE the moment a memory
        slot flips (the discovery terminal from P3.2). Every one of those conditions is a
        function of fields already in OrchestratorObs, so this adds no information — it
        changes when the module is ASKED, not what it can see. WHICH option to start is
        still entirely learned; only when to stop is structural, which is the standard
        options setup rather than a re-implementation of the oracle.

        `hold_consume` is off by default. CONSUME is not a navigational intention, and
        holding it would overfill: DRINK_AMOUNT=0.15 needs ~4 ticks to fill from ideal,
        so a 15-tick hold spends 11 ticks past target. Left as a flag so the cost can be
        measured rather than assumed.
        """
        if net is None and weights is None:
            raise ValueError(
                "LearnedOrchestrator needs weights='<tag>' or an explicit net= (trainer only)."
            )
        self.rng = rng if rng is not None else np.random.default_rng()
        self.explorer = explorer
        self.explore_eps = float(explore_eps)
        self.tag = weights
        self.n_calls = 0
        self.hold = int(hold)
        self.hold_natural = bool(hold_natural)
        self.hold_consume = bool(hold_consume)
        # hold_explore=False isolates GOAL commitment from uninterrupted exploring. Held
        # EXPLORE is temporally-extended exploration, which 05-H2 already showed raises
        # crossing supply on its own — so without this control the two are confounded.
        self.hold_explore = bool(hold_explore)
        self._held: int | None = None
        self._ticks_left = 0
        self._known_at_start = (False, False)
        self.n_redecisions = 0
        self._init_recorder()

        if net is not None:
            self.net = net
        else:
            from model_modules.checkpoint_v1 import load_net

            self.net, payload = load_net("orchestrator", weights)
            if payload["obs_fields"] != OBS_FIELDS:
                raise ValueError(
                    f"orchestrator '{weights}' obs_fields mismatch. Retrain."
                )

    def reset(self) -> None:
        self._held = None
        self._ticks_left = 0
        if hasattr(self.explorer, "reset"):
            self.explorer.reset()

    def _terminated(self, k: int, obs: OrchestratorObs) -> bool:
        """Task-condition termination, all from fields already in the observation."""
        if k == GO_WATER:
            return obs.tile_water_lvl > 0
        if k == GO_FOOD:
            return obs.tile_food_lvl > 0
        if k == EXPLORE:
            # discovery terminal: a memory slot flipped since this option began
            return (bool(obs.water_known), bool(obs.food_known)) != self._known_at_start
        return False

    def _continue(self, obs: OrchestratorObs, legal: np.ndarray) -> int | None:
        """The still-running option, or None if it has terminated and we must re-decide."""
        if self.hold <= 1 or self._held is None:
            return None
        k = self._held
        if not legal[k]:
            return None                                  # became illegal
        if k == CONSUME and not self.hold_consume:
            return None                                  # never held; see __init__
        if k == EXPLORE and not self.hold_explore:
            return None                                  # control arm; see __init__
        if self._ticks_left <= 0:
            return None                                  # horizon spent
        if self.hold_natural and self._terminated(k, obs):
            return None
        self._ticks_left -= 1
        return k

    def act(self, obs: OrchestratorObs) -> Action:
        global CALLS
        CALLS += 1
        self.n_calls += 1

        x = encode_orchestrator_obs(obs)
        legal = abstract_mask(obs)

        k = self._continue(obs, legal)
        if k is None:
            self.n_redecisions += 1
            if self.explore_eps > 0.0 and self.rng.random() < self.explore_eps:
                k = int(self.rng.choice(np.flatnonzero(legal)))
            else:
                with torch.no_grad():
                    q = self.net(torch.from_numpy(x))
                q = q.masked_fill(~torch.from_numpy(legal), -1e9)
                k = int(torch.argmax(q).item())
            self._held = k
            self._ticks_left = self.hold - 1
            self._known_at_start = (bool(obs.water_known), bool(obs.food_known))

        # recorded every tick, held or not, so the training rig's record->tick join keeps
        # one entry per call; the value recorded is the action actually emitted
        self.record(x, k)

        if k == GO_WATER:
            return Action.GO_WATER
        if k == GO_FOOD:
            return Action.GO_FOOD
        if k == CONSUME:
            return Action.CONSUME
        return self._delegate_explore(obs)

    def _delegate_explore(self, obs: OrchestratorObs) -> Action:
        """Hand off to the explorer module — identical to the oracle's delegation path."""
        mask = Orchestrator.legal_mask(obs)
        legal_moves = np.array(
            [bool(mask[Action.EXPLORE_0 + k]) for k in range(6)], dtype=bool
        )
        if self.explorer is None:
            choices = np.flatnonzero(legal_moves)
            k = int(self.rng.choice(choices)) if len(choices) else 0
            return Action(int(Action.EXPLORE_0) + k)

        move = self.explorer.act(ExplorerObs(
            legal_moves=legal_moves,
            water_smell=obs.water_smell,
            food_smell=obs.food_smell,
            need_water=not obs.water_known,
            need_food=not obs.food_known,
        ))
        return Action(int(Action.EXPLORE_0) + int(move))


def make_orchestrator_learned(rng=None, explorer=None, weights: str = None, net=None,
                              explore_eps: float = 0.0, hold: int = 1,
                              hold_natural: bool = True, hold_consume: bool = False,
                              hold_explore: bool = True, **kw) -> LearnedOrchestrator:
    return LearnedOrchestrator(rng=rng, explorer=explorer, weights=weights, net=net,
                               explore_eps=explore_eps, hold=hold,
                               hold_natural=hold_natural, hold_consume=hold_consume,
                               hold_explore=hold_explore, **kw)
