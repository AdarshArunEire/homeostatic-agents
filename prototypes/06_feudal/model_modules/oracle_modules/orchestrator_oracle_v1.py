"""
orchestrator_oracle_v1.py  —  v0 oracle orchestrator for Proto 06.

Tiny on purpose.

This module does NOT parse vision.
This module does NOT use smell.
This module does NOT know coordinates.
This module does NOT pathfind.
This module does NOT update memory.
This module does NOT implement exploration internals.

Memory is upstream state. The composer/env observation layer fills resource memory from
touch/vision, then this orchestrator only sees the booleans:
    water_known, food_known

Policy:
    - if on useful tile and not filled: CONSUME
    - if both resources are known: cycle water <-> food through GO_* actions
    - otherwise: delegate to explorer module

That makes the oracle structurally replaceable by learned modules later.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import (
    Orchestrator,
    OrchestratorObs,
    Explorer,
    ExplorerObs,
    Action,
)

class RandomExplorer:
    """
    Fallback explorer.

    No coords.
    No map memory.
    Just random legal EXPLORE_k.

    This keeps old v0 behaviour if no explorer module is passed.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self) -> None:
        pass

    def act(self, mask: np.ndarray) -> Action:
        choices = [
            Action.EXPLORE_0,
            Action.EXPLORE_1,
            Action.EXPLORE_2,
            Action.EXPLORE_3,
            Action.EXPLORE_4,
            Action.EXPLORE_5,
        ]
        choices = [a for a in choices if mask[a]]

        if not choices:
            return Action.EXPLORE_0

        return Action(self.rng.choice(choices))


class OracleOrchestrator(Orchestrator):
    def __init__(
        self,
        h_fill: float = 1.25,
        s_fill: float = 1.25,
        h_crit: float = 0.7,
        s_crit: float = 0.7,
        rng: np.random.Generator | None = None,
        explorer: Explorer | None = None,
    ):
        self.h_fill = h_fill
        self.s_fill = s_fill
        # Critical thresholds: below these, detour to a KNOWN resource even while the
        # OTHER resource is still unknown (i.e. mid-exploration). Must sit well below
        # *_fill: too high and the agent camps on the known drive and never explores;
        # too low and it starves before the detour triggers. Policy knob, not physics.
        self.h_crit = h_crit
        self.s_crit = s_crit
        self.rng = rng if rng is not None else np.random.default_rng()

        self.target: str | None = None
        self.explorer = explorer

    def reset(self) -> None:
        """
        Called on respawn/map reset.

        Important for fair curriculum:
        each new life starts with blank high-level intent and blank explorer state.
        """
        self.target = None

        if hasattr(self.explorer, "reset"):
            self.explorer.reset()

    def act(self, obs: OrchestratorObs) -> Action:
        mask = self.legal_mask(obs)

        on_water = obs.tile_water_lvl > 0
        on_food = obs.tile_food_lvl > 0

        # 1) If standing on a useful resource, consume until that drive is good.
        if on_water and obs.h < self.h_fill and mask[Action.CONSUME]:
            self.target = "water"
            return Action.CONSUME

        if on_food and obs.s < self.s_fill and mask[Action.CONSUME]:
            self.target = "food"
            return Action.CONSUME

        # 1b) Critical-drive interrupt: if a drive is at risk and we already KNOW where
        # to satisfy it, detour there even if the other resource is still unknown. Without
        # this the agent explores single-mindedly for the missing resource and starves of
        # the known one (dies of thirst next to remembered water). Thresholds sit well
        # below *_fill so this only fires in genuine danger, preserving exploration.
        if obs.h < self.h_crit and obs.water_known and mask[Action.GO_WATER]:
            self.target = "water"
            return Action.GO_WATER

        if obs.s < self.s_crit and obs.food_known and mask[Action.GO_FOOD]:
            self.target = "food"
            return Action.GO_FOOD

        # 2) If memory is incomplete, delegate exploration.
        if not (obs.water_known and obs.food_known):
            return self._explore(mask, obs)

        # 3) Memory complete: cycle mode.
        # If just finished water, go food.
        if on_water and mask[Action.GO_FOOD]:
            self.target = "food"
            return Action.GO_FOOD

        # If just finished food, go water.
        if on_food and mask[Action.GO_WATER]:
            self.target = "water"
            return Action.GO_WATER

        # In transit / neutral tile: keep current target if possible.
        if self.target == "water" and mask[Action.GO_WATER]:
            return Action.GO_WATER

        if self.target == "food" and mask[Action.GO_FOOD]:
            return Action.GO_FOOD

        # No sticky target yet: pick the lower drive.
        if obs.h <= obs.s and mask[Action.GO_WATER]:
            self.target = "water"
            return Action.GO_WATER

        if mask[Action.GO_FOOD]:
            self.target = "food"
            return Action.GO_FOOD

        # Should only happen if mask/memory contract is broken.
        return self._explore(mask, obs)

    def _explore(self, mask: np.ndarray, obs: OrchestratorObs) -> Action:
        if self.explorer is None:
            choices = [
                Action.EXPLORE_0,
                Action.EXPLORE_1,
                Action.EXPLORE_2,
                Action.EXPLORE_3,
                Action.EXPLORE_4,
                Action.EXPLORE_5,
            ]
            choices = [a for a in choices if mask[a]]

            if not choices:
                return Action.EXPLORE_0

            return Action(self.rng.choice(choices))

        legal_moves = np.array(
            [
                bool(mask[Action.EXPLORE_0]),
                bool(mask[Action.EXPLORE_1]),
                bool(mask[Action.EXPLORE_2]),
                bool(mask[Action.EXPLORE_3]),
                bool(mask[Action.EXPLORE_4]),
                bool(mask[Action.EXPLORE_5]),
            ],
            dtype=bool,
        )

        move = self.explorer.act(
            ExplorerObs(
                legal_moves=legal_moves,
                water_smell=obs.water_smell,
                food_smell=obs.food_smell,
                need_water=not obs.water_known,
                need_food=not obs.food_known,
            )
        )

        return Action.EXPLORE_0 + int(move)