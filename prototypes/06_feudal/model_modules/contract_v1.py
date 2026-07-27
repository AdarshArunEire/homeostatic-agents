"""
contract.py  —  module interface for the modular homeostatic agent (Proto 06).

The spine of the whole prototype. Every module has an *oracle* impl and a *learned*
impl; they MUST share the signatures below, or nothing composes. Freeze this first,
then smoke-test that oracle and learned produce different-but-same-shaped outputs on
one state before wiring any training (same dispatch check as the two-arm sweeps).

Four modules:  orchestrator, pathfinder, eat, drink.
Memory is STATE, not a trained module (single-slot per resource for v1).

FLIP POINTS (the two defaults you can override):
  [A] Exploration = masked direct moves (EXPLORE_0..5 bypass the pathfinder).
      The pathfinder therefore never sees a null/phantom goal. To instead route
      exploration through the pathfinder as a phantom coordinate, delete the EXPLORE_*
      members, add an EXPLORE action, and have the composer synthesise a goal.
  [B] The orchestrator emits ONE consume primitive (CONSUME); the composer procs the
      consumer matching the tile (drink on water, eat on food). Each consumer returns a
      *fill fraction*, not a bool — graded because the overfill penalty makes "stop at
      ideal" the real skill. To split behaviour, keep eat and drink as separate Consumer
      impls (they already share ConsumerObs); to merge, point both slots at one impl.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


# --- shared types ---------------------------------------------------------------

Coord = tuple[int, int]          # axial hex (q, r)
Displacement = tuple[int, int]   # goal - pos, in axial hex; the pathfinder's only signal


class HexMove(IntEnum):          # the 6 hex directions; every module's motor output
    D0 = 0; D1 = 1; D2 = 2; D3 = 3; D4 = 4; D5 = 5


class Action(IntEnum):           # orchestrator action space — 9 discrete
    GO_WATER   = 0               # resolve water coord from memory -> pathfinder
    GO_FOOD    = 1               # resolve food coord from memory  -> pathfinder
    EXPLORE_0  = 2               # [flip A] direct move, bypasses pathfinder
    EXPLORE_1  = 3
    EXPLORE_2  = 4
    EXPLORE_3  = 5
    EXPLORE_4  = 6
    EXPLORE_5  = 7
    CONSUME    = 8               # [flip B] procs the tile's consumer (drink/eat)


# --- observations ---------------------------------------------------------------

def zero3() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)

@dataclass(frozen=True)
class OrchestratorObs:
    h: float
    s: float

    # Backward-compatible old smell slot.
    # In Proto 06 this can be food smell history or left unused by orchestrator.
    smell: np.ndarray

    vision: np.ndarray

    water_known: bool
    food_known: bool

    tile_water_lvl: float
    tile_food_lvl: float

    # New Proto 06 smell histories, newest first:
    # [current_reading, previous_reading, previous_previous_reading]
    water_smell: np.ndarray = field(default_factory=zero3)
    food_smell: np.ndarray = field(default_factory=zero3)


@dataclass(frozen=True)
class PathfinderObs:
    to_goal: Displacement        # goal is ALWAYS a real remembered coord (never null)
    # senses deliberately omitted: the goal bearing dominates, keep it a clean ablation.
    # add `smell`/`vision` here only if the learned pathfinder underperforms the oracle.


@dataclass(frozen=True)
class ConsumerObs:               # shared by eat and drink
    brightness: float
    h: float
    s: float
    tile_water_lvl: float
    tile_food_lvl: float

@dataclass(frozen=True)
class ExplorerObs:
    """
    Observation for exploration-only movement.

    legal_moves[k] says whether EXPLORE_k / HexMove.Dk is legal.

    smell histories are newest first:
        [current, previous, previous_previous]

    The need_* flags tell the explorer which resource memory is still missing.
    """
    legal_moves: np.ndarray  # shape (6,), bool

    water_smell: np.ndarray = field(default_factory=zero3)
    food_smell: np.ndarray = field(default_factory=zero3)

    need_water: bool = True
    need_food: bool = True

# --- module interfaces (oracle and learned impls both subclass these) -----------

class Orchestrator(ABC):
    @abstractmethod
    def act(self, obs: OrchestratorObs) -> Action:
        """Pick an action. Composer masks GO_* (empty slot) and CONSUME (nothing on
        tile), so a valid impl may assume it only ever emits a legal action.
        Oracle explore: vision -> smell-gradient -> rand(0..5); flat smell in the
        dead band falls through to random, same as the learned agent will face."""

    @staticmethod
    def legal_mask(obs: OrchestratorObs) -> np.ndarray:
        """9-length bool mask. EXPLORE_* always legal; GO_* legal iff slot filled;
        CONSUME legal iff the tile holds something consumable."""
        m = np.ones(len(Action), dtype=bool)
        m[Action.GO_WATER] = obs.water_known
        m[Action.GO_FOOD]  = obs.food_known
        m[Action.CONSUME]  = (obs.tile_water_lvl > 0) or (obs.tile_food_lvl > 0)
        return m


class Pathfinder(ABC):
    @abstractmethod
    def move(self, obs: PathfinderObs) -> HexMove:
        """Step toward the goal. Oracle = greedy hex step down |to_goal|.
        Learned = goal-conditioned net, trained with HER (relabel goal = tile actually reached)."""


class Consumer(ABC):
    @abstractmethod
    def consume(self, obs: ConsumerObs) -> float:
        """Return fill fraction in [0, 1]. Procced by the orchestrator's CONSUME on the
        matching tile. Learned impl argmaxes over discrete bins (e.g. 0/.25/.5/.75/1, or
        ~20) internally and returns the chosen fraction; oracle fills toward ideal using
        the decay scaling. Graded because slamming 1.0 near ideal eats the overfill penalty."""

class Explorer(ABC):
    @abstractmethod
    def act(self, obs: ExplorerObs) -> HexMove:
        """
        Choose one local movement direction for exploration.

        Returns:
            HexMove.D0..D5
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional per-life reset hook.
        """
        pass

# --- module spec: how the composer is told which impl fills each slot ------------

@dataclass(frozen=True)
class ModuleSpec:
    orchestrator: str
    pathfinder: str
    eat: str
    drink: str
    explorer: str = "random"
    # e.g. ModuleSpec("oracle", "learned", "oracle", "oracle") trains the pathfinder;
    #      ModuleSpec("learned","learned","learned","learned") is the all-learned rung.