"""
pathfinder_rig_v1.py  —  standalone goal-reaching rig. No drives, no decay, no resources.

PathfinderObs carries `to_goal` and nothing else, so the pathfinder's whole world is hex
geometry: where am I going, did I get there. This rig is that and no more.

WHY GEOMETRY IS IMPORTED, NOT REIMPLEMENTED. It pulls `_make_coords` and
`LOCAL_DIRECTIONS` straight from hex_world_cached rather than writing its own hex maths.
The oracle pathfinder's docstring flags the live risk here: if its AXIAL_DIRS table drifts
from the env's neighbour ordering, HexMove.Dk means two different physical steps in two
places and the agent walks away from goals. Training against the same table the sim moves
on removes that failure mode from the rig by construction, leaving the oracle's own table
as the only place it can still occur — which is exactly what the diff test checks.

Importing hex_world_cached does NOT build a world: module import is just constants and
class definitions, and `_make_coords` is a pure static method. No Bridson, no resource
placement, no scaffold cost.

REWARD IS SPARSE, DELIBERATELY. Potential-based shaping on |to_goal| would be legal —
to_goal is in the observation, so distance-shaping leaks nothing the module cannot already
see, and the standing "proximity-as-reward is cheating" rule is about the whole agent
being rewarded for approaching resources it cannot sense. But shaping here hands over
"descend the gradient", which is the entire policy, and turns the result vacuous. Sparse
+1 on arrival with a small step cost keeps it an honest result, and stays cheap because
the horizon is short and the bearing is observed. HER does the rest.
"""

from __future__ import annotations

import numpy as np

import hex_world_cached as hex_world
from model_modules.learned_modules.pathfinder_learned_v1 import (
    encode_to_goal,
    hex_len,
    N_ACT,
)

# Same table, same order, same physical meaning as the sim.
LOCAL_DIRECTIONS = hex_world.LOCAL_DIRECTIONS


class PathfinderRig:
    """
    One goal-reaching episode at a time.

    reset()  -> obs
    step(a)  -> obs, reward, done, info

    `arrived` in info is the only success signal; there is no partial credit.
    """

    def __init__(
        self,
        radius: int = 20,
        rng: np.random.Generator | None = None,
        step_cost: float = -0.01,
        arrive_reward: float = 1.0,
        step_cap: int | None = None,
        distance_sampling: str = "uniform_dist",
        mask_illegal: bool = False,
    ):
        self.radius = int(radius)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.step_cost = float(step_cost)
        self.arrive_reward = float(arrive_reward)
        # max hex distance across a radius-R disc is 2R; 3R gives slack without
        # letting a broken policy wander forever
        self.step_cap = int(step_cap) if step_cap is not None else 3 * self.radius
        self.distance_sampling = distance_sampling
        self.mask_illegal = bool(mask_illegal)

        self.coords = hex_world.HexWorld._make_coords(self.radius)
        self.coord_set = set(self.coords)

        # coords bucketed by distance from origin, for flat-in-distance goal sampling
        self._by_dist: dict[int, list] = {}
        for c in self.coords:
            self._by_dist.setdefault(hex_len(c[0], c[1]), []).append(c)
        self._dists = sorted(self._by_dist)

        self.pos = None
        self.goal = None
        self.t = 0

    # --- geometry ---------------------------------------------------------------

    def board_mask(self, pos=None) -> np.ndarray:
        """6-bool: does move k land on the board? Diagnostic — see legal_mask."""
        q, r = self.pos if pos is None else pos
        m = np.zeros(N_ACT, dtype=bool)
        for k, (dq, dr) in LOCAL_DIRECTIONS.items():
            m[k] = (q + dq, r + dr) in self.coord_set
        return m

    def legal_mask(self, pos=None) -> np.ndarray:
        """
        All six moves are legal by default, and that is deliberate.

        `compose_to_queued_action` does NOT mask the pathfinder — it takes whatever
        HexMove comes back and puts it straight into a_que. PathfinderObs carries only
        `to_goal`, so at inference the module has no idea where the rim is. Masking during
        training would therefore train a different regime from the one it runs in, and
        leave the net with never-visited Q-values on exactly the actions an unmasked
        argmax can still pick.

        Off-board moves are not impossible, they are wasteful: the agent attempts them and
        stands still, paying step_cost. Modelling them that way makes training and
        inference the same problem. In practice the constraint barely binds — greedy
        descent toward an interior goal points inward from the rim anyway — but matching
        it costs nothing and removes a silent divergence.

        Set mask_illegal=True to recover the masked regime for comparison.
        """
        if self.mask_illegal:
            return self.board_mask(pos)
        return np.ones(N_ACT, dtype=bool)

    def to_goal(self, pos=None, goal=None):
        p = self.pos if pos is None else pos
        g = self.goal if goal is None else goal
        return (g[0] - p[0], g[1] - p[1])

    def obs(self, pos=None, goal=None) -> np.ndarray:
        return encode_to_goal(self.to_goal(pos, goal), scale=self.radius)

    # --- episode ----------------------------------------------------------------

    def _sample_goal(self, pos):
        """
        Flat in DISTANCE by default, not flat in cell.

        Uniform-over-cells would put most mass on large displacements (a hex ring at
        distance d has ~6d cells), and the sim needs precision most at SMALL
        displacements — the last few steps of an approach are where a sloppy pathfinder
        loses path_efficiency. Flattening over distance buys coverage where it matters.
        """
        if self.distance_sampling == "uniform_cell":
            while True:
                g = self.coords[self.rng.integers(len(self.coords))]
                if g != pos:
                    return g

        for _ in range(64):
            d = self._dists[self.rng.integers(1, len(self._dists))]
            ring = self._by_dist[d]
            g = ring[self.rng.integers(len(ring))]
            if g != pos:
                return g
        # degenerate fallback (radius 0/1); keeps reset total
        return self.coords[self.rng.integers(len(self.coords))]

    def reset(self, pos=None, goal=None) -> np.ndarray:
        self.pos = tuple(pos) if pos is not None else self.coords[self.rng.integers(len(self.coords))]
        self.goal = tuple(goal) if goal is not None else self._sample_goal(self.pos)
        self.t = 0
        return self.obs()

    def step(self, move_id: int):
        dq, dr = LOCAL_DIRECTIONS[int(move_id)]
        nxt = (self.pos[0] + dq, self.pos[1] + dr)
        # off-board moves are masked at selection; if one arrives anyway, stand still
        # rather than teleport — same as the env clipping at the rim
        if nxt in self.coord_set:
            self.pos = nxt

        self.t += 1
        arrived = self.pos == self.goal
        timeout = self.t >= self.step_cap

        reward = self.step_cost + (self.arrive_reward if arrived else 0.0)
        done = arrived or timeout

        return self.obs(), reward, done, {"arrived": arrived, "timeout": timeout, "t": self.t}

    # --- HER --------------------------------------------------------------------

    def relabel(self, positions: list, achieved_goal) -> tuple:
        """
        Hindsight relabel: recompute a transition against a goal that WAS reached.

        `positions` is the visited sequence [p_0 .. p_T]. Returns arrays the trainer can
        splice into the buffer. Every failed episode becomes a successful one for some
        goal, which is what makes a sparse arrival reward tractable here.
        """
        g = tuple(achieved_goal)
        obs = [encode_to_goal((g[0] - p[0], g[1] - p[1]), scale=self.radius) for p in positions]
        arrived = [p == g for p in positions]
        return obs, arrived


def oracle_step_count(rig: PathfinderRig, pos, goal) -> int:
    """
    Steps a perfect greedy descent needs. On an unobstructed hex board this is just the
    hex distance — used as the denominator for path efficiency in the diff test, so the
    learned module is scored against the optimum rather than against 'did it arrive'.
    """
    return hex_len(goal[0] - pos[0], goal[1] - pos[1])
