"""
pathfinder_oracle.py  —  the reference (oracle) pathfinder for Proto 06.

Unambiguous by design: given a displacement to a real goal, take the one hex step that
reduces the hex-distance to it the most. This is the greedy descent the learned
pathfinder (HER) has to match, and the behaviour the all-oracle composer smoke test
asserts (perfect execution MUST score near-perfect route-success, or the composer is
broken). No senses, no memory — just the goal bearing, exactly the PathfinderObs contract.

INTEGRATION POINT TO WATCH: AXIAL_DIRS below must match hex_world's neighbour ordering
so HexMove.Dk means the same physical step everywhere. If the smoke test shows the
oracle walking away from goals, this table is misaligned with the env — fix it here.
"""

from __future__ import annotations
from model_modules.contract_v1 import Pathfinder, PathfinderObs, HexMove

# axial (q, r) unit steps, indexed to match HexMove.D0..D5  <-- confirm against hex_world
AXIAL_DIRS: tuple[tuple[int, int], ...] = (
    ( 1, -1),   # D0
    ( 1,  0),   # D1
    ( 0,  1),   # D2
    (-1,  1),   # D3
    (-1,  0),   # D4
    ( 0, -1),   # D5
)


def hex_len(q: int, r: int) -> int:
    """Hex distance from the origin in axial coords = cube norm."""
    return (abs(q) + abs(r) + abs(q + r)) // 2


class OraclePathfinder(Pathfinder):
    def move(self, obs: PathfinderObs) -> HexMove:
        gq, gr = obs.to_goal                       # goal - pos
        # stepping by direction d changes the displacement to (to_goal - d)
        best_k, best_d = 0, None
        for k, (dq, dr) in enumerate(AXIAL_DIRS):
            d = hex_len(gq - dq, gr - dr)           # distance remaining after this step
            if best_d is None or d < best_d:
                best_d, best_k = d, k               # first-wins tie-break = deterministic
        return HexMove(best_k)