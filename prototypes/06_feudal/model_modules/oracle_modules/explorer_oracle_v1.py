"""
explorer_oracle_v1.py  —  coordinate-free oracle explorers for Proto 06.

These explorers are deliberately local.

They do NOT know coordinates.
They do NOT know water/food locations.
They do NOT keep a visited map.

RandomExplorer:
    old flailing baseline.

MomentumExplorer:
    anti-dithering / EZ-style exploration.
    Keeps moving in the same direction when legal.
    Avoids immediate reversal when alternatives exist.

SmellMomentumExplorer:
    chemotaxis-style local exploration.
    Uses only smell history:
        [current, previous, previous_previous]
    If smell improves after moving, keep going.
    If smell worsens, try reversing/turning.
    This is learnable from smell history + recent action history.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Explorer, ExplorerObs, HexMove


# Must match HexWorld LOCAL_DIRECTIONS order:
# 0: ( 1, -1)
# 1: ( 1,  0)
# 2: ( 0,  1)
# 3: (-1,  1)
# 4: (-1,  0)
# 5: ( 0, -1)
REVERSE_DIR = {
    0: 3,
    1: 4,
    2: 5,
    3: 0,
    4: 1,
    5: 2,
}


class RandomExplorer(Explorer):
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.n_calls = 0

    def reset(self) -> None:
        pass

    def act(self, obs: ExplorerObs) -> HexMove:
        self.n_calls += 1

        legal = [k for k in range(6) if bool(obs.legal_moves[k])]

        if not legal:
            return HexMove.D0

        return HexMove(int(self.rng.choice(legal)))


class MomentumExplorer(Explorer):
    def __init__(
        self,
        rng: np.random.Generator | None = None,
        persist_p: float = 0.85,
        avoid_reverse: bool = True,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.persist_p = float(persist_p)
        self.avoid_reverse = bool(avoid_reverse)

        self.last_dir: int | None = None

        self.n_calls = 0
        self.n_persist = 0
        self.n_avoid_reverse = 0
        self.n_random_turns = 0

    def reset(self) -> None:
        self.last_dir = None

    def act(self, obs: ExplorerObs) -> HexMove:
        self.n_calls += 1

        legal = [k for k in range(6) if bool(obs.legal_moves[k])]

        if not legal:
            self.last_dir = None
            return HexMove.D0

        # 1. Continue in same direction if possible.
        if (
            self.last_dir is not None
            and self.last_dir in legal
            and self.rng.random() < self.persist_p
        ):
            self.n_persist += 1
            return HexMove(self.last_dir)

        # 2. Otherwise avoid immediately undoing the last move, if possible.
        candidate_dirs = legal

        if self.avoid_reverse and self.last_dir is not None:
            rev = REVERSE_DIR[self.last_dir]
            non_reverse = [k for k in legal if k != rev]

            if non_reverse:
                self.n_avoid_reverse += 1
                candidate_dirs = non_reverse

        # 3. Pick a new legal direction.
        choice = int(self.rng.choice(candidate_dirs))
        self.last_dir = choice
        self.n_random_turns += 1

        return HexMove(choice)


class SmellMomentumExplorer(MomentumExplorer):
    def __init__(
        self,
        rng: np.random.Generator | None = None,
        persist_p: float = 0.70,
        avoid_reverse: bool = True,
        trend_eps: float = 0.01,
        follow_p: float = 0.95,
        reverse_on_drop_p: float = 0.75,
    ):
        super().__init__(
            rng=rng,
            persist_p=persist_p,
            avoid_reverse=avoid_reverse,
        )

        self.trend_eps = float(trend_eps)
        self.follow_p = float(follow_p)
        self.reverse_on_drop_p = float(reverse_on_drop_p)

        self.n_smell_follow = 0
        self.n_smell_reverse = 0

    def _active_smell_now_prev(self, obs: ExplorerObs) -> tuple[float, float]:
        """
        Select the smell signal for the currently missing resource.

        If only food is missing, use food smell.
        If only water is missing, use water smell.
        If both are missing, use the stronger available smell.
        """
        candidates: list[tuple[float, float]] = []

        if obs.need_water:
            candidates.append((float(obs.water_smell[0]), float(obs.water_smell[1])))

        if obs.need_food:
            candidates.append((float(obs.food_smell[0]), float(obs.food_smell[1])))

        # Defensive fallback if caller accidentally says nothing is needed.
        if not candidates:
            candidates = [
                (float(obs.water_smell[0]), float(obs.water_smell[1])),
                (float(obs.food_smell[0]), float(obs.food_smell[1])),
            ]

        # Follow whichever missing-resource smell is currently stronger.
        return max(candidates, key=lambda x: x[0])

    def act(self, obs: ExplorerObs) -> HexMove:
        self.n_calls += 1

        legal = [k for k in range(6) if bool(obs.legal_moves[k])]

        if not legal:
            self.last_dir = None
            return HexMove.D0

        smell_now, smell_prev = self._active_smell_now_prev(obs)
        smell_delta = smell_now - smell_prev

        # If the last movement improved the target smell, keep going.
        if (
            self.last_dir is not None
            and self.last_dir in legal
            and smell_delta > self.trend_eps
            and self.rng.random() < self.follow_p
        ):
            self.n_smell_follow += 1
            return HexMove(self.last_dir)

        # If the last movement made smell worse, try reversing.
        # This is the scalar-smell equivalent of chemotaxis.
        if (
            self.last_dir is not None
            and smell_delta < -self.trend_eps
        ):
            rev = REVERSE_DIR[self.last_dir]

            if rev in legal and self.rng.random() < self.reverse_on_drop_p:
                self.last_dir = rev
                self.n_smell_reverse += 1
                return HexMove(rev)

        # Otherwise fall back to anti-dither momentum.
        return super().act(obs)


def make_random(
    rng: np.random.Generator | None = None,
    **kwargs,
) -> RandomExplorer:
    return RandomExplorer(rng=rng)


def make_momentum(
    rng: np.random.Generator | None = None,
    persist_p: float = 0.85,
    avoid_reverse: bool = True,
    **kwargs,
) -> MomentumExplorer:
    return MomentumExplorer(
        rng=rng,
        persist_p=persist_p,
        avoid_reverse=avoid_reverse,
    )


def make_smell_momentum(
    rng: np.random.Generator | None = None,
    persist_p: float = 0.70,
    avoid_reverse: bool = True,
    trend_eps: float = 0.01,
    follow_p: float = 0.95,
    reverse_on_drop_p: float = 0.75,
    **kwargs,
) -> SmellMomentumExplorer:
    return SmellMomentumExplorer(
        rng=rng,
        persist_p=persist_p,
        avoid_reverse=avoid_reverse,
        trend_eps=trend_eps,
        follow_p=follow_p,
        reverse_on_drop_p=reverse_on_drop_p,
    )