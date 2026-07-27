import numpy as np

import hex_world_cached as hex_world


class MapCurriculum:
    """
    Proto 05 curriculum: difficulty = commute length, maps resampled.

    The schedule controls exactly one thing — the band (food<->water ring
    distance), i.e. how long the commute is. The band sets map *generation*
    and never enters the agent's observation, so it reveals how far, not
    where: non-leaking by construction.

    Two faults from Proto 04 are designed against directly:

      * route-memorisation (H1). Every build() makes a *fresh* map from a
        disjoint seed, and train/eval seed streams are kept disjoint via a
        shared `seen` set, so no single route survives and eval maps are
        genuinely held out.

      * widening-handoff forgetting (3b's curriculum, 6/10 -> 1/10). There
        is no discrete phase step-up. The band floor is sampled each map
        from [c_min, c_now], where c_now ramps over training. Easy maps keep
        appearing at every stage, so a replay archive that is *not* cleared
        never loses the short-commute regime. (Clearing replay at handoff is
        the thing this is built to avoid; see sim_instance.)

    Radius is held fixed (one-variable discipline): the hex grid, and hence
    coord_to_idx / novelty array size, is invariant across resamples — only
    resource placement changes.
    """

    def __init__(
        self,
        seed,
        base_env_kwargs,
        EB,
        c_min=2,
        c_max=9,
        band_width=2,
        ramp_frac=0.6,
        sampling="uniform",          # "uniform" | "frontier"
        eval_band=None,              # None -> hardest, (c_max, c_max+band_width)
        eval_seed_offset=1_000_000,
        build_max_tries=50,
    ):
        # strip band/seed — curriculum owns the band, sim owns the seed
        self.base = {
            k: v for k, v in base_env_kwargs.items() if k not in ("band", "seed")
        }
        self.EB = EB
        self.c_min = int(c_min)
        self.c_max = int(c_max)
        self.band_width = int(band_width)
        self.ramp_frac = float(ramp_frac)
        self.sampling = sampling
        self.build_max_tries = int(build_max_tries)

        budget = hex_world.WATER_R_MIN
        hardest_sum = 2 * self.c_max + self.band_width
        if hardest_sum > budget:
            raise ValueError(
                f"hardest band sum {hardest_sum} (= 2*c_max+width) exceeds "
                f"WATER_R_MIN={budget}; lower c_max/band_width or raise WATER_R_MIN."
            )
        if self.c_min < 1:
            raise ValueError(f"c_min must be >= 1, got {self.c_min}")

        self.eval_band = tuple(eval_band) if eval_band is not None else (
            self.c_max, self.c_max + self.band_width
        )

        # independent, reproducible streams; shared seen-set keeps them disjoint
        self.train_rng = np.random.default_rng(seed)
        self.eval_rng = np.random.default_rng(seed + eval_seed_offset)
        self.seen = set()

        self.n_train = 0
        self.n_eval = 0

    # ---- band schedule ---------------------------------------------------

    def floor_ceiling(self, t):
        """Highest commute floor unlocked at tick t (a real number)."""
        ramp = min(1.0, t / max(1.0, self.ramp_frac * self.EB))
        return self.c_min + ramp * (self.c_max - self.c_min)

    def sample_band(self, t):
        """Band for a training map built at tick t. Floor drawn from the
        widening window [c_min, c_now] — easy maps never stop appearing."""
        c_now = self.floor_ceiling(t)
        lo, hi = self.c_min, int(round(c_now))

        if hi <= lo:
            floor = lo
        elif self.sampling == "frontier":
            # triangular peaked at the current frontier, still covers easy
            floor = int(round(self.train_rng.triangular(lo, hi, hi)))
        else:
            floor = int(self.train_rng.integers(lo, hi + 1))

        floor = max(self.c_min, min(self.c_max, floor))
        return (floor, floor + self.band_width)

    # ---- map construction ------------------------------------------------

    def _fresh_seed(self, rng):
        # train/eval disjoint because both draw against the same seen-set
        while True:
            s = int(rng.integers(0, 2**31 - 1))
            if s not in self.seen:
                self.seen.add(s)
                return s

    def build(self, t, eval_mode=False):
        """Return (env, band, map_seed). Retries past degenerate worlds —
        with thousands of resamples some seeds will hit HexWorld's
        crash-on-violation guards, and one bad seed must not kill the run."""
        rng = self.eval_rng if eval_mode else self.train_rng
        band = self.eval_band if eval_mode else self.sample_band(t)

        last_err = None
        for _ in range(self.build_max_tries):
            map_seed = self._fresh_seed(rng)
            try:
                env = hex_world.HexWorld(seed=map_seed, band=band, **self.base)
            except ValueError as e:        # dead hub / no food / band violation
                last_err = e
                continue
            if eval_mode:
                self.n_eval += 1
            else:
                self.n_train += 1
            return env, band, map_seed

        raise RuntimeError(
            f"no valid map for band={band} in {self.build_max_tries} tries: {last_err}"
        )


if __name__ == "__main__":
    # smoke test: eyeball the widening window without running the sim
    EB = 480_000
    cur = MapCurriculum(seed=0, base_env_kwargs={"radius": 20}, EB=EB)
    print(f"eval band (held-out): {cur.eval_band}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = int(frac * EB)
        floors = [cur.sample_band(t)[0] for _ in range(2000)]
        vals, counts = np.unique(floors, return_counts=True)
        dist = {int(v): round(c / len(floors), 2) for v, c in zip(vals, counts)}
        print(f"t/EB={frac:>4}: c_now={cur.floor_ceiling(t):4.1f}  floor dist={dist}")
    # one real build per mode, prove train/eval seeds are disjoint
    e_tr, b_tr, s_tr = cur.build(t=EB // 2, eval_mode=False)
    e_ev, b_ev, s_ev = cur.build(t=EB, eval_mode=True)
    print(f"train build: band={b_tr} waters={len(e_tr.water_coords)} foods={len(e_tr.food_coords)}")
    print(f"eval  build: band={b_ev} waters={len(e_ev.water_coords)} foods={len(e_ev.food_coords)}")
    print(f"seed overlap (must be empty): {set([s_tr]) & set([s_ev])}")
