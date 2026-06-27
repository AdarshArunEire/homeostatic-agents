import numpy as np
from collections import defaultdict

WAIT = 0

WATER_R_MIN = 20         # water-water spacing (Bridson). Scarcity knob.

DRINK_ACTIONS = slice(1, 4)
EAT_ACTIONS = slice(4, 7)
MOVE_OFFSET = 7

LOCAL_DIRECTIONS = {
    0: (1, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 1),
    4: (-1, 0),
    5: (0, -1),
}

MOVE_ACTIONS = {
    MOVE_OFFSET + move_id: direction
    for move_id, direction in LOCAL_DIRECTIONS.items()
}

ACTION_EFFECTS = [
    (0.0, 0.0, None),
    (1.0, 0.0, None),
    (0.5, 0.0, None),
    (0.25, 0.0, None),
    (0.0, 1.0, None),
    (0.0, 0.5, None),
    (0.0, 0.25, None),
    (0.0, 0.0, 0),
    (0.0, 0.0, 1),
    (0.0, 0.0, 2),
    (0.0, 0.0, 3),
    (0.0, 0.0, 4),
    (0.0, 0.0, 5),
]

N_ACT = len(ACTION_EFFECTS)


class HexWorld:
    def __init__(
        self,
        seed,
        radius=1,
        band=(9, 11),
        w_min_sep=None,        # unused now; water-water spacing is WATER_R_MIN
        start_coord=(0, 0)
    ):
        
        if WATER_R_MIN < band[0] + band[1]:
            raise ValueError(
                f"WATER_R_MIN={WATER_R_MIN} < band_min+band_max={band[0]+band[1]}: "
                f"raise WATER_R_MIN."
            )

        self.radius = radius
        self.start_coord = tuple(start_coord)
        self.coord = tuple(start_coord)
        water_level = 1.0
        food_level = 1.0

        rng = np.random.default_rng(seed)

        self.coords = self._make_coords(radius)

        self.nodes = {
            coord: {"water": 0.0, "food": 0.0}
            for coord in self.coords
        }

        self.coord_to_idx = {tuple(c): i for i, c in enumerate(self.coords)}
        self.idx_to_coord = {i: tuple(c) for i, c in enumerate(self.coords)}

        if self.start_coord not in self.nodes:
            raise ValueError(f"Invalid start_coord for radius {radius}: {self.start_coord}")

        # 1. only water positions that have >=1 on-map cell in their band
        node_set = self.nodes 
        MARGIN = 1   # hubs camt be on edge, kills the masking crutch

        water_candidates = [
            c for c in self.coords
            if HexWorld.hex_dist(c, (0, 0)) <= self.radius - MARGIN    
            and any(n in node_set for n in HexWorld.hex_neighbors(c, band[0], band[1]))
        ]

        # 2. scarce water via Bridson over the eligible candidates
        water_coords = HexWorld.bridson(rng, water_candidates, r_min=WATER_R_MIN, k=30)

        # 3. each water gets 2-3 foods sampled from its band ring
        food_coords = []
        for water in water_coords:
            band_cells = [c for c in HexWorld.hex_neighbors(water, band[0], band[1])
                          if c in node_set]
            if not band_cells:
                continue                       # edge hub with no on-map ring; caught below
            n_food = int(rng.integers(2, 4))   # 2 or 3 — food-density knob
            idx = rng.choice(len(band_cells),
                             size=min(n_food, len(band_cells)),
                             replace=False)
            food_coords.extend(band_cells[i] for i in idx)

        food_coords = list(dict.fromkeys(food_coords))

        # affirm a well-formed world
        if not water_coords:
            raise ValueError("no water placed — broken world")
        if not food_coords:
            raise ValueError("no food placed — broken world")

        # no dead water, at least one in-band food
        for w in water_coords:
            if not any(HexWorld.hex_dist(w, f) <= band[1] for f in food_coords):
                raise ValueError(f"water {w} has no food within band {band} — dead hub")

        min_pair = min(HexWorld.hex_dist(f, w) for f in food_coords for w in water_coords)
        if min_pair < band[0]:
            raise ValueError(
                f"band violated: a food is {min_pair} from a water (floor {band[0]}). "
                f"WATER_R_MIN={WATER_R_MIN}; need >= {band[0]+band[1]}."
            )

        self.water_coords = water_coords
        self.food_coords = food_coords

        self.move_dirs = [LOCAL_DIRECTIONS[i] for i in range(6)]
        self.water_set = set(map(tuple, self.water_coords))
        self.food_set = set(map(tuple, self.food_coords))

        # spawn pool: hexes within reach of at least one water
        SPAWN_REACH = band[1]          # 11; within a commute of some hub
        self._spawn_pool = [
            c for c in self.coords
            if any(HexWorld.hex_dist(c, w) <= SPAWN_REACH for w in self.water_coords)
        ]
        if not self._spawn_pool:                    # degenerate world guard
            self._spawn_pool = list(self.coords)

    @staticmethod
    def cube_s(coord):
        q, r = coord
        return -q - r

    @staticmethod
    def add_coord(a, b, dist=1):
        return (a[0] + b[0]*dist, a[1] + b[1]*dist)

    @staticmethod
    def _make_coords(radius):
        coords = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= radius:
                    coords.append((q, r))
        return coords

    @staticmethod
    def hex_dist(a, b):
        dq = a[0] - b[0]
        dr = a[1] - b[1]
        ds = HexWorld.cube_s((dq, dr))
        return max(abs(dq), abs(dr), abs(ds))

    @staticmethod
    def hex_neighbors(coord, r_min, r_max):
        q, r = coord
        neighbours = []
        for dq in range(-r_max, r_max + 1):
            for dr in range(-r_max, r_max + 1):
                d = max(abs(dq), abs(dr), abs(HexWorld.cube_s((dq, dr))))
                if r_min <= d <= r_max and (dq, dr) != (0, 0):
                    neighbours.append((q + dq, r + dr))
        return neighbours

    @staticmethod
    def coord_to_bucket(coord, bucket_size):
        q, r = coord
        return (q // bucket_size, r // bucket_size)

    @staticmethod
    def bridson(rng, coords, r_min, k=30):
        coords_set = set(coords)
        grid = defaultdict(list)
        bucket_size = int(r_min / np.sqrt(2))
        start = coords[rng.integers(0, len(coords))]
        active = [start]
        placed = [start]
        grid[HexWorld.coord_to_bucket(start, bucket_size)].append(start)

        while active:
            center = active[rng.integers(0, len(active))]

            annulus = [c for c in HexWorld.hex_neighbors(center, r_min, 2*r_min) if c in coords_set]
            if not annulus:
                active.remove(center)
                continue

            for attempt in range(k):
                dart = annulus[rng.integers(0, len(annulus))]
                dart_bin = HexWorld.coord_to_bucket(dart, bucket_size)

                valid = True
                for bq in range(dart_bin[0]-2, dart_bin[0]+3):
                    for br in range(dart_bin[1]-2, dart_bin[1]+3):
                        for coord in grid[(bq, br)]:
                            if HexWorld.hex_dist(dart, coord) < r_min:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    grid[dart_bin].append(dart)
                    active.append(dart)
                    placed.append(dart)
                    break
            else:
                active.remove(center)

        return placed
    
    def reset_position(self, random_start=True, coord=None):
        if coord is None:
            if not random_start:
                coord = self.start_coord
            else:
                pool = self._spawn_pool
                idx = np.random.randint(0, len(pool))
                coord = pool[idx]

        coord = tuple(coord)
        if coord not in self.nodes:
            raise ValueError(f"Cannot reset to invalid hex coordinate: {coord}")
        self.coord = coord

    def get_vision_features(self):
        q, r = self.coord
        feats = []

        # current tile resource levels
        here = tuple(self.coord)
        feats.extend([
            float(self.nodes[here]["water"]),
            float(self.nodes[here]["food"]),
        ])

        # six neighbouring tiles
        for dq, dr in self.move_dirs:
            c = (q + dq, r + dr)
            legal = c in self.coord_to_idx

            if legal:
                water_level = self.nodes[c]["water"]
                food_level = self.nodes[c]["food"]
            else:
                water_level = 0.0
                food_level = 0.0

            feats.extend([
                1.0 if legal else 0.0,
                float(water_level),
                float(food_level),
            ])

        return np.array(feats, dtype=np.float32)

    def get_food_smell(self, radius=3):
        here = tuple(self.coord)

        d = min(HexWorld.hex_dist(here, tuple(food)) for food in self.food_coords)

        if d > radius:
            return 0.0

        return (radius + 1 - d) / (radius + 1)


    def move_local(self, move_id):
        direction = LOCAL_DIRECTIONS[move_id]
        next_coord = self.add_coord(self.coord, direction)

        if next_coord not in self.nodes:
            return False

        self.coord = next_coord
        return True

    def apply_action_movement(self, action_id):
        _, _, move_id = ACTION_EFFECTS[action_id]

        if move_id is None:
            return False

        return self.move_local(move_id)

    def get_action_mask(self, coord=None):
        if coord is None:
            coord = self.coord

        coord = tuple(coord)

        if coord not in self.nodes:
            raise ValueError(f"Cannot make mask for invalid hex coordinate: {coord}")

        mask = np.zeros(N_ACT, dtype=bool)
        q, r = coord

        for action_id, (dq, dr) in MOVE_ACTIONS.items():
            mask[action_id] = (q + dq, r + dr) in self.nodes

        mask[DRINK_ACTIONS] = self.nodes[coord]["water"] > 0
        mask[EAT_ACTIONS] = self.nodes[coord]["food"] > 0
        mask[WAIT] = True

        return mask

    def make_state(self, hydration, satiation, brightness, m_que_d, m_que_e, m_que_m):
        q, r = self.coord
        cube_s = self.cube_s(self.coord)

        q_scaled = q / self.radius
        r_scaled = r / self.radius
        s_scaled = cube_s / self.radius

        return np.array(
            [
                q_scaled,
                r_scaled,
                s_scaled,
                hydration,
                satiation,
                brightness,
                *m_que_d,
                *m_que_e,
                *m_que_m,
            ],
            dtype=np.float32,
        )