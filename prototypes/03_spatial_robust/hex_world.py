
import numpy as np

WAIT = 0

DRINK_ACTIONS = slice(1, 4)  # 1, 2, 3
EAT_ACTIONS = slice(4, 7)    # 4, 5, 6

MOVE_OFFSET = 7              # movement actions are 7..12


LOCAL_DIRECTIONS = {
    0: (1, -1),    # E
    1: (1, 0),     # NE
    2: (0, 1),     # NW
    3: (-1, 1),    # W
    4: (-1, 0),    # SW
    5: (0, -1),    # SE
}

MOVE_ACTIONS = {
    MOVE_OFFSET + move_id: direction
    for move_id, direction in LOCAL_DIRECTIONS.items()
}


ACTION_EFFECTS = [
    (0.0, 0.0, None),  # 0 wait
    (1.0, 0.0, None),  # 1 drink full
    (0.5, 0.0, None),  # 2 drink half
    (0.25, 0.0, None), # 3 drink quarter
    (0.0, 1.0, None),  # 4 eat full
    (0.0, 0.5, None),  # 5 eat half
    (0.0, 0.25, None), # 6 eat quarter
    (0.0, 0.0, 0),     # 7 move E
    (0.0, 0.0, 1),     # 8 move NE
    (0.0, 0.0, 2),     # 9 move NW
    (0.0, 0.0, 3),     # 10 move W
    (0.0, 0.0, 4),     # 11 move SW
    (0.0, 0.0, 5),     # 12 move SE
]

N_ACT = len(ACTION_EFFECTS)


class HexWorld:
    def __init__(
        self,
        radius=1,
        start_coord=(0, 0),
        water_coord=None,
        food_coord=None,
        water_level=1.0,
        food_level=1.0,
    ):
        self.radius = radius
        self.start_coord = tuple(start_coord)
        self.coord = tuple(start_coord)

        self.coords = self._make_coords(radius)

        self.nodes = {
            coord: {"water": 0.0, "food": 0.0}
            for coord in self.coords
        }

        if water_coord is None:
            water_coord = (-radius, 0)      # left corner-ish

        if food_coord is None:
            food_coord = (0, radius)        # upper-left / different corner, not opposite

        water_coord = tuple(water_coord)
        food_coord = tuple(food_coord)

        if self.start_coord not in self.nodes:
            raise ValueError(f"Invalid start_coord for radius {radius}: {self.start_coord}")

        if water_coord not in self.nodes:
            raise ValueError(f"Invalid water_coord for radius {radius}: {water_coord}")

        if food_coord not in self.nodes:
            raise ValueError(f"Invalid food_coord for radius {radius}: {food_coord}")

        if water_coord == food_coord:
            raise ValueError("water_coord and food_coord cannot be the same.")
        
        self.water_coord = water_coord
        self.food_coord = food_coord

        self.nodes[water_coord]["water"] = float(water_level)
        self.nodes[food_coord]["food"] = float(food_level)

    @staticmethod
    def cube_s(coord):
        q, r = coord
        return -q - r

    @staticmethod
    def add_coord(a, b):
        return (a[0] + b[0], a[1] + b[1])

    @staticmethod
    def _make_coords(radius):
        coords = []

        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r

                if max(abs(q), abs(r), abs(s)) <= radius:
                    coords.append((q, r))

        return coords

    def reset_position(self, coord=None):
        if coord is None:
            coord = self.start_coord

        coord = tuple(coord)

        if coord not in self.nodes:
            raise ValueError(f"Cannot reset to invalid hex coordinate: {coord}")

        self.coord = coord

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