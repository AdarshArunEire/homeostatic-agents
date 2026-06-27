MOVE_DELTAS = {
    7:  (1, -1),   # move_E
    8:  (1, 0),    # move_NE
    9:  (0, 1),    # move_NW
    10: (-1, 1),   # move_W
    11: (-1, 0),   # move_SW
    12: (0, -1),   # move_SE
}

DEFAULT_WATER_COORDS = None
DEFAULT_FOOD_COORDS = None
DEFAULT_FILL_H = 1.7
DEFAULT_FILL_S = 1.7


def configure(water_coords=None, food_coords=None, fill_h=1.7, fill_s=1.7):
    if water_coords is None or food_coords is None:
        raise ValueError("oracle.configure must be called with water_coords and food_coords before use")
    
    global DEFAULT_WATER_COORDS, DEFAULT_FOOD_COORDS, DEFAULT_FILL_H, DEFAULT_FILL_S

    DEFAULT_WATER_COORDS = water_coords
    DEFAULT_FOOD_COORDS = food_coords
    DEFAULT_FILL_H = fill_h
    DEFAULT_FILL_S = fill_s


def hex_dist(a, b):
    aq, ar = a
    bq, br = b
    return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))

def nearest(coord, coord_list):
    return min(coord_list, key=lambda c: hex_dist(coord, c))

class OraclePolicy:
    def __init__(self, water_coords, food_coords, fill_h, fill_s):
        self.water_coords = water_coords
        self.food_coords = food_coords
        self.fill_h = fill_h
        self.fill_s = fill_s

    def eval(self):
        pass

    def move_towards(self, coord, target_coord, action_mask):
        cur_q, cur_r = coord

        best_action = 0  # wait fallback
        best_dist = hex_dist(coord, target_coord)

        for act, (dq, dr) in MOVE_DELTAS.items():
            if not action_mask[act]:
                continue

            nxt = (cur_q + dq, cur_r + dr)
            d = hex_dist(nxt, target_coord)

            if d < best_dist:
                best_dist = d
                best_action = act

        return best_action
    
    def set_context(self, coord, hydration, satiation):
        self.coord = tuple(coord)
        self.hydration = float(hydration)
        self.satiation = float(satiation)

    def select_action(self, action_mask):
        coord = self.coord
        h = float(self.hydration)
        s = float(self.satiation)

        nearest_water = nearest(coord, self.water_coords)
        nearest_food = nearest(coord, self.food_coords)

        if coord in self.water_coords and h < self.fill_h and action_mask[1]:
            return 1

        if coord in self.food_coords and s < self.fill_s and action_mask[4]:
            return 4

        if s <= h:
            return self.move_towards(coord, nearest_food, action_mask)

        return self.move_towards(coord, nearest_water, action_mask)


def make_model(n_input, n_hidden, n_act, lr):
    policy = OraclePolicy(
        water_coords=DEFAULT_WATER_COORDS,
        food_coords=DEFAULT_FOOD_COORDS,
        fill_h=DEFAULT_FILL_H,
        fill_s=DEFAULT_FILL_S,
    )
    return policy, None


def make_target(model):
    return None


def sync_target(target, model):
    return None

def learn_step(model, target, optimiser, batch, gamma):
    return 0.0

def select_action(model, state, action_mask):
    return model.select_action(action_mask)