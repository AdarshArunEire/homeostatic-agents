MOVE_DELTAS = {
    7:  (1, -1),   # move_E
    8:  (1, 0),    # move_NE
    9:  (0, 1),    # move_NW
    10: (-1, 1),   # move_W
    11: (-1, 0),   # move_SW
    12: (0, -1),   # move_SE
}

DEFAULT_WATER_COORD = (-1, 0)
DEFAULT_FOOD_COORD = (1, 0)
DEFAULT_FILL_H = 1.7
DEFAULT_FILL_S = 1.7


def configure(water_coord=(-1, 0), food_coord=(1, 0), fill_h=1.7, fill_s=1.7):
    global DEFAULT_WATER_COORD, DEFAULT_FOOD_COORD, DEFAULT_FILL_H, DEFAULT_FILL_S

    DEFAULT_WATER_COORD = tuple(water_coord)
    DEFAULT_FOOD_COORD = tuple(food_coord)
    DEFAULT_FILL_H = fill_h
    DEFAULT_FILL_S = fill_s


def hex_dist(a, b):
    aq, ar = a
    bq, br = b
    return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))


class OraclePolicy:
    def __init__(self, water_coord, food_coord, fill_h, fill_s):
        self.water_coord = tuple(water_coord)
        self.food_coord = tuple(food_coord)
        self.fill_h = fill_h
        self.fill_s = fill_s

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
        h = self.hydration
        s = self.satiation

        q = int(round(float(coord[0])))
        r = int(round(float(coord[1])))
        h = float(h)
        s = float(s)

        coord = (q, r)

        if coord == self.water_coord and h < self.fill_h and action_mask[1]:
            return 1  # drink_full

        if coord == self.food_coord and s < self.fill_s and action_mask[4]:
            return 4  # eat_full

        if s <= h:
            return self.move_towards(coord, self.food_coord, action_mask)

        return self.move_towards(coord, self.water_coord, action_mask)


def make_model(n_input, n_hidden, n_act, lr):
    policy = OraclePolicy(
        water_coord=DEFAULT_WATER_COORD,
        food_coord=DEFAULT_FOOD_COORD,
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