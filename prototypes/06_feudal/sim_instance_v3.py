"""
sim_instance_v0.py  —  Proto 06 oracle/module runner.

Purpose:
    Run the modular Proto 06 stack without replay/learning code.

Keeps from old sim_instance:
    - old env construction style
    - old a_que temporal semantics
    - old delayed absorption profiles
    - old m_que_d/e/m action-history inputs
    - old touch-memory slots for remembered water/food coords
    - old train/eval split metrics

Changes:
    - no DQN register / target / replay / learn step
    - no arbitrary action_effects index table
    - a_que stores resolved physical effects directly
    - module registry selects orchestrator/pathfinder/eat/drink impls
"""

from __future__ import annotations
import numpy as np
import random
from collections import deque
from dataclasses import dataclass

import hex_world_cached as hex_world
import map_curriculum as mc
import world_v1 as world

from model_modules.contract_v1 import (
    Action,
    ConsumerObs,
    ModuleSpec,
    OrchestratorObs,
    PathfinderObs,
)

from model_modules.oracle_modules.orchestrator_oracle_v1 import OracleOrchestrator

from model_modules.oracle_modules.orchestrator_noisy_v1 import (
    make_orchestrator_noisy,
)

from model_modules.oracle_modules.pathfinder_oracle_v1 import OraclePathfinder

from model_modules.oracle_modules.pathfinder_noisy_v1 import (
    make_noisy_pathfinder,
)

from model_modules.oracle_modules.consumer_oracle_v1 import (
    make_drink,
    make_eat,
)

from model_modules.oracle_modules.consumer_noisy_v1 import (
    make_eat_noisy,
    make_drink_noisy,
)

from model_modules.oracle_modules.explorer_oracle_v1 import (
    make_random as make_random_explorer,
    make_momentum as make_momentum_explorer,
    make_smell_momentum as make_smell_momentum_explorer,
)

# Learned impls. Each is constructed from a STRING TAG in oracle_params and loads its own
# weights inside the worker — configs cross a spawn boundary under sweep_parallel and live
# torch objects do not survive that pickle.
from model_modules.learned_modules.pathfinder_learned_v1 import (
    make_pathfinder as make_learned_pathfinder,
)

from model_modules.learned_modules.consumer_learned_v1 import (
    make_eat_learned,
    make_drink_learned,
)

from model_modules.learned_modules.explorer_learned_v1 import (
    make_explorer_learned,
)

from model_modules.learned_modules.orchestrator_learned_v1 import (
    make_orchestrator_learned,
)

from model_modules.oracle_modules.explorer_noisy_v1 import (
    make_explorer_noisy,
)


# Must match old env movement action convention.
OLD_MOVE_OFFSET = 7


@dataclass(frozen=True)
class QueuedAction:
    """
    Resolved physical action stored in a_que.

    This replaces old arbitrary action ids. The queue still means:
        newest action moves once
        drink/eat fractions absorb over future ticks
    """
    drink_frac: float = 0.0
    eat_frac: float = 0.0
    move_dir: int | None = None          # 0..5, or None
    abstract_action: int | None = None   # optional logging: Action enum int
    # which consumer was procced this tick: "eat", "drink", or None.
    # Recorded explicitly because it CANNOT be inferred from the fracs: a learned consumer
    # may legitimately return 0.0, and so may the oracle when the drive is already at
    # target, leaving a zero-frac CONSUME tick indistinguishable between the two slots.
    # The in-sim training rig joins module records onto ticks through this field, and a
    # wrong join trains every transition against the wrong reward without erroring.
    consumer: str | None = None


MODULE_REGISTRY = {
    "orchestrator": {
        "oracle": OracleOrchestrator,
        # instrument: epsilon (general) + crit_scale (the arbitration threshold, whose
        # ablation has a recorded effect size — hydration deaths 43 -> 10)
        "noisy_oracle": make_orchestrator_noisy,
        # 4-way abstract head (GO_WATER / GO_FOOD / CONSUME / EXPLORE); exploration is
        # DELEGATED to the explorer module so arbitration is not confounded with a task
        # already established as unlearnable
        "learned": make_orchestrator_learned,
    },
    "pathfinder": {
        "oracle": OraclePathfinder,
        # needs oracle_params["pathfinder"] = {"weights": "<tag>", "scale": radius}
        "learned": make_learned_pathfinder,
        # measurement instrument, not a candidate: oracle + epsilon random steps, used to
        # calibrate how much pathfinder degradation a given metric can actually detect
        "noisy_oracle": make_noisy_pathfinder,
    },
    "eat": {
        "oracle": make_eat,
        # needs oracle_params["eat"] = {"weights": "<tag>"}
        "learned": make_eat_learned,
        # instrument: oracle fill + sigma (graded) and/or epsilon (rare, large)
        "noisy_oracle": make_eat_noisy,
    },
    "drink": {
        "oracle": make_drink,
        # needs oracle_params["drink"] = {"weights": "<tag>"}
        "learned": make_drink_learned,
        "noisy_oracle": make_drink_noisy,
    },
    "explorer": {
        "random": make_random_explorer,
        "momentum": make_momentum_explorer,
        "smell_momentum": make_smell_momentum_explorer,
        # needs oracle_params["explorer"] = {"weights": "<tag>"}
        "learned": make_explorer_learned,
        # instrument: smell_momentum + epsilon uniform legal moves. epsilon=1 degrades
        # toward the random floor, so the sweep spans the known 0.17-0.48 ladder.
        "noisy_oracle": make_explorer_noisy,
    },
}


def default_oracle_params():
    return {
        "orchestrator": {
            "h_fill": 1.25,
            "s_fill": 1.25,
        },
        "pathfinder": {},
        "eat": {
            "fill_target": 1.25,
        },
        "drink": {
            "fill_target": 1.25,
        },        
        "explorer": {
            "persist_p": 0.70,
            "avoid_reverse": True,
            "trend_eps": 0.01,
            "follow_p": 0.95,
            "reverse_on_drop_p": 0.75,
        },
    }

def build_modules(
    module_spec: ModuleSpec,
    oracle_params: dict,
    rng: np.random.Generator,
    radius: int,
):
    orch_make = MODULE_REGISTRY["orchestrator"][module_spec.orchestrator]
    path_make = MODULE_REGISTRY["pathfinder"][module_spec.pathfinder]
    eat_make = MODULE_REGISTRY["eat"][module_spec.eat]
    drink_make = MODULE_REGISTRY["drink"][module_spec.drink]
    explorer_make = MODULE_REGISTRY["explorer"][module_spec.explorer]

    #print("DEBUG build module_spec:", module_spec)
    #print("DEBUG module_spec.explorer:", module_spec.explorer)

    explorer = explorer_make(
        rng=rng,
        **oracle_params.get("explorer", {}),
    )

    modules = {
        "explorer": explorer,

        "orchestrator": orch_make(
            rng=rng,
            explorer=explorer,
            **oracle_params.get("orchestrator", {}),
        ),

        "pathfinder": path_make(
            **oracle_params.get("pathfinder", {}),
        ),

        "eat": eat_make(
            **oracle_params.get("eat", {}),
        ),

        "drink": drink_make(
            **oracle_params.get("drink", {}),
        ),
    }

    return modules

def compose_to_queued_action(
    orch_action: Action,
    modules: dict,
    memory: dict,
    pos: tuple[int, int],
    consumer_obs: ConsumerObs,
) -> QueuedAction:
    """
    Lower modular abstract action into one physical queued effect.

    This is the bridge:
        orchestrator Action
          -> optional pathfinder/consumer
          -> QueuedAction
          -> a_que physics
    """
    pathfinder = modules["pathfinder"]
    eat = modules["eat"]
    drink = modules["drink"]

    if Action.EXPLORE_0 <= orch_action <= Action.EXPLORE_5:
        move_k = int(orch_action - Action.EXPLORE_0)
        return QueuedAction(
            move_dir=move_k,
            abstract_action=int(orch_action),
        )

    if orch_action == Action.GO_WATER:
        if memory["water"] is None:
            return QueuedAction(abstract_action=int(orch_action))

        goal = memory["water"]
        to_goal = (goal[0] - pos[0], goal[1] - pos[1])
        move = pathfinder.move(PathfinderObs(to_goal=to_goal))

        return QueuedAction(
            move_dir=int(move),
            abstract_action=int(orch_action),
        )

    if orch_action == Action.GO_FOOD:
        if memory["food"] is None:
            return QueuedAction(abstract_action=int(orch_action))

        goal = memory["food"]
        to_goal = (goal[0] - pos[0], goal[1] - pos[1])
        move = pathfinder.move(PathfinderObs(to_goal=to_goal))

        return QueuedAction(
            move_dir=int(move),
            abstract_action=int(orch_action),
        )

    if orch_action == Action.CONSUME:
        if consumer_obs.tile_water_lvl > 0:
            frac = drink.consume(consumer_obs)
            return QueuedAction(
                drink_frac=float(np.clip(frac, 0.0, 1.0)),
                abstract_action=int(orch_action),
                consumer="drink",
            )

        if consumer_obs.tile_food_lvl > 0:
            frac = eat.consume(consumer_obs)
            return QueuedAction(
                eat_frac=float(np.clip(frac, 0.0, 1.0)),
                abstract_action=int(orch_action),
                consumer="eat",
            )

        return QueuedAction(abstract_action=int(orch_action))

    raise ValueError(f"unknown orchestrator action {orch_action!r}")



def sim_instance(
        seed=None,

        # --- run length ---
        sim_len=1000000,
        eval_len=20000,

        # --- env / world ---
        env_kwargs=None,
        comfort_surface="exponential",
        random_start="uniform",
        hmax=3,
        smax=3,
        memory_len=10,

        # --- physics tuning ---
        decay_mult=1.0,

        # --- smell / sensory cues ---
        smell_radius=12,

        # --- eval probe: spawn on/adjacent to water at eval only ---
        # Isolates food-seeking: water is handed to the agent (findable turn 1 via
        # vision, infinite tile), so the only death cause left is satiation. Changes
        # the eval initial-state distribution only; no observation/privilege change.
        eval_spawn_beside_water=False,

        # --- eval probe: reject DOOMED spawns (keeps full 2-resource nav) ---
        # A spawn is doomed if, even playing perfectly (go to nearest water, top up,
        # then head to nearest food), a drive kills you before you can secure both.
        # We check the expected-decay survival clocks against the required hex travel,
        # plus spawn_leeway steps of slack. Removes unfair deaths WITHOUT handing water
        # over -- the agent still has to find water then food from a cold, non-adjacent
        # start. Takes precedence over eval_spawn_beside_water.
        eval_spawn_nondoomed=False,
        spawn_leeway=10,

        # --- eval probe: GOD navigation (Proto-05 oracle power) ---
        # Pre-populate memory with the true nearest water/food every eval tick, so the
        # agent never has to EXPLORE -- it just pathfinds to known coords, exactly like
        # the Proto-05 god oracle. Isolates "is the world survivable with perfect
        # navigation" (fairness) from "can a legal agent FIND the resources" (the task).
        eval_god_memory=False,

        # --- curriculum / cadence ---
        curriculum_mode="band",
        c_min=2,
        c_max=9,
        band_width=2,
        curriculum_ramp_frac=0.6,
        curriculum_sampling="uniform",
        terminals_per_map=1,
        life_cap=1000,

        # --- modules ---
        module_spec=None,

        # --- oracle params ---
        oracle_params=None,

        # --- logging ---
        log_every=50_000,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    rng = np.random.default_rng(seed)

    EB = sim_len - eval_len

    if env_kwargs is None:
        raise ValueError("env_kwargs must be provided")

    # -------------------------------------------------------------------------
    # world construction / curriculum
    # -------------------------------------------------------------------------
    eval_segments = []
    train_terminal_count = 0

    def record_eval_segment(t_global):
        eval_segments.append({
            "t_start": int(t_global),
            "band": tuple(current_band),
            "water_coords": [tuple(c) for c in env.water_coords],
            "food_coords": [tuple(c) for c in env.food_coords],
        })
        
    if curriculum_mode == "disabled":
        curr = None
        env = hex_world.HexWorld(seed=seed, **env_kwargs)
        current_band = tuple(env_kwargs.get("band", (9, 11)))
        decay_band = current_band

    elif curriculum_mode == "band":
        curr = mc.MapCurriculum(
            seed=seed,
            base_env_kwargs=env_kwargs,
            EB=EB,
            c_min=c_min,
            c_max=c_max,
            band_width=band_width,
            ramp_frac=curriculum_ramp_frac,
            sampling=curriculum_sampling,
        )
        env, current_band, _ = curr.build(t=0, eval_mode=False)
        decay_band = (c_max, c_max + band_width)

    water_set = set(map(tuple, env.water_coords))
    food_set = set(map(tuple, env.food_coords))

    def bind_env(new_env):
        nonlocal env, water_set, food_set
        env = new_env
        water_set = set(map(tuple, env.water_coords))
        food_set = set(map(tuple, env.food_coords))

    h_scale, s_scale = world.decay_scaling(env.radius)#
    h_scale *= decay_mult
    s_scale *= decay_mult

    # -------------------------------------------------------------------------
    # module construction
    # -------------------------------------------------------------------------
    oracle_params = default_oracle_params() if oracle_params is None else oracle_params

    if module_spec is None:
        module_spec = ModuleSpec(
            orchestrator="oracle",
            pathfinder="oracle",
            eat="oracle",
            drink="oracle",
            explorer="momentum",
        )

    modules = build_modules(
        module_spec=module_spec,
        oracle_params=oracle_params,
        rng=rng,
        radius=env.radius,
    )

    #print("DEBUG explorer:", type(modules["explorer"]).__name__)
    #print("DEBUG orch explorer:", type(modules["orchestrator"].explorer).__name__)
    #print("DEBUG same object:", modules["orchestrator"].explorer is modules["explorer"])

    # -------------------------------------------------------------------------
    # state
    # -------------------------------------------------------------------------
    h = world.IDEAL_H
    s = world.IDEAL_S
    day_length = 50
    life_t = 0

    # Physical queue. Holds direct effects, not action ids.
    a_que = deque(maxlen=11)

    # Semantic action-history queues for model inputs.
    # These replace old action_effects[action] decoding.
    m_que_d = deque([0.0] * memory_len, maxlen=memory_len)
    m_que_e = deque([0.0] * memory_len, maxlen=memory_len)
    m_que_m = deque([-1] * memory_len, maxlen=memory_len)

    # Touch/resource memory.
    last_water_seen = None
    last_food_seen = None
    last_resource = None

    train_wf_trips = 0
    train_fw_trips = 0
    eval_water_ticks = 0
    eval_food_ticks = 0
    n_timeouts = 0
    n_timeouts_eval = 0
    death_events = []

    lifetime_stats = {
        "comfort": [],
        "abstract_action": [],
        "drink_frac": [],
        "eat_frac": [],
        "move_dir": [],
        "consumer_slot": [],
        "hydration": [],
        "satiation": [],
        "brightness": [],
        "dead": [],
        "coord": [],
    }

    def _water_adjacent_pool():
        """Coords on or within 1 hex of a water tile (eval probe spawn pool)."""
        coord_set = set(map(tuple, env.coords))
        pool = set()
        for w in env.water_coords:
            w = tuple(w)
            pool.add(w)
            for dq, dr in env.move_dirs:
                c = (w[0] + dq, w[1] + dr)
                if c in coord_set:
                    pool.add(c)
        return [np.array(c) for c in pool]

    # Survival clocks under expected (noise-free) decay, at mean brightness. Used only
    # to screen doomed spawns; h_scale/s_scale already fold in decay_mult.
    _B_REP = 0.5

    def _thirst_clock(h0, s_hold):
        """Steps until thirst death from h0 with no drinking (s held ~constant)."""
        h = float(h0)
        for t in range(2000):
            h = world.expected_decay_hydration(h, s_hold, _B_REP, h_scale)
            if h <= world.DEATH_THRESH:
                return t + 1
        return 2000

    def _hunger_clock(s0):
        """Steps until starvation from s0, assuming hydration is kept at ideal."""
        s = float(s0)
        for t in range(2000):
            s = world.expected_decay_satiation(s, world.IDEAL_H, _B_REP, s_scale)
            if s <= world.DEATH_THRESH:
                return t + 1
        return 2000

    def _not_doomed(coord, h0, s0):
        """Can perfect water-first play secure both resources with leeway to spare?"""
        hd = hex_world.HexWorld.hex_dist
        d_w = min(hd(tuple(coord), tuple(w)) for w in env.water_coords)
        w_star = min(env.water_coords, key=lambda w: hd(tuple(coord), tuple(w)))
        d_wf = min(hd(tuple(w_star), tuple(f)) for f in env.food_coords)
        # 1) reach the first drink before dying of thirst
        if _thirst_clock(h0, s0) < d_w + spawn_leeway:
            return False
        # 2) reach food (via that water) before starving
        if _hunger_clock(s0) < d_w + d_wf + spawn_leeway:
            return False
        return True

    def sample_uniform_start(beside_water=False, nondoomed=False):
        pool = _water_adjacent_pool() if beside_water else env._spawn_pool
        last = None
        for _ in range(500):
            ang = np.random.uniform(0, 2 * np.pi)
            rad = np.sqrt(np.random.uniform(0.1 ** 2, 2.9 ** 2))
            nh = world.IDEAL_H + rad * np.cos(ang)
            ns = world.IDEAL_S + rad * np.sin(ang)
            if not (
                world.DEATH_THRESH < nh < hmax
                and world.DEATH_THRESH < ns < smax
                and not (nh < 0.35 and ns < 0.35)
            ):
                continue
            coord = tuple(pool[np.random.randint(len(pool))])
            last = (float(nh), float(ns), coord)
            if nondoomed and not _not_doomed(coord, nh, ns):
                continue
            return float(nh), float(ns), coord
        # Fallback: no admissible draw in budget -> return last basic-valid sample.
        return last
            
    water_smell_hist = deque([0.0, 0.0, 0.0], maxlen=3)
    food_smell_hist = deque([0.0, 0.0, 0.0], maxlen=3)

    def resource_smell(coord, resource_coords, radius):
        """
        Local smell intensity from nearest resource.

        This is not coordinate sensing by the agent.
        The env computes a scalar sensory reading at the current tile.

        Triangular falloff:
            d = 0        -> 1.0
            d > radius   -> 0.0
        """
        coord = tuple(coord)
        d = min(hex_world.HexWorld.hex_dist(coord, tuple(c)) for c in resource_coords)

        if d > radius:
            return 0.0

        return float((radius + 1 - d) / (radius + 1))

    def reset_smell_history():
        water_smell_hist.clear()
        food_smell_hist.clear()

        water_smell_hist.extend([0.0, 0.0, 0.0])
        food_smell_hist.extend([0.0, 0.0, 0.0])

    def update_smell_history(coord):
        water_smell_hist.appendleft(
            resource_smell(coord, env.water_coords, smell_radius)
        )
        food_smell_hist.appendleft(
            resource_smell(coord, env.food_coords, smell_radius)
        )

    def smell_vecs():
        return (
            np.asarray(water_smell_hist, dtype=np.float32),
            np.asarray(food_smell_hist, dtype=np.float32),
        )  

    def respawn(t, eval_mode):
        nonlocal h, s, life_t, last_water_seen, last_food_seen, last_resource

        h, s, new_coord = sample_uniform_start(
            beside_water=(eval_mode and eval_spawn_beside_water and not eval_spawn_nondoomed),
            nondoomed=(eval_mode and eval_spawn_nondoomed),
        )
        env.reset_position(coord=new_coord)

        life_t = 0

        a_que.clear()

        m_que_d.clear()
        m_que_e.clear()
        m_que_m.clear()
        m_que_d.extend([0.0] * memory_len)
        m_que_e.extend([0.0] * memory_len)
        m_que_m.extend([-1] * memory_len)

        last_water_seen = None
        last_food_seen = None
        last_resource = None

        reset_smell_history()

        if hasattr(modules["orchestrator"], "reset"):
            modules["orchestrator"].reset()

    def make_agent_state(h, s, b):
        """
        Keep this if/when learned modules need the old-style flat state.

        Important:
            Models do not receive raw a_que objects.
            Models receive semantic histories m_que_d/e/m.

        If env.make_state already understands these three queues, this stays compatible.
        If you later remove env.make_state, build the flat vector directly here.
        """
        base = env.make_state(h, s, b, m_que_d, m_que_e, m_que_m)
        parts = [np.asarray(base, dtype=np.float32)]

        # Touch-memory features, same spirit as old sim.
        cq, cr = env.coord
        R = env.radius

        if last_water_seen is None:
            wq, wr, wf = 0.0, 0.0, 0.0
        else:
            wq, wr, wf = (
                (last_water_seen[0] - cq) / R,
                (last_water_seen[1] - cr) / R,
                1.0,
            )

        if last_food_seen is None:
            fq, fr, ff = 0.0, 0.0, 0.0
        else:
            fq, fr, ff = (
                (last_food_seen[0] - cq) / R,
                (last_food_seen[1] - cr) / R,
                1.0,
            )

        parts.append(np.array([wq, wr, wf, fq, fr, ff], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------
    for t in range(sim_len):
        # t > 0 guard: 0 % anything == 0, so the t=0 line fired on every run regardless of
        # log_every, including log_every=10**9 which is the idiom for "silence". It printed
        # an all-zero row before the sim had done anything, which is noise in every sweep.
        if t > 0 and t % log_every == 0:
            print(
                f"    [seed {seed}] t={t} deaths={int(np.sum(lifetime_stats['dead']))} "
                f"timeouts={n_timeouts} wf={train_wf_trips} fw={train_fw_trips}",
                flush=True,
            )

        training = t < EB
        life_t += 1

        if t == EB:
            if curr is not None:
                new_env, current_band, _ = curr.build(t=t, eval_mode=True)
                bind_env(new_env)

            respawn(t=t, eval_mode=True)
            record_eval_segment(t)

        training = t < EB

        # ---------------------------------------------------------------------
        # physics: decay first
        # ---------------------------------------------------------------------
        b = world.brightness(t, day_len=day_length, noise=True)

        h = world.decay_hydration(h, s, b, h_scale)
        s = world.decay_satiation(s, h, b, s_scale)

        # ---------------------------------------------------------------------
        # physics: queued effects
        # ---------------------------------------------------------------------
        for age, qa in enumerate(a_que):
            # Newest queued move applies once.
            if age == 0 and qa.move_dir is not None:
                env.apply_action_movement(OLD_MOVE_OFFSET + int(qa.move_dir))

            # Drink absorption over profile.
            if age < len(world.DRINK_PROFILE):
                h += world.DRINK_AMOUNT * qa.drink_frac * world.DRINK_PROFILE[age] * (0.8 + 0.3 * s)

            # Eat absorption over profile.
            if age < len(world.EAT_PROFILE):
                s += world.EAT_AMOUNT * qa.eat_frac * world.EAT_PROFILE[age]

        h = float(np.clip(h, 0.0, hmax))
        s = float(np.clip(s, 0.0, smax))

        cur_comfort = world.comfort(h, s, surface=comfort_surface)
        cur_dead = int((h <= world.DEATH_THRESH) or (s <= world.DEATH_THRESH))

        cur_coord = tuple(env.coord)
        on_water = cur_coord in water_set
        on_food = cur_coord in food_set

        update_smell_history(cur_coord)

        # ---------------------------------------------------------------------
        # memory update lives here
        # ---------------------------------------------------------------------
        if on_water:
            last_water_seen = cur_coord

        if on_food:
            last_food_seen = cur_coord

        vision = env.get_vision_features()
        
        # current tile memory
        if vision[0] > 0:
            last_water_seen = cur_coord
        if vision[1] > 0:
            last_food_seen = cur_coord

        # neighbour vision memory
        triples = vision[2:].reshape(6, 3)
        q, r = cur_coord

        for k, (legal, water_lvl, food_lvl) in enumerate(triples):
            if legal <= 0:
                continue

            dq, dr = env.move_dirs[k]
            c = (q + dq, r + dr)

            if water_lvl > 0:
                last_water_seen = c
            if food_lvl > 0:
                last_food_seen = c

        # GOD navigation probe: overwrite memory with the true nearest resources at eval,
        # so exploration is removed and only navigation/survival is tested.
        if (not training) and eval_god_memory:
            _hd = hex_world.HexWorld.hex_dist
            last_water_seen = tuple(min(env.water_coords, key=lambda cc: _hd(cur_coord, tuple(cc))))
            last_food_seen = tuple(min(env.food_coords, key=lambda cc: _hd(cur_coord, tuple(cc))))

        # ---------------------------------------------------------------------
        # trip metrics
        # ---------------------------------------------------------------------
        if training:
            if on_water:
                if last_resource == "food":
                    train_fw_trips += 1
                last_resource = "water"

            elif on_food:
                if last_resource == "water":
                    train_wf_trips += 1
                last_resource = "food"

        else:
            if on_water:
                eval_water_ticks += 1
            if on_food:
                eval_food_ticks += 1

        # ---------------------------------------------------------------------
        # terminal handling
        # ---------------------------------------------------------------------
        timeout = (
            (curr is not None)
            and (life_cap is not None)
            and (life_t >= life_cap)
            and (cur_dead == 0)
        )

        terminal = bool(cur_dead) or bool(timeout)

        # For future learned modules:
        # death is a true value-target terminal;
        # timeout is only an episode/map boundary.
        done_for_target = bool(cur_dead)
        is_boundary = bool(terminal)

        if timeout:
            if training:
                n_timeouts += 1
            else:
                n_timeouts_eval += 1

        if cur_dead and t != 0:
            if h <= world.DEATH_THRESH and s <= world.DEATH_THRESH:
                cause = "both"
            elif h <= world.DEATH_THRESH:
                cause = "hydration"
            else:
                cause = "satiation"

            death_events.append({
                "t": int(t),
                "cause": cause,
                "h": float(h),
                "s": float(s),
            })

        log_coord = tuple(env.coord)
        log_h = h
        log_s = s

        if terminal:
            if curr is not None:
                if training:
                    train_terminal_count += 1

                    if train_terminal_count % terminals_per_map == 0:
                        new_env, current_band, _ = curr.build(t=t, eval_mode=False)
                        bind_env(new_env)

                else:
                    new_env, current_band, _ = curr.build(t=t, eval_mode=True)
                    bind_env(new_env)
                    record_eval_segment(t)

            respawn(t=t, eval_mode=(t >= EB))

            # Refresh after respawn/map switch before module decision.
            cur_coord = tuple(env.coord)
            on_water = cur_coord in water_set
            on_food = cur_coord in food_set
            vision = env.get_vision_features()
            update_smell_history(cur_coord)

        # ---------------------------------------------------------------------
        # module observations
        # ---------------------------------------------------------------------
        tile_water_lvl = 1.0 if on_water else 0.0
        tile_food_lvl = 1.0 if on_food else 0.0

        water_smell_vec, food_smell_vec = smell_vecs()

        orch_obs = OrchestratorObs(
            h=h,
            s=s,

            # Backward-compatible old smell slot.
            # Use food smell history because old code had food smell only.
            smell=food_smell_vec,

            vision=vision,

            water_known=last_water_seen is not None,
            food_known=last_food_seen is not None,

            tile_water_lvl=tile_water_lvl,
            tile_food_lvl=tile_food_lvl,

            water_smell=water_smell_vec,
            food_smell=food_smell_vec,
        )

        consumer_obs = ConsumerObs(
            brightness=b,
            h=h,
            s=s,
            tile_water_lvl=tile_water_lvl,
            tile_food_lvl=tile_food_lvl,
        )

        # ---------------------------------------------------------------------
        # module decision
        # ---------------------------------------------------------------------
        orch_action = modules["orchestrator"].act(orch_obs)

        queued_action = compose_to_queued_action(
            orch_action=orch_action,
            modules=modules,
            memory={
                "water": last_water_seen,
                "food": last_food_seen,
            },
            pos=cur_coord,
            consumer_obs=consumer_obs,
        )

        # Queue direct effect.
        a_que.appendleft(queued_action)

        # Feed semantic history to future learned models.
        m_que_d.appendleft(float(queued_action.drink_frac))
        m_que_e.appendleft(float(queued_action.eat_frac))
        m_que_m.appendleft(-1 if queued_action.move_dir is None else int(queued_action.move_dir))

        # Optional for future learned modules. For now, just proves the state builds.
        cur_x = make_agent_state(h, s, b)

        # ---------------------------------------------------------------------
        # logging
        # ---------------------------------------------------------------------
        lifetime_stats["comfort"].append(float(cur_comfort))
        lifetime_stats["abstract_action"].append(
            -1 if queued_action.abstract_action is None else int(queued_action.abstract_action)
        )
        lifetime_stats["drink_frac"].append(float(queued_action.drink_frac))
        lifetime_stats["eat_frac"].append(float(queued_action.eat_frac))
        lifetime_stats["move_dir"].append(-1 if queued_action.move_dir is None else int(queued_action.move_dir))
        # 0 = none, 1 = drink, 2 = eat. Exact dispatch record for the training rig's join.
        lifetime_stats["consumer_slot"].append(
            0 if queued_action.consumer is None
            else (1 if queued_action.consumer == "drink" else 2)
        )
        lifetime_stats["hydration"].append(float(log_h))
        lifetime_stats["satiation"].append(float(log_s))
        lifetime_stats["brightness"].append(float(b))
        lifetime_stats["dead"].append(int(cur_dead))
        lifetime_stats["coord"].append(log_coord)

    # -------------------------------------------------------------------------
    # return summary
    # -------------------------------------------------------------------------
    comfort_T = np.asarray(lifetime_stats["comfort"], dtype=np.float32)
    hydration_T = np.asarray(lifetime_stats["hydration"], dtype=np.float32)

    # --- legacy action_T shim -------------------------------------------------------
    #
    # sweep_fn_v5.compute_eval_metrics keys off `action_T` in the pre-Proto-06 action-id
    # scheme. Proto 06 replaced action ids with QueuedAction (continuous fracs), so the key
    # was absent and compute_eval_metrics returned {} on EVERY Proto 06 run -- silently
    # taking the whole metric suite with it, including the named verdict metrics
    # `eat_rate_at_food` and `two_way_route_success_min`.
    #
    # Legacy scheme (hex_world_cached.ACTION_EFFECTS):
    #   0        WAIT
    #   1,2,3    drink at frac 1.0 / 0.5 / 0.25
    #   4,5,6    eat   at frac 1.0 / 0.5 / 0.25
    #   7..12    move  in direction 0..5
    #
    # LOSSY, and only in one direction. The consumer now returns a CONTINUOUS fraction, so
    # binning to three levels cannot round-trip. What the shim preserves exactly is what
    # the metrics actually test: "did a drink/eat happen at all" (membership of DRINK_IDS /
    # EAT_IDS) and "was it a full eat" (== FULL_EAT_ID). The 0.5-vs-0.25 split is a nearest
    # -bin approximation and carries no meaning beyond it -- do not read
    # `half_eat`/`quarter_eat` distinctions off this array.
    _drink_f = np.asarray(lifetime_stats["drink_frac"], dtype=np.float32)
    _eat_f = np.asarray(lifetime_stats["eat_frac"], dtype=np.float32)
    _move_d = np.asarray(lifetime_stats["move_dir"], dtype=np.int32)

    _FULL_EPS = 1e-6
    action_T = np.zeros(len(_drink_f), dtype=np.int32)  # default WAIT

    _moved = _move_d >= 0
    action_T[_moved] = hex_world.MOVE_OFFSET + _move_d[_moved]

    def _frac_to_level(f):
        """1.0 -> level 0 (full), else nearest of {0.5, 0.25} -> level 1 / 2."""
        lvl = np.where(f >= 1.0 - _FULL_EPS, 0,
                       np.where(np.abs(f - 0.5) <= np.abs(f - 0.25), 1, 2))
        return lvl.astype(np.int32)

    _drinking = _drink_f > 0.0
    action_T[_drinking] = 1 + _frac_to_level(_drink_f[_drinking])

    # drink and eat are mutually exclusive per QueuedAction (compose_to_queued_action
    # dispatches on tile type), so ordering here cannot mask a simultaneous consume
    _eating = _eat_f > 0.0
    action_T[_eating] = 4 + _frac_to_level(_eat_f[_eating])
    satiation_T = np.asarray(lifetime_stats["satiation"], dtype=np.float32)
    death_T = np.asarray(lifetime_stats["dead"], dtype=np.int32)

    return {
        "eval_boundary": EB,

        "comfort_T": comfort_T,
        "hydration_T": hydration_T,
        "satiation_T": satiation_T,
        "death_T": death_T,

        "abstract_action_T": np.asarray(lifetime_stats["abstract_action"], dtype=np.int32),
        # legacy-scheme shim so sweep_fn_v5.compute_eval_metrics works; see construction
        # above for what it preserves exactly and what it approximates
        "action_T": action_T,
        "drink_frac_T": np.asarray(lifetime_stats["drink_frac"], dtype=np.float32),
        "eat_frac_T": np.asarray(lifetime_stats["eat_frac"], dtype=np.float32),
        "move_dir_T": np.asarray(lifetime_stats["move_dir"], dtype=np.int32),
        "consumer_slot_T": np.asarray(lifetime_stats["consumer_slot"], dtype=np.int32),
        "brightness_T": np.asarray(lifetime_stats["brightness"], dtype=np.float32),
        "coordinates_T": np.asarray(lifetime_stats["coord"], dtype=object),

        "mean_comfort": float(comfort_T[EB:].mean()),
        "min_comfort": float(comfort_T[EB:].min()),
        "std_comfort": float(comfort_T[EB:].std()),
        "mean_hydration": float(hydration_T[EB:].mean()),
        "mean_satiation": float(satiation_T[EB:].mean()),

        "death_count": int(death_T.sum()),
        "death_rate": float(death_T.mean()),
        "death_count_eval": int(death_T[EB:].sum()),
        "death_rate_eval": float(death_T[EB:].mean()),

        "ticks_at_water_eval": int(eval_water_ticks),
        "ticks_at_food_eval": int(eval_food_ticks),

        "curriculum_mode": curriculum_mode,
        "decay_band": tuple(decay_band),
        "n_train_maps": (curr.n_train if curr is not None else 1),
        "n_eval_maps": (curr.n_eval if curr is not None else 1),
        "eval_segments": eval_segments,
        "life_cap": life_cap,
        "terminals_per_map": terminals_per_map,
        "n_timeouts": int(n_timeouts),
        "n_timeouts_eval": int(n_timeouts_eval),

        "train_wf_trips": int(train_wf_trips),
        "train_fw_trips": int(train_fw_trips),

        "death_events": death_events,

        "water_coords": env.water_coords,
        "food_coords": env.food_coords,

        "module_spec": module_spec,
        "oracle_params": oracle_params,

        "decay_mult": float(decay_mult),

        "explorer_type": type(modules["explorer"]).__name__,
        "smell_radius": int(smell_radius),
        "explorer_type": type(modules["explorer"]).__name__,
        "explorer_stats": {
            "n_calls": getattr(modules["explorer"], "n_calls", None),
            "n_persist": getattr(modules["explorer"], "n_persist", None),
            "n_avoid_reverse": getattr(modules["explorer"], "n_avoid_reverse", None),
            "n_random_turns": getattr(modules["explorer"], "n_random_turns", None),
            "n_smell_follow": getattr(modules["explorer"], "n_smell_follow", None),
            "n_smell_reverse": getattr(modules["explorer"], "n_smell_reverse", None),
        },
    }