import numpy as np
import random
from collections import deque
import torch

pi = np.pi

import drqn as dqn
import hex_world_cached as hex_world
import oracle_v2 as oracle
import map_curriculum as mc


def sim_instance(
        seed=None,
        model_type="noisy_DQN",
        n_step=10,
        sigma_0=0.5,
        curriculum_mode="band",         
        comfort_surface="exponential",
        random_start="targeted",
        senses=("vision", "smell", "touch_memory"),
        midpoint_probe=False,
        novelty_rewards=True,
        beta=0.05,
        beta_floor=0.0,
        sim_len=500000,
        env_kwargs=None,
        hmax=3,
        smax=3,
        memory_len=10,
        replay_archive_len=50000,
        n_hidden=64,
        over_w=0.02,
        under_w=0.5,
        epsilon_start=0.3,
        gamma=0.99,
        alpha=0.01,
        death_penalty_k=0.5,
        batch_size=512,
        update_ticks=500,
        replay_warmup=500,
        learn_every=20,
        drqn_burn_in=5,
        drqn_learn_len=20,
        eval_len=20000,
        # --- proto 05 curriculum / cadence ---
        c_min=2,
        c_max=9,                        
        band_width=2,
        curriculum_ramp_frac=0.6,
        curriculum_sampling="uniform",   
        terminals_per_map=1,             
        life_cap=1000,        
        ez_zeta=0.0,
        sil_frac=0.0,             
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    EB = sim_len - eval_len

    if env_kwargs is None:
        raise ValueError("env_kwargs must be provided")

    if midpoint_probe and curriculum_mode != "disabled":
        raise ValueError(
            "midpoint_probe is not curriculum-aware (the pool is built once on map 0). "
            "Use curriculum_mode='disabled', or recompute the pool inside bind_env."
        )

    train_wf_trips = 0
    train_fw_trips = 0
    last_resource = None

    n_timeouts = 0
    n_timeouts_eval = 0
    eval_water_ticks = 0
    eval_food_ticks = 0
    eval_segments = []
    train_terminal_count = 0
    life_t = 0

    register = {
        "vanilla_DQN": {
            "fn_make_model": dqn.vanilla_make_model,
            "fn_make_target": dqn.vanilla_make_target,
            "fn_sync_target": dqn.vanilla_sync_target,
            "fn_learn_step": dqn.vanilla_learn_step,
            "fn_select_action": dqn.vanilla_select_action,
        },
        "double_DQN": {
            "fn_make_model": dqn.vanilla_make_model,
            "fn_make_target": dqn.vanilla_make_target,
            "fn_sync_target": dqn.vanilla_sync_target,
            "fn_learn_step": dqn.double_learn_step,
            "fn_select_action": dqn.vanilla_select_action,
        },
        "oracle": {
            "fn_make_model": oracle.make_model,
            "fn_make_target": oracle.make_target,
            "fn_sync_target": oracle.sync_target,
            "fn_learn_step": oracle.learn_step,
            "fn_select_action": oracle.select_action,
        },
        "noisy_DQN": {
            "fn_make_model": dqn.noisy_make_model,
            "fn_make_target": dqn.noisy_make_target,
            "fn_sync_target": dqn.noisy_sync_target,
            "fn_learn_step": dqn.noisy_learn_step,
            "fn_select_action": dqn.vanilla_select_action,
        },
        "drqn_DQN": {
            "fn_make_model": dqn.drqn_make_model,
            "fn_make_target": dqn.drqn_make_target,
            "fn_sync_target": dqn.drqn_sync_target,
            "fn_learn_step": dqn.drqn_learn_step,
            "fn_select_action": dqn.drqn_select_action,
            "replay_mode": "sequence",
        },
        "noisy_drqn_DQN": {
            "fn_make_model": dqn.noisy_drqn_make_model,
            "fn_make_target": dqn.noisy_drqn_make_target,
            "fn_sync_target": dqn.drqn_sync_target,
            "fn_learn_step": dqn.drqn_learn_step,
            "fn_select_action": dqn.drqn_select_action,
            "replay_mode": "sequence",
        },
    }

    ################### world construction #############################
    if curriculum_mode == "disabled":
        curr = None
        env = hex_world.HexWorld(seed=seed, **env_kwargs)
        current_band = tuple(env_kwargs.get("band", (9, 11)))
        decay_band = current_band
    elif curriculum_mode == "band":
        curr = mc.MapCurriculum(
            seed=seed, base_env_kwargs=env_kwargs, EB=EB,
            c_min=c_min, c_max=c_max, band_width=band_width,
            ramp_frac=curriculum_ramp_frac, sampling=curriculum_sampling,
        )
        env, current_band, _ = curr.build(t=0, eval_mode=False)
        decay_band = (c_max, c_max + band_width)
    else:
        raise ValueError(f"unknown curriculum_mode {curriculum_mode!r}")

    water_set = set(map(tuple, env.water_coords))
    food_set = set(map(tuple, env.food_coords))

    def hex_dist(a, b):
        aq, ar = a
        bq, br = b
        return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))

    midpoint_pool = None
    if midpoint_probe:
        MID_TOL = 2
        midpoint_pool = []
        for f in map(tuple, env.food_coords):
            w = min(map(tuple, env.water_coords), key=lambda w: hex_dist(f, w))
            d_fw = hex_dist(f, w)
            for c in env.coords:
                df, dw = hex_dist(c, f), hex_dist(c, w)
                if abs(df - dw) <= MID_TOL and (df + dw) <= d_fw + MID_TOL:
                    midpoint_pool.append(tuple(c))
        if not midpoint_pool:
            midpoint_pool = list(env._spawn_pool)

    coord_to_idx = env.coord_to_idx
    novelty_N = np.zeros(len(env.coords), dtype=np.int64)

    r_model = register[model_type]
    fn_make_model = r_model["fn_make_model"]
    fn_make_target = r_model["fn_make_target"]
    fn_sync_target = r_model["fn_sync_target"]
    fn_learn_step = r_model["fn_learn_step"]
    fn_select_action = r_model["fn_select_action"]
    replay_mode = r_model.get("replay_mode", "regular")

    if model_type == "oracle":
        oracle.configure(
            water_coords=env.water_coords, food_coords=env.food_coords,
            fill_h=1.7, fill_s=1.7,
        )

    def bind_env(new_env, reset_novelty):
        # refresh every map-derived binding on resample. radius is fixed, so the
        # grid (coord_to_idx, novelty size) is invariant — only placement changes.
        nonlocal env, water_set, food_set, coord_to_idx
        assert len(new_env.coords) == len(novelty_N), "grid size changed across maps"
        env = new_env
        water_set = set(map(tuple, env.water_coords))
        food_set = set(map(tuple, env.food_coords))
        coord_to_idx = env.coord_to_idx
        if reset_novelty:
            novelty_N[:] = 0
        if model_type == "oracle":
            oracle.configure(
                water_coords=env.water_coords, food_coords=env.food_coords,
                fill_h=1.7, fill_s=1.7,
            )

    def record_eval_segment(t_global):
        eval_segments.append({
            "t_start": int(t_global),
            "band": tuple(current_band),
            "water_coords": [tuple(c) for c in env.water_coords],
            "food_coords": [tuple(c) for c in env.food_coords],
        })

    eps_start = epsilon_start
    eps_end = 0.05
    eps_decay_frac = 0.7
    eps_decay_ticks = max(1, int(eps_decay_frac * EB))

    ideal_h = 1
    ideal_s = 1
    day_length = 50

    h = ideal_h
    s = ideal_s

    drink_amount = 0.15
    eat_amount = 0.3

    action_effects = np.array([
        [0.0, 0.0, None],
        [1.0, 0.0, None],
        [0.5, 0.0, None],
        [0.25, 0.0, None],
        [0.0, 1.0, None],
        [0.0, 0.5, None],
        [0.0, 0.25, None],
        [0.0, 0.0, 0],
        [0.0, 0.0, 1],
        [0.0, 0.0, 2],
        [0.0, 0.0, 3],
        [0.0, 0.0, 4],
        [0.0, 0.0, 5],
    ], dtype=object)

    positions1 = np.arange(1, 8)
    weights1 = (11 - positions1) ** 2
    proportions1 = weights1 / weights1.sum()

    positions2 = np.arange(1, 11)
    proportions2 = positions2 / positions2.sum()

    r_que = deque(maxlen=replay_archive_len)

    wf_traj_window = deque(maxlen=40)   # rolling recent transitions
    success_archive = deque(maxlen=2000) # non-evicting 

    m_que_d = deque([0] * memory_len, maxlen=memory_len)
    m_que_e = deque([0] * memory_len, maxlen=memory_len)
    m_que_m = deque([-1] * memory_len, maxlen=memory_len)

    n_step_que = deque(maxlen=n_step)

    a_que = deque(maxlen=11) # action que

    # simple memory: last-seen resource coords (q, r, seen-flag), radius-scaled
    last_water_seen = None   # tuple (q, r) or None
    last_food_seen = None

    def sample_midpoint_start():
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            rad = np.sqrt(np.random.uniform(0.1 ** 2, 2.9 ** 2))
            nh = ideal_h + rad * np.cos(angle)
            ns = ideal_s + rad * np.sin(angle)
            if (0.05 < nh < hmax) and (0.05 < ns < smax) and not (nh < 0.35 and ns < 0.35):
                return float(nh), float(ns), midpoint_pool[np.random.randint(len(midpoint_pool))]

    death_penalty = -death_penalty_k / (1 - gamma)

    N_ACT = len(action_effects)

    if senses is None:
        senses = set()
    elif isinstance(senses, str):
        senses = {x.strip().lower() for x in senses.split(",")}
    else:
        senses = {x.strip().lower() for x in senses}

    BASE_INPUT = 6 + memory_len * 3

    VISION_FEATURES = 0
    SMELL_FEATURES = 0
    MEMORY_FEATURES = 0
    if "vision" in senses:
        VISION_FEATURES = 2 + 6 * 3
    if "smell" in senses:
        SMELL_FEATURES = 1
    if "touch_memory" in senses:
        MEMORY_FEATURES = 6

    N_INPUT = BASE_INPUT + VISION_FEATURES + SMELL_FEATURES + MEMORY_FEATURES

    carry_x = ()
    carry_act = 0

    drqn_seq_len = drqn_burn_in + drqn_learn_len
    seq_replay = dqn.SequenceReplay(max_episodes=replay_archive_len, seq_len=drqn_seq_len)
    drqn_hidden = None

    if model_type in ("noisy_DQN", "noisy_drqn_DQN"):
        model, optimiser = fn_make_model(N_INPUT, n_hidden, N_ACT, lr=alpha, sigma_0=sigma_0)
    else:
        model, optimiser = fn_make_model(N_INPUT, n_hidden, N_ACT, lr=alpha)

    target = fn_make_target(model)

    if model_type == "oracle":
        model.set_context(coord=tuple(env.coord), hydration=h, satiation=s)

    lifetime_stats = [[], [], [], [], [], [], [], [], [], []]
    death_events = []
    sigma_w_mean = []

    def make_agent_state(h, s, b):
        base = env.make_state(h, s, b, m_que_d, m_que_e, m_que_m)
        parts = [np.asarray(base, dtype=np.float32)]
        if "vision" in senses:
            parts.append(env.get_vision_features())
        if "smell" in senses:
            parts.append(np.array([env.get_food_smell(radius=3)], dtype=np.float32))
        if "touch_memory" in senses:
            cq, cr = env.coord            # agent's current position
            R = env.radius
            if last_water_seen is None:
                wq, wr, wf = 0., 0., 0.
            else:
                wq, wr, wf = (last_water_seen[0]-cq)/R, (last_water_seen[1]-cr)/R, 1.
            if last_food_seen is None:
                fq, fr, ff = 0., 0., 0.
            else:
                fq, fr, ff = (last_food_seen[0]-cq)/R, (last_food_seen[1]-cr)/R, 1.
            parts.append(np.array([wq, wr, wf, fq, fr, ff], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def get_brightness(time, day_len=100):
        a = 0.5
        b = 0.3
        c = (2 * pi) / day_len
        brightness = a + b * np.sin(c * time)
        brightness += np.random.normal(0, 0.05)
        return min(1, max(0, brightness))

    effective_size = env.radius / 2 + 1
    hydration_decay_scaling = 0.05 + 1.45 / ((1 + 1.0426 * (effective_size - 1)) ** 0.7122)
    satiation_decay_scaling = 0.8 * hydration_decay_scaling

    def decay_hydration(hydration, satiation, brightness):
        decay = max(0, (0.15 * brightness) - (0.03 * satiation))
        decay += np.random.normal(0.05, 0.03)
        return hydration - decay * hydration_decay_scaling

    def decay_satiation(satiation, hydration, brightness):
        decay = max(0, ((0.05 - 0.05 * brightness) + (0.04 - 0.04 * hydration) + 0.1 * (ideal_h - hydration)))
        decay += np.random.normal(0.01, 0.005)
        return satiation - decay * satiation_decay_scaling

    def the_meaning_of_life_exp(hydration, satiation):
        h_over = max(0, hydration - ideal_h)
        h_under = min(0, hydration - ideal_h)
        s_over = max(0, satiation - ideal_s)
        s_under = min(0, satiation - ideal_s)
        d2 = (under_w * h_under ** 2 + over_w * h_over ** 2 +
              under_w * s_under ** 2 + over_w * s_over ** 2)
        return 2 * np.exp(-3 * d2) - 1

    def the_meaning_of_life_quad(hydration, satiation):
        h_over = max(0, hydration - ideal_h)
        h_under = min(0, hydration - ideal_h)
        s_over = max(0, satiation - ideal_s)
        s_under = min(0, satiation - ideal_s)
        d2 = (under_w * h_under ** 2 + over_w * h_over ** 2 +
              under_w * s_under ** 2 + over_w * s_over ** 2)
        return 1 - d2

    if comfort_surface == "quadratic":
        the_meaning_of_life = the_meaning_of_life_quad
    elif comfort_surface == "exponential":
        the_meaning_of_life = the_meaning_of_life_exp

    def sample_box(h_lo, h_hi, s_lo, s_hi):
        return np.random.uniform(h_lo, h_hi), np.random.uniform(s_lo, s_hi)

    def coord_at_dist(anchor, d):
        ring = [c for c in env.coords if hex_dist(c, anchor) == d]
        if not ring:
            ring = [c for c in env.coords if hex_dist(c, anchor) <= max(1, d)]
        return ring[np.random.randint(len(ring))]

    def sample_uniform_start():
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            rad = np.sqrt(np.random.uniform(0.1 ** 2, 2.9 ** 2))
            nh = ideal_h + rad * np.cos(angle)
            ns = ideal_s + rad * np.sin(angle)
            if (0.05 < nh < hmax) and (0.05 < ns < smax) and not (nh < 0.35 and ns < 0.35):
                pool = env._spawn_pool
                return float(nh), float(ns), tuple(pool[np.random.randint(len(pool))])

    targeted_names = ["hungry_food", "hungry_water", "thirsty_water", "thirsty_food", "overfull", "both_low"]
    targeted_weights = np.array([3, 3, 1, 1, 1, 1], dtype=float)
    targeted_weights /= targeted_weights.sum()

    def sample_targeted_start(reach):
        pick = targeted_names[np.random.choice(len(targeted_names), p=targeted_weights)]
        if pick == "hungry_food":
            h0, s0 = sample_box(0.9, 1.3, 0.15, 0.45)
            anchor = "food"
        elif pick == "hungry_water":
            h0, s0 = sample_box(0.9, 1.3, 0.15, 0.45)
            anchor = "water"
        elif pick == "thirsty_water":
            h0, s0 = sample_box(0.15, 0.45, 0.9, 1.3)
            anchor = "water"
        elif pick == "thirsty_food":
            h0, s0 = sample_box(0.15, 0.45, 0.9, 1.3)
            anchor = "food"
        elif pick == "overfull":
            h0, s0 = sample_box(1.3, 1.7, 1.3, 1.7)
            anchor = None
        else:
            h0, s0 = sample_box(0.4, 0.7, 0.4, 0.7)
            anchor = None

        if anchor is None:
            coord = random.choice(env.coords)
        else:
            if anchor == "food":
                anchor = random.choice(env.food_coords)
            elif anchor == "water":
                anchor = random.choice(env.water_coords)
            coord = coord_at_dist(anchor, np.random.randint(0, reach + 1))

        return float(h0), float(s0), tuple(coord)

    def reach_at(t):
        return 1 + int((t / EB) * env.radius)

    def targeted_prob(t):
        end = 0.8 * EB
        return 0.0 if t >= end else 0.8 * (1 - t / end)


    def respawn(t, eval_mode):
        # shared respawn for every terminal death OR timeout.
        nonlocal h, s, life_t, last_resource, drqn_hidden, carry_x
        nonlocal last_water_seen, last_food_seen
        nonlocal ez_repeat_left, ez_repeat_action

        use_targeted = (random_start == "targeted" and not eval_mode
                        and np.random.uniform() < targeted_prob(t))
        if use_targeted:
            h, s, new_coord = sample_targeted_start(reach_at(t))
        elif eval_mode and midpoint_probe:
            h, s, new_coord = sample_midpoint_start()
        else:
            h, s, new_coord = sample_uniform_start()
        env.reset_position(coord=new_coord)
        life_t = 0
        a_que.clear()
        m_que_d.clear(); m_que_e.clear(); m_que_m.clear()
        m_que_d.extend([0] * memory_len)
        m_que_e.extend([0] * memory_len)
        m_que_m.extend([-1] * memory_len)
        last_resource = None
        last_water_seen = None                             
        last_food_seen = None
        ez_repeat_left = 0                                
        ez_repeat_action = None                               
        if replay_mode == "sequence":
            drqn_hidden = None

    # εz-greedy temporally-extended exploration state
    ez_repeat_left = 0
    ez_repeat_action = None

    def sample_ez_duration():
        # zeta(μ): P(n) ∝ n^(-μ) + Cap
        n = int(np.random.zipf(ez_zeta))
        return min(n, 2 * c_max)     # cap at ~2× the hardest commute
    
    ez_runs = []          # (start_coord, end_coord, length) per completed repeat
    _ez_run_start = None
    _ez_run_len = 0

####################################################################################################################
    for t in range(sim_len):#########################} SIM LOOP {###################################################
        if t == EB:#################################################################################################
            model.eval()
            if replay_mode == "sequence":
                drqn_hidden = None
            if curr is not None:
                # held-out evaluation: switch to the disjoint eval map stream
                new_env, current_band, _ = curr.build(t=t, eval_mode=True)
                bind_env(new_env, reset_novelty=False)
                respawn(t, eval_mode=True)
            record_eval_segment(t)

        training = True if t < EB else False

        if t % 50_000 == 0:
            n_maps = curr.n_train if curr is not None else 1
            print(f"    [seed {seed}] t={t} deaths={int(np.sum(lifetime_stats[8]))} "
                  f"timeouts={n_timeouts} maps={n_maps} wf={train_wf_trips} fw={train_fw_trips}",
                  flush=True)

        life_t += 1

        b = get_brightness(t, day_length)
        h = decay_hydration(h, s, b)
        s = decay_satiation(s, h, b)

        for age, act in enumerate(a_que):
            drink_choice, eat_choice, move_choice = action_effects[act]
            if age == 0:
                env.apply_action_movement(act)
            if age < len(proportions1):
                h += drink_amount * drink_choice * proportions1[age] * (0.8 + 0.3 * s)
            if age < len(proportions2):
                s += eat_amount * eat_choice * proportions2[age]

        h = min(hmax, max(0, h))
        s = min(smax, max(0, s))

        cur_comfort = the_meaning_of_life(h, s)
        cur_dead = int((h <= 0.05) or (s <= 0.05))

        timeout = (curr is not None) and (life_cap is not None) and (life_t >= life_cap) and (cur_dead == 0)
        terminal = bool(cur_dead) or bool(timeout)

        done_for_target = bool(cur_dead)
        is_boundary = bool(terminal)

        if timeout:
            if training:
                n_timeouts += 1
            else:
                n_timeouts_eval += 1

        novelty_r = 0
        if novelty_rewards and training:
            beta_t = beta_floor + (beta - beta_floor) * (1.0 - t / EB)
            novelty_idx = coord_to_idx[tuple(env.coord)]
            novelty_r = beta_t / np.sqrt(novelty_N[novelty_idx] + 1)
            novelty_N[novelty_idx] += 1

        cur_reward = cur_comfort + novelty_r + death_penalty * cur_dead

        if cur_dead and t != 0:
            if h <= 0.05 and s <= 0.05:
                death_cause = "both"
            elif h <= 0.05:
                death_cause = "hydration"
            else:
                death_cause = "satiation"
            death_events.append({"t": t, "cause": death_cause, "h": h, "s": s})

        cur_coord = tuple(env.coord)
        on_water = cur_coord in water_set
        on_food = cur_coord in food_set

        if on_water:
            last_water_seen = cur_coord
            if training:
                wf_traj_window.clear()     # start a fresh crossing record
        if on_food:
            last_food_seen = cur_coord
        
        if training:
            if on_water:
                if last_resource == "food":
                    train_fw_trips += 1
                last_resource = "water"
            elif on_food:
                if last_resource == "water":
                    train_wf_trips += 1
                    for tup in wf_traj_window:       # the crossing that just completed
                        success_archive.append(tup)
                last_resource = "food"
        else:
            cc = tuple(env.coord)
            if cc in water_set:
                eval_water_ticks += 1
            if cc in food_set:
                eval_food_ticks += 1

        cur_x = make_agent_state(h, s, b)
        cur_mask = env.get_action_mask()

        if t != 0 and training and model_type != "oracle":

            if replay_mode == "sequence": # DRQN IS NOT HANDLING TIMEOUT, NEED EXTRA FLAG
                seq_replay.append(carry_x, carry_act, cur_reward, cur_x, is_boundary, cur_mask)

                if t % learn_every == 0 and len(seq_replay) >= replay_warmup:
                    if t % (update_ticks * learn_every) == 0:
                        fn_sync_target(target, model)
                    batch = seq_replay.sample(min(batch_size, len(seq_replay)))
                    fn_learn_step(model, target, optimiser, batch, gamma,
                                  burn_in=drqn_burn_in, n_step=n_step)

            else:
                n_step_que.append([carry_x, carry_act, cur_reward, cur_x,
                                   done_for_target, cur_mask, is_boundary])

                while n_step_que:
                    state_0 = list(n_step_que[0][:6])
                    states_added = 0
                    state_0[2] = 0
                    completed_state = False

                    for i, transition in enumerate(n_step_que.copy()):
                        state_0[2] += transition[2] * gamma ** i
                        for n in [3, 4, 5]:
                            state_0[n] = transition[n]
                        states_added += 1
                        completed_state = (states_added == n_step) or transition[6]
                        if completed_state:
                            n_step_que.popleft()
                            break

                    if completed_state:
                        packed = tuple(state_0 + [states_added]) 
                        r_que.append(packed)
                        if training:
                            wf_traj_window.append(packed)
                    else:
                        break

                if t % learn_every == 0 and len(r_que) >= replay_warmup:
                    if t % (update_ticks * learn_every) == 0:
                        fn_sync_target(target, model)
                    main_k = batch_size
                    if success_archive and sil_frac > 0.0:
                        n_sil = min(int(batch_size * sil_frac), len(success_archive))
                        main_k = batch_size - n_sil
                        batch = random.sample(r_que, min(main_k, len(r_que))) \
                                + random.sample(list(success_archive), n_sil)
                    else:
                        batch = random.sample(r_que, min(batch_size, len(r_que)))
                    fn_learn_step(model, target, optimiser, batch, gamma)

        log_coord = tuple(env.coord)
        log_h = h
        log_s = s

        if terminal: # archive is NOT cleared
            if curr is not None:
                if training:
                    train_terminal_count += 1
                    if train_terminal_count % terminals_per_map == 0:
                        new_env, current_band, _ = curr.build(t=t, eval_mode=False)
                        bind_env(new_env, reset_novelty=True)
                else:
                    new_env, current_band, _ = curr.build(t=t, eval_mode=True)
                    bind_env(new_env, reset_novelty=False)
                    record_eval_segment(t)

            respawn(t, eval_mode=(t >= EB))
            cur_x = make_agent_state(h, s, b)
            cur_mask = env.get_action_mask()

        if training:
            frac = min(1.0, t / eps_decay_ticks)
            epsilon_t = eps_start + frac * (eps_end - eps_start)
        else:
            epsilon_t = 0.0

        if model_type == "oracle":
            model.set_context(coord=tuple(env.coord), hydration=h, satiation=s)
            action = fn_select_action(model, cur_x, cur_mask)
        else:
            if replay_mode == "sequence":
                greedy_action, drqn_hidden = fn_select_action(model, cur_x, cur_mask, drqn_hidden)
                if np.random.uniform(0, 1) < epsilon_t:
                    valid_actions = np.flatnonzero(cur_mask)
                    action = int(np.random.choice(valid_actions))
                else:
                    action = greedy_action
            else:
                if ez_repeat_left > 0 and ez_repeat_action is not None and cur_mask[ez_repeat_action]:
                    action = ez_repeat_action
                    ez_repeat_left -= 1
                    _ez_run_len += 1
                    if ez_repeat_left == 0:                      # run just ended
                        ez_runs.append((_ez_run_start, tuple(env.coord), _ez_run_len))
                elif np.random.uniform(0, 1) < epsilon_t:
                    valid_actions = np.flatnonzero(cur_mask)
                    action = int(np.random.choice(valid_actions))
                    if ez_zeta > 1.0:
                        ez_repeat_action = action
                        ez_repeat_left = sample_ez_duration() - 1
                        _ez_run_start = tuple(env.coord)         # mark run origin
                        _ez_run_len = 1
                else:
                    action = fn_select_action(model, cur_x, cur_mask)
                    ez_repeat_left = 0
                    ez_repeat_action = None

        a_que.appendleft(action)

        drink_memory, eat_memory, move_memory = action_effects[action]
        m_que_d.appendleft(float(drink_memory))
        m_que_e.appendleft(float(eat_memory))
        m_que_m.appendleft(-1 if move_memory is None else int(move_memory))

        carry_x = cur_x
        carry_act = action

        lifetime_stats[0].append(cur_comfort)
        lifetime_stats[1].append(action)
        lifetime_stats[3].append(log_h)
        lifetime_stats[4].append(log_s)
        lifetime_stats[5].append(b)
        lifetime_stats[7].append(cur_reward)
        lifetime_stats[8].append(cur_dead)
        lifetime_stats[9].append(log_coord)

        if model_type in ("noisy_DQN", "noisy_drqn_DQN") and t % learn_every == 0 and training:
            sigma_w_mean.append(model.fc2.sigma_w.abs().mean().item())

    lifetime_comfort = np.array(lifetime_stats[0])
    lifetime_action = np.array(lifetime_stats[1])
    lifetime_coordinates = np.array(lifetime_stats[9])
    lifetime_hydration = np.array(lifetime_stats[3])
    lifetime_satiation = np.array(lifetime_stats[4])
    death_T = np.array(lifetime_stats[8])

    return {
        "eval_boundary": EB,

        "comfort_T": lifetime_comfort,
        "action_T": lifetime_action,
        "coordinates_T": lifetime_coordinates,
        "hydration_T": lifetime_hydration,
        "satiation_T": lifetime_satiation,
        "death_T": death_T,

        "mean_comfort": lifetime_comfort[EB:].mean(),
        "min_comfort": lifetime_comfort[EB:].min(),
        "std_comfort": lifetime_comfort[EB:].std(),
        "mean_hydration": lifetime_hydration[EB:].mean(),
        "mean_satiation": lifetime_satiation[EB:].mean(),

        "ticks_at_water_eval": eval_water_ticks,
        "ticks_at_food_eval": eval_food_ticks,

        "death_count": int(death_T.sum()),
        "death_rate": float(death_T.mean()),
        "death_count_eval": int(death_T[EB:].sum()),
        "death_rate_eval": float(death_T[EB:].mean()),

        "n_timeouts": n_timeouts,
        "n_timeouts_eval": n_timeouts_eval,

        "sigma_w_mean": sigma_w_mean,

        "water_coords": env.water_coords,
        "food_coords": env.food_coords,
        "eval_segments": eval_segments,

        "senses": tuple(sorted(senses)),
        "n_input": N_INPUT,

        "curriculum_mode": curriculum_mode,
        "decay_band": tuple(decay_band),
        "n_train_maps": (curr.n_train if curr is not None else 1),
        "n_eval_maps": (curr.n_eval if curr is not None else 1),

        "train_wf_trips": train_wf_trips,
        "train_fw_trips": train_fw_trips,

        "ez_runs": ez_runs,

        "success_archive_size": len(success_archive)
    }