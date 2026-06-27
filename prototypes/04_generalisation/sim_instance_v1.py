import numpy as np 
import random
from collections import deque
import torch

pi = np.pi

import drqn as dqn
import hex_world_procedural_senses as hex_world
import oracle_v2 as oracle

def sim_instance(
        seed=None,
        model_type="vanilla_DQN",
        n_step = 10,
        sigma_0= 0.5,
        curriculum_mode="disabled",
        comfort_surface="exponential",
        random_start="targeted",
        senses=("vision", "smell"),
        midpoint_probe=False,
        novelty_rewards=True,
        beta=0.05,
        sim_len=500000,
        env_kwargs=None,
        hmax = 3,
        smax = 3,
        memory_len=10,
        replay_archive_len=5000,
        n_hidden=64,
        over_w = 0.02,
        under_w = 0.5,
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
        eval_len=20000
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    EB = sim_len - eval_len

    train_wf_trips = 0          # water→food completions during training
    train_fw_trips = 0          # food→water completions during training
    last_resource = None        # "water" / "food" / None

    if env_kwargs is None:
        raise ValueError("env_kwargs must be provided — no default map for proto 04")


    register = {
        "vanilla_DQN" : {
            "fn_make_model" : dqn.vanilla_make_model,
            "fn_make_target" : dqn.vanilla_make_target,
            "fn_sync_target" : dqn.vanilla_sync_target,
            "fn_learn_step" : dqn.vanilla_learn_step,
            "fn_select_action" : dqn.vanilla_select_action
        },

        "double_DQN" : {
            "fn_make_model" : dqn.vanilla_make_model,
            "fn_make_target" : dqn.vanilla_make_target,
            "fn_sync_target" : dqn.vanilla_sync_target,
            "fn_learn_step" : dqn.double_learn_step,
            "fn_select_action" : dqn.vanilla_select_action
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
    
    curriculum = {
        "disabled" : [
            {},
        ],

        "coordinate" : [
            {"water_coord": (-2, 0), "food_coord": (0, 2)},
            {"water_coord": (-3, 0), "food_coord": (0, 3)},
            {"water_coord": (-4, 0), "food_coord": (0, 4)},
            {"water_coord": (-5, 0), "food_coord": (0, 5)},
        ]
    }

    phase_i = -99
    phase_edges = np.linspace(0, EB, len(curriculum[curriculum_mode]) + 1, dtype=int)[1:-1]
    base_env_kwargs = env_kwargs.copy()

    def curriculum_phase(t, curr_env, phase_index):
        new_phase_i = int(np.digitize(t, phase_edges))

        need_build = curr_env is None
        need_phase_change = (new_phase_i != phase_index and curriculum_mode != "disabled")

        if need_build or need_phase_change:
            phase = curriculum[curriculum_mode][new_phase_i]

            current_env_kwargs = {}
            for k, v in (base_env_kwargs | phase).items():
                if k != "seed":
                    current_env_kwargs[k] = v
            new_env = hex_world.HexWorld(seed=seed, **current_env_kwargs)

            return new_env, new_phase_i, (not need_build)

        return curr_env, phase_index, False

    env, phase_i, _ = curriculum_phase(t=0, curr_env=None, phase_index=phase_i)

    water_set = set(map(tuple, env.water_coords))
    food_set = set(map(tuple, env.food_coords))
    
    def hex_dist(a, b):
        aq, ar = a
        bq, br = b
        return max(abs(aq - bq), abs(ar - br), abs((-aq - ar) - (-bq - br)))

    MID_TOL = 2
    midpoint_pool = []
    for f in map(tuple, env.food_coords):
        w = min(map(tuple, env.water_coords), key=lambda w: hex_dist(f, w))  # parent = nearest
        d_fw = hex_dist(f, w)
        for c in env.coords:
            df, dw = hex_dist(c, f), hex_dist(c, w)
            if abs(df - dw) <= MID_TOL and (df + dw) <= d_fw + MID_TOL:  # on the line, halfway
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
            water_coords=env.water_coords,
            food_coords=env.food_coords,
            fill_h=1.7,
            fill_s=1.7,
        )

    eps_start = epsilon_start
    eps_end = 0.05
    full_phase_edges = np.concatenate([[0], phase_edges, [EB]])
    eps_decay_frac = 0.7
    phase_start_t = 0
    cur_phase_decay_ticks = int(eps_decay_frac * (full_phase_edges[phase_i + 1] - full_phase_edges[phase_i]))

    ideal_h = 1
    ideal_s = 1
    day_length = 50 # ticks

    h = ideal_h
    s = ideal_s

    drink_amount = 0.15
    eat_amount = 0.3

    action_effects = np.array([
        [0.0, 0.0, None],   # 0 wait
        [1.0, 0.0, None],   # 1 drink full
        [0.5, 0.0, None],   # 2 drink half
        [0.25, 0.0, None],  # 3 drink quarter
        [0.0, 1.0, None],   # 4 eat full
        [0.0, 0.5, None],   # 5 eat half
        [0.0, 0.25, None],  # 6 eat quarter
        [0.0, 0.0, 0],      # 7 move E
        [0.0, 0.0, 1],      # 8 move NE
        [0.0, 0.0, 2],      # 9 move NW
        [0.0, 0.0, 3],      # 10 move W
        [0.0, 0.0, 4],      # 11 move SW
        [0.0, 0.0, 5],      # 12 move SE
    ], dtype=object)

    positions1 = np.arange(1, 8)
    weights1 = (11 - positions1) ** 2
    proportions1 = weights1 / weights1.sum()

    positions2 = np.arange(1, 11)
    ##weights2 = positions2 ** 2
    proportions2 = positions2 / positions2.sum()

    r_que = deque(maxlen=replay_archive_len)
    m_que_d = deque([0] * memory_len, maxlen=memory_len)
    m_que_e = deque([0] * memory_len, maxlen=memory_len)
    m_que_m = deque([-1] * memory_len, maxlen=memory_len) # -1 means no movement action remembered
    n_step_que = deque(maxlen=n_step)
    a_que = deque(maxlen=11)

    '''
    action_names = [
    "wait",
    "drink_full", "drink_half", "drink_quarter",
    "eat_full", "eat_half", "eat_quarter",
    "move_E", "move_NE", "move_NW", "move_W", "move_SW", "move_SE",
    ]
    '''

    def sample_midpoint_start():
        while True:
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.sqrt(np.random.uniform(0.1**2, 2.9**2))
            nh = ideal_h + rad*np.cos(angle)
            ns = ideal_s + rad*np.sin(angle)
            if (0.05 < nh < hmax) and (0.05 < ns < smax) and not (nh < 0.35 and ns < 0.35):
                return float(nh), float(ns), midpoint_pool[np.random.randint(len(midpoint_pool))]

    '''    
    print(len(midpoint_pool))
    print(sorted(min(hex_dist(c, w) for w in map(tuple, env.water_coords))
                for c in midpoint_pool)[:10])      # nearest-water dist for the 10 closest pool tiles
    print(sorted(min(hex_dist(c, f) for f in map(tuple, env.food_coords))
                for c in midpoint_pool)[:10])      # nearest-food dist
    ''' 

    death_penalty = -death_penalty_k / (1 - gamma)

    N_ACT = len(action_effects)

    if senses is None:
        senses = set()
    elif isinstance(senses, str):
        senses = {x.strip().lower() for x in senses.split(",")}
    else:
        senses = {x.strip().lower() for x in senses}

    BASE_INPUT = 6 + memory_len * 3     # state = q, r, cube_s, hydration, satiation, brightness, drink memory, eat memory, move memory

    VISION_FEATURES = 0
    SMELL_FEATURES = 0

    if "vision" in senses:
         #current tile water level + food level + 6 surrounding tiles × [legal, water level, food level]
        VISION_FEATURES = 2 + 6 * 3

    if "smell" in senses:
        # scalar proximity-to-food smell
        SMELL_FEATURES = 1

    N_INPUT = BASE_INPUT + VISION_FEATURES + SMELL_FEATURES

    carry_x = ()
    carry_act = 0

    drqn_seq_len = drqn_burn_in + drqn_learn_len
    seq_replay = dqn.SequenceReplay(
        max_episodes=replay_archive_len,
        seq_len=drqn_seq_len,
    )

    drqn_hidden = None

    if model_type in ("noisy_DQN", "noisy_drqn_DQN"):
        model, optimiser = fn_make_model(N_INPUT, n_hidden, N_ACT, lr=alpha, sigma_0=sigma_0)
    else:
        model, optimiser = fn_make_model(N_INPUT, n_hidden, N_ACT, lr=alpha) 
        
    target = fn_make_target(model)

    if model_type == "oracle":
        model.set_context(
            coord=tuple(env.coord),
            hydration=h,
            satiation=s,
        )

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

        return np.concatenate(parts).astype(np.float32)

    def get_brightness(time, day_len=100):
        a = 0.5
        b = 0.3
        c = (2*pi) / day_len

        brightness = a + b*np.sin(c*time)
        brightness += np.random.normal(0, 0.05)
        return min(1, max(0, brightness))
    
    band = env_kwargs.get("band", (9, 11))
    band_mid = sum(band) / 2          # 10
    effective_size = band_mid / 2 + 1  # maps band-10 commute onto the r=5 scaling → ~0.5
    hydration_decay_scaling = 0.05 + 1.45 / ((1 + 1.0426*(effective_size - 1))**0.7122)
    satiation_decay_scaling = 0.8*hydration_decay_scaling
    
    # print(band_mid, hydration_decay_scaling, satiation_decay_scaling)

    def decay_hydration(hydration, satiation, brightness):
        decay = max(0, (0.15*brightness) - (0.03*satiation))
        decay += np.random.normal(0.05, 0.03)
        return hydration - decay * hydration_decay_scaling

    def decay_satiation(satiation, hydration, brightness):
        decay = max(0, ((0.05 - 0.05*brightness) + (0.04 - 0.04*hydration) + 0.1*(ideal_h - hydration)))        
        decay += np.random.normal(0.01, 0.005)
        return satiation - decay * satiation_decay_scaling

    def the_meaning_of_life_exp(hydration, satiation):

        h_over = max(0, hydration - ideal_h)
        h_under = min(0, hydration - ideal_h)

        s_over = max(0, satiation - ideal_s)
        s_under = min(0, satiation - ideal_s)

        d2 = (
            under_w * h_under**2 +
            over_w  * h_over**2 +
            under_w * s_under**2 +
            over_w  * s_over**2
        )

        return 2 * np.exp(-3 * d2) - 1
    
    def the_meaning_of_life_quad(hydration, satiation):

        h_over = max(0, hydration - ideal_h)
        h_under = min(0, hydration - ideal_h)

        s_over = max(0, satiation - ideal_s)
        s_under = min(0, satiation - ideal_s)

        d2 = (
            under_w * h_under**2 +
            over_w  * h_over**2 +
            under_w * s_under**2 +
            over_w  * s_over**2
        )

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
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.sqrt(np.random.uniform(0.1**2, 2.9**2))
            nh = ideal_h + rad*np.cos(angle)
            ns = ideal_s + rad*np.sin(angle)
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

####################################################################################################################  
    for t in range(sim_len):#########################} SIM LOOP {###################################################
        if t == EB:#################################################################################################
            model.eval()

            if replay_mode == "sequence":
                drqn_hidden = None

        if t % 50_000 == 0:
            print(f"    [seed {seed}] t={t} deaths={int(np.sum(lifetime_stats[8]))} "
                  f"wf={train_wf_trips} fw={train_fw_trips}", flush=True)

        b = get_brightness(t, day_length)
        h = decay_hydration(h, s, b)
        s = decay_satiation(s, h, b)

        env, phase_i, new_phase = curriculum_phase(t=t, curr_env=env, phase_index=phase_i)

        if new_phase:
            env.reset_position()

            n_step_que.clear()

            r_que.clear()

            if replay_mode == "sequence":
                seq_replay.clear()
                drqn_hidden = None

            m_que_d.clear()
            m_que_e.clear()
            m_que_m.clear()

            m_que_d.extend([0] * memory_len)
            m_que_e.extend([0] * memory_len)
            m_que_m.extend([-1] * memory_len)

            a_que.clear()
            a_que.extend([0] * memory_len)

            fn_sync_target(target, model)

            phase_start_t = t
            cur_phase_decay_ticks = int(eps_decay_frac * (full_phase_edges[phase_i + 1] - full_phase_edges[phase_i]))

            new_phase = False

        for age, act in enumerate(a_que):
            drink_choice, eat_choice, move_choice = action_effects[act]

            if age == 0:
                env.apply_action_movement(act) # movement is handled by the physical world now

            if age < len(proportions1):
                h += drink_amount * drink_choice * proportions1[age] * (0.8 + 0.3*s) 

            if age < len(proportions2):
                s += eat_amount * eat_choice * proportions2[age]

        h = min(hmax, max(0, h))
        s = min(smax, max(0, s))

        cur_comfort = the_meaning_of_life(h, s)
        cur_dead = int((h <= 0.05) or (s <= 0.05))

        novelty_r = 0
        if novelty_rewards and t < EB: 
            novelty_idx = coord_to_idx[tuple(env.coord)]
            novelty_r = beta / np.sqrt(novelty_N[novelty_idx] + 1)
            novelty_N[novelty_idx] += 1

        cur_reward = cur_comfort + novelty_r + death_penalty*cur_dead

        if cur_dead and t != 0:
            if h <= 0.05 and s <= 0.05:
                death_cause = "both"
            elif h <= 0.05:
                death_cause = "hydration"
            else:
                death_cause = "satiation"

            death_events.append({
                "t": t,
                "cause": death_cause,
                "h": h,
                "s": s,
            })

        if t < EB:
            cur_coord = tuple(env.coord)
            on_water = cur_coord in water_set
            on_food  = cur_coord in food_set
            if on_water:
                if last_resource == "food":
                    train_fw_trips += 1
                last_resource = "water"
            elif on_food:
                if last_resource == "water":
                    train_wf_trips += 1
                last_resource = "food"

        cur_x = make_agent_state(h, s, b)
        cur_mask = env.get_action_mask()

        if t != 0 and t < EB and model_type != "oracle": # dont learn if we dont know anything

            if replay_mode == "sequence":

                # one real ordered transition:
                # previous state/action -> current state/reward
                seq_replay.append(
                    carry_x,
                    carry_act,
                    cur_reward,
                    cur_x,
                    bool(cur_dead),
                    cur_mask,
                )

                if t % learn_every == 0 and len(seq_replay) >= replay_warmup:
                    if t % (update_ticks * learn_every) == 0:
                        fn_sync_target(target, model)

                    batch = seq_replay.sample(min(batch_size, len(seq_replay)))

                    fn_learn_step(
                        model,
                        target,
                        optimiser,
                        batch,
                        gamma,
                        burn_in=drqn_burn_in,
                        n_step=n_step
                    )

            else:

                n_step_que.append([carry_x, carry_act, cur_reward, cur_x, bool(cur_dead), cur_mask])

                while n_step_que:

                    state_0 = n_step_que[0].copy()
                    states_added = 0
                    state_0[2] = 0
                    completed_state = False

                    for i, transition in enumerate(n_step_que.copy()):
                        state_0[2] += transition[2] * gamma**i
                        for n in [3, 4, 5]:
                            state_0[n] = transition[n]
                        states_added += 1

                        completed_state = (states_added == n_step) or (transition[4])
                        if completed_state:
                            n_step_que.popleft()
                            break

                    if completed_state:
                        r_que.append(tuple(state_0 + [states_added]))

                    else:
                        break

                if t % learn_every == 0 and len(r_que) >= replay_warmup:
                    if t % (update_ticks * learn_every) == 0:
                        fn_sync_target(target, model)

                    batch = random.sample(r_que, min(batch_size, len(r_que)))

                    fn_learn_step(model, target, optimiser, batch, gamma)

        
        log_coord = tuple(env.coord)
        log_h = h
        log_s = s
        #log_m_d = list(m_que_d)
        #log_m_e = list(m_que_e)
        #log_m_m = list(m_que_m)
                
        if cur_dead:
            use_targeted = (random_start == "targeted" and t < EB and np.random.uniform() < targeted_prob(t))
            if use_targeted:
                h, s, new_coord = sample_targeted_start(reach_at(t))
            elif t >= EB and midpoint_probe:        
                h, s, new_coord = sample_midpoint_start()
            else:
                h, s, new_coord = sample_uniform_start()
                
            env.reset_position(coord=new_coord)

            a_que.clear()
            m_que_d.clear()
            m_que_e.clear()
            m_que_m.clear()

            last_resource = None  

            m_que_d.extend([0] * memory_len)
            m_que_e.extend([0] * memory_len)
            m_que_m.extend([-1] * memory_len)

            if replay_mode == "sequence":
                drqn_hidden = None

            cur_x = make_agent_state(h, s, b)
            cur_mask = env.get_action_mask()

        if t < EB:
            frac = min(1.0, (t - phase_start_t) / cur_phase_decay_ticks)
            epsilon_t = eps_start + frac * (eps_end - eps_start)
        else:
            epsilon_t = 0.0

        if model_type == "oracle":
            model.set_context(
                coord=tuple(env.coord),
                hydration=h,
                satiation=s,
            )
            action = fn_select_action(model, cur_x, cur_mask)

        else:
            if replay_mode == "sequence":

                # Always fwd pass the recurrent model so hidden state updates even if epsilon later overrides the chosen action.
                greedy_action, drqn_hidden = fn_select_action(
                    model,
                    cur_x,
                    cur_mask,
                    drqn_hidden,
                )

                if np.random.uniform(0, 1) < epsilon_t:
                    valid_actions = np.flatnonzero(cur_mask)
                    action = int(np.random.choice(valid_actions))
                else:
                    action = greedy_action

            else:
                if np.random.uniform(0, 1) < epsilon_t:
                    valid_actions = np.flatnonzero(cur_mask)
                    action = int(np.random.choice(valid_actions))
                else:
                    action = fn_select_action(model, cur_x, cur_mask)

        a_que.appendleft(action)

        drink_memory, eat_memory, move_memory = action_effects[action]

        m_que_d.appendleft(float(drink_memory))
        m_que_e.appendleft(float(eat_memory))
        m_que_m.appendleft(-1 if move_memory is None else int(move_memory))

        carry_x = cur_x
        carry_act = action

        lifetime_stats[0].append(cur_comfort)
        lifetime_stats[1].append(action)
        #lifetime_stats[2].append([log_m_d, log_m_e, log_m_m])
        lifetime_stats[3].append(log_h)
        lifetime_stats[4].append(log_s)
        lifetime_stats[5].append(b)
        lifetime_stats[7].append(cur_reward)
        lifetime_stats[8].append(cur_dead)
        lifetime_stats[9].append(log_coord)

        if model_type in ("noisy_DQN", "noisy_drqn_DQN") and t % learn_every == 0 and t < EB:
            sigma_w_mean.append(model.fc2.sigma_w.abs().mean().item())

    lifetime_comfort = np.array(lifetime_stats[0])
    lifetime_action = np.array(lifetime_stats[1])
    #lifetime_memory = np.array(lifetime_stats[2])
    lifetime_coordinates = np.array(lifetime_stats[9])
    lifetime_hydration = np.array(lifetime_stats[3])
    lifetime_satiation = np.array(lifetime_stats[4])
    #lifetime_brightness = np.array(lifetime_stats[5])

    coord_eval = lifetime_coordinates[EB:]

    ticks_at_water_eval = sum(tuple(c) in water_set for c in coord_eval)
    ticks_at_food_eval = sum(tuple(c) in food_set for c in coord_eval)

    #lifetime_reward = np.array(lifetime_stats[7])
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

    "ticks_at_water_eval": ticks_at_water_eval,
    "ticks_at_food_eval": ticks_at_food_eval,

    "death_count": int(death_T.sum()),
    "death_rate": float(death_T.mean()),
    "death_count_eval": int(death_T[EB:].sum()),
    "death_rate_eval": float(death_T[EB:].mean()),

    "sigma_w_mean": sigma_w_mean,

    "water_coords": env.water_coords,
    "food_coords": env.food_coords,

    "senses": tuple(sorted(senses)),
    "n_input": N_INPUT,

    "train_wf_trips": train_wf_trips,
    "train_fw_trips": train_fw_trips,
}
