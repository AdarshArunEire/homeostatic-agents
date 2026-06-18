import pickle
import itertools
import copy
import inspect
from pathlib import Path
import numpy as np
import time

__all__ = [
    "fmt_time",
    "fmt_value",
    "slugify",
    "hex_dist",

    "build_config",
    "make_sweep_configs",
    "make_config_name",

    "sweep_dir",
    "run_file_path",
    "atomic_pickle_dump",

    "eval_boundary",
    "get_eval_array",
    "extract_water_to_food_trips",
    "compute_eval_metrics",
    "slim_run",

    "sweep",
    "run_one_config_seed",

    "DRINK_IDS",
    "EAT_IDS",
    "FULL_EAT_ID",
    "HALF_EAT_ID",
    "QUARTER_EAT_ID",
    "MOVE_MIN_ID",

    "inspect"
]

# -----------------------------
# basic utilities
# -----------------------------

def fmt_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def repo_root():
    cwd = Path.cwd()
    return next((p for p in [cwd, *cwd.parents] if (p / ".git").exists()), cwd)


def sweep_dir(experiment_name, PROTOTYPE_NAME=None):
    PROTOTYPE_DIR = repo_root() / "prototypes" / PROTOTYPE_NAME
    return PROTOTYPE_DIR / "results" / "sweeps" / experiment_name

def atomic_pickle_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "wb") as f:
        pickle.dump(obj, f)

    tmp.replace(path)


def slugify(x):
    x = str(x)
    bad = [" ", "/", "\\", ":", ";", ",", "=", "(", ")", "[", "]", "{", "}"]
    for b in bad:
        x = x.replace(b, "")
    x = x.replace(".", "p")
    return x

def fmt_value(v, percent_style=False):
    if isinstance(v, bool):
        return "T" if v else "F"

    if isinstance(v, str):
        v = v.replace("_DQN", "")
        v = v.replace("DQN", "")
        return slugify(v)

    if isinstance(v, int):
        if abs(v) >= 1_000_000:
            if v % 1_000_000 == 0:
                return f"{v // 1_000_000}m"
            return f"{v / 1_000_000:g}m".replace(".", "p")

        if abs(v) >= 1_000:
            return f"{v // 1_000}k"

        return str(v)

    if isinstance(v, float):
        if percent_style and 0 <= v < 1:
            return f"{int(round(v * 100)):02d}"
        if v.is_integer():
            return str(int(v))
        return f"{v:.3g}".replace(".", "p")

    if isinstance(v, (tuple, list)):
        return "-".join(fmt_value(a) for a in v)

    return slugify(v)

KEY_ABBR = {
    "env.radius": "r",
    "env.water_coord": "w",
    "env.food_coord": "f",

    "sim.sim_len": "len",
    "sim.eval_len": "eval",
    "sim.decay_version": "decay",

    "agent.batch_size": "b",
    "agent.gamma": "g",
    "agent.epsilon_start": "e",
    "agent.alpha": "a",
    "agent.replay_archive_len": "buf",
    "agent.replay_warmup": "wu",
    "agent.update_ticks": "u",
    "agent.learn_every": "le",
    "agent.n_hidden": "h",
    "agent.under_w": "under",
    "agent.over_w": "over",
}

PERCENT_STYLE_KEYS = {
    "agent.gamma",
    "agent.epsilon_start",
    "agent.alpha",
}


def gamma_tag(gamma):
    s = f"{gamma:.6f}".rstrip("0").rstrip(".")
    if s.startswith("0."):
        return "g_" + s[2:]
    return "g_" + s.replace(".", "p")


def make_config_name(overrides):
    if not overrides:
        return "base"

    parts = []
    for key, value in overrides.items():
        abbr = KEY_ABBR.get(key, key.split(".")[-1])

        if key == "agent.gamma":
            parts.append(gamma_tag(value))
            continue

        val = fmt_value(value, percent_style=(key in PERCENT_STYLE_KEYS))

        if key == "agent.model_type":
            parts.append(val)
        else:
            parts.append(f"{abbr}_{val}")

    return slugify("__".join(parts))

# -----------------------------
# config system
# -----------------------------

GROUP_MAP = {
    "env": "env_kwargs",
    "sim": "sim_kwargs",
    "agent": "agent_kwargs",
}


def build_config(
    *,
    base_env,
    base_sim,
    base_agent,
    overrides=None,
    name=None,
):
    overrides = overrides or {}

    config = {
        "name": name if name is not None else make_config_name(overrides),
        "env_kwargs": copy.deepcopy(base_env),
        "sim_kwargs": copy.deepcopy(base_sim),
        "agent_kwargs": copy.deepcopy(base_agent),
        "overrides": copy.deepcopy(overrides),
    }

    for dotted_key, value in overrides.items():
        if "." not in dotted_key:
            raise ValueError(
                f"Override key '{dotted_key}' must be prefixed, e.g. "
                f"'sim.sim_len', 'agent.batch_size', or 'env.radius'."
            )

        group, key = dotted_key.split(".", 1)

        if group not in GROUP_MAP:
            raise ValueError(
                f"Unknown override group '{group}' in key '{dotted_key}'. "
                f"Use one of {list(GROUP_MAP)}."
            )

        config[GROUP_MAP[group]][key] = value

    return config

def make_sweep_configs(
    *,
    base_env,
    base_sim,
    base_agent,
    sweep_grid,
):
    """
    sweep_grid example:

    sweep_grid = {
        "sim.sim_len": [300_000, 600_000, 1_000_000],
    }

    or later:

    sweep_grid = {
        "sim.sim_len": [600_000, 1_000_000],
        "agent.batch_size": [32, 64],
    }
    """
    keys = list(sweep_grid.keys())
    value_lists = [list(sweep_grid[k]) for k in keys]

    configs = []

    for values in itertools.product(*value_lists):
        overrides = dict(zip(keys, values))
        configs.append(
            build_config(
                base_env=base_env,
                base_sim=base_sim,
                base_agent=base_agent,
                overrides=overrides,
            )
        )

    return configs


# -----------------------------
# eval helpers / metrics
# -----------------------------

DRINK_IDS = np.array([1, 2, 3])
EAT_IDS = np.array([4, 5, 6])
FULL_EAT_ID = 4
HALF_EAT_ID = 5
QUARTER_EAT_ID = 6
MOVE_MIN_ID = 7


def hex_dist(a, b):
    aq, ar = tuple(a)
    bq, br = tuple(b)

    return max(
        abs(aq - bq),
        abs(ar - br),
        abs((-aq - ar) - (-bq - br)),
    )


def eval_boundary(run):
    if "eval_boundary" in run:
        return int(run["eval_boundary"])

    if "comfort_train" in run:
        return len(run["comfort_train"])

    raise KeyError("Cannot infer eval boundary. Need run['eval_boundary'] or run['comfort_train'].")


def get_eval_array(run, key_T, key_eval=None):
    if key_T in run:
        eb = eval_boundary(run)
        return np.asarray(run[key_T])[eb:]

    if key_eval is not None and key_eval in run:
        return np.asarray(run[key_eval])

    return None


def extract_water_to_food_trips(
    coords_eval,
    death_eval,
    water_coord,
    food_coord,
    max_trip_ticks=300,
):
    water_coord = tuple(water_coord)
    food_coord = tuple(food_coord)

    trips = []
    active = False
    start_t = None
    ticks = 0
    moves = 0

    for t in range(len(coords_eval) - 1):
        cur = tuple(coords_eval[t])
        nxt = tuple(coords_eval[t + 1])

        if death_eval is not None and death_eval[t]:
            if active:
                trips.append({
                    "success": False,
                    "reason": "death",
                    "start": start_t,
                    "end": t,
                    "ticks": ticks,
                    "moves": moves,
                })
            active = False
            continue

        if not active:
            if cur == water_coord and nxt != water_coord:
                active = True
                start_t = t
                ticks = 0
                moves = 0
            else:
                continue

        step_dist = hex_dist(cur, nxt)
        ticks += 1
        moves += step_dist

        if nxt == food_coord:
            trips.append({
                "success": True,
                "reason": "food",
                "start": start_t,
                "end": t + 1,
                "ticks": ticks,
                "moves": moves,
            })
            active = False

        elif nxt == water_coord:
            trips.append({
                "success": False,
                "reason": "returned_water",
                "start": start_t,
                "end": t + 1,
                "ticks": ticks,
                "moves": moves,
            })
            active = False

        elif ticks >= max_trip_ticks:
            trips.append({
                "success": False,
                "reason": "timeout",
                "start": start_t,
                "end": t + 1,
                "ticks": ticks,
                "moves": moves,
            })
            active = False

    return trips


def compute_eval_metrics(run, env_kwargs):
    eb = eval_boundary(run)

    coords_eval = get_eval_array(run, "coordinates_T", "coordinates_eval")
    actions_eval = get_eval_array(run, "action_T", "action_eval")
    death_eval = get_eval_array(run, "death_T", "death_eval")

    if coords_eval is None or actions_eval is None:
        return {}

    coords_eval = np.asarray(coords_eval, dtype=int)
    actions_eval = np.asarray(actions_eval)
    death_eval = np.asarray(death_eval).astype(bool) if death_eval is not None else np.zeros(len(coords_eval), dtype=bool)

    radius = env_kwargs.get("radius", None)

    water_coord = env_kwargs.get("water_coord", None)
    food_coord = env_kwargs.get("food_coord", None)

    if water_coord is None and radius is not None:
        water_coord = (-radius, 0)

    if food_coord is None and radius is not None:
        food_coord = (0, radius)

    metrics = {}

    metrics["eval_len"] = int(len(coords_eval))
    metrics["eval_deaths"] = int(death_eval.sum())

    # non-neighbour jumps check
    # Raw includes death/reset jumps. Clean ignores jumps touching a death tick.
    jumps_raw = 0
    jumps_clean = 0

    for t in range(len(coords_eval) - 1):
        d = hex_dist(coords_eval[t], coords_eval[t + 1])

        if d > 1:
            jumps_raw += 1

            if not (death_eval[t] or death_eval[t + 1]):
                jumps_clean += 1

    metrics["non_neighbor_jumps_raw"] = int(jumps_raw)
    metrics["non_neighbor_jumps"] = int(jumps_clean)

    wait = actions_eval == 0
    drink = np.isin(actions_eval, DRINK_IDS)
    eat = np.isin(actions_eval, EAT_IDS)
    move = actions_eval >= MOVE_MIN_ID

    metrics["wait_rate_eval"] = float(wait.mean())
    metrics["drink_rate_eval"] = float(drink.mean())
    metrics["eat_rate_eval"] = float(eat.mean())
    metrics["move_rate_eval"] = float(move.mean())

    if water_coord is not None:
        water_coord = tuple(water_coord)
        at_water = np.all(coords_eval == water_coord, axis=1)

        metrics["water_coord"] = water_coord
        metrics["water_visit_pct"] = float(100 * at_water.mean())
        metrics["drink_rate_at_water"] = float(drink[at_water].mean()) if at_water.any() else np.nan
        metrics["move_rate_at_water"] = float(move[at_water].mean()) if at_water.any() else np.nan
    else:
        at_water = None

    if food_coord is not None:
        food_coord = tuple(food_coord)
        at_food = np.all(coords_eval == food_coord, axis=1)

        metrics["food_coord"] = food_coord
        metrics["food_visit_pct"] = float(100 * at_food.mean())
        metrics["eat_rate_at_food"] = float(eat[at_food].mean()) if at_food.any() else np.nan
        metrics["full_eat_rate_at_food"] = float((actions_eval[at_food] == FULL_EAT_ID).mean()) if at_food.any() else np.nan
        metrics["half_eat_rate_at_food"] = float((actions_eval[at_food] == HALF_EAT_ID).mean()) if at_food.any() else np.nan
        metrics["quarter_eat_rate_at_food"] = float((actions_eval[at_food] == QUARTER_EAT_ID).mean()) if at_food.any() else np.nan
    else:
        at_food = None

    if water_coord is not None and food_coord is not None:
        shortest = hex_dist(water_coord, food_coord)
        trips = extract_water_to_food_trips(
            coords_eval=coords_eval,
            death_eval=death_eval,
            water_coord=water_coord,
            food_coord=food_coord,
        )

        successful = [tr for tr in trips if tr["success"]]
        success_moves = np.array([tr["moves"] for tr in successful], dtype=float)

        metrics["water_to_food_shortest_dist"] = int(shortest)
        metrics["water_to_food_trip_count"] = int(len(trips))
        metrics["water_to_food_success_count"] = int(len(successful))
        metrics["water_to_food_success_rate"] = float(len(successful) / len(trips)) if trips else np.nan
        metrics["median_success_trip_moves"] = float(np.median(success_moves)) if len(success_moves) else np.nan
        metrics["path_efficiency"] = float(shortest / np.median(success_moves)) if len(success_moves) else np.nan
        metrics["perfectish_trip_rate"] = float((success_moves <= shortest + 1).mean()) if len(success_moves) else np.nan

    return metrics



def effective_kwargs_from_signature(train_fn, config):
    """
    Build the actual saved metadata by reading train_fn defaults,
    then overlaying config kwargs.

    So if sim_instance has gamma=0.98 by default, saved runs will show gamma=0.98
    even if BASE_AGENT does not explicitly contain gamma.
    """
    sig = inspect.signature(train_fn)

    defaults = {}
    for name, p in sig.parameters.items():
        if p.default is not inspect._empty:
            defaults[name] = copy.deepcopy(p.default)

    # Keep this split simple and explicit.
    # These are simulation-control args, not agent hyperparams.
    sim_keys = {
        "sim_len",
        "eval_len",
        "check_eval_states",
    }

    # These should not be saved inside agent_kwargs.
    ignored_keys = {
        "seed",
        "env_kwargs",
        "sim_kwargs",
        "agent_kwargs",
    }

    effective_sim = {
        k: v for k, v in defaults.items()
        if k in sim_keys
    }

    effective_agent = {
        k: v for k, v in defaults.items()
        if k not in sim_keys and k not in ignored_keys
    }

    # Config values override defaults.
    effective_sim.update(copy.deepcopy(config.get("sim_kwargs", {})))
    effective_agent.update(copy.deepcopy(config.get("agent_kwargs", {})))

    return effective_sim, effective_agent

# -----------------------------
# run slimming
# -----------------------------

ARRAY_DTYPES = {
    "comfort_T": np.float32,
    "hydration_T": np.float32,
    "satiation_T": np.float32,
    "death_T": bool,
    "coordinates_T": np.int16,
    "action_T": np.int16,
}

def slim_run(raw_run, config, seed, run_time=None, train_fn=None):
    eb = eval_boundary(raw_run)

    if train_fn is not None:
        effective_sim, effective_agent = effective_kwargs_from_signature(train_fn, config)
    else:
        effective_sim = copy.deepcopy(config["sim_kwargs"])
        effective_agent = copy.deepcopy(config["agent_kwargs"])

    slim = {
        "eval_boundary": eb,
        "seed": int(seed),
        "config_name": config["name"],
        "env_kwargs": copy.deepcopy(config["env_kwargs"]),
        "sim_kwargs": effective_sim,
        "agent_kwargs": effective_agent,
        "overrides": copy.deepcopy(config.get("overrides", {})),
    }

    if run_time is not None:
        slim["run_time_seconds"] = float(run_time)

    for key, dtype in ARRAY_DTYPES.items():
        if key in raw_run:
            slim[key] = np.asarray(raw_run[key]).astype(dtype)

    if "comfort_T" in slim:
        slim["comfort_train"] = slim["comfort_T"][:eb].astype(np.float32)
        slim["comfort_eval"] = slim["comfort_T"][eb:].astype(np.float32)

    if "hydration_T" in slim:
        slim["hydration_eval"] = slim["hydration_T"][eb:].astype(np.float32)

    if "satiation_T" in slim:
        slim["satiation_eval"] = slim["satiation_T"][eb:].astype(np.float32)

    if "mean_comfort" in raw_run:
        slim["mean_comfort"] = float(raw_run["mean_comfort"])
    elif "comfort_eval" in slim:
        slim["mean_comfort"] = float(np.mean(slim["comfort_eval"]))

    if "mean_reward" in raw_run:
        slim["mean_reward"] = float(raw_run["mean_reward"])

    if "death_count_eval" in raw_run:
        slim["death_count_eval"] = int(raw_run["death_count_eval"])
    elif "death_T" in slim:
        slim["death_count_eval"] = int(np.asarray(slim["death_T"][eb:]).sum())

    if "death_rate_eval" in raw_run:
        slim["death_rate_eval"] = float(raw_run["death_rate_eval"])
    elif "death_T" in slim:
        slim["death_rate_eval"] = float(np.asarray(slim["death_T"][eb:]).mean())

    slim["metrics"] = compute_eval_metrics(slim, config["env_kwargs"])

    return slim


# -----------------------------
# calling the training function
# -----------------------------

def call_train_fn(train_fn, config, seed):
    """
    This adapts to two possible sim_instance styles:

    1. New clean style:
       sim_instance(seed=seed, env_kwargs={...}, **sim_kwargs, **agent_kwargs)

    2. Flat old style:
       sim_instance(seed=seed, radius=3, sim_len=..., batch_size=..., etc.)

    It uses the function signature so TypeErrors inside sim_instance do not get hidden.
    """
    sig = inspect.signature(train_fn)
    params = sig.parameters
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs = {"seed": seed}

    def add_group(group_name, group_kwargs):
        if group_name in params:
            kwargs[group_name] = copy.deepcopy(group_kwargs)
            return

        for k, v in group_kwargs.items():
            if accepts_varkw or k in params:
                kwargs[k] = v

    add_group("env_kwargs", config["env_kwargs"])
    add_group("sim_kwargs", config["sim_kwargs"])
    add_group("agent_kwargs", config["agent_kwargs"])

    return train_fn(**kwargs)


def run_one_config_seed(config, seed, train_fn):
    start = time.perf_counter()

    raw_run = call_train_fn(train_fn, config, seed)

    run_time = time.perf_counter() - start

    slim = slim_run(
        raw_run=raw_run,
        config=config,
        seed=seed,
        run_time=run_time,
        train_fn=train_fn,
    )

    return slim

# -----------------------------
# sweep runner
# -----------------------------

def run_file_path(experiment_name, config_name, seed, PROTOTYPE_NAME):
    return sweep_dir(experiment_name, PROTOTYPE_NAME) / "runs" / f"{config_name}__seed{seed}.pkl"


def write_manifest(experiment_name, configs, seeds, PROTOTYPE_NAME):
    manifest = {
        "experiment_name": experiment_name,
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": list(seeds),
        "configs": configs,
    }

    atomic_pickle_dump(manifest, sweep_dir(experiment_name, PROTOTYPE_NAME) / "manifest.pkl")


def sweep(
    *,
    experiment_name,
    configs,
    seeds,
    train_fn=None,
    resume=False,
    PROTOTYPE_NAME=None,
):
    """
    Runs configs × seeds and saves every run separately.

    Later load cell will rebuild res from:
        results/sweeps/<experiment_name>/runs/*.pkl
    """
    if train_fn is None:
        train_fn = globals().get("sim_instance", None)

    if train_fn is None:
        raise ValueError("No train_fn provided and no global sim_instance found.")

    seeds = list(seeds)
    configs = list(configs)

    write_manifest(experiment_name, configs, seeds, PROTOTYPE_NAME)

    total_runs = len(configs) * len(seeds)
    completed_now = 0
    skipped = 0
    global_start = time.perf_counter()

    print(f"experiment: {experiment_name}")
    print(f"configs: {len(configs)}")
    print(f"seeds: {seeds}")
    print(f"total planned runs: {total_runs}")
    print(f"output: {sweep_dir(experiment_name, PROTOTYPE_NAME)}")

    for config in configs:
        name = config["name"]

        print(f"\n=== config: {name} ===")
        print("env:", config["env_kwargs"])
        print("sim:", config["sim_kwargs"])
        print("agent:", config["agent_kwargs"])

        for seed in seeds:
            out_path = run_file_path(experiment_name, name, seed, PROTOTYPE_NAME)

            if resume and out_path.exists():
                skipped += 1
                print(f"skip existing: config={name} seed={seed}")
                continue

            run_start = time.perf_counter()

            try:
                slim = run_one_config_seed(
                    config=config,
                    seed=seed,
                    train_fn=train_fn,
                )

                atomic_pickle_dump(slim, out_path)

            except Exception as e:
                print(f"FAILED: config={name} seed={seed}")
                print(repr(e))
                raise

            run_time = time.perf_counter() - run_start
            completed_now += 1

            finished_total = skipped + completed_now
            elapsed = time.perf_counter() - global_start
            avg = elapsed / max(completed_now, 1)
            remaining = total_runs - finished_total
            eta = avg * remaining

            mean = slim.get("mean_comfort", np.nan)
            deaths = slim.get("death_count_eval", np.nan)
            food_pct = slim.get("metrics", {}).get("food_visit_pct", np.nan)
            water_pct = slim.get("metrics", {}).get("water_visit_pct", np.nan)
            path_eff = slim.get("metrics", {}).get("path_efficiency", np.nan)

            print(
                f"[{finished_total}/{total_runs}] "
                f"config={name:<24} seed={seed:<3} "
                f"mean={mean:.3f} deaths={deaths} "
                f"water={water_pct:.1f}% food={food_pct:.1f}% "
                f"path_eff={path_eff:.3f} "
                f"run={fmt_time(run_time)} "
                f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}"
            )

    total_time = time.perf_counter() - global_start

    print(f"\nDONE: {experiment_name}")
    print(f"new runs: {completed_now}")
    print(f"skipped existing: {skipped}")
    print(f"time: {fmt_time(total_time)}")