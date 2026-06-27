import pickle
import itertools
import copy
import inspect
from pathlib import Path
import numpy as np
import time
import pandas as pd
from IPython.display import display

import os
import traceback
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

__all__ = [
    # utilities
    "fmt_time",
    "fmt_value",
    "slugify",
    "hex_dist",

    # config
    "build_config",
    "make_sweep_configs",
    "make_config_name",

    # io / paths
    "sweep_dir",
    "run_file_path",
    "atomic_pickle_dump",

    # eval / metrics
    "eval_boundary",
    "get_eval_array",
    "extract_resource_trips",
    "summarise_trips",
    "compute_eval_metrics",
    "slim_run",

    # sweep runner
    "sweep",
    "sweep_parallel",
    "run_one_config_seed",

    # action ids
    "DRINK_IDS",
    "EAT_IDS",
    "FULL_EAT_ID",
    "HALF_EAT_ID",
    "QUARTER_EAT_ID",
    "MOVE_MIN_ID",

    # formatting / stats
    "fmt",
    "fmt_pct",
    "wilson_interval",

    # loading + reporting
    "load_pickle",
    "load_sweep",
    "safe_metric",
    "config_value",
    "make_run_row",
    "make_runs_df",
    "solved_rate",
    "make_summary_df",
    "make_summary_display_df",
    "print_config_reports",
    "load_and_report",

    "inspect",
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


def longest_true_run(mask):
    longest = 0
    cur = 0

    for x in mask:
        if x:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    return int(longest)


def safe_ratio(a, b):
    return float(a / b) if b not in [0, 0.0] else np.nan


def extract_resource_trips(
    coords_eval,
    death_eval,
    source_coords,
    target_coords,
    max_trip_ticks=300,
):
    source_set = set(map(tuple, source_coords))
    target_set = set(map(tuple, target_coords))

    trips = []
    active = False
    start_t = None
    start_coord = None
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
                    "shortest": None,
                })
            active = False
            continue

        if not active:
            if cur in source_set and nxt not in source_set:
                active = True
                start_t = t
                start_coord = cur
                ticks = 0
                moves = 0
            else:
                continue

        step_dist = hex_dist(cur, nxt)
        ticks += 1
        moves += step_dist

        if nxt in target_set:
            trips.append({
                "success": True,
                "reason": "target",
                "start": start_t,
                "end": t + 1,
                "ticks": ticks,
                "moves": moves,
                "shortest": hex_dist(start_coord, nxt),
            })
            active = False

        elif nxt in source_set:
            trips.append({
                "success": False,
                "reason": "returned_source",
                "start": start_t,
                "end": t + 1,
                "ticks": ticks,
                "moves": moves,
                "shortest": None,
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
                "shortest": None,
            })
            active = False

    return trips


def summarise_trips(trips, prefix):
    metrics = {}

    successful = [tr for tr in trips if tr["success"]]
    success_moves = np.array([tr["moves"] for tr in successful], dtype=float)
    success_ticks = np.array([tr["ticks"] for tr in successful], dtype=float)
    success_shortest = np.array([tr["shortest"] for tr in successful], dtype=float)

    if len(successful):
        per_trip_eff = success_shortest / np.maximum(success_moves, 1)
    else:
        per_trip_eff = np.array([])

    reasons = {}
    for tr in trips:
        reasons[tr["reason"]] = reasons.get(tr["reason"], 0) + 1

    metrics[f"{prefix}_trip_count"] = int(len(trips))
    metrics[f"{prefix}_success_count"] = int(len(successful))
    metrics[f"{prefix}_success_rate"] = float(len(successful) / len(trips)) if trips else np.nan

    metrics[f"{prefix}_death_count"] = int(reasons.get("death", 0))
    metrics[f"{prefix}_timeout_count"] = int(reasons.get("timeout", 0))
    metrics[f"{prefix}_returned_source_count"] = int(reasons.get("returned_source", 0))

    metrics[f"{prefix}_median_success_ticks"] = float(np.median(success_ticks)) if len(success_ticks) else np.nan
    metrics[f"{prefix}_median_success_moves"] = float(np.median(success_moves)) if len(success_moves) else np.nan
    metrics[f"{prefix}_median_shortest"] = float(np.median(success_shortest)) if len(success_shortest) else np.nan

    metrics[f"{prefix}_path_efficiency"] = float(np.median(per_trip_eff)) if len(per_trip_eff) else np.nan
    metrics[f"{prefix}_perfectish_trip_rate"] = float((success_moves <= success_shortest + 1).mean()) if len(success_moves) else np.nan

    return metrics


def compute_eval_metrics(run, env_kwargs):
    coords_eval = get_eval_array(run, "coordinates_T", "coordinates_eval")
    actions_eval = get_eval_array(run, "action_T", "action_eval")
    death_eval = get_eval_array(run, "death_T", "death_eval")
    comfort_eval = get_eval_array(run, "comfort_T", "comfort_eval")
    hydration_eval = get_eval_array(run, "hydration_T", "hydration_eval")
    satiation_eval = get_eval_array(run, "satiation_T", "satiation_eval")

    if coords_eval is None or actions_eval is None:
        return {}

    coords_eval = np.asarray(coords_eval, dtype=int)
    actions_eval = np.asarray(actions_eval)
    death_eval = np.asarray(death_eval).astype(bool) if death_eval is not None else np.zeros(len(coords_eval), dtype=bool)

    metrics = {}

    # -----------------------------
    # basic eval health
    # -----------------------------
    metrics["eval_len"] = int(len(coords_eval))
    metrics["eval_deaths"] = int(death_eval.sum())
    metrics["eval_death_rate"] = float(death_eval.mean())
    metrics["eval_alive_rate"] = float(1.0 - death_eval.mean())
    metrics["longest_no_death_run"] = longest_true_run(~death_eval)
    metrics["longest_death_free_frac"] = float(metrics["longest_no_death_run"] / max(len(death_eval), 1))

    if comfort_eval is not None:
        comfort_eval = np.asarray(comfort_eval, dtype=float)
        metrics["mean_comfort"] = float(np.mean(comfort_eval))
        metrics["median_comfort"] = float(np.median(comfort_eval))
        metrics["min_comfort"] = float(np.min(comfort_eval))
        metrics["p05_comfort"] = float(np.percentile(comfort_eval, 5))
        metrics["p25_comfort"] = float(np.percentile(comfort_eval, 25))
        metrics["p75_comfort"] = float(np.percentile(comfort_eval, 75))
        metrics["comfort_stability"] = float(np.std(comfort_eval))

    if hydration_eval is not None:
        hydration_eval = np.asarray(hydration_eval, dtype=float)
        metrics["mean_hydration"] = float(np.mean(hydration_eval))
        metrics["median_hydration"] = float(np.median(hydration_eval))
        metrics["p05_hydration"] = float(np.percentile(hydration_eval, 5))
        metrics["p95_hydration"] = float(np.percentile(hydration_eval, 95))

    if satiation_eval is not None:
        satiation_eval = np.asarray(satiation_eval, dtype=float)
        metrics["mean_satiation"] = float(np.mean(satiation_eval))
        metrics["median_satiation"] = float(np.median(satiation_eval))
        metrics["p05_satiation"] = float(np.percentile(satiation_eval, 5))
        metrics["p95_satiation"] = float(np.percentile(satiation_eval, 95))

    # -----------------------------
    # movement sanity
    # -----------------------------
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

    # -----------------------------
    # action mix
    # -----------------------------
    wait = actions_eval == 0
    drink = np.isin(actions_eval, DRINK_IDS)
    eat = np.isin(actions_eval, EAT_IDS)
    move = actions_eval >= MOVE_MIN_ID

    metrics["wait_rate_eval"] = float(wait.mean())
    metrics["drink_rate_eval"] = float(drink.mean())
    metrics["eat_rate_eval"] = float(eat.mean())
    metrics["move_rate_eval"] = float(move.mean())
    metrics["action_entropy_proxy"] = float(
        -sum(
            p * np.log(p)
            for p in [
                wait.mean(),
                drink.mean(),
                eat.mean(),
                move.mean(),
            ]
            if p > 0
        )
    )

    # -----------------------------
    # resource sets
    # -----------------------------
    water_coords = run.get("water_coords", None)
    food_coords = run.get("food_coords", None)

    if water_coords is None:
        wc = env_kwargs.get("water_coord", None)
        if wc is None and env_kwargs.get("radius"):
            wc = (-env_kwargs["radius"], 0)
        water_coords = [wc] if wc is not None else None

    if food_coords is None:
        fc = env_kwargs.get("food_coord", None)
        if fc is None and env_kwargs.get("radius"):
            fc = (0, env_kwargs["radius"])
        food_coords = [fc] if fc is not None else None

    water_set = set(map(tuple, water_coords)) if water_coords else set()
    food_set = set(map(tuple, food_coords)) if food_coords else set()

    at_water = np.array([tuple(c) in water_set for c in coords_eval], dtype=bool)
    at_food = np.array([tuple(c) in food_set for c in coords_eval], dtype=bool)
    at_resource = at_water | at_food

    metrics["water_visit_pct"] = float(100 * at_water.mean())
    metrics["food_visit_pct"] = float(100 * at_food.mean())
    metrics["resource_visit_pct"] = float(100 * at_resource.mean())
    metrics["neutral_visit_pct"] = float(100 * (~at_resource).mean())

    metrics["water_food_visit_ratio"] = safe_ratio(metrics["water_visit_pct"], metrics["food_visit_pct"])
    metrics["food_water_visit_ratio"] = safe_ratio(metrics["food_visit_pct"], metrics["water_visit_pct"])

    metrics["longest_water_run"] = longest_true_run(at_water)
    metrics["longest_food_run"] = longest_true_run(at_food)
    metrics["longest_neutral_run"] = longest_true_run(~at_resource)

    metrics["drink_rate_at_water"] = float(drink[at_water].mean()) if at_water.any() else np.nan
    metrics["move_rate_at_water"] = float(move[at_water].mean()) if at_water.any() else np.nan

    metrics["eat_rate_at_food"] = float(eat[at_food].mean()) if at_food.any() else np.nan
    metrics["move_rate_at_food"] = float(move[at_food].mean()) if at_food.any() else np.nan

    metrics["full_eat_rate_at_food"] = float((actions_eval[at_food] == FULL_EAT_ID).mean()) if at_food.any() else np.nan
    metrics["half_eat_rate_at_food"] = float((actions_eval[at_food] == HALF_EAT_ID).mean()) if at_food.any() else np.nan
    metrics["quarter_eat_rate_at_food"] = float((actions_eval[at_food] == QUARTER_EAT_ID).mean()) if at_food.any() else np.nan

    # -----------------------------
    # camp diagnostics
    # -----------------------------
    coord_counts = {}
    for c in coords_eval:
        c = tuple(c)
        coord_counts[c] = coord_counts.get(c, 0) + 1

    if coord_counts:
        dominant_coord, dominant_count = max(coord_counts.items(), key=lambda kv: kv[1])
        dominant_pct = 100 * dominant_count / len(coords_eval)
    else:
        dominant_coord, dominant_pct = None, np.nan

    metrics["dominant_cell"] = dominant_coord
    metrics["dominant_cell_pct"] = float(dominant_pct)

    metrics["dominant_cell_is_water"] = bool(dominant_coord in water_set) if dominant_coord is not None else False
    metrics["dominant_cell_is_food"] = bool(dominant_coord in food_set) if dominant_coord is not None else False

    # continuous camp score: high when one resource dominates and the other is ignored
    metrics["water_camp_score"] = float(metrics["water_visit_pct"] - metrics["food_visit_pct"])
    metrics["food_camp_score"] = float(metrics["food_visit_pct"] - metrics["water_visit_pct"])
    metrics["resource_balance_abs_gap"] = float(abs(metrics["water_visit_pct"] - metrics["food_visit_pct"]))

    # -----------------------------
    # route diagnostics
    # -----------------------------
    if water_coords and food_coords:
        wf_trips = extract_resource_trips(
            coords_eval=coords_eval,
            death_eval=death_eval,
            source_coords=water_coords,
            target_coords=food_coords,
        )

        fw_trips = extract_resource_trips(
            coords_eval=coords_eval,
            death_eval=death_eval,
            source_coords=food_coords,
            target_coords=water_coords,
        )

        metrics.update(summarise_trips(wf_trips, "water_to_food"))
        metrics.update(summarise_trips(fw_trips, "food_to_water"))

        metrics["two_way_route_success_min"] = float(np.nanmin([
            metrics["water_to_food_success_rate"],
            metrics["food_to_water_success_rate"],
        ]))

        metrics["total_successful_resource_trips"] = int(
            metrics["water_to_food_success_count"] +
            metrics["food_to_water_success_count"]
        )

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

    if "water_coords" in raw_run:
        slim["water_coords"] = raw_run["water_coords"]
    if "food_coords" in raw_run:
        slim["food_coords"] = raw_run["food_coords"]

    if "sigma_w_mean" in raw_run:
        slim["sigma_w_mean"] = list(raw_run["sigma_w_mean"])

    if "train_wf_trips" in raw_run:        
        slim["train_wf_trips"] = int(raw_run["train_wf_trips"])
    if "train_fw_trips" in raw_run:        
        slim["train_fw_trips"] = int(raw_run["train_fw_trips"])

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
                
    env_kw = copy.deepcopy(config["env_kwargs"])
    env_kw["seed"] = seed  # env seed matches run seed

    add_group("env_kwargs", env_kw)
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
                f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}",
                flush=True
            )

    total_time = time.perf_counter() - global_start

    print(f"\nDONE: {experiment_name}")
    print(f"new runs: {completed_now}")
    print(f"skipped existing: {skipped}")
    print(f"time: {fmt_time(total_time)}")

    # =====================================================================
# loading + reporting
# append to the bottom of the sweep module (the `import pickle` file)
#
# also add at the TOP of that module:
#     import pandas as pd
#     try:
#         from IPython.display import display
#     except ImportError:
#         display = print
#
# and extend __all__ with:
#     "fmt", "fmt_pct", "wilson_interval",
#     "load_sweep", "make_runs_df", "make_summary_df",
#     "make_summary_display_df", "print_config_reports",
#     "load_and_report",
# =====================================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import os


def _parallel_sweep_worker(job):
    """
    One process = one config/seed run.

    Important:
    - train_fn must be importable/pickleable.
    - On Windows, this works best from a .py script, not inside Jupyter.
    """

    # stop every worker from trying to use all CPU threads internally
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    experiment_name = job["experiment_name"]
    config = job["config"]
    seed = job["seed"]
    train_fn = job["train_fn"]
    PROTOTYPE_NAME = job["PROTOTYPE_NAME"]

    name = config["name"]
    out_path = run_file_path(experiment_name, name, seed, PROTOTYPE_NAME)

    print(f"  START config={name} seed={seed}", flush=True)

    start = time.perf_counter()

    try:
        slim = run_one_config_seed(
            config=config,
            seed=seed,
            train_fn=train_fn,
        )

        atomic_pickle_dump(slim, out_path)

        run_time = time.perf_counter() - start

        mean = slim.get("mean_comfort", np.nan)
        deaths = slim.get("death_count_eval", np.nan)
        food_pct = slim.get("metrics", {}).get("food_visit_pct", np.nan)
        water_pct = slim.get("metrics", {}).get("water_visit_pct", np.nan)
        path_eff = slim.get("metrics", {}).get("water_to_food_path_efficiency", np.nan)

        return {
            "status": "ok",
            "config": name,
            "seed": seed,
            "mean": mean,
            "deaths": deaths,
            "food_pct": food_pct,
            "water_pct": water_pct,
            "path_eff": path_eff,
            "run_time": run_time,
            "out_path": str(out_path),
        }

    except Exception as e:
        return {
            "status": "fail",
            "config": name,
            "seed": seed,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "out_path": str(out_path),
        }

def sweep_parallel(
    *,
    experiment_name,
    configs,
    seeds,
    train_fn=None,
    resume=False,
    PROTOTYPE_NAME=None,
    max_workers=4,
):
    """
    Parallel version of sweep().

    Runs configs × seeds across multiple processes.
    Saves the same per-run .pkl files as sweep(), so load_and_report still works.
    """

    if train_fn is None:
        train_fn = globals().get("sim_instance", None)

    if train_fn is None:
        raise ValueError("No train_fn provided and no global sim_instance found.")

    seeds = list(seeds)
    configs = list(configs)

    write_manifest(experiment_name, configs, seeds, PROTOTYPE_NAME)

    jobs = []
    skipped = 0

    for config in configs:
        for seed in seeds:
            out_path = run_file_path(
                experiment_name,
                config["name"],
                seed,
                PROTOTYPE_NAME,
            )

            if resume and out_path.exists():
                skipped += 1
                continue

            jobs.append({
                "experiment_name": experiment_name,
                "config": config,
                "seed": seed,
                "train_fn": train_fn,
                "PROTOTYPE_NAME": PROTOTYPE_NAME,
            })

    total_runs = len(configs) * len(seeds)
    total_to_run = len(jobs)

    print(f"experiment: {experiment_name}")
    print(f"configs: {len(configs)}")
    print(f"seeds: {seeds}")
    print(f"total planned runs: {total_runs}")
    print(f"skipped existing: {skipped}")
    print(f"new runs to launch: {total_to_run}")
    print(f"max_workers: {max_workers}")
    print(f"output: {sweep_dir(experiment_name, PROTOTYPE_NAME)}")

    if total_to_run == 0:
        print("\nDONE: nothing new to run.")
        return

    global_start = time.perf_counter()
    completed_now = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_parallel_sweep_worker, job) for job in jobs]

        for fut in as_completed(futures):
            row = fut.result()
            completed_now += 1

            finished_total = skipped + completed_now
            elapsed = time.perf_counter() - global_start
            avg = elapsed / max(completed_now, 1)
            remaining = total_to_run - completed_now
            eta = avg * remaining

            if row["status"] == "ok":
                print(
                    f"[{finished_total}/{total_runs}] "
                    f"config={row['config']:<24} seed={row['seed']:<3} "
                    f"mean={row['mean']:.3f} deaths={row['deaths']} "
                    f"water={row['water_pct']:.1f}% food={row['food_pct']:.1f}% "
                    f"path_eff={row['path_eff']:.3f} "
                    f"run={fmt_time(row['run_time'])} "
                    f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}",
                    flush=True
                )

            else:
                print(
                    f"\nFAILED: config={row['config']} seed={row['seed']}"
                )
                print(row["error"])
                print(row["traceback"])
                raise RuntimeError(
                    f"Parallel sweep failed for config={row['config']} seed={row['seed']}"
                )

    total_time = time.perf_counter() - global_start

    print(f"\nDONE: {experiment_name}")
    print(f"new runs: {completed_now}")
    print(f"skipped existing: {skipped}")
    print(f"time: {fmt_time(total_time)}")

# -----------------------------
# formatting / stats helpers
# -----------------------------

def fmt(x, ndp=3):
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return "—"
    return f"{xf:.{ndp}f}"


def fmt_pct(x, ndp=1):
    """x is a proportion in [0, 1]."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return "—"
    return f"{100 * xf:.{ndp}f}%"


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for k successes / n trials. Returns (lo, hi) as proportions."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - margin, center + margin)


# -----------------------------
# loading
# -----------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sweep(experiment_name, PROTOTYPE_NAME, recompute_metrics=True):
    folder = sweep_dir(experiment_name, PROTOTYPE_NAME)
    runs_dir = folder / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs folder found: {runs_dir}")

    run_files = sorted(runs_dir.glob("*.pkl"))
    if not run_files:
        raise FileNotFoundError(f"No run files found in: {runs_dir}")

    manifest_path = folder / "manifest.pkl"
    manifest = load_pickle(manifest_path) if manifest_path.exists() else None

    runs = []
    for path in run_files:
        run = load_pickle(path)
        if recompute_metrics:
            run["metrics"] = compute_eval_metrics(run, run.get("env_kwargs", {}))
        runs.append(run)

    grouped = {}
    for run in runs:
        name = run.get("config_name", "unknown")
        if name not in grouped:
            grouped[name] = {
                "runs": [],
                "ranked": [],
                "config": {
                    "env_kwargs": run.get("env_kwargs", {}),
                    "sim_kwargs": run.get("sim_kwargs", {}),
                    "agent_kwargs": run.get("agent_kwargs", {}),
                    "overrides": run.get("overrides", {}),
                },
            }
        grouped[name]["runs"].append(run)

    for name, item in grouped.items():
        ranked = sorted(item["runs"], key=lambda r: r.get("mean_comfort", -np.inf))
        n = len(ranked)
        item["ranked"] = ranked
        item["worst"] = ranked[0]
        item["best"] = ranked[-1]
        item["median"] = ranked[n // 2]
        item["upper_median"] = ranked[n // 2]

    return grouped, manifest


# -----------------------------
# row / dataframe building
# -----------------------------

def safe_metric(run, key, default=np.nan):
    return run.get("metrics", {}).get(key, default)


def config_value(run, group, key, default=np.nan):
    return run.get(f"{group}_kwargs", {}).get(key, default)


def make_run_row(run):
    m = run.get("metrics", {})

    pe = m.get("water_to_food_path_efficiency", m.get("path_efficiency", np.nan))
    pf = m.get("water_to_food_perfectish_trip_rate", m.get("perfectish_trip_rate", np.nan))
    sr = m.get("water_to_food_success_rate", m.get("trip_success_rate", np.nan))

    return {
        "config": run.get("config_name"),
        "seed": run.get("seed"),

        # core config
        "sim_len": config_value(run, "sim", "sim_len"),
        "eval_len": config_value(run, "sim", "eval_len"),
        "radius": config_value(run, "env", "radius"),
        "batch_size": config_value(run, "agent", "batch_size"),
        "gamma": config_value(run, "agent", "gamma"),
        "epsilon_start": config_value(run, "agent", "epsilon_start"),
        "buffer": config_value(run, "agent", "replay_archive_len"),
        "warmup": config_value(run, "agent", "replay_warmup"),
        "update_ticks": config_value(run, "agent", "update_ticks"),
        "learn_every": config_value(run, "agent", "learn_every"),

        # headline
        "mean_comfort": run.get("mean_comfort", m.get("mean_comfort", np.nan)),
        "mean_reward": run.get("mean_reward", np.nan),
        "eval_deaths": run.get("death_count_eval", m.get("eval_deaths", np.nan)),
        "death_rate_eval": run.get("death_rate_eval", m.get("eval_death_rate", np.nan)),
        "run_time_seconds": run.get("run_time_seconds", np.nan),

        # spatial / action
        "water_visit_pct": m.get("water_visit_pct", np.nan),
        "food_visit_pct": m.get("food_visit_pct", np.nan),
        "water_food_visit_ratio": m.get("water_food_visit_ratio", np.nan),
        "water_camp_score": m.get("water_camp_score", np.nan),
        "dominant_cell_pct": m.get("dominant_cell_pct", np.nan),

        "drink_rate_at_water": m.get("drink_rate_at_water", np.nan),
        "eat_rate_at_food": m.get("eat_rate_at_food", np.nan),
        "full_eat_rate_at_food": m.get("full_eat_rate_at_food", np.nan),
        "move_rate_eval": m.get("move_rate_eval", np.nan),
        "drink_rate_eval": m.get("drink_rate_eval", np.nan),
        "eat_rate_eval": m.get("eat_rate_eval", np.nan),

        # route
        "water_to_food_trip_count": m.get("water_to_food_trip_count", np.nan),
        "water_to_food_success_count": m.get("water_to_food_success_count", np.nan),
        "water_to_food_success_rate": sr,
        "water_to_food_path_efficiency": pe,
        "water_to_food_perfectish_trip_rate": pf,

        "food_to_water_trip_count": m.get("food_to_water_trip_count", np.nan),
        "food_to_water_success_count": m.get("food_to_water_success_count", np.nan),
        "food_to_water_success_rate": m.get("food_to_water_success_rate", np.nan),
        "food_to_water_path_efficiency": m.get("food_to_water_path_efficiency", np.nan),

        "total_successful_resource_trips": m.get("total_successful_resource_trips", np.nan),
        "two_way_route_success_min": m.get("two_way_route_success_min", np.nan),

        "non_neighbor_jumps": m.get("non_neighbor_jumps", np.nan),
    }


def make_runs_df(res):
    rows = [make_run_row(run) for item in res.values() for run in item["runs"]]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["config", "seed"]).reset_index(drop=True)
        df["total_ticks"] = df["sim_len"] + df["eval_len"]
    return df


def solved_rate(series):
    return float(np.mean(series)) if len(series) else np.nan


def make_summary_df(runs_df):
    if runs_df.empty:
        return pd.DataFrame()

    df = runs_df.copy()

    pe = pd.to_numeric(df["water_to_food_path_efficiency"], errors="coerce")
    pf = pd.to_numeric(df["water_to_food_perfectish_trip_rate"], errors="coerce")
    dd = pd.to_numeric(df["eval_deaths"], errors="coerce")

    # clean-solve  = efficient route + some near-perfect trips (crossed the valley)
    # solved (3b)  = clean-solve AND survived it (<= 5 eval deaths)
    df["clean_solve"] = (pe >= 0.9) & (pf > 0)
    df["solved"] = df["clean_solve"] & (dd <= 5)

    g = df.groupby("config", dropna=False)

    summary = g.agg(
        solved_rate=("solved", solved_rate),
        solved_seeds=("solved", lambda x: int(x.sum())),
        clean_solve_rate=("clean_solve", solved_rate),
        clean_solve_seeds=("clean_solve", lambda x: int(x.sum())),
        n_seeds=("seed", "count"),

        sim_len=("sim_len", "first"),
        radius=("radius", "first"),
        batch_size=("batch_size", "first"),
        gamma=("gamma", "first"),
        epsilon_start=("epsilon_start", "first"),
        buffer=("buffer", "first"),
        warmup=("warmup", "first"),
        update_ticks=("update_ticks", "first"),
        learn_every=("learn_every", "first"),

        mean_comfort_mean=("mean_comfort", "mean"),
        mean_comfort_median=("mean_comfort", "median"),
        mean_comfort_std=("mean_comfort", "std"),
        mean_comfort_min=("mean_comfort", "min"),
        mean_comfort_max=("mean_comfort", "max"),

        eval_deaths_mean=("eval_deaths", "mean"),
        eval_deaths_median=("eval_deaths", "median"),
        eval_deaths_min=("eval_deaths", "min"),
        eval_deaths_max=("eval_deaths", "max"),
        zero_death_seeds=("eval_deaths", lambda x: int((x == 0).sum())),

        water_visit_pct_median=("water_visit_pct", "median"),
        food_visit_pct_median=("food_visit_pct", "median"),
        water_food_visit_ratio_median=("water_food_visit_ratio", "median"),
        water_camp_score_median=("water_camp_score", "median"),
        dominant_cell_pct_median=("dominant_cell_pct", "median"),

        eat_rate_at_food_median=("eat_rate_at_food", "median"),
        full_eat_rate_at_food_median=("full_eat_rate_at_food", "median"),

        water_to_food_success_rate_median=("water_to_food_success_rate", "median"),
        water_to_food_success_count_median=("water_to_food_success_count", "median"),
        water_to_food_path_efficiency_median=("water_to_food_path_efficiency", "median"),
        water_to_food_perfectish_trip_rate_median=("water_to_food_perfectish_trip_rate", "median"),

        food_to_water_success_rate_median=("food_to_water_success_rate", "median"),
        food_to_water_success_count_median=("food_to_water_success_count", "median"),
        food_to_water_path_efficiency_median=("food_to_water_path_efficiency", "median"),

        total_successful_resource_trips_median=("total_successful_resource_trips", "median"),
        two_way_route_success_min_median=("two_way_route_success_min", "median"),

        non_neighbor_jumps_max=("non_neighbor_jumps", "max"),

        run_time_seconds_mean=("run_time_seconds", "mean"),
        run_time_seconds_median=("run_time_seconds", "median"),
    ).reset_index()

    summary["zero_death_rate"] = summary["zero_death_seeds"] / summary["n_seeds"]

    ci = [wilson_interval(int(k), int(n))
          for k, n in zip(summary["solved_seeds"], summary["n_seeds"])]
    summary["solved_ci_low"] = [lo for lo, _ in ci]
    summary["solved_ci_high"] = [hi for _, hi in ci]

    summary = summary.sort_values(
        ["solved_rate", "mean_comfort_median", "eval_deaths_median", "food_visit_pct_median"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return summary


# -----------------------------
# compact table
# -----------------------------

def make_summary_display_df(summary_df):
    if summary_df.empty:
        return pd.DataFrame()
    d = summary_df
    return pd.DataFrame({
        "config": d["config"],
        "solved": [f"{fmt_pct(r, 0)} ({int(k)}/{int(n)})"
                   for r, k, n in zip(d["solved_rate"], d["solved_seeds"], d["n_seeds"])],
        "95% wilson": [f"[{fmt_pct(lo, 0)}–{fmt_pct(hi, 0)}]"
                       for lo, hi in zip(d["solved_ci_low"], d["solved_ci_high"])],
        "clean": [fmt_pct(r, 0) for r in d["clean_solve_rate"]],
        "comfort_med": [fmt(x, 2) for x in d["mean_comfort_median"]],
        "deaths_med": [fmt(x, 1) for x in d["eval_deaths_median"]],
        "zero_death": [fmt_pct(x, 0) for x in d["zero_death_rate"]],
        "food%": [fmt(x, 1) for x in d["food_visit_pct_median"]],
        "water%": [fmt(x, 1) for x in d["water_visit_pct_median"]],
    })


# -----------------------------
# prose per-config detail
# -----------------------------

def print_config_reports(summary_df):
    if summary_df.empty:
        return

    print("\nconfig reports:")

    for _, r in summary_df.iterrows():
        k, n = int(r["solved_seeds"]), int(r["n_seeds"])
        lo, hi = r["solved_ci_low"], r["solved_ci_high"]

        print("\n" + "=" * 90)
        print(f"CONFIG: {r['config']}")
        print("=" * 90)

        print(
            f"Solved rate: {fmt_pct(r['solved_rate'], 1)} ({k}/{n} seeds), "
            f"95% Wilson [{fmt_pct(lo, 1)}, {fmt_pct(hi, 1)}]."
        )
        print(
            f"Clean-solve (crossed the valley, no death cap): "
            f"{fmt_pct(r['clean_solve_rate'], 1)} ({int(r['clean_solve_seeds'])}/{n}) — "
            f"the gap to solved is the survival cost of the crossing."
        )
        print(
            f"Comfort: median {fmt(r['mean_comfort_median'])}, mean {fmt(r['mean_comfort_mean'])}, "
            f"std {fmt(r['mean_comfort_std'])}, range {fmt(r['mean_comfort_min'])} to {fmt(r['mean_comfort_max'])}."
        )
        print(
            f"Deaths: median {fmt(r['eval_deaths_median'], 1)}, mean {fmt(r['eval_deaths_mean'], 1)}, "
            f"range {fmt(r['eval_deaths_min'], 0)} to {fmt(r['eval_deaths_max'], 0)}, "
            f"zero-death rate {fmt_pct(r['zero_death_rate'], 1)}."
        )
        print(
            f"Resource occupancy: food {fmt(r['food_visit_pct_median'], 2)}%, "
            f"water {fmt(r['water_visit_pct_median'], 2)}%, "
            f"water:food ratio {fmt(r['water_food_visit_ratio_median'], 2)}."
        )
        print(
            f"Camp shape: water-camp score {fmt(r['water_camp_score_median'], 2)}, "
            f"dominant-cell occupancy {fmt(r['dominant_cell_pct_median'], 2)}%."
        )
        print(
            f"Water→food route: success rate {fmt(r['water_to_food_success_rate_median'], 3)}, "
            f"success count median {fmt(r['water_to_food_success_count_median'], 1)}, "
            f"path efficiency {fmt(r['water_to_food_path_efficiency_median'], 3)}, "
            f"perfect-ish trip rate {fmt(r['water_to_food_perfectish_trip_rate_median'], 3)}."
        )
        print(
            f"Food→water route: success rate {fmt(r['food_to_water_success_rate_median'], 3)}, "
            f"success count median {fmt(r['food_to_water_success_count_median'], 1)}, "
            f"path efficiency {fmt(r['food_to_water_path_efficiency_median'], 3)}."
        )
        print(
            f"Two-way route floor: {fmt(r['two_way_route_success_min_median'], 3)}; "
            f"total successful resource trips median {fmt(r['total_successful_resource_trips_median'], 1)}."
        )
        print(f"Bug check: max non-neighbour jumps = {fmt(r['non_neighbor_jumps_max'], 0)}.")


# -----------------------------
# one-call orchestrator
# -----------------------------

def load_and_report(experiment_name, PROTOTYPE_NAME, recompute_metrics=True, save_csv=True, detail=True):
    res, manifest = load_sweep(experiment_name, PROTOTYPE_NAME, recompute_metrics=recompute_metrics)

    runs_df = make_runs_df(res)
    summary_df = make_summary_df(runs_df)

    print(f"loaded experiment: {experiment_name}")
    print(f"configs: {len(res)}   runs: {len(runs_df)}")

    display(make_summary_display_df(summary_df))

    if detail:
        print_config_reports(summary_df)

    if save_csv:
        out = sweep_dir(experiment_name, PROTOTYPE_NAME)
        runs_df.to_csv(out / "runs_df.csv", index=False)
        summary_df.to_csv(out / "summary_df.csv", index=False)
        print(f"\nsaved summary CSVs to: {out}")

    return res, manifest, runs_df, summary_df