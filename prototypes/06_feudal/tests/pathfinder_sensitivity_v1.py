"""
pathfinder_sensitivity_v1.py  —  does pathfinder quality show up in any metric we have?

    python tests/pathfinder_sensitivity_v1.py --tag v1
    python tests/pathfinder_sensitivity_v1.py --tag v1 --eval beside_water
    python tests/pathfinder_sensitivity_v1.py --tag v1 --eval both --seeds 8

THE PROBLEM THIS SOLVES. Swapping the learned pathfinder in moved solveScore by exactly
0.000 (35 eval deaths in both arms, 34,697 invocations). That reads as "the module is
fine", but it equally supports "the metric cannot see this module". On the nondoomed eval
deaths split hydration 29 / satiation 6 — failures to FIND water — and the pathfinder only
runs once a goal is already known. The metric is close to orthogonal to commute execution.

So a null result here is uninterpretable on its own. The fix is a POSITIVE CONTROL: sweep
`noisy_oracle` over epsilon, find where each metric starts to move, and read that as the
metric's detection floor. Then place the learned module against it.

  learned effect below the floor -> metric certifies nothing; use a sharper one, or accept
                                    the module is indistinguishable from oracle at this
                                    resolution and say so
  learned effect above the floor -> the null is real and the module is genuinely fine
  no epsilon moves the metric    -> the metric is blind to pathfinders entirely; stop
                                    quoting it about them

TWO EVALS, because they load the pathfinder differently:
  nondoomed     the standing config (solveScore 0.48). Exploration-dominated.
  beside_water  food isolation, water handed over (solveScore 0.86). The commute IS the
                task, so pathfinder quality has somewhere to show up.

METRICS ARE COMPUTED HERE, NOT VIA compute_eval_metrics. That function returns {} on any
Proto 06 run: it requires `action_T`, and sim_instance emits `abstract_action_T` under the
new QueuedAction scheme. `extract_resource_trips` needs only coordinates and deaths, so the
trip metrics are reachable — the segment-aware slicing below mirrors sweep_fn_v5 lines
652-717 with the action-mix section (the part that needs action_T) dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

from model_modules.contract_v1 import ModuleSpec  # noqa: E402
from sweep_fn_v5 import extract_resource_trips, summarise_trips  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20

# metrics reported, in the order they are printed
TRIP_KEYS = [
    "water_to_food_success_rate",
    "water_to_food_path_efficiency",
    "water_to_food_perfectish_trip_rate",
    "water_to_food_median_success_moves",
    "food_to_water_success_rate",
    "food_to_water_path_efficiency",
    "food_to_water_perfectish_trip_rate",
]


def _oracle_params(pathfinder_kwargs: dict | None = None) -> dict:
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": pathfinder_kwargs or {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": {
            "persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
            "follow_p": 0.95, "reverse_on_drop_p": 0.75,
        },
    }


EVALS = {
    "nondoomed": dict(eval_spawn_nondoomed=True, spawn_leeway=10,
                      eval_spawn_beside_water=False),
    # nondoomed takes precedence in sim_instance, so it must be off for this one to apply
    "beside_water": dict(eval_spawn_nondoomed=False, eval_spawn_beside_water=True),
}


def _run(seed: int, spec: ModuleSpec, oracle_params: dict, eval_kind: str,
         sim_len=7000, eval_len=5000):
    import sim_instance_v3 as S

    return S.sim_instance(
        seed=seed, sim_len=sim_len, eval_len=eval_len,
        env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7, smell_radius=3,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=oracle_params, module_spec=spec, log_every=10**9,
        eval_god_memory=False,
        **EVALS[eval_kind],
    )


# --- trip metrics, segment-aware ------------------------------------------------

def trip_metrics(run) -> dict:
    """
    Pooled water->food and food->water trip stats, each segment scored against its OWN
    map's resource coordinates. Pooling across segments without re-basing the coordinates
    would score a trip on map B against map A's water, which is the bug the segment-aware
    rewrite of compute_eval_metrics fixed.
    """
    eb = int(run["eval_boundary"])
    coords_eval = np.asarray(run["coordinates_T"][eb:], dtype=object)
    death_eval = np.asarray(run["death_T"][eb:]).astype(bool)
    segments = run.get("eval_segments") or []

    bounds = []
    if segments:
        for i, seg in enumerate(segments):
            a = max(0, int(seg["t_start"]) - eb)
            z = (int(segments[i + 1]["t_start"]) - eb
                 if i + 1 < len(segments) else len(coords_eval))
            z = min(len(coords_eval), z)
            if z > a:
                bounds.append((seg["water_coords"], seg["food_coords"], a, z))
    else:
        bounds.append((run["water_coords"], run["food_coords"], 0, len(coords_eval)))

    wf, fw = [], []
    for wcoords, fcoords, a, z in bounds:
        wset = list(set(map(tuple, wcoords)))
        fset = list(set(map(tuple, fcoords)))
        sc, sd = coords_eval[a:z], death_eval[a:z]
        wf += extract_resource_trips(coords_eval=sc, death_eval=sd,
                                     source_coords=wset, target_coords=fset)
        fw += extract_resource_trips(coords_eval=sc, death_eval=sd,
                                     source_coords=fset, target_coords=wset)

    m = {}
    m.update(summarise_trips(wf, "water_to_food"))
    m.update(summarise_trips(fw, "food_to_water"))
    m["two_way_route_success_min"] = float(np.nanmin([
        m["water_to_food_success_rate"], m["food_to_water_success_rate"]]))
    m["n_eval_segments"] = len(bounds)
    return m


def score_arm(spec, oracle_params, eval_kind: str, seeds) -> dict:
    """Pools trips across seeds rather than averaging per-seed rates: a seed with two
    trips should not carry the same weight as one with two hundred."""
    D = T = 0
    per_seed = []
    agg = {k: [] for k in TRIP_KEYS}
    agg["two_way_route_success_min"] = []

    for s in seeds:
        run = _run(s, spec, oracle_params, eval_kind)
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        tm = trip_metrics(run)
        per_seed.append(tm)
        for k in agg:
            agg[k].append(tm.get(k, np.nan))

    out = {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "n_seeds": len(per_seed),
    }
    for k, v in agg.items():
        out[k] = float(np.nanmedian(v)) if len(v) else float("nan")
    return out


# --- calibration ----------------------------------------------------------------

def calibrate(eval_kind: str, seeds, epsilons) -> dict:
    """
    Sweep the positive control. epsilon is the probability of a random step, so expected
    optimality is roughly 1 - (5/6)*epsilon: eps 0.02 ~ 0.983, eps 0.06 ~ 0.95, eps 0.12 ~
    0.90. Those bracket the learned module's measured 0.9804 on the commute band.
    """
    rows = {}
    for eps in epsilons:
        spec = ModuleSpec("oracle", "noisy_oracle", "oracle", "oracle",
                          explorer="smell_momentum")
        rows[eps] = score_arm(spec, _oracle_params({"epsilon": eps, "seed": 0}),
                              eval_kind, seeds)
    return rows


def run_eval(tag: str, eval_kind: str, seeds, epsilons) -> dict:
    cal = calibrate(eval_kind, seeds, epsilons)
    learned = score_arm(
        ModuleSpec("oracle", "learned", "oracle", "oracle", explorer="smell_momentum"),
        _oracle_params({"weights": tag, "scale": RADIUS}),
        eval_kind, seeds,
    )
    return {"calibration": cal, "learned": learned}


def report(eval_kind: str, res: dict, tag: str) -> None:
    cal, learned = res["calibration"], res["learned"]
    base = cal[0.0]

    print(f"\n--- eval: {eval_kind} " + "-" * (52 - len(eval_kind)))
    cols = ["solveScore", "water_to_food_path_efficiency",
            "water_to_food_perfectish_trip_rate", "water_to_food_success_rate",
            "two_way_route_success_min"]
    short = ["solveScr", "wf_pathEff", "wf_perfect", "wf_success", "twoWay_min"]

    print(f"{'arm':<18} {'deaths':>7} " + " ".join(f"{s:>11}" for s in short))
    for eps in sorted(cal):
        r = cal[eps]
        label = "oracle (eps 0)" if eps == 0.0 else f"noisy eps={eps:g}"
        print(f"{label:<18} {r['eval_deaths']:>7} "
              + " ".join(f"{r[c]:>11.4f}" for c in cols))
    print(f"{'learned ' + tag:<18} {learned['eval_deaths']:>7} "
          + " ".join(f"{learned[c]:>11.4f}" for c in cols))

    # Detection floor, GATED ON MONOTONICITY.
    #
    # A deviation threshold alone is not enough. Degradation is applied in increasing
    # doses, so a metric that genuinely tracks pathfinder quality must fall as epsilon
    # rises. One that wanders is responding to seed noise, and the first epsilon to clear
    # a 2% threshold is then meaningless — it reports a floor for a metric that cannot
    # rank pathfinders at all. solveScore does exactly this: it scored a knowably WORSE
    # pathfinder higher than the oracle, so any "floor" computed from it is an artifact.
    print("\n  detection floor (monotone-gated; >2% deviation from oracle):")
    eps_sorted = sorted(cal)
    for c, s in zip(cols, short):
        series = [cal[e][c] for e in eps_sorted]
        finite = [v for v in series if np.isfinite(v)]
        # allow a small tolerance so quantised metrics do not fail on flat stretches
        monotone = all(b <= a + 1e-9 for a, b in zip(finite, finite[1:]))

        d_learn = (abs(learned[c] - base[c]) / abs(base[c])
                   if np.isfinite(base[c]) and base[c] else float("nan"))

        if not monotone:
            print(f"    {s:<12} NOT MONOTONE in epsilon -> cannot rank pathfinders; "
                  f"any floor here is noise. Do not quote about this module.")
            continue

        floor = None
        for eps in eps_sorted:
            if eps == 0.0 or not np.isfinite(base[c]) or base[c] == 0:
                continue
            if abs(cal[eps][c] - base[c]) / abs(base[c]) > 0.02:
                floor = eps
                break

        if floor is None:
            print(f"    {s:<12} monotone but never moves >2% -> too coarse to certify")
            continue

        floor_effect = abs(cal[floor][c] - base[c]) / abs(base[c])
        ratio = floor_effect / d_learn if d_learn > 0 else float("inf")
        call = "CERTIFIED (learned well inside floor)" if ratio >= 3 else "inconclusive"
        print(f"    {s:<12} floor eps={floor:g} (effect {floor_effect*100:.1f}%); "
              f"learned {d_learn*100:.1f}%  -> {call}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v1")
    p.add_argument("--eval", default="both",
                   choices=["nondoomed", "beside_water", "both"])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[0.0, 0.02, 0.06, 0.12, 0.25])
    a = p.parse_args()

    kinds = ["nondoomed", "beside_water"] if a.eval == "both" else [a.eval]
    print("=" * 78)
    print(f"pathfinder sensitivity — tag '{a.tag}', {a.seeds} seeds, "
          f"epsilons {a.epsilons}")
    print("positive control: does ANY metric move when the pathfinder is knowably worse?")
    for k in kinds:
        report(k, run_eval(a.tag, k, range(a.seeds), a.epsilons), a.tag)
    print("=" * 78)


if __name__ == "__main__":
    main()
