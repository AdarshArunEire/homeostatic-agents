"""
reactive_ceiling_v1.py  —  where is the reactive optimum actually?

    python tests/reactive_ceiling_v1.py                 # coordinate sweep (fast)
    python tests/reactive_ceiling_v1.py --mode grid     # full grid (slower)

WHY THIS EXISTS. Every explorer result so far has been quoted against 0.4776, described as
"the reactive floor" and treated as the bar a learned explorer must clear. That number is ONE
hand-picked point in a five-parameter space. It was never swept. The class default is
`persist_p=0.70`; the standing config uses 0.75 — so it was nudged once, not optimised.

Three different things have been conflated:

  the number smell_momentum happens to score with these parameters   (measured: 0.4776)
  the best achievable by ANY setting of the same parameters          (unknown)
  the best achievable by any reactive policy on this observation     (unknown, >= the above)

Only the first is known. This sweeps the second, which brackets the third from below.

THE COMPARISON IS INFORMATION-FAIR, and that is what makes the target legitimate.
SmellMomentumExplorer reads: legal_moves, water_smell[0]/[1], food_smell[0]/[1], need_water,
need_food, and its own last_dir. The learned explorer receives all of it and more — three
smell readings rather than two, three steps of action history rather than one. So a learned
policy is not handicapped relative to the baseline; if it cannot match it, that is a
statement about the learning method, not about access to information.

WHAT THE ANSWER CHANGES.

  baseline sits at/near the peak   0.4776 IS the reactive ceiling. Beating it requires
                                   leaving the reactive class — spatial memory, coverage —
                                   which the current observation contract cannot support.
                                   That becomes a clean Proto 06 verdict and a specification
                                   for Proto 07.
  a better setting exists          The bar was mis-stated. Every "fails to beat the floor"
                                   verdict was measured against the wrong number, and the
                                   real target is higher still.
  the surface is flat              Parameters barely matter, and the 0.48-vs-0.10 gap is
                                   about policy STRUCTURE, not tuning.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

from model_modules.contract_v1 import ModuleSpec  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20
RANDOM_FLOOR = 0.1712

BASELINE = {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
            "follow_p": 0.95, "reverse_on_drop_p": 0.75}

AXES = {
    "persist_p": [0.40, 0.55, 0.70, 0.75, 0.85, 0.95],
    "follow_p": [0.60, 0.75, 0.85, 0.95, 1.00],
    "reverse_on_drop_p": [0.25, 0.45, 0.60, 0.75, 0.90],
    "trend_eps": [0.001, 0.01, 0.05, 0.15],
}


def score(kw, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    D = T = 0
    for s in seeds:
        run = S.sim_instance(
            seed=s, sim_len=sim_len, eval_len=eval_len,
            env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
            decay_mult=0.7, smell_radius=3, curriculum_mode="band",
            c_min=2, c_max=9, band_width=2, life_cap=1000,
            oracle_params={
                "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL,
                                 "h_crit": 0.7, "s_crit": 0.7},
                "pathfinder": {}, "eat": {"fill_target": S_FILL},
                "drink": {"fill_target": H_FILL}, "explorer": kw,
            },
            module_spec=ModuleSpec("oracle", "oracle", "oracle", "oracle",
                                   explorer="smell_momentum"),
            log_every=10**9, eval_spawn_nondoomed=True, spawn_leeway=10,
            eval_god_memory=False,
        )
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
    from sweep_fn_v5 import wilson_interval

    n = T + D
    lo, hi = wilson_interval(T, n) if n > 0 else (float("nan"), float("nan"))
    return {"eval_deaths": int(D), "n_events": int(n),
            "solveScore": (T / n) if n > 0 else float("nan"),
            "ci": (lo, hi)}


def coord_sweep(seeds):
    """
    Vary one parameter at a time from the baseline. ~20 configs rather than 600.

    Cannot find interactions, but answers the question that matters first: is the baseline
    even at a LOCAL maximum? If any single-parameter change improves it, 0.4776 is not the
    reactive optimum and no further argument is needed.
    """
    t0 = time.time()
    base = score(dict(BASELINE), seeds)
    print(f"  baseline {BASELINE}")
    print(f"    solveScore {base['solveScore']:.4f}  deaths {base['eval_deaths']}  "
          f"({time.time()-t0:.0f}s)\n")

    results = {("baseline", None): base}
    for axis, values in AXES.items():
        print(f"  {axis}:")
        for v in values:
            kw = dict(BASELINE)
            kw[axis] = v
            r = score(kw, seeds)
            results[(axis, v)] = r
            flag = "  <- baseline" if v == BASELINE[axis] else ""
            delta = r["solveScore"] - base["solveScore"]
            # overlap with the baseline CI is the only thing that makes a delta a result
            sep = "" if (r["ci"][0] <= base["ci"][1] and base["ci"][0] <= r["ci"][1]) \
                else "  SEPARATED"
            print(f"    {v:<8g} solve {r['solveScore']:.4f} "
                  f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  deaths {r['eval_deaths']:>4}  "
                  f"{delta:+.4f}{flag}{sep}   ({time.time()-t0:.0f}s)", flush=True)
        print()
    return base, results


def grid_sweep(seeds):
    t0 = time.time()
    keys = ["persist_p", "follow_p", "reverse_on_drop_p"]
    combos = list(itertools.product(*[AXES[k] for k in keys]))
    print(f"  {len(combos)} configs over {keys} (trend_eps held at baseline)\n")

    rows = []
    for i, vals in enumerate(combos, 1):
        kw = dict(BASELINE)
        kw.update(dict(zip(keys, vals)))
        r = score(kw, seeds)
        rows.append((vals, r))
        if i % 10 == 0 or i == len(combos):
            print(f"    {i}/{len(combos)}  ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda x: -x[1]["solveScore"])
    print(f"\n  top 10 of {len(rows)}:")
    print(f"    {'persist':>8} {'follow':>8} {'revDrop':>8} {'solve':>9} {'deaths':>7}")
    for vals, r in rows[:10]:
        print(f"    {vals[0]:>8g} {vals[1]:>8g} {vals[2]:>8g} "
              f"{r['solveScore']:>9.4f} {r['eval_deaths']:>7}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="coord", choices=["coord", "grid"])
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()

    seeds = range(a.seeds)
    print("=" * 78)
    print(f"reactive ceiling — is 0.4776 the optimum, or just the point we picked?")
    print(f"smell 3, band (9,11), nondoomed, {a.seeds} seeds. random floor {RANDOM_FLOOR}")
    print("=" * 78)

    if a.mode == "coord":
        base, results = coord_sweep(seeds)
        best_key, best = max(results.items(), key=lambda kv: kv[1]["solveScore"])
        gain = best["solveScore"] - base["solveScore"]
        print("-" * 78)
        print(f"baseline {base['solveScore']:.4f} [{base['ci'][0]:.3f},{base['ci'][1]:.3f}]"
              f" | best single change {best['solveScore']:.4f} "
              f"[{best['ci'][0]:.3f},{best['ci'][1]:.3f}] "
              f"({best_key[0]}={best_key[1]}) {gain:+.4f}")

        # A POINT ESTIMATE IS NOT A RESULT. The max of ~20 noisy configs sits ~2sd above
        # the mean even when every config is identical, which is the 03b lesson
        # ("apparent winners at n<=10 are often the same config with wide uncertainty").
        # Only a config whose CI clears the baseline's counts.
        separated = [(k, r) for k, r in results.items()
                     if k != ("baseline", None) and r["ci"][0] > base["ci"][1]]
        if separated:
            print(f"  -> {len(separated)} config(s) SEPARATED from the baseline CI:")
            for k, r in separated:
                print(f"       {k[0]}={k[1]}  {r['solveScore']:.4f} "
                      f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
            print("     The bar was mis-stated; re-anchor before drawing conclusions.")
        else:
            print(f"  -> NO config separates from the baseline CI "
                  f"[{base['ci'][0]:.3f},{base['ci'][1]:.3f}] at these seeds. The spread is "
                  f"sampling noise, not a tuning gradient. Consistent with the baseline "
                  f"already sitting at the reactive optimum; raise --seeds to tighten.")
    else:
        rows = grid_sweep(seeds)
        best_vals, best = rows[0]
        base = score(dict(BASELINE), seeds)
        print("-" * 78)
        print(f"baseline {base['solveScore']:.4f} | grid best {best['solveScore']:.4f} "
              f"at persist={best_vals[0]:g} follow={best_vals[1]:g} "
              f"revDrop={best_vals[2]:g}  ({best['solveScore']-base['solveScore']:+.4f})")
        spread = rows[0][1]["solveScore"] - rows[-1][1]["solveScore"]
        print(f"surface spread across the grid: {spread:.4f}")
        if spread < 0.05:
            print("  -> the surface is FLAT. Parameters barely matter, so the gap between "
                  "0.48 and the learned ~0.10 is about policy STRUCTURE, not tuning.")
    print("=" * 78)


if __name__ == "__main__":
    main()
