"""
consumer_sensitivity_v1.py  —  can any metric see consumer quality?

    python tests/consumer_sensitivity_v1.py --slot drink --tag v1
    python tests/consumer_sensitivity_v1.py --slot eat --tag v1 --seeds 8

Same discipline as pathfinder_sensitivity_v1, applied before any consumer result is read.
P1.1 established why: solveScore LOOKED like it had a detection floor until a controlled
degradation showed it was non-monotone. Nothing errored; the number was simply meaningless.

TWO DEGRADATION AXES, because the pathfinder showed error CHARACTER dominates error RATE.

  sigma    gaussian offset on every fill  — small errors, everywhere
  epsilon  random fill with probability e — rare errors, large

Sweeping one scalar cannot separate those, and they are not interchangeable: the consumer's
comfort surface has a free buffer to ideal+OVER_TOL, so a small offset may cost nothing at
all while an occasional wild fill overshoots the band. If sigma moves nothing and epsilon
moves a lot, the module needs checking against epsilon-shaped error, and its mean absolute
deviation from the oracle is the wrong summary of it.

A metric qualifies as a consumer detector only if it is MONOTONE in the degradation. The
learned module is then placed against the floor of a qualifying metric, never against a
metric that merely produced a different number.
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
from sweep_fn_v5 import compute_eval_metrics  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20

SLOT_CONFIG = {
    "drink": dict(eval_spawn_nondoomed=True, spawn_leeway=10,
                  eval_spawn_beside_water=False),
    "eat": dict(eval_spawn_nondoomed=False, eval_spawn_beside_water=True),
}

# consumer-plausible metrics. Overfill and underfill show up in drive levels and comfort
# long before they show up in deaths, so the drive percentiles are included deliberately.
COLS = [
    ("solveScore", "solveScr"),
    ("mean_comfort", "comfort"),
    ("p05_hydration", "p05_hyd"),
    ("p95_hydration", "p95_hyd"),
    ("p05_satiation", "p05_sat"),
    ("p95_satiation", "p95_sat"),
    ("drink_rate_at_water", "drinkRate"),
    ("eat_rate_at_food", "eatRate"),
]


def _oracle_params(slot: str, kwargs: dict) -> dict:
    p = {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                     "follow_p": 0.95, "reverse_on_drop_p": 0.75},
    }
    p[slot] = kwargs
    return p


def _spec(slot: str, impl: str) -> ModuleSpec:
    return ModuleSpec(
        orchestrator="oracle", pathfinder="oracle",
        eat=impl if slot == "eat" else "oracle",
        drink=impl if slot == "drink" else "oracle",
        explorer="smell_momentum",
    )


def score_arm(slot, impl, kwargs, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    D = T = 0
    rows = []
    for s in seeds:
        run = S.sim_instance(
            seed=s, sim_len=sim_len, eval_len=eval_len,
            env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
            decay_mult=0.7, smell_radius=3, curriculum_mode="band",
            c_min=2, c_max=9, band_width=2, life_cap=1000,
            oracle_params=_oracle_params(slot, kwargs), module_spec=_spec(slot, impl),
            log_every=10**9, eval_god_memory=False, **SLOT_CONFIG[slot],
        )
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        rows.append(compute_eval_metrics(run, dict(radius=RADIUS, band=(9, 11))))

    out = {"eval_deaths": int(D),
           "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan")}
    for key, _ in COLS:
        if key == "solveScore":
            continue
        vals = [r.get(key, np.nan) for r in rows]
        with np.errstate(all="ignore"):
            out[key] = float(np.nanmedian(vals)) if len(vals) else float("nan")
    return out


def sweep(slot, seeds, sigmas, epsilons, tag):
    base_kw = {"fill_target": H_FILL if slot == "drink" else S_FILL}
    arms = {("oracle", 0.0): score_arm(slot, "oracle", dict(base_kw), seeds)}

    for sg in sigmas:
        if sg == 0.0:
            continue
        arms[("sigma", sg)] = score_arm(
            slot, "noisy_oracle", dict(base_kw, sigma=sg), seeds)
    for ep in epsilons:
        if ep == 0.0:
            continue
        arms[("epsilon", ep)] = score_arm(
            slot, "noisy_oracle", dict(base_kw, epsilon=ep), seeds)

    learned = score_arm(slot, "learned", {"weights": tag}, seeds)
    return arms, learned


def report(slot, arms, learned, tag):
    base = arms[("oracle", 0.0)]
    keys = [k for k, _ in COLS]
    shorts = [s for _, s in COLS]

    print(f"\n--- consumer sensitivity: {slot} " + "-" * (44 - len(slot)))
    print(f"{'arm':<16} {'deaths':>7} " + " ".join(f"{s:>10}" for s in shorts))

    def row(label, r):
        print(f"{label:<16} {r['eval_deaths']:>7} "
              + " ".join(f"{r[k]:>10.4f}" if np.isfinite(r[k]) else f"{'nan':>10}"
                         for k in keys))

    row("oracle", base)
    for (kind, val) in sorted(a for a in arms if a != ("oracle", 0.0)):
        row(f"{kind}={val:g}", arms[(kind, val)])
    row(f"learned {tag}", learned)

    for axis in ("sigma", "epsilon"):
        levels = sorted(v for (k, v) in arms if k == axis)
        if not levels:
            continue
        print(f"\n  {axis} axis — monotone-gated detection floor:")
        for key, short in COLS:
            series = [base[key]] + [arms[(axis, v)][key] for v in levels]
            finite = [x for x in series if np.isfinite(x)]
            if len(finite) < 2:
                print(f"    {short:<10} no data")
                continue
            # comfort/solveScore fall as things worsen; drive percentiles can move either
            # way, so accept monotone in EITHER direction
            down = all(b <= a + 1e-9 for a, b in zip(finite, finite[1:]))
            up = all(b >= a - 1e-9 for a, b in zip(finite, finite[1:]))
            if not (down or up):
                print(f"    {short:<10} NOT MONOTONE -> cannot rank consumers; ignore it here")
                continue
            if not np.isfinite(base[key]) or base[key] == 0:
                print(f"    {short:<10} oracle baseline is 0/nan -> no usable ratio")
                continue

            floor = next((v for v in levels
                          if abs(arms[(axis, v)][key] - base[key]) / abs(base[key]) > 0.02),
                         None)
            d_learn = abs(learned[key] - base[key]) / abs(base[key])
            if floor is None:
                print(f"    {short:<10} monotone but never moves >2% -> too coarse")
                continue
            eff = abs(arms[(axis, floor)][key] - base[key]) / abs(base[key])
            ratio = eff / d_learn if d_learn > 0 else float("inf")
            call = "CERTIFIED" if ratio >= 3 else "inconclusive"
            print(f"    {short:<10} floor {axis}={floor:g} (effect {eff*100:.1f}%); "
                  f"learned {d_learn*100:.1f}%  -> {call}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", required=True, choices=["eat", "drink"])
    p.add_argument("--tag", default="v1")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.15, 0.35])
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.08, 0.25])
    a = p.parse_args()

    print("=" * 82)
    print(f"consumer sensitivity — slot '{a.slot}', tag '{a.tag}', {a.seeds} seeds")
    print(f"sigma (graded) {a.sigmas}   epsilon (rare, large) {a.epsilons}")
    arms, learned = sweep(a.slot, range(a.seeds), a.sigmas, a.epsilons, a.tag)
    report(a.slot, arms, learned, a.tag)
    print("=" * 82)


if __name__ == "__main__":
    main()
