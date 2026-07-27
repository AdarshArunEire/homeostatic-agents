"""
explorer_sensitivity_v1.py  —  calibrate FIRST, then read the thesis result.

    python tests/explorer_sensitivity_v1.py --calibrate-only
    python tests/explorer_sensitivity_v1.py --tags e0 e1 e2

`--calibrate-only` needs no trained module and should be run BEFORE training. The consumer
thread cost four training runs discovering afterwards that the environment could barely
resolve the module; the rule taken from it is calibrate first, train second.

The control is `smell_momentum` + epsilon uniform legal moves, so epsilon spans the ladder
already on record:

    random 0.17  <  momentum 0.33  <  smell_momentum 0.48  <<  god 1.00

At epsilon=1.0 the control should land near the random floor (~0.17). That is an ANCHOR: if
it does not, the instrument is wrong and nothing downstream can be trusted. Unlike the
consumer, this metric is expected to have real dynamic range — 0.83 of it.

VERDICT PROTOCOL, fixed before any learned number exists:

  beats the floor   min solveScore across 3 seeds > 0.48, with the seed spread reported
  moves the needle  the gain is large relative to the detection floor of the same metric
  fails             does not clear 0.48 across seeds

  A fail is only "memory does not help" if the training discovery counter was RISING. If it
  was flat, the correct reading is "the reward never reached the policy" — a different
  claim with a different fix, and the distinction Proto 04's Go-Explore probe existed to
  make.
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
from training.rigs.insim_rig_v1 import count_discoveries  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20
REACTIVE_FLOOR = 0.48
RANDOM_ANCHOR = 0.17


def _oracle_params(explorer_kwargs):
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": explorer_kwargs,
    }


def score_arm(impl, explorer_kwargs, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    D = T = 0
    disc = 0
    causes = {}
    for s in seeds:
        run = S.sim_instance(
            seed=s, sim_len=sim_len, eval_len=eval_len,
            env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
            decay_mult=0.7, smell_radius=3, curriculum_mode="band",
            c_min=2, c_max=9, band_width=2, life_cap=1000,
            oracle_params=_oracle_params(explorer_kwargs),
            module_spec=ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer=impl),
            log_every=10**9, eval_spawn_nondoomed=True, spawn_leeway=10,
            eval_god_memory=False,
        )
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        disc += count_discoveries(run)
        eb = int(run["eval_boundary"])
        for e in run.get("death_events", []):
            if e["t"] >= eb:
                causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "discoveries": disc,
        "causes": causes,
    }


REACTIVE_KW = {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
               "follow_p": 0.95, "reverse_on_drop_p": 0.75}


def calibrate(seeds, epsilons):
    """
    Prints each arm as it lands. 3 ladder arms + len(epsilons) noisy arms, each over every
    seed at 7000 ticks — silence for minutes is indistinguishable from a hang, and the
    ladder arms are also a live sanity check (random should come out near 0.17,
    smell_momentum near 0.48) so seeing them early is worth the noise.
    """
    import time

    plan = [
        (("ladder", "random"), "random", {}),
        # persist_p 0.75 to match the recorded ladder: god_vs_legal_check applies one
        # explorer param block to every arm. The factory default of 0.85 scores 0.278
        # instead of the ledger's 0.33 and is not comparable.
        (("ladder", "momentum"), "momentum", {"persist_p": 0.75}),
        (("ladder", "smell_momentum"), "smell_momentum", dict(REACTIVE_KW)),
    ] + [
        (("noisy", eps), "noisy_oracle", dict(REACTIVE_KW, epsilon=eps))
        for eps in epsilons
    ]

    rows = {}
    t0 = time.time()
    for i, (key, impl, kw) in enumerate(plan, 1):
        label = key[1] if key[0] == "ladder" else f"noisy eps={key[1]:g}"
        print(f"  [{i}/{len(plan)}] {label} ...", end="", flush=True)
        rows[key] = score_arm(impl, kw, seeds)
        print(f" solve {rows[key]['solveScore']:.4f}  "
              f"deaths {rows[key]['eval_deaths']}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def report_calibration(rows, seeds):
    print(f"\n--- calibration ({len(list(seeds))} seeds) " + "-" * 34)
    print(f"{'arm':<26} {'deaths':>7} {'solveScore':>11} {'disc':>7}  causes")
    for key in [("ladder", "random"), ("ladder", "momentum"), ("ladder", "smell_momentum")]:
        r = rows[key]
        print(f"{key[1]:<26} {r['eval_deaths']:>7} {r['solveScore']:>11.4f} "
              f"{r['discoveries']:>7}  {r['causes']}")
    for key in sorted(k for k in rows if k[0] == "noisy"):
        r = rows[key]
        print(f"{'noisy eps=' + format(key[1], 'g'):<26} {r['eval_deaths']:>7} "
              f"{r['solveScore']:>11.4f} {r['discoveries']:>7}  {r['causes']}")

    base = rows[("ladder", "smell_momentum")]["solveScore"]
    eps_keys = sorted(k for k in rows if k[0] == "noisy")
    series = [base] + [rows[k]["solveScore"] for k in eps_keys]
    finite = [x for x in series if np.isfinite(x)]
    monotone = all(b <= a + 1e-9 for a, b in zip(finite, finite[1:]))

    print("\n  instrument checks:")
    print(f"    monotone in epsilon: {'YES' if monotone else 'NO — instrument unusable'}")
    if eps_keys:
        top = rows[eps_keys[-1]]["solveScore"]
        near = abs(top - RANDOM_ANCHOR) < 0.10
        print(f"    eps={eps_keys[-1][1]:g} lands {top:.3f} vs random anchor "
              f"{RANDOM_ANCHOR:.2f}: {'consistent' if near else 'OFF — check the control'}")
    floor = None
    for k in eps_keys:
        if np.isfinite(base) and base and abs(rows[k]["solveScore"] - base) / base > 0.02:
            floor = k[1]
            break
    print(f"    detection floor: {('eps=' + format(floor, 'g')) if floor else 'none <2%'}")
    print(f"    dynamic range: {min(finite):.3f} - {max(finite):.3f} "
          f"({max(finite) - min(finite):.3f})")
    return {"monotone": monotone, "floor": floor, "base": base}


POLICY_MODE = {"on": False}


def report_learned(tags, seeds, cal):
    print(f"\n--- learned explorers ({len(tags)} seeds) " + "-" * 30)
    print(f"{'tag':<12} {'deaths':>7} {'solveScore':>11} {'disc':>7}  vs floor 0.48")
    scores = []
    for tag in tags:
        kw = {"weights": tag}
        if POLICY_MODE["on"]:
            kw["policy_mode"] = True
        r = score_arm("learned", kw, seeds)
        scores.append(r["solveScore"])
        delta = r["solveScore"] - REACTIVE_FLOOR
        print(f"{tag:<12} {r['eval_deaths']:>7} {r['solveScore']:>11.4f} "
              f"{r['discoveries']:>7}  {delta:+.4f}")

    finite = [s for s in scores if np.isfinite(s)]
    if not finite:
        print("  no finite scores")
        return
    lo, hi = min(finite), max(finite)
    print(f"\n  across seeds: min {lo:.4f}  median {np.median(finite):.4f}  max {hi:.4f}"
          f"   (spread {hi - lo:.4f})")
    if lo > REACTIVE_FLOOR:
        print(f"  VERDICT: beats the reactive floor on ALL seeds "
              f"(worst {lo:.3f} > {REACTIVE_FLOOR:.2f}). Memory earns its keep.")
    elif hi <= REACTIVE_FLOOR:
        print(f"  VERDICT: fails to beat the floor on any seed (best {hi:.3f}). "
              f"Check the training discovery counter before reading this as "
              f"'memory does not help' — a flat counter means the reward never arrived.")
    else:
        print(f"  VERDICT: seed-dependent ({lo:.3f}-{hi:.3f} straddles {REACTIVE_FLOOR:.2f}). "
              f"Not a result — widen seeds before claiming either way.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.15, 0.4, 0.7, 1.0])
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--policy", action="store_true",
                   help="tags were trained by train_explorer_pg_v1 (policy net, sampled)")
    a = p.parse_args()
    POLICY_MODE["on"] = a.policy

    seeds = range(a.seeds)
    print("=" * 78)
    print(f"explorer sensitivity — smell 3, band (9,11), nondoomed, {a.seeds} seeds")
    print("calibrate first: can this metric rank explorers at all?")
    cal_rows = calibrate(seeds, a.epsilons)
    cal = report_calibration(cal_rows, seeds)

    if not a.calibrate_only and a.tags:
        if not cal["monotone"]:
            print("\n  REFUSING to score learned arms: the metric is not monotone in a "
                  "controlled degradation, so it cannot rank explorers.")
        else:
            report_learned(a.tags, seeds, cal)
    print("=" * 78)


if __name__ == "__main__":
    main()
