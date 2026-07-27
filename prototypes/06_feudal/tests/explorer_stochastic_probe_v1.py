"""
explorer_stochastic_probe_v1.py  —  is the policy CLASS the problem?

    python tests/explorer_stochastic_probe_v1.py --tags f0 f1 f2

No retraining. The same weights are evaluated as a greedy policy (temperature 0, what was
already measured) and as stochastic policies sampling `softmax(Q/T)` over legal moves. If
the trained Q-function contains useful knowledge that argmax is destroying, this finds it in
minutes.

THE HYPOTHESIS BEING TESTED.

Q-learning returns a greedy deterministic policy. For memoryless POMDP policies the best
deterministic policy can be arbitrarily suboptimal, with the optimum requiring stochasticity
(Singh, Jaakkola & Jordan 1994). Two facts say this task is in that regime:

  - In the dead band both smells read zero, so the observation collapses to (legality,
    recent actions). Physically distinct cells alias onto one input, and a deterministic map
    from that either walks straight or cycles. Nothing in the TD loss prefers the former.
  - Every baseline that beats the learned module is stochastic: random (fully),
    momentum (persist_p 0.75), smell_momentum (persist_p 0.75, follow_p 0.95,
    reverse_on_drop_p 0.75).

PRE-REGISTERED READINGS.

  inverted-U, peak > argmax     Confirms the class problem. Q learned something argmax
                                cannot express. Fix is policy-gradient (a distribution is
                                the native output), not more Q-learning.
  flat, or peak == argmax       Class is not the binding constraint. The Q-function has no
                                usable structure, and the aliasing/credit problem is
                                upstream of how actions are selected.
  peak still < random 0.1712    The Q-function is actively anti-informative — worse than no
                                information at all. That points at the 147x oversampling of
                                positives without importance correction, which biases values
                                everywhere.

High T must degenerate toward uniform-over-legal, which should land near the random floor
(0.1712). That is the sanity anchor: if T=10 does NOT approach it, the sampler is wrong and
nothing else here is readable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

from model_modules.contract_v1 import ModuleSpec  # noqa: E402
from training.rigs.insim_rig_v1 import count_discoveries  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20
REACTIVE_FLOOR = 0.4776
RANDOM_FLOOR = 0.1712


def _oracle_params(explorer_kwargs):
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": explorer_kwargs,
    }


def score(tag, temperature, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    D = T = 0
    disc = 0
    for s in seeds:
        run = S.sim_instance(
            seed=s, sim_len=sim_len, eval_len=eval_len,
            env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
            decay_mult=0.7, smell_radius=3, curriculum_mode="band",
            c_min=2, c_max=9, band_width=2, life_cap=1000,
            oracle_params=_oracle_params({"weights": tag, "temperature": temperature}),
            module_spec=ModuleSpec("oracle", "oracle", "oracle", "oracle",
                                   explorer="learned"),
            log_every=10**9, eval_spawn_nondoomed=True, spawn_leeway=10,
            eval_god_memory=False,
        )
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        disc += count_discoveries(run)

    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "discoveries": disc,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--temps", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    a = p.parse_args()

    seeds = range(a.seeds)
    print("=" * 78)
    print(f"stochastic policy probe — {a.seeds} seeds, no retraining")
    print(f"argmax (T=0) is the number already on record; T>0 samples softmax(Q/T)")
    print(f"anchors: random {RANDOM_FLOOR:.4f} | reactive floor {REACTIVE_FLOOR:.4f}")

    t0 = time.time()
    table = {}
    for tag in a.tags:
        print(f"\n{tag}")
        print(f"  {'T':>6} {'deaths':>7} {'solveScore':>11} {'disc':>6}  vs argmax")
        base = None
        for T in a.temps:
            r = score(tag, T, seeds)
            table[(tag, T)] = r
            if base is None:
                base = r["solveScore"]
            delta = r["solveScore"] - base
            mark = "  <- argmax" if T == 0.0 else f"  {delta:+.4f}"
            print(f"  {T:>6.2f} {r['eval_deaths']:>7} {r['solveScore']:>11.4f} "
                  f"{r['discoveries']:>6}{mark}   ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "-" * 78)
    for tag in a.tags:
        vals = {T: table[(tag, T)]["solveScore"] for T in a.temps}
        argmax_v = vals[a.temps[0]]
        best_T = max(vals, key=lambda t: vals[t])
        best_v = vals[best_T]
        gain = best_v - argmax_v
        hi_T = max(a.temps)
        print(f"{tag}: argmax {argmax_v:.4f} -> best {best_v:.4f} at T={best_T:g} "
              f"({gain:+.4f})")
        # ORDER MATTERS: beating random is checked BEFORE a gain over argmax is called a
        # result. v1 declared "class was binding" on gain alone and mis-read f0, whose peak
        # sat at the MOST uniform temperature — the gain was the policy degenerating toward
        # random, not the Q-function being unlocked. A rise that asymptotes at the random
        # floor means the best use of the Q-function is to ignore it.
        peak_is_uniform = best_T == max(a.temps)
        if best_v < RANDOM_FLOOR + 0.02:
            print(f"   peak {best_v:.4f} does not clear random {RANDOM_FLOOR:.4f} at ANY "
                  f"temperature -> the Q-function is anti-informative, not merely "
                  f"inexpressible by argmax.")
            if peak_is_uniform:
                print(f"   peak sits at the MOST uniform T -> the gain is degeneration "
                      f"toward random, not recovered knowledge.")
        elif gain > 0.05 and not peak_is_uniform:
            print(f"   stochasticity recovers performance ABOVE random at an intermediate "
                  f"T -> the POLICY CLASS was binding. Move to policy gradient.")
        else:
            print(f"   no material gain -> class is not the constraint; the problem is "
                  f"upstream (aliasing / credit), not action selection.")
        print(f"   sanity: T={hi_T:g} lands {vals[hi_T]:.4f} vs uniform anchor "
              f"{RANDOM_FLOOR:.4f} "
              f"({'consistent' if abs(vals[hi_T] - RANDOM_FLOOR) < 0.08 else 'OFF'})")
    print("=" * 78)


if __name__ == "__main__":
    main()
