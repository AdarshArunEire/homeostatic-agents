"""
orchestrator_hold_sweep_v1.py  —  does forcing commitment undo the collapse?

    python tests/orchestrator_hold_sweep_v1.py --tag o0
    python tests/orchestrator_hold_sweep_v1.py --tag o0 --blind
    python tests/orchestrator_hold_sweep_v1.py --tag o0 --hold-consume

P4.2 measured median GO_FOOD run length 1.0 against the oracle's 10.0 — the learned
orchestrator is a set of options with termination beta=1 everywhere, and options with
beta=1 are primitive actions. This sweeps the commitment horizon and asks whether that
alone recovers survival.

NO RETRAINING. The existing weights are re-evaluated with the horizon applied at
inference. That makes this the cheapest experiment that could produce a non-null, in the
P3.0 sense: if holding an already-collapsed policy's own first choice lifts survival, the
collapse WAS the failure and a retrain is worth doing. If it does not, the preference is
real and the retrain would have been wasted.

The horizon applies to GO_WATER / GO_FOOD / EXPLORE only. CONSUME is re-decided every
tick as before (`--hold-consume` to include it): the collapse is navigational, and
DRINK_AMOUNT=0.15 needs ~4 ticks to fill from ideal, so a 15-tick CONSUME hold would sit
11 ticks past target. That is the obvious way for this sweep to produce a spurious null.

By default options also terminate on their task condition — GO_* on arrival, EXPLORE on
a memory slot flipping. `--blind` disables that and runs a pure counter, which is the
cleaner single variable but wastes ticks standing on a resource it already reached.

METRIC WARNING. Median GO_FOOD run length is NOT a valid outcome here: the intervention
sets it to `hold` by construction. Read solveScore, the death split, and crossings per
life instead. The detector calibrated in P4.2 is invalidated by the thing it motivated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
for _p in (_PROTO_DIR, _PROTO_DIR / "figures"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from make_agent_gif_v1 import eval_segments, run  # noqa: E402

GO_WATER, GO_FOOD = 0, 1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def measure(res) -> dict:
    """solveScore plus the things the horizon should move if collapse was the failure."""
    EB = int(res["eval_boundary"])
    deaths = int(np.asarray(res["death_T"])[EB:].sum())
    timeouts = int(res["n_timeouts_eval"])

    causes = {"hydration": 0, "satiation": 0, "both": 0}
    for e in res["death_events"]:
        if int(e["t"]) >= EB:
            causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    act = np.asarray(res["abstract_action_T"])
    fk = np.asarray(res["food_known_T"]).astype(bool)
    coords = res["coordinates_T"]

    wf = lives = 0
    for t0, t1, seg in eval_segments(res):
        water = {tuple(c) for c in seg["water_coords"]}
        food = {tuple(c) for c in seg["food_coords"]}
        lives += 1
        last = None
        for p in (tuple(c) for c in coords[t0:t1]):
            if p in water:
                last = "water"
            elif p in food:
                if last == "water":
                    wf += 1
                last = "food"

    return dict(deaths=deaths, timeouts=timeouts, lives=lives, wf=wf,
                fk=int(fk[EB:].sum()), ticks=len(act) - EB,
                go_w=int((act[EB:] == GO_WATER).sum()),
                go_f=int((act[EB:] == GO_FOOD).sum()),
                **{f"c_{k}": v for k, v in causes.items()})


def pooled(ms: list[dict]) -> dict:
    return {k: sum(m[k] for m in ms) for k in ms[0]}


def row(name: str, m: dict) -> str:
    n = m["timeouts"] + m["deaths"]
    ss = m["timeouts"] / n if n else 0.0
    lo, hi = wilson(m["timeouts"], n)
    ratio = f'{m["go_w"] / m["go_f"]:.1f}:1' if m["go_f"] else "inf"
    return (f"{name:>16}  {ss:>10.4f}  [{lo:.2f},{hi:.2f}]  {m['deaths']:>7}"
            f"  {m['c_hydration']:>4}/{m['c_satiation']:<4}"
            f"  {m['wf'] / max(1, m['lives']):>8.2f}  {ratio:>8}"
            f"  {100 * m['fk'] / max(1, m['ticks']):>7.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="o0")
    p.add_argument("--holds", default="1,3,5,10,15")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--blind", action="store_true", help="counter only, no task termination")
    p.add_argument("--hold-consume", action="store_true", help="hold CONSUME too")
    p.add_argument("--eps", type=float, default=0.0,
                   help="CONTROL A. Probability a re-decision is a uniform random legal "
                        "action instead of the argmax. --eps 1.0 holds a RANDOM option "
                        "for the horizon, which separates 'commitment to this policy's "
                        "choices' from 'consulting a flat Q-function less often'. If the "
                        "random arm matches the learned arm, the policy is not what is "
                        "doing the work.")
    p.add_argument("--no-hold-explore", action="store_true",
                   help="CONTROL B. Never hold EXPLORE, only GO_*. Held EXPLORE is "
                        "temporally-extended exploration, which 05-H2 showed lifts "
                        "crossing supply on its own; this separates goal commitment from "
                        "an explorer with a longer leash.")
    p.add_argument("--skip-oracle", action="store_true")
    a = p.parse_args()

    holds = [int(h) for h in a.holds.split(",")]
    natural = not a.blind

    arm_note = ""
    if a.eps > 0:
        arm_note += f"  [CONTROL A: eps={a.eps}]"
    if a.no_hold_explore:
        arm_note += "  [CONTROL B: GO_* only]"

    print(f"tag={a.tag}  seeds={a.seeds}  termination="
          f"{'counter only' if a.blind else 'task condition or counter'}"
          f"  hold_consume={a.hold_consume}{arm_note}\n")
    hdr = (f"{'arm':>16}  {'solveScore':>10}  {'wilson':>13}  {'deaths':>7}"
           f"  {'hyd/sat':>9}  {'w->f/life':>8}  {'GO_W:F':>8}  {'food_kn':>8}")
    print(hdr)
    print("-" * len(hdr))

    if not a.skip_oracle:
        ms = [measure(run(s)) for s in range(a.seeds)]
        print(row("oracle", pooled(ms)))
        print("-" * len(hdr))

    suffix = ("+rand" if a.eps > 0 else "") + ("+goals" if a.no_hold_explore else "")
    best = None
    for h in holds:
        kw = dict(hold=h, hold_natural=natural, hold_consume=a.hold_consume,
                  hold_explore=not a.no_hold_explore, explore_eps=a.eps)
        try:
            ms = [measure(run(s, orchestrator=a.tag, orch_kw=kw))
                  for s in range(a.seeds)]
        except FileNotFoundError as e:
            raise SystemExit(f"missing weights: {e}")
        m = pooled(ms)
        print(row(f"hold={h}{suffix}", m))
        n = m["timeouts"] + m["deaths"]
        ss = m["timeouts"] / n if n else 0.0
        if best is None or ss > best[0]:
            best = (ss, h, m)

    print("-" * len(hdr))
    ss, h, m = best
    base = None
    print(f"\nbest: hold={h} at solveScore {ss:.4f}")
    print("Reminder: GO_FOOD run length is set by `hold` here and is not an outcome.")
    print("A lift on THESE weights means collapse was the failure and a retrain at this")
    print("horizon is worth running. A flat result means the water lean is real and the")
    print("retrain would have chased the wrong mechanism.")


if __name__ == "__main__":
    main()
