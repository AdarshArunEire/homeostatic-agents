"""
orchestrator_legality_v1.py  —  is the water-cult attractor a PREFERENCE or a TRAP?

    python tests/orchestrator_legality_v1.py --tag o0
    python tests/orchestrator_legality_v1.py --tag o0 --seeds 8 --verbose

P4.1 reported GO_WATER : GO_FOOD at 7.6 : 1 for the learned orchestrator against the
oracle's 1.0 : 1, and read it as a preference — the module chooses water. That reading
assumes both policies were choosing from the same menu.

They may not have been. `Orchestrator.legal_mask` masks GO_FOOD unless `food_known`, and
`food_known` only becomes True by EXPLORING into a food tile's vision. So a policy that
stops exploring once it has water can never make GO_FOOD legal again — and in the
aggregate that is indistinguishable from a policy that could pick food and doesn't.

Two hypotheses, one measurement:

  PREFERENCE  food_known is often True, GO_FOOD is legal and rarely chosen.
              -> a valuing failure. Proto 05's SIL archive is the on-target lever.

  TRAP        food_known is ~never True. GO_FOOD is not in the action set at all.
              -> consolidation cannot help; there is no crossing to consolidate.
                 The lever is upstream: why does EXPLORE stop?

The script also measures two things the GIF made visible:

  * GO_* emitted while ALREADY STANDING ON the goal tile. `compose_to_queued_action`
    still calls the pathfinder, `to_goal` is (0,0), every direction ties at distance 1,
    and OraclePathfinder's first-wins tie-break returns D0 every time. The agent steps
    off and straight back — a 2-cycle that looks like a departure and is not one.
    OracleOrchestrator never reaches this path (rule 1 fires CONSUME first), so no
    existing probe exercises it.

  * EXPLORE split before and after the first water contact of each life. A share
    averaged over a whole episode hides a policy that front-loads exploration and then
    stops dead — the same way mean comfort hid the bimodal failure in 03b.

Requires `water_known_T` / `food_known_T` from sim_instance_v3.
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

# Reusing the runner rather than copying the standing config. Same argument world_v1
# makes about decay constants: two copies drift, one does not.
from make_agent_gif_v1 import eval_segments, run  # noqa: E402

GO_WATER, GO_FOOD, CONSUME = 0, 1, 8
EXPLORE = range(2, 8)


def run_lengths(arr, pred) -> list[int]:
    """Lengths of maximal runs where pred(v) holds."""
    out, n = [], 0
    for v in arr:
        if pred(v):
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def analyse(res) -> dict:
    if "food_known_T" not in res:
        raise SystemExit(
            "food_known_T missing — apply the sim_instance_v3 logging patch first "
            "(lifetime_stats['water_known'/'food_known'] and the two return keys)."
        )

    act = np.asarray(res["abstract_action_T"])
    fk = np.asarray(res["food_known_T"]).astype(bool)
    wk = np.asarray(res["water_known_T"]).astype(bool)
    coords = res["coordinates_T"]

    acc = dict(ticks=0, lives=0, lives_with_food=0, fk=0, wk=0,
               go_w=0, go_f=0, consume=0, explore=0,
               both_legal=0, bl_w=0, bl_f=0,
               bl_off=0, blo_w=0, blo_f=0,
               go_on_goal=0, go_total=0,
               wf_trips=0, fw_trips=0,
               pre_ticks=0, pre_explore=0, post_ticks=0, post_explore=0,
               f_runs=[], w_runs=[])

    for t0, t1, seg in eval_segments(res):
        if t1 <= t0:
            continue
        water = {tuple(c) for c in seg["water_coords"]}
        food = {tuple(c) for c in seg["food_coords"]}

        a = act[t0:t1]
        f = fk[t0:t1]
        w = wk[t0:t1]
        pos = [tuple(c) for c in coords[t0:t1]]

        acc["ticks"] += len(a)
        acc["lives"] += 1
        acc["lives_with_food"] += int(f.any())
        acc["fk"] += int(f.sum())
        acc["wk"] += int(w.sum())

        acc["go_w"] += int((a == GO_WATER).sum())
        acc["go_f"] += int((a == GO_FOOD).sum())
        acc["consume"] += int((a == CONSUME).sum())
        acc["explore"] += int(np.isin(a, list(EXPLORE)).sum())

        on_w = np.array([p in water for p in pos], dtype=bool)
        on_f = np.array([p in food for p in pos], dtype=bool)
        off_resource = ~(on_w | on_f)

        # ticks where BOTH goals were selectable
        both = f & w
        acc["both_legal"] += int(both.sum())
        acc["bl_w"] += int((a[both] == GO_WATER).sum())
        acc["bl_f"] += int((a[both] == GO_FOOD).sum())

        # ...and standing on NEITHER resource. This is the only set of ticks where the
        # agent is making a genuine travel decision. On a resource tile, GO_* naming that
        # tile is a no-op (see below), so including those ticks counts jitter as choice.
        both_off = both & off_resource
        acc["bl_off"] += int(both_off.sum())
        acc["blo_w"] += int((a[both_off] == GO_WATER).sum())
        acc["blo_f"] += int((a[both_off] == GO_FOOD).sum())

        # GO_* issued while standing on the tile it names -> pathfinder gets to_goal=(0,0),
        # all six directions tie at distance 1, OraclePathfinder's first-wins tie-break
        # returns D0, and the agent steps off and back next tick. A 2-cycle, not a trip.
        gw, gf = (a == GO_WATER), (a == GO_FOOD)
        acc["go_total"] += int((gw | gf).sum())
        acc["go_on_goal"] += int((gw & on_w).sum() + (gf & on_f).sum())

        # completed crossings, by the sim's own definition (sim_instance_v3 trip metrics).
        # This is SIL's precondition: an archive of successful crossings needs successful
        # crossings. 05-H3 checked archive fill before trusting its result; same check.
        last = None
        for i in range(len(pos)):
            if on_w[i]:
                if last == "food":
                    acc["fw_trips"] += 1
                last = "water"
            elif on_f[i]:
                if last == "water":
                    acc["wf_trips"] += 1
                last = "food"

        # exploration before vs after this life's first water contact
        first_w = int(np.argmax(w)) if w.any() else len(w)
        pre, post = a[:first_w], a[first_w:]
        acc["pre_ticks"] += len(pre)
        acc["pre_explore"] += int(np.isin(pre, list(EXPLORE)).sum())
        acc["post_ticks"] += len(post)
        acc["post_explore"] += int(np.isin(post, list(EXPLORE)).sum())

        acc["f_runs"] += run_lengths(a, lambda v: v == GO_FOOD)
        acc["w_runs"] += run_lengths(a, lambda v: v == GO_WATER)

    return acc


def merge(accs: list[dict]) -> dict:
    out = {}
    for k in accs[0]:
        if isinstance(accs[0][k], list):
            out[k] = [x for acc in accs for x in acc[k]]
        else:
            out[k] = sum(acc[k] for acc in accs)
    return out


def ratio(a: int, b: int) -> str:
    if b == 0:
        return "inf : 1" if a else "n/a"
    return f"{a / b:.2f} : 1"


def pct(a: int, b: int) -> str:
    return "n/a" if b == 0 else f"{100 * a / b:5.1f}%"


def med(xs) -> str:
    return "n/a" if not xs else f"{float(np.median(xs)):.1f}"


def report(rows: dict[str, dict], verbose: bool):
    names = list(rows)
    w = 26

    def line(label, fn):
        print(f"  {label:<38}" + "".join(f"{fn(rows[n]):>{w}}" for n in names))

    print("=" * (40 + w * len(names)))
    print("orchestrator legality probe")
    print("=" * (40 + w * len(names)))
    print(f"  {'':<38}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (40 + w * len(names)))

    print("\n  MEMORY / LEGALITY")
    line("eval ticks with water_known", lambda a: pct(a["wk"], a["ticks"]))
    line("eval ticks with food_known", lambda a: pct(a["fk"], a["ticks"]))
    line("lives that ever found food", lambda a: f'{a["lives_with_food"]}/{a["lives"]}')
    line("ticks where BOTH GO_* were legal", lambda a: pct(a["both_legal"], a["ticks"]))

    print("\n  ARBITRATION")
    line("GO_WATER : GO_FOOD  (raw, as published)",
         lambda a: ratio(a["go_w"], a["go_f"]))
    line("GO_WATER : GO_FOOD  (both legal)",
         lambda a: ratio(a["bl_w"], a["bl_f"]))
    line("GO_WATER : GO_FOOD  (both legal, off-tile)",
         lambda a: ratio(a["blo_w"], a["blo_f"]))
    line("  ...over this many ticks", lambda a: f'{a["bl_off"]}')

    print("\n  COMMITMENT")
    line("median GO_FOOD run length", lambda a: med(a["f_runs"]))
    line("median GO_WATER run length", lambda a: med(a["w_runs"]))
    line("GO_* issued while ON the goal tile",
         lambda a: pct(a["go_on_goal"], a["go_total"]))

    print("\n  CROSSINGS  (SIL's precondition)")
    line("completed water -> food trips", lambda a: f'{a["wf_trips"]}')
    line("completed food -> water trips", lambda a: f'{a["fw_trips"]}')
    line("water -> food per life", lambda a: f'{a["wf_trips"] / max(1, a["lives"]):.2f}')

    print("\n  EXPLORATION IN TIME")
    line("EXPLORE share, whole episode", lambda a: pct(a["explore"], a["ticks"]))
    line("EXPLORE share, before first water", lambda a: pct(a["pre_explore"], a["pre_ticks"]))
    line("EXPLORE share, after first water", lambda a: pct(a["post_explore"], a["post_ticks"]))
    line("mean ticks before first water", lambda a: f'{a["pre_ticks"] / max(1, a["lives"]):.1f}')
    line("mean ticks after first water", lambda a: f'{a["post_ticks"] / max(1, a["lives"]):.1f}')
    line("mean life length (eval)", lambda a: f'{a["ticks"] / max(1, a["lives"]):.1f}')

    if verbose:
        print("\n  RAW COUNTS")
        for n in names:
            a = rows[n]
            print(f"    {n}: ticks={a['ticks']} lives={a['lives']} "
                  f"GO_W={a['go_w']} GO_F={a['go_f']} CONSUME={a['consume']} "
                  f"EXPLORE={a['explore']}")

    print("\n" + "=" * (40 + w * len(names)))
    verdict(rows)


def verdict(rows: dict[str, dict]):
    """State which hypothesis the numbers support, in the terms set before running."""
    learned = [n for n in rows if n != "oracle"]
    if not learned:
        return
    a = rows[learned[0]]
    fk_share = a["fk"] / max(1, a["ticks"])
    bl_share = a["both_legal"] / max(1, a["ticks"])

    if bl_share < 0.05:
        print("VERDICT: TRAP.")
        print(f"  Both GO_* were legal on only {100*bl_share:.1f}% of eval ticks, so the")
        print("  published 7.6:1 is mostly an availability constraint, not a preference.")
        print("  SIL is OFF-TARGET here: there is no crossing in experience to protect.")
        print("  Next lever is upstream — why EXPLORE stops. Check the discount horizon")
        print("  (gamma=0.99 ~= 100 ticks) against time-to-starvation before anything else.")
    elif fk_share > 0.25:
        print("VERDICT: PREFERENCE.")
        print(f"  food_known held on {100*fk_share:.1f}% of ticks and GO_FOOD was legal")
        print("  on a real share of them, so the module could pick food and did not.")
        print("  That is a valuing failure and Proto 05 H3's SIL archive is on-target.")
        per_life = a["wf_trips"] / max(1, a["lives"])
        print(f"\n  SIL precondition: {a['wf_trips']} completed water->food crossings, "
              f"{per_life:.2f} per life.")
        if per_life < 0.5:
            print("  THIN. An archive needs crossings to hold. 05-H3 confirmed archive fill")
            print("  before trusting its dose-response; do the same or the null is unreadable.")
    else:
        print("VERDICT: MIXED — neither hypothesis is clean at this sample size.")
        print(f"  food_known {100*fk_share:.1f}% of ticks, both-legal {100*bl_share:.1f}%.")
        print("  Raise --seeds before drawing a conclusion.")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="o0", help="learned orchestrator weight tag")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    arms = {} if a.skip_oracle else {"oracle": {}}
    arms[f"learned {a.tag}"] = {"orchestrator": a.tag}

    rows = {}
    for name, kw in arms.items():
        print(f"running {name} over {a.seeds} seeds ...", flush=True)
        try:
            accs = [analyse(run(s, **kw)) for s in range(a.seeds)]
        except FileNotFoundError as e:
            raise SystemExit(f"missing weights for '{name}': {e}")
        rows[name] = merge(accs)

    print()
    report(rows, a.verbose)


if __name__ == "__main__":
    main()
