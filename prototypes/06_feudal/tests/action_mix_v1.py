"""
action_mix_v1.py  —  measure the orchestrator's action distribution.

    python tests/action_mix_v1.py                 # oracle only
    python tests/action_mix_v1.py --tags o0       # oracle vs learned

Needed because the P4.1 water-cult claim compares the learned orchestrator's action shares
against the oracle's — and the oracle's were never actually measured. No instrumentation is
required: `abstract_action_T` already records the Action enum every tick, so the mix is a
histogram over a completed run.

Collapses the nine-member Action space to the four abstract choices the learned orchestrator
uses, so the two are directly comparable:

    GO_WATER   GO_FOOD   CONSUME   EXPLORE (= EXPLORE_0..5 pooled)

Measured over the EVAL segment only, matching every other number in the ledger.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

from model_modules.contract_v1 import Action, ModuleSpec  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20
LABELS = ["GO_WATER", "GO_FOOD", "CONSUME", "EXPLORE"]

EXPLORER_KW = {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
               "follow_p": 0.95, "reverse_on_drop_p": 0.75}


def mix(impl, orch_kw, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    counts = np.zeros(4, dtype=float)
    D = T = 0
    causes = {}

    for s in seeds:
        run = S.sim_instance(
            seed=s, sim_len=sim_len, eval_len=eval_len,
            env_kwargs=dict(radius=RADIUS, band=(9, 11), start_coord=(0, 0)),
            decay_mult=0.7, smell_radius=3, curriculum_mode="band",
            c_min=2, c_max=9, band_width=2, life_cap=1000,
            oracle_params={
                "orchestrator": dict(orch_kw), "pathfinder": {},
                "eat": {"fill_target": S_FILL}, "drink": {"fill_target": H_FILL},
                "explorer": dict(EXPLORER_KW),
            },
            module_spec=ModuleSpec(impl, "oracle", "oracle", "oracle",
                                   explorer="smell_momentum"),
            log_every=10**9, eval_spawn_nondoomed=True, spawn_leeway=10,
            eval_god_memory=False,
        )
        eb = int(run["eval_boundary"])
        ab = np.asarray(run["abstract_action_T"])[eb:]

        counts[0] += int((ab == int(Action.GO_WATER)).sum())
        counts[1] += int((ab == int(Action.GO_FOOD)).sum())
        counts[2] += int((ab == int(Action.CONSUME)).sum())
        counts[3] += int(((ab >= int(Action.EXPLORE_0))
                          & (ab <= int(Action.EXPLORE_5))).sum())

        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        for e in run.get("death_events", []):
            if e["t"] >= eb:
                causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    total = counts.sum()
    return {"shares": counts / max(1.0, total), "counts": counts,
            "eval_deaths": int(D), "causes": causes,
            "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan")}


def show(name, r):
    print(f"\n{name}")
    print("  " + "  ".join(f"{l:>9}" for l in LABELS))
    print("  " + "  ".join(f"{v:>9.3f}" for v in r["shares"]))
    print(f"  deaths {r['eval_deaths']}   solveScore {r['solveScore']:.4f}   "
          f"causes {r['causes']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--tags", nargs="*", default=[])
    a = p.parse_args()

    seeds = range(a.seeds)
    print("=" * 62)
    print(f"orchestrator action mix — eval segment, {a.seeds} seeds")
    print("=" * 62)

    base_kw = {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7}
    orc = mix("oracle", base_kw, seeds)
    show("oracle", orc)

    for tag in a.tags:
        r = mix("learned", {"weights": tag}, seeds)
        show(f"learned {tag}", r)
        ratio = (r["shares"][0] / r["shares"][1]) if r["shares"][1] > 0 else float("inf")
        oratio = (orc["shares"][0] / orc["shares"][1]) if orc["shares"][1] > 0 else float("inf")
        print(f"  GO_WATER:GO_FOOD  learned {ratio:.1f}:1   oracle {oratio:.1f}:1")

    print("\nPaste the shares into figures/make_figures_v1.py (ORCH_MIX / ORACLE_MIX).")
    print("=" * 62)


if __name__ == "__main__":
    main()
