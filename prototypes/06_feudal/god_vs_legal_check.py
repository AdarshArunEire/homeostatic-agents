"""
god_vs_legal_check.py — is the changed world trivial / fair / is the oracle the true ceiling?

Runs THREE policies in the SAME fully-changed world (tolerance comfort, nondoomed eval
spawns, smell 3, decay 0.7, h_fill 1.6 / s_fill 1.3):

  GOD        : eval_god_memory=True  -> Proto-05 oracle power (knows nearest water/food,
               pure navigation, NO exploration). Upper bound: is the world survivable?
  reactive   : legal smell_momentum explorer (must FIND food by chemotaxis). The Proto-06
               "oracle" -- perfect exploit, realistic explore.
  random     : legal random explorer. Triviality floor.

Reading:
  - GOD ~solves (deaths ~0)  => world is FAIR (physically survivable, nondoomed guarantees it)
  - random far below GOD     => task is NOT trivial
  - reactive << GOD          => the Proto-06 oracle is NOT the true best-case ceiling; the
                                gap GOD-minus-reactive is pure EXPLORATION difficulty, which
                                a learned memory-explorer could still close.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import sim_instance_v3 as S
from model_modules.contract_v1 import ModuleSpec

SEEDS = range(8)
H_FILL, S_FILL = 1.6, 1.3


def op():
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL}, "drink": {"fill_target": H_FILL},
        "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                     "follow_p": 0.95, "reverse_on_drop_p": 0.75},
    }


def run(seed, explorer, god):
    spec = ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer=explorer)
    o = S.sim_instance(
        seed=seed, sim_len=7000, eval_len=5000,
        env_kwargs=dict(radius=20, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7, smell_radius=3,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=op(), module_spec=spec, log_every=10**9,
        eval_spawn_nondoomed=True, spawn_leeway=10, eval_god_memory=god,
    )
    EB = 7000 - 5000
    causes = {}
    for e in o.get("death_events", []):
        if e["t"] >= EB:
            causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    return o["death_count_eval"], o["n_timeouts_eval"], causes


if __name__ == "__main__":
    print(f"{'policy':22s} {'evalDeaths':10s} {'solveScore':10s} causes(eval)")
    rows = [("GOD (god_memory)", "smell_momentum", True),
            ("reactive smell_mom", "smell_momentum", False),
            ("random", "random", False)]
    for label, expl, god in rows:
        D = T = 0
        causes = {}
        for s in SEEDS:
            d, t, ce = run(s, expl, god)
            D += d; T += t
            for k, v in ce.items():
                causes[k] = causes.get(k, 0) + v
        score = T / (T + D) if (T + D) > 0 else float("nan")
        print(f"{label:22s} {int(D):<10} {score:<10.2f} {causes}")
