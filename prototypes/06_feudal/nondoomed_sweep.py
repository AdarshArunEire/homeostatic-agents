"""
nondoomed_sweep.py — full two-resource navigation eval with DOOMED spawns removed.

eval_spawn_nondoomed=True: agent spawns anywhere in the pool (cold start, must find
water THEN food), but spawns where perfect water-first play can't secure both in time
(+ spawn_leeway) are rejected. This is the honest "remove doomed spawns" eval — it does
NOT hand water over, unlike eval_spawn_beside_water.

smell 3, tolerance-band comfort (world.OVER_TOL), decay 0.7/0.9, smell_momentum,
h_crit=s_crit=0.7. Compares against the beside-water ceiling you already have
(0.86 @ smell3) — expect nondoomed to sit BELOW it, since the agent now also has to
find water from cold. Also sweeps spawn_leeway to show sensitivity of the filter.

Deaths here should be near-evenly split hydration/satiation (both resources in play),
not satiation-only — that's the tell that it's genuine 2-resource nav again.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import world_v1 as world
import sim_instance_v3 as S
from model_modules.contract_v1 import ModuleSpec

SMELL = 3
SEEDS = range(8)
H_FILL, S_FILL = 1.6, 1.3          # best-survival cell from the fill sweep
H_CRIT = S_CRIT = 0.7


def op():
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": H_CRIT, "s_crit": S_CRIT},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                     "follow_p": 0.95, "reverse_on_drop_p": 0.75},
    }


def run(seed, dm, leeway, explorer="smell_momentum"):
    spec = ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer=explorer)
    o = S.sim_instance(
        seed=seed, sim_len=7000, eval_len=5000,
        env_kwargs=dict(radius=20, band=(9, 11), start_coord=(0, 0)),
        decay_mult=dm, smell_radius=SMELL,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=op(), module_spec=spec, log_every=10**9,
        eval_spawn_nondoomed=True, spawn_leeway=leeway,
    )
    EB = 7000 - 5000
    causes = {}
    for e in o.get("death_events", []):
        if e["t"] >= EB:
            causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    return o["mean_comfort"], o["death_count_eval"], o["n_timeouts_eval"], causes


def sweep(title, cells):
    print(f"\n=== {title} ===")
    print(f"{'dm':4s} {'leeway':6s} {'expl':15s} {'mComfort':8s} {'evDeaths':8s} "
          f"{'solveScore':10s} causes(eval)")
    for dm, lw, ex in cells:
        C = D = T = 0.0
        causes = {}
        for s in SEEDS:
            mc, d, t, ce = run(s, dm, lw, ex)
            C += mc; D += d; T += t
            for k, v in ce.items():
                causes[k] = causes.get(k, 0) + v
        n = len(list(SEEDS))
        score = T / (T + D) if (T + D) > 0 else float("nan")
        print(f"{dm:<4} {lw:<6} {ex:15s} {C/n:<8.3f} {int(D):<8} {score:<10.2f} {causes}")


if __name__ == "__main__":
    print(f"OVER_TOL={world.OVER_TOL} OVER_W={world.OVER_W}  smell={SMELL}  h_fill={H_FILL} s_fill={S_FILL}  seeds={len(list(SEEDS))}")
    # A) explorer comparison at the operating point
    sweep("explorers @ dm0.7, leeway10",
          [(0.7, 10, "random"), (0.7, 10, "momentum"), (0.7, 10, "smell_momentum")])
    # B) decay sensitivity (smell_momentum)
    sweep("smell_momentum decay sweep, leeway10",
          [(0.7, 10, "smell_momentum"), (0.9, 10, "smell_momentum")])
    # C) leeway sensitivity — how much the filter's slack moves the numbers
    sweep("smell_momentum leeway sweep, dm0.7",
          [(0.7, 0, "smell_momentum"), (0.7, 10, "smell_momentum"), (0.7, 25, "smell_momentum")])
