"""
test_module_diff_v1.py  —  acceptance gate for learned modules.

    python tests/test_module_diff_v1.py --tag v1
    python tests/test_module_diff_v1.py --tag v1 --in-sim      # slower, the real check

Three checks, cheapest first.

1. AXIAL ALIGNMENT (static, free)
   OraclePathfinder.AXIAL_DIRS against hex_world_cached.LOCAL_DIRECTIONS. Its own
   docstring flags this as the integration point to watch: if the tables drift, HexMove.Dk
   means two different physical steps in two places and the oracle walks away from goals.
   The rig imports the env's table directly, so this can now only break on the oracle side
   — which is precisely why it is still worth asserting.

2. OPTIMALITY, exhaustive (seconds)
   Every displacement a radius-R world can produce, scored two ways:

     optimality_rate   does the move reduce hex distance by 1?  <- PRIMARY
     oracle_agreement  is it the same move the oracle picked?   <- secondary

   Agreement understates. Several moves are often equally optimal and the oracle
   tie-breaks first-wins by construction; a learned module that breaks ties differently is
   not worse. Judge on optimality, report agreement for information — the same reason
   Proto 04 judged crossings on path_efficiency rather than "did it arrive".

3. IN-SIM CONTRACT INTEGRITY (minutes)
   The one that matters. The pathfinder is trained with NO access to the sim, because
   PathfinderObs claims to need nothing but `to_goal`. Drop it into the full stack with
   oracles in the other three slots and compare solveScore against all-oracle.

     no change      -> the observation contract is closed. Nothing undeclared leaked in.
     degradation    -> PathfinderObs has a dependency it does not name. That is a contract
                       bug, and finding it here is the entire point of training out of sim.

   Pre-register the reading before running it, per BUILDNOTES discipline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

import hex_world_cached as hex_world  # noqa: E402
from model_modules.contract_v1 import ModuleSpec, PathfinderObs  # noqa: E402
from model_modules.oracle_modules.pathfinder_oracle_v1 import (  # noqa: E402
    AXIAL_DIRS,
    OraclePathfinder,
)
from model_modules.learned_modules.pathfinder_learned_v1 import (  # noqa: E402
    LearnedPathfinder,
    hex_len,
)

H_FILL, S_FILL = 1.6, 1.3


# --- 1. static alignment --------------------------------------------------------

def check_axial_alignment() -> dict:
    env_dirs = tuple(hex_world.LOCAL_DIRECTIONS[k] for k in range(6))
    ok = tuple(AXIAL_DIRS) == env_dirs
    return {
        "pass": ok,
        "oracle_AXIAL_DIRS": tuple(AXIAL_DIRS),
        "env_LOCAL_DIRECTIONS": env_dirs,
    }


# --- 2. exhaustive optimality ---------------------------------------------------

def all_displacements(radius: int):
    """
    Every to_goal a radius-R disc can produce. Both endpoints lie inside the disc, so the
    magnitude tops out at 2R. Zero is excluded: no move reduces it, and the composer never
    calls the pathfinder when it is already on the goal.
    """
    lim = 2 * radius
    out = []
    for dq in range(-lim, lim + 1):
        for dr in range(-lim, lim + 1):
            d = hex_len(dq, dr)
            if 0 < d <= lim:
                out.append((dq, dr))
    return out


def optimal_moves(to_goal) -> set:
    """Move indices that reduce hex distance by exactly 1 — the greedy-optimal set."""
    gq, gr = to_goal
    d0 = hex_len(gq, gr)
    return {
        k for k, (dq, dr) in enumerate(AXIAL_DIRS)
        if hex_len(gq - dq, gr - dr) == d0 - 1
    }


def move_severity(to_goal, k: int) -> str:
    """
    How expensive is a non-optimal move?

    A pooled optimality rate treats every error alike, and that hides the thing that
    actually matters. On a hex grid a move either closes the distance, holds it, or opens
    it — costing 0, 1 or 2 extra steps respectively. The eps-noise control takes uniformly
    random steps, so 1-in-6 of its errors are reversals; a learned module whose errors are
    mostly sideways does far less damage at the same optimality number. This is the
    measurement that separates those two cases.
    """
    gq, gr = to_goal
    dq, dr = AXIAL_DIRS[k]
    delta = hex_len(gq - dq, gr - dr) - hex_len(gq, gr)
    if delta == -1:
        return "optimal"
    if delta == 0:
        return "sideways"     # +1 step to recover
    return "backwards"        # +2 steps to recover


# Distance bands. A pooled rate over the whole disc is dominated by the far tail — a ring
# at distance d holds 6d cells, so most of the input space sits at displacements the agent
# never actually experiences. The commute band is (9,11) by construction, so `operational`
# is what the gate is set on; `far` is reported but not gated, because reaching it requires
# both endpoints near opposite rims.
def band_of(d: int, radius: int) -> str:
    if d <= 5:
        return "approach"      # last hexes of an arrival; where path_efficiency is won
    if d <= 11:
        return "commute"       # the band the world is built around
    if d <= radius:
        return "long"          # plausible after wandering
    return "far"               # rim-to-rim; not reachable in a band-(9,11) commute


GATED_BANDS = ("approach", "commute", "long")


def check_pathfinder_optimality(module, radius: int = 20, gate: float = 0.99) -> dict:
    oracle = OraclePathfinder()
    disps = all_displacements(radius)

    bands = {
        b: {"n": 0, "opt": 0, "agree": 0, "sideways": 0, "backwards": 0}
        for b in ("approach", "commute", "long", "far")
    }
    n_opt = n_agree = 0
    worst = []

    for d in disps:
        obs = PathfinderObs(to_goal=d)
        got = int(module.move(obs))
        ref = int(oracle.move(obs))
        opt = optimal_moves(d)
        b = bands[band_of(hex_len(*d), radius)]

        b["n"] += 1
        sev = move_severity(d, got)
        if sev == "optimal":
            n_opt += 1
            b["opt"] += 1
        else:
            b[sev] += 1
            if len(worst) < 10:
                worst.append({"to_goal": d, "dist": hex_len(*d), "got": got,
                              "optimal": sorted(opt), "severity": sev})
        if got == ref:
            n_agree += 1
            b["agree"] += 1

    for b in bands.values():
        b["optimality"] = b["opt"] / b["n"] if b["n"] else float("nan")
        b["agreement"] = b["agree"] / b["n"] if b["n"] else float("nan")
        # expected extra steps per move: sideways costs 1, backwards costs 2. This is the
        # number that predicts in-sim damage, not the raw optimality rate — a module at
        # 98% optimality whose errors are all sideways is half as costly as one whose
        # errors are all reversals, and uniform random noise is ~1/6 reversals.
        b["excess_steps_per_move"] = (
            (b["sideways"] + 2 * b["backwards"]) / b["n"] if b["n"] else float("nan")
        )

    gated = [bands[k]["optimality"] for k in GATED_BANDS if bands[k]["n"]]
    n = len(disps)
    return {
        "n_displacements": n,
        "optimality_rate": n_opt / n,
        "oracle_agreement": n_agree / n,
        "bands": bands,
        "gate": gate,
        "failures_sample": worst,
        "pass": all(x >= gate for x in gated),
    }


def check_oracle_is_optimal(radius: int = 20) -> dict:
    """
    Sanity: the oracle should itself be 100% optimal. If it is not, the reference is
    broken and every agreement number above is measured against a bad ruler.
    """
    oracle = OraclePathfinder()
    bad = [
        d for d in all_displacements(radius)
        if int(oracle.move(PathfinderObs(to_goal=d))) not in optimal_moves(d)
    ]
    return {"pass": not bad, "n_suboptimal": len(bad), "sample": bad[:10]}


# --- 3. in-sim contract integrity -----------------------------------------------

def _oracle_params(pathfinder_kwargs: dict | None = None) -> dict:
    """Mirrors god_vs_legal_check.op() so the comparison sits on the standing config."""
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


def _run(seed: int, spec: ModuleSpec, oracle_params: dict, sim_len=7000, eval_len=5000):
    import sim_instance_v3 as S

    o = S.sim_instance(
        seed=seed, sim_len=sim_len, eval_len=eval_len,
        env_kwargs=dict(radius=20, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7, smell_radius=3,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=oracle_params, module_spec=spec, log_every=10**9,
        eval_spawn_nondoomed=True, spawn_leeway=10, eval_god_memory=False,
    )
    return o["death_count_eval"], o["n_timeouts_eval"]


def check_pathfinder_in_sim(tag: str, seeds=range(8), radius: int = 20) -> dict:
    arms = {
        "all_oracle": (
            ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer="smell_momentum"),
            _oracle_params(),
        ),
        "learned_pathfinder": (
            ModuleSpec("oracle", "learned", "oracle", "oracle", explorer="smell_momentum"),
            _oracle_params({"weights": tag, "scale": radius}),
        ),
    }

    import model_modules.learned_modules.pathfinder_learned_v1 as PL

    out = {}
    for label, (spec, op) in arms.items():
        PL.reset_instrumentation(instrument=True)
        D = T = 0
        for s in seeds:
            d, t = _run(s, spec, op)
            D += d
            T += t
        out[label] = {
            "eval_deaths": int(D),
            "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
            "instrumentation": PL.instrumentation(),
        }

    delta = out["learned_pathfinder"]["solveScore"] - out["all_oracle"]["solveScore"]
    calls = out["learned_pathfinder"]["instrumentation"]["calls"]

    out["delta_solveScore"] = delta
    out["learned_calls"] = calls
    # A zero delta only means anything if the module ran. Without this, "no degradation"
    # and "never invoked" produce the same PASS.
    out["invoked"] = calls > 0
    # 8 seeds is small; treat anything inside a couple of deaths as noise rather than
    # signal, and re-run wider before reading a real gap into it
    out["pass"] = out["invoked"] and delta > -0.05
    return out


# --- report ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v1")
    p.add_argument("--radius", type=int, default=20)
    p.add_argument("--in-sim", action="store_true", help="run the slow contract check too")
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()

    print("=" * 68)
    print("1. axial alignment (oracle table vs env table)")
    r = check_axial_alignment()
    print(f"   {'PASS' if r['pass'] else 'FAIL'}")
    if not r["pass"]:
        print(f"   oracle: {r['oracle_AXIAL_DIRS']}")
        print(f"   env:    {r['env_LOCAL_DIRECTIONS']}")
        print("   -> HexMove.Dk means different steps in different places. Fix before anything else.")
        return

    print("\n2a. oracle self-check (is the ruler straight?)")
    r = check_oracle_is_optimal(a.radius)
    print(f"   {'PASS' if r['pass'] else 'FAIL'}  suboptimal: {r['n_suboptimal']}")

    print(f"\n2b. learned pathfinder '{a.tag}' — exhaustive, radius {a.radius}")
    module = LearnedPathfinder(weights=a.tag, scale=a.radius)
    r = check_pathfinder_optimality(module, a.radius)
    print(f"   displacements {r['n_displacements']}   pooled optimality "
          f"{r['optimality_rate']:.4f}  agreement {r['oracle_agreement']:.4f}")
    print(f"   {'band':<10} {'n':>6} {'optimality':>11} {'agree':>7} "
          f"{'sideway':>8} {'backwd':>7} {'excess/mv':>10}   gated")
    for name in ("approach", "commute", "long", "far"):
        b = r["bands"][name]
        star = "yes" if name in GATED_BANDS else "no (unreachable)"
        print(f"   {name:<10} {b['n']:>6} {b['optimality']:>11.4f} {b['agreement']:>7.3f} "
              f"{b['sideways']:>8} {b['backwards']:>7} {b['excess_steps_per_move']:>10.4f}   {star}")
    print(f"   gate {r['gate']} on {', '.join(GATED_BANDS)} -> {'PASS' if r['pass'] else 'FAIL'}")
    print("   excess/mv = expected extra steps per move (sideways 1, backwards 2).")
    print("   Uniform random noise is ~1/6 backwards; an all-sideways module at the same")
    print("   optimality does roughly half the damage. This is what predicts in-sim cost.")
    for f in r["failures_sample"]:
        print(f"     to_goal {f['to_goal']} (d={f['dist']})  got D{f['got']}  "
              f"optimal {f['optimal']}  [{f['severity']}]")

    if a.in_sim:
        print(f"\n3. in-sim contract integrity ({a.seeds} seeds)")
        r = check_pathfinder_in_sim(a.tag, seeds=range(a.seeds), radius=a.radius)
        for k in ("all_oracle", "learned_pathfinder"):
            print(f"   {k:20s} deaths {r[k]['eval_deaths']:<5} solveScore {r[k]['solveScore']:.3f}")
        print(f"   delta {r['delta_solveScore']:+.3f}")
        print(f"   learned pathfinder invoked: {r['learned_calls']} times")
        if not r["invoked"]:
            print("   -> NOT INVOKED. A zero delta here means nothing: both arms ran the")
            print("      same code. Check GO_WATER/GO_FOOD are firing before reading this.")
        else:
            hist = r["learned_pathfinder"]["instrumentation"]["dist_hist"]
            tot = sum(hist.values()) or 1
            cum, p95 = 0, max(hist) if hist else 0
            for d in sorted(hist):
                cum += hist[d]
                if cum / tot >= 0.95:
                    p95 = d
                    break
            print(f"   observed |to_goal|: max {max(hist) if hist else 0}, p95 {p95}")
            print("      ^ this is the range the optimality gate should be set against")
        print(f"   {'PASS' if r['pass'] else 'FAIL'}")
        if r["invoked"] and not r["pass"]:
            print("   -> trained out of sim, degrades in sim: PathfinderObs has an undeclared dependency.")
    else:
        print("\n3. in-sim contract integrity: skipped (--in-sim)")

    print("=" * 68)


if __name__ == "__main__":
    main()
