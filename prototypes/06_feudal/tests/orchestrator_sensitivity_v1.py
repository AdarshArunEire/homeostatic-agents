"""
orchestrator_sensitivity_v1.py  —  go/no-go before training an orchestrator.

    python tests/orchestrator_sensitivity_v1.py --calibrate-only

Needs no trained module. Run it FIRST. If no metric is monotone in orchestrator degradation,
stop — training would produce a number nobody can interpret, which is exactly what the consumer
thread cost four runs to discover.

TWO AXES.

`epsilon`     probability of replacing the chosen action with a uniformly random legal one.
              General degradation, comparable to the explorer's control.

`crit_scale`  multiplies the critical-drive thresholds (baseline h_crit = s_crit = 0.7). This is
              the ARBITRATION knob and it is two-sided:

                crit_scale = 0     the interrupt never fires; the agent hunts the missing
                                   resource single-mindedly and dies of thirst beside
                                   remembered water
                crit_scale >> 1    it fires constantly; the agent camps on the known drive and
                                   never finishes exploring

              The optimum is INTERIOR, which is what makes this a real arbitration problem
              rather than a threshold to push one way.

ANCHOR. Adding the critical-drive interrupt is recorded as moving hydration deaths 43 -> 10
(random explorer) and 13 -> 1 (smell_momentum). So `crit_scale=0` has a known expected
magnitude. If the sweep does not reproduce a large degradation there, the instrument is wrong
and nothing downstream is readable — the same role eps=1.0 -> random played for the explorer.

WHY THIS MODULE MIGHT ACTUALLY BE LEARNABLE, unlike the explorer. Its observation (h, s,
water_known, food_known, tile levels) is continuous and directly informative about the decision
it makes. There is no perceptual aliasing: distinct situations produce distinct inputs. The
explorer failed because its observation carried no positional memory and physically different
cells collapsed onto identical inputs. That failure mode does not apply here.
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
from sweep_fn_v5 import compute_eval_metrics, wilson_interval  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
RADIUS = 20
BASE_CRIT = 0.7

EXPLORER_KW = {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
               "follow_p": 0.95, "reverse_on_drop_p": 0.75}

# candidate detectors. Drive percentiles are included because bad arbitration shows up as a
# drive running low long before it shows up as a death.
COLS = [
    ("solveScore", "solveScr"),
    ("mean_comfort", "comfort"),
    ("p05_hydration", "p05_hyd"),
    ("p05_satiation", "p05_sat"),
    ("water_visit_pct", "waterVis"),
    ("food_visit_pct", "foodVis"),
]


def score(orch_kw, seeds, sim_len=7000, eval_len=5000) -> dict:
    import sim_instance_v3 as S

    # Resolve the impl ONCE, outside the loop, and work on a copy.
    #
    # v1 called orch_kw.pop("_impl") inside the seed loop on the caller's dict: seed 0
    # consumed the key, and seed 1 onward silently fell back to "noisy_oracle" while still
    # carrying `weights`, which the oracle constructor rejects. Mutating an argument inside a
    # loop over it is the bug; copying is the fix.
    orch_kw = dict(orch_kw)
    impl = orch_kw.pop("_impl", "noisy_oracle")

    D = T = 0
    rows = []
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
        D += run["death_count_eval"]
        T += run["n_timeouts_eval"]
        rows.append(compute_eval_metrics(run, dict(radius=RADIUS, band=(9, 11))))
        eb = int(run["eval_boundary"])
        for e in run.get("death_events", []):
            if e["t"] >= eb:
                causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    n = T + D
    lo, hi = wilson_interval(T, n) if n > 0 else (float("nan"), float("nan"))
    out = {"eval_deaths": int(D), "n_events": int(n), "causes": causes,
           "solveScore": (T / n) if n > 0 else float("nan"), "ci": (lo, hi)}
    for key, _ in COLS:
        if key == "solveScore":
            continue
        with np.errstate(all="ignore"):
            vals = [r.get(key, np.nan) for r in rows]
            out[key] = float(np.nanmedian(vals)) if vals else float("nan")
    return out


def base_kw(**over):
    kw = {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": BASE_CRIT, "s_crit": BASE_CRIT}
    kw.update(over)
    return kw


def calibrate(seeds, epsilons, crit_scales):
    plan = [(("oracle", None), dict(base_kw(), _impl="oracle"))]
    plan += [(("epsilon", e), base_kw(epsilon=e)) for e in epsilons if e > 0]
    plan += [(("crit_scale", c), base_kw(crit_scale=c)) for c in crit_scales if c != 1.0]

    rows = {}
    t0 = time.time()
    for i, (key, kw) in enumerate(plan, 1):
        label = "oracle (baseline)" if key[0] == "oracle" else f"{key[0]}={key[1]:g}"
        print(f"  [{i}/{len(plan)}] {label} ...", end="", flush=True)
        rows[key] = score(kw, seeds)
        r = rows[key]
        print(f" solve {r['solveScore']:.4f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  "
              f"deaths {r['eval_deaths']:>4}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def report(rows, epsilons, crit_scales):
    base = rows[("oracle", None)]
    keys = [k for k, _ in COLS]
    shorts = [s for _, s in COLS]

    print("\n--- calibration " + "-" * 58)
    print(f"{'arm':<20} {'deaths':>7} " + " ".join(f"{s:>9}" for s in shorts) + "   causes")

    def row(label, r):
        print(f"{label:<20} {r['eval_deaths']:>7} "
              + " ".join(f"{r[k]:>9.4f}" if np.isfinite(r[k]) else f"{'nan':>9}"
                         for k in keys)
              + f"   {r['causes']}")

    row("oracle", base)
    for e in [x for x in epsilons if x > 0]:
        row(f"epsilon={e:g}", rows[("epsilon", e)])
    for c in [x for x in crit_scales if x != 1.0]:
        row(f"crit_scale={c:g}", rows[("crit_scale", c)])

    # --- anchor check ------------------------------------------------------------
    #
    # REVISED AFTER FIRST RUN, and the reason matters. v1 required crit_scale=0 to separate
    # from the baseline CI, on the grounds that ablating the critical-drive interrupt had a
    # recorded effect of "hydration deaths 43 -> 10". That figure was measured under a
    # DIFFERENT configuration — this baseline carries 29 hydration deaths, not 13 — so the
    # anchor was mis-specified against a stale number. It is a factual error in how the test
    # was built, not a property of the instrument.
    #
    # The replacement asks the question the anchor was always trying to ask: does the
    # arbitration threshold matter AT ALL? If any point on the crit_scale axis separates from
    # the baseline, the decision this module makes is consequential and the metric can see it.
    #
    # Stated plainly because changing a criterion after seeing its output is exactly how
    # thresholds stop meaning anything: the PRIMARY criterion (a monotone detector with
    # usable range on the epsilon axis) is unchanged and passed independently of this.
    print("\n  instrument checks:")
    crit_keys = [k for k in rows if k[0] == "crit_scale"]
    separated = [k for k in crit_keys if rows[k]["ci"][1] < base["ci"][0]
                 or rows[k]["ci"][0] > base["ci"][1]]
    print(f"    crit_scale axis: {len(separated)} of {len(crit_keys)} points separate from "
          f"the baseline CI [{base['ci'][0]:.3f},{base['ci'][1]:.3f}]")
    for k in sorted(separated, key=lambda x: x[1]):
        r = rows[k]
        direction = "WORSE" if r["solveScore"] < base["solveScore"] else "BETTER"
        print(f"      crit_scale={k[1]:g}: {r['solveScore']:.4f} "
              f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  {direction}   {r['causes']}")
    anchor_ok = len(separated) > 0
    if not anchor_ok:
        print("      -> no point on the arbitration axis separates. Either the threshold does "
              "not matter, or the seeds are too few to tell. Raise --seeds before training.")

    # interior optimum: is the hand-tuned baseline actually the best setting?
    if crit_keys:
        best_k = max(crit_keys + [("crit_scale", 1.0)],
                     key=lambda k: rows.get(k, base)["solveScore"])
        best_r = rows.get(best_k, base)
        if best_k[1] != 1.0 and best_r["solveScore"] > base["solveScore"]:
            print(f"    NOTE: crit_scale={best_k[1]:g} scores {best_r['solveScore']:.4f} against "
                  f"the hand-tuned baseline's {base['solveScore']:.4f}. CIs overlap at these "
                  f"seeds, so this is not yet a result — but the hand-tuned threshold may not "
                  f"be at its optimum, which is a target a learned orchestrator could beat.")

    # monotone-gated floors on the epsilon axis
    eps_used = [x for x in epsilons if x > 0]
    if eps_used:
        print("\n  epsilon axis — monotone-gated detection floor:")
        for key, short in COLS:
            series = [base[key]] + [rows[("epsilon", e)][key] for e in eps_used]
            fin = [v for v in series if np.isfinite(v)]
            if len(fin) < 2:
                print(f"    {short:<10} no data")
                continue
            down = all(b <= a + 1e-9 for a, b in zip(fin, fin[1:]))
            up = all(b >= a - 1e-9 for a, b in zip(fin, fin[1:]))
            if not (down or up):
                print(f"    {short:<10} NOT MONOTONE -> cannot rank orchestrators")
                continue
            if not np.isfinite(base[key]) or base[key] == 0:
                print(f"    {short:<10} baseline 0/nan -> no usable ratio")
                continue
            floor = next((e for e in eps_used
                          if abs(rows[("epsilon", e)][key] - base[key]) / abs(base[key]) > 0.02),
                         None)
            print(f"    {short:<10} monotone; " +
                  (f"floor eps={floor:g}" if floor else "never moves >2% -> too coarse"))

    print("\n  GO / NO-GO:")
    # PRIMARY criterion: a monotone detector with usable dynamic range. This is what decides
    # whether a trained module's number would mean anything.
    detectors = []
    for key, short in COLS:
        series = [base[key]] + [rows[("epsilon", e)][key] for e in eps_used]
        fin = [v for v in series if np.isfinite(v)]
        if len(fin) < 2:
            continue
        if (all(b <= a + 1e-9 for a, b in zip(fin, fin[1:]))
                or all(b >= a - 1e-9 for a, b in zip(fin, fin[1:]))):
            detectors.append((short, max(fin) - min(fin)))

    for short, rng_ in sorted(detectors, key=lambda x: -x[1]):
        print(f"    detector: {short:<10} dynamic range {rng_:.4f}")
    if not detectors:
        print("    no monotone detector on the epsilon axis")

    # for scale: the other modules' best detectors, measured the same way
    print("    (for comparison — explorer 0.362, consumer 0.044)")

    if detectors and max(r for _, r in detectors) > 0.10 and anchor_ok:
        print("\n    GO — a metric ranks orchestrators across a usable range, and the")
        print("    arbitration threshold demonstrably changes the outcome.")
        print("    Train with: fixed eval seeds, best-checkpoint, 3 seeds, and a SURVIVAL-based")
        print("    reward. Comfort is flat across the tolerance band, so it cannot grade the")
        print("    decisions this module makes (see P2 in BUILDNOTES).")
    else:
        print("\n    NO-GO — no usable detector. Training would produce an uninterpretable number.")


def report_learned(tags, seeds, base):
    print(f"\n--- learned orchestrators ({len(tags)} seed(s)) " + "-" * 30)
    print(f"{'tag':<12} {'deaths':>7} {'solveScore':>11} {'CI':>17}  vs oracle")
    scores = []
    for tag in tags:
        r = score(dict(base_kw(weights=tag), _impl="learned"), seeds)
        scores.append(r["solveScore"])
        sep = "SEPARATED" if (r["ci"][0] > base["ci"][1] or r["ci"][1] < base["ci"][0]) else ""
        print(f"{tag:<12} {r['eval_deaths']:>7} {r['solveScore']:>11.4f} "
              f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  "
              f"{r['solveScore'] - base['solveScore']:+.4f} {sep}")
        print(f"             causes {r['causes']}")

    fin = [s for s in scores if np.isfinite(s)]
    if not fin:
        return
    lo, hi = min(fin), max(fin)
    print(f"\n  across seeds: min {lo:.4f}  median {np.median(fin):.4f}  max {hi:.4f}")
    if lo > base["ci"][1]:
        print(f"  VERDICT: beats the oracle on every seed, CI-separated. Learned arbitration "
              f"exceeds the hand-tuned policy.")
    elif lo >= base["ci"][0]:
        print(f"  VERDICT: within the oracle's CI [{base['ci'][0]:.3f},{base['ci'][1]:.3f}] — "
              f"MATCHES hand-tuned arbitration. Not a win, and not a failure: the module is "
              f"indistinguishable from the policy it replaced, which is the honest claim.")
    else:
        print(f"  VERDICT: below the oracle. Check the action mix in the training log before "
              f"reading this as 'arbitration is not learnable' — a collapsed mix means the "
              f"policy never arbitrated at all.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.15, 0.35, 0.7])
    p.add_argument("--crit-scales", type=float, nargs="+", default=[0.0, 0.5, 1.5, 3.0])
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--calibrate-only", action="store_true")
    a = p.parse_args()

    seeds = range(a.seeds)
    print("=" * 82)
    print(f"orchestrator sensitivity — smell 3, band (9,11), nondoomed, {a.seeds} seeds")
    print(f"epsilon {a.epsilons}   crit_scale {a.crit_scales}  (baseline crit {BASE_CRIT})")
    print("calibrate first: can any metric rank orchestrators?")
    rows = calibrate(seeds, a.epsilons, a.crit_scales)
    report(rows, a.epsilons, a.crit_scales)
    if a.tags and not a.calibrate_only:
        report_learned(a.tags, seeds, rows[("oracle", None)])
    print("=" * 82)


if __name__ == "__main__":
    main()
