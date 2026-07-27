"""
probe_infra_v1.py  —  does the machinery do what it claims?

    python tests/probe_infra_v1.py
    python tests/probe_infra_v1.py --verbose

Not a results check. Every probe here guards an invariant whose violation is SILENT: the
run completes, numbers come out, and they are wrong. Those are the failures that cost a
whole verdict, so they get asserted rather than assumed.

P1.1 is the cautionary tale — solveScore appeared to have a detection floor for pages
until a controlled degradation showed it was non-monotone. Nothing errored. The number was
simply meaningless, and would have gone into the ledger as a pass.

Probes, cheapest first:

  1  privilege boundary   no learned module imports world_v1 (static grep)
  2  checkpoint round-trip save -> load -> identical outputs
  3  registry integrity    every MODULE_REGISTRY entry constructs
  4  log silence           log_every=10**9 prints nothing
  5  determinism           same seed twice -> identical arrays
  6  action_T shim         legacy ids agree with the fracs they were built from
  7  compute_eval_metrics  returns a populated dict again, with the named verdict keys
  8  consumer_slot_T       dispatch record agrees with fracs where fracs are unambiguous
  9  rig alignment         records join to the ticks they actually came from
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hex_world_cached as hex_world  # noqa: E402
from model_modules.contract_v1 import Action, ConsumerObs, ModuleSpec  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def probe(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw) or "ok"
                RESULTS.append((name, True, msg))
            except AssertionError as e:
                RESULTS.append((name, False, str(e)))
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return wrapped
    return deco


SHORT_SIM = dict(sim_len=1200, eval_len=600)


def _oracle_params(**over):
    p = {
        "orchestrator": {"h_fill": 1.6, "s_fill": 1.3, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": 1.3},
        "drink": {"fill_target": 1.6},
        "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                     "follow_p": 0.95, "reverse_on_drop_p": 0.75},
    }
    p.update(over)
    return p


def _run(spec=None, oracle_params=None, seed=0, **over):
    import sim_instance_v3 as S

    kw = dict(
        seed=seed, env_kwargs=dict(radius=20, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7, smell_radius=3, curriculum_mode="band",
        c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=oracle_params or _oracle_params(),
        module_spec=spec or ModuleSpec("oracle", "oracle", "oracle", "oracle",
                                       explorer="smell_momentum"),
        log_every=10**9, eval_spawn_nondoomed=True, spawn_leeway=10,
        **SHORT_SIM,
    )
    kw.update(over)
    return S.sim_instance(**kw)


# --- 1. privilege boundary ------------------------------------------------------

@probe("privilege boundary (no learned module imports world_v1)")
def probe_privilege():
    """
    world_v1's own docstring: "an oracle is *allowed* to know the true physics ... but a
    learned module importing `world` is a leak -- grep for it." This is that grep.

    A learned module that imports world can read DRINK_AMOUNT and the (0.8+0.3*s) coupling
    straight out of the file. It would train beautifully and prove nothing, because the
    dynamics it was supposed to discover were handed to it in closed form.
    """
    d = _PROTO_DIR / "model_modules" / "learned_modules"
    pat = re.compile(r"^\s*(?:import\s+world_v1|from\s+world_v1\s+import)", re.M)
    bad = [p.name for p in d.glob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert not bad, f"LEAK: learned modules import world_v1: {bad}"
    return f"{len(list(d.glob('*.py')))} learned modules clean"


# --- 2. checkpoint round-trip ---------------------------------------------------

@probe("checkpoint round-trip (save -> load -> identical outputs)")
def probe_checkpoint():
    from model_modules.checkpoint_v1 import clear_cache, load_net, resolve, save_module
    from model_modules.learned_modules.nets_v1 import build_net

    arch = {"kind": "mlp", "n_input": 5, "n_hidden": 8, "n_act": 6}
    net = build_net(arch)
    net.eval()
    x = torch.randn(4, 5)
    with torch.no_grad():
        before = net(x).clone()

    tag = "__probe_tmp__"
    path = save_module(net, kind="pathfinder", tag=tag, arch=arch,
                       obs_fields=["a", "b", "c", "d", "e"], metrics={"probe": True},
                       train_config={"probe": True}, train_seed=0, notes="probe artefact")
    try:
        clear_cache()
        loaded, payload = load_net("pathfinder", tag)
        with torch.no_grad():
            after = loaded(x)
        assert torch.allclose(before, after, atol=0), "weights changed across save/load"
        assert payload["arch"] == arch, "arch not preserved"
        assert not any(p.requires_grad for p in loaded.parameters()), \
            "loaded net is not frozen; a sweep could silently train it"
        assert not loaded.training, "loaded net is in train mode (NoisyNet would inject noise)"
    finally:
        clear_cache()
        p = resolve("pathfinder", tag)
        if p.exists():
            p.unlink()
    return "weights, arch and frozen/eval state all preserved"


# --- 3. registry integrity ------------------------------------------------------

@probe("registry integrity (every entry constructs)")
def probe_registry():
    """
    A registry key that raises on construction only fails when a sweep reaches it, which
    may be an hour in. Constructing each one here costs milliseconds.
    """
    import sim_instance_v3 as S

    rng = np.random.default_rng(0)
    needs_weights = {("pathfinder", "learned"), ("eat", "learned"), ("drink", "learned"),
                     ("explorer", "learned")}
    built, skipped = 0, []

    for slot, impls in S.MODULE_REGISTRY.items():
        for name, factory in impls.items():
            if (slot, name) in needs_weights:
                skipped.append(f"{slot}/{name}")
                continue
            kwargs = {}
            if slot == "explorer":
                kwargs["rng"] = rng
            elif slot == "orchestrator":
                kwargs.update(rng=rng, explorer=S.MODULE_REGISTRY["explorer"]["random"](rng=rng))
            factory(**kwargs)
            built += 1
    return f"{built} built; {len(skipped)} need weights (skipped: {', '.join(skipped)})"


# --- 4. log silence -------------------------------------------------------------

@probe("log silence (log_every=10**9 prints nothing)")
def probe_log_silence():
    """t=0 satisfies t % log_every == 0 for every log_every, so the guard is t > 0."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run()
    out = buf.getvalue().strip()
    assert not out, f"expected silence, got:\n{out[:400]}"
    return "silent"


# --- 5. determinism -------------------------------------------------------------

@probe("determinism (same seed -> identical run)")
def probe_determinism():
    a, b = _run(seed=3), _run(seed=3)
    for k in ("comfort_T", "hydration_T", "death_T", "action_T", "consumer_slot_T"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f"{k} differs across runs"
    return "comfort/hydration/death/action/consumer_slot all identical"


# --- 6. action_T shim -----------------------------------------------------------

@probe("action_T shim (legacy ids agree with the fracs they came from)")
def probe_action_shim():
    """
    The shim is lossy by design (continuous frac -> 3 legacy levels). What must hold
    exactly is what the metrics key off: drink/eat membership, full-eat identity, and
    move ids. The 0.5-vs-0.25 split is approximate and deliberately NOT asserted.
    """
    from sweep_fn_v5 import DRINK_IDS, EAT_IDS, FULL_EAT_ID, MOVE_MIN_ID

    r = _run()
    a = np.asarray(r["action_T"])
    df, ef = np.asarray(r["drink_frac_T"]), np.asarray(r["eat_frac_T"])
    md = np.asarray(r["move_dir_T"])

    assert np.array_equal(np.isin(a, DRINK_IDS), df > 0), "DRINK_IDS != (drink_frac > 0)"
    assert np.array_equal(np.isin(a, EAT_IDS), ef > 0), "EAT_IDS != (eat_frac > 0)"
    assert np.array_equal(a == FULL_EAT_ID, ef >= 1.0 - 1e-6), "FULL_EAT_ID != (eat_frac == 1)"
    moved = md >= 0
    assert np.array_equal(a >= MOVE_MIN_ID, moved), "move ids != (move_dir >= 0)"
    assert np.array_equal(a[moved], hex_world.MOVE_OFFSET + md[moved]), "move id offset wrong"
    assert np.array_equal(a == 0, (df <= 0) & (ef <= 0) & ~moved), "WAIT id leaks"
    return f"{int((df > 0).sum())} drinks, {int((ef > 0).sum())} eats, {int(moved.sum())} moves"


# --- 7. compute_eval_metrics ----------------------------------------------------

@probe("compute_eval_metrics returns populated metrics")
def probe_eval_metrics():
    """
    Before the shim this returned {} on every Proto 06 run, silently taking the named
    verdict metrics with it. Empty is the bug being guarded against.
    """
    from sweep_fn_v5 import compute_eval_metrics

    r = _run()
    m = compute_eval_metrics(r, dict(radius=20, band=(9, 11)))
    assert m, "compute_eval_metrics returned {} -- action_T shim not reaching it"
    for k in ("eval_deaths", "eat_rate_at_food", "drink_rate_at_water",
              "water_to_food_success_rate", "two_way_route_success_min"):
        assert k in m, f"missing named metric '{k}'"
    return f"{len(m)} metrics; eat_rate_at_food={m['eat_rate_at_food']}"


# --- 8. consumer_slot_T ---------------------------------------------------------

@probe("consumer_slot_T (dispatch record agrees with unambiguous fracs)")
def probe_consumer_slot():
    """
    The rig joins module records onto ticks through this field. A wrong join trains every
    transition against the wrong reward and raises nothing, so it is checked directly.

    Only the unambiguous direction is asserted: a nonzero frac implies its slot. The
    converse does not hold -- a zero-frac consume is a real event and is exactly why this
    field exists instead of frac inference.
    """
    r = _run()
    cs = np.asarray(r["consumer_slot_T"])
    df, ef = np.asarray(r["drink_frac_T"]), np.asarray(r["eat_frac_T"])
    ab = np.asarray(r["abstract_action_T"])

    assert np.all(cs[df > 0] == 1), "drink_frac > 0 at a tick not marked drink"
    assert np.all(cs[ef > 0] == 2), "eat_frac > 0 at a tick not marked eat"
    assert np.all(ab[cs > 0] == int(Action.CONSUME)), "consumer dispatched without CONSUME"
    assert not np.any((cs == 1) & (ef > 0)), "tick marked drink also ate"
    assert not np.any((cs == 2) & (df > 0)), "tick marked eat also drank"
    n0 = int(((cs > 0) & (df <= 0) & (ef <= 0)).sum())
    return f"{int((cs == 1).sum())} drink / {int((cs == 2).sum())} eat dispatches, {n0} zero-frac"


# --- 9. rig alignment -----------------------------------------------------------

@probe("rig alignment (records join to the ticks they came from)")
def probe_rig_alignment():
    """
    End-to-end on the join the consumer trainer depends on. Recording consumers are run
    in-sim, then their records are re-derived from the sim's own logged fracs and compared.
    If these disagree, every reward in training is attached to the wrong decision.

    BOTH slots are wired and coverage is REPORTED rather than assumed. Food encounters are
    rare -- a 1200-tick run produced 279 drinks and 0 eats -- so a probe that silently
    exercised only one slot would claim to certify machinery it never touched. A longer
    sim on the beside-water eval gives eat a chance; if it still gets nothing, the return
    message says so instead of implying full coverage.
    """
    from model_modules.learned_modules.consumer_learned_v1 import (
        BINS, LearnedConsumer, default_arch,
    )
    from model_modules.learned_modules.nets_v1 import build_net
    from training.rigs.insim_rig_v1 import align_records

    net = build_net(default_arch(n_hidden=8))
    net.eval()
    mods = {"eat": LearnedConsumer("food", net=net),
            "drink": LearnedConsumer("water", net=net)}
    for m in mods.values():
        m.start_recording()

    import sim_instance_v3 as S
    orig = {k: S.MODULE_REGISTRY[k]["learned"] for k in mods}
    for k, m in mods.items():
        S.MODULE_REGISTRY[k]["learned"] = (lambda mm: (lambda **kw: mm))(m)
    try:
        r = _run(
            spec=ModuleSpec("oracle", "oracle", "learned", "learned",
                            explorer="smell_momentum"),
            sim_len=4000, eval_len=2000,
            eval_spawn_nondoomed=False, eval_spawn_beside_water=True,
        )
    finally:
        for k in mods:
            S.MODULE_REGISTRY[k]["learned"] = orig[k]
    for m in mods.values():
        m.stop_recording()

    frac_key = {"eat": "eat_frac_T", "drink": "drink_frac_T"}
    covered, notes = [], []
    for slot, m in mods.items():
        if not m.records:
            notes.append(f"{slot}: 0 calls (not exercised)")
            continue
        ticks = align_records(r, m.records, slot)   # raises on count mismatch
        f = np.asarray(r[frac_key[slot]])
        for i, t in enumerate(ticks):
            want = float(BINS[m.records[i][1]])
            assert abs(f[int(t)] - want) < 1e-5, (
                f"{slot} join wrong at record {i}: module chose {want}, sim logged "
                f"{f[int(t)]} at tick {t}"
            )
        covered.append(slot)
        notes.append(f"{slot}: {len(ticks)} records joined, fracs match")

    assert covered, (
        "neither consumer was invoked; the join is UNVERIFIED. Lengthen the probe sim or "
        "use a config that reaches a resource — do not read this probe as a pass."
    )
    return "; ".join(notes)


# --- run ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    probes = [
        probe_privilege, probe_checkpoint, probe_registry, probe_log_silence,
        probe_determinism, probe_action_shim, probe_eval_metrics,
        probe_consumer_slot, probe_rig_alignment,
    ]
    print("=" * 78)
    print("infrastructure probes")
    print("=" * 78)
    for fn in probes:
        fn()
        name, ok, msg = RESULTS[-1]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if a.verbose or not ok:
            print(f"       {msg}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
