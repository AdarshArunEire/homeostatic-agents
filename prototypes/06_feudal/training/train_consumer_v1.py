"""
train_consumer_v1.py  —  in-sim RL for the eat and drink consumers.

    python training/train_consumer_v1.py --slot drink --tag v1
    python training/train_consumer_v1.py --slot eat   --tag v1

ONE SLOT AT A TIME, AND UNDER DIFFERENT CONFIGS. Measured on the standing nondoomed config,
a 1200-tick oracle run gave 279 drink dispatches and 0 eat dispatches: the agent finds water
readily and rarely reaches food. Training both heads under one config would hand the eat
head a few dozen samples and a confident-looking loss curve. So:

  drink  nondoomed     — the standing config, ample on-policy data
  eat    beside_water  — food isolation; water is handed over so food encounters are the
                         point. Raised eat records from 0 to 47 in the rig probe.

The eat head remains the data-poor one even so. Its rollout budget is higher by default,
and `--rounds` should be raised before anything is concluded from it.

REWARD IS WINDOWED, NOT INSTANTANEOUS. A fill absorbs through `a_que` (maxlen 11) and the
cost of overfilling only materialises later as decay wastes the excess. Scoring the tick of
the consume alone rewards slamming 1.0 every time — the module would learn the single worst
policy the graded action space exists to avoid. `comfort_window_reward` credits the module
over the absorption window instead.

THE ORACLE IS NOT GROUND TRUTH HERE. Its `fill_target` (1.6 / 1.3 in the standing config) is
a POLICY choice that was swept, not physics. So `mean |learned - oracle|` is reported as a
diagnostic only; a learned consumer that disagrees with the oracle is not thereby wrong, and
this number must never become the training objective. The verdict comes from
`tests/consumer_sensitivity_v1.py`, against a calibrated noisy control.
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

from model_modules.checkpoint_v1 import prepare_worker, save_module  # noqa: E402
from model_modules.contract_v1 import ModuleSpec  # noqa: E402
from model_modules.learned_modules.nets_v1 import make_trainable  # noqa: E402
from model_modules.learned_modules.consumer_learned_v1 import (  # noqa: E402
    BINS, N_BINS, OBS_FIELDS, LearnedConsumer, default_arch,
)
from training.rigs.insim_rig_v1 import (  # noqa: E402
    comfort_window_reward, consumer_reward, harvest, rollout, use_module,
)

# A FIXED set of eval seeds, held apart from training seeds.
#
# v1 evaluated on seed=10_000+round, so every checkpoint met a different map and the curve
# mixed learning with map variation -- deaths read 3, 2, 2, 7, 4, 9, 3, which is a random
# walk, not a learning curve. Same mistake caught and fixed in the pathfinder rig, then
# reintroduced here. Fixed seeds mean a change in the number is a change in the policy.
EVAL_SEEDS = (10_001, 10_002, 10_003, 10_004)
from training import rl_core_v1 as rl  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3

# per-slot eval config; see module docstring for why these differ
SLOT_CONFIG = {
    "drink": dict(eval_spawn_nondoomed=True, spawn_leeway=10,
                  eval_spawn_beside_water=False),
    "eat": dict(eval_spawn_nondoomed=False, eval_spawn_beside_water=True),
}
SLOT_RESOURCE = {"drink": "water", "eat": "food"}


def _oracle_params(**over):
    p = {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                     "follow_p": 0.95, "reverse_on_drop_p": 0.75},
    }
    p.update(over)
    return p


def _spec(slot: str) -> ModuleSpec:
    """Learned in one slot, oracle everywhere else — the standing training protocol."""
    return ModuleSpec(
        orchestrator="oracle",
        pathfinder="oracle",
        eat="learned" if slot == "eat" else "oracle",
        drink="learned" if slot == "drink" else "oracle",
        explorer="smell_momentum",
    )


def collect(slot, module, seed, sim_len, eval_len, reward_fn, gamma, n_step):
    """One frozen-weights rollout -> transitions. The module records; rewards join after."""
    module.start_recording()
    with use_module(**{slot: module}):
        run = rollout(_spec(slot), _oracle_params(), seed=seed,
                      sim_len=sim_len, eval_len=eval_len, **SLOT_CONFIG[slot])
    module.stop_recording()

    trans = harvest(run, module, slot, reward_fn, n_act=N_BINS, gamma=gamma, n_step=n_step)
    return run, trans


def evaluate(slot, net, sim_len, eval_len, seeds=EVAL_SEEDS):
    """
    Frozen greedy rollout over the FIXED eval seeds, pooled.

    `oracle_gap` is a DIAGNOSTIC, not a score. The oracle's fill_target is a swept policy
    knob, so disagreeing with it is not an error — the number is here to show whether the
    module has learned anything shaped at all, not to be minimised.
    """
    import world_v1 as world  # trainer-side only; the learned module must not import this

    D = T = 0
    comforts, fracs, gaps, calls = [], [], [], 0

    for seed in seeds:
        module = LearnedConsumer(SLOT_RESOURCE[slot], net=net)
        module.start_recording()
        with use_module(**{slot: module}):
            run = rollout(_spec(slot), _oracle_params(), seed=seed,
                          sim_len=sim_len, eval_len=eval_len, **SLOT_CONFIG[slot])
        module.stop_recording()

        D += int(run["death_count_eval"])
        T += int(run["n_timeouts_eval"])
        eb = int(run["eval_boundary"])
        comforts.append(float(np.asarray(run["comfort_T"], dtype=float)[eb:].mean()))
        calls += len(module.records)

        for x, a in module.records:
            h, s = float(x[0]) * 3.0, float(x[1]) * 3.0
            ref = (world.drink_frac_for(h, s, H_FILL) if slot == "drink"
                   else world.eat_frac_for(s, S_FILL))
            fracs.append(float(BINS[a]))
            gaps.append(abs(float(BINS[a]) - ref))

    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "mean_comfort_eval": float(np.mean(comforts)) if comforts else float("nan"),
        "n_calls": calls,
        "mean_frac": float(np.mean(fracs)) if fracs else float("nan"),
        "oracle_gap": float(np.mean(gaps)) if gaps else float("nan"),
    }


def train(
    slot: str = "drink",
    tag: str = "v1",
    rounds: int = 60,
    sim_len: int = 7000,
    eval_len: int = 5000,
    seed: int = 0,
    n_hidden: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 3,
    batch_size: int = 128,
    buffer_size: int = 200_000,
    target_sync: int = 200,
    updates_per_round: int = 400,
    warmup: int = 1000,
    eps_start: float = 0.6,
    eps_end: float = 0.05,
    eps_frac: float = 0.5,
    window: int = 25,
    cap_per_round: int = 2500,
    reward: str = "survival",
    death_penalty: float = 20.0,
    tick_scale: float = 50.0,
    eval_every: int = 10,
    save: bool = True,
):
    if slot not in SLOT_CONFIG:
        raise ValueError(f"slot must be one of {sorted(SLOT_CONFIG)}, got {slot!r}")

    prepare_worker()
    rng = rl.seed_everything(seed)

    arch = default_arch(n_hidden=n_hidden, kind="mlp")
    net, target, optimiser = make_trainable(arch, lr)
    buffer = rl.ReplayBuffer(buffer_size, rng=rng)
    if reward == "survival":
        reward_fn = consumer_reward(death_penalty=death_penalty, tick_scale=tick_scale,
                                    window=window, gamma=gamma)
    elif reward == "comfort":
        # v1's reward, kept only to reproduce the failure it produced
        reward_fn = comfort_window_reward(window=window, gamma=gamma)
    else:
        raise ValueError(f"reward must be 'survival' or 'comfort', got {reward!r}")

    history = []
    updates = 0
    t0 = time.time()

    # Best-checkpoint selection.
    #
    # v1(survival) oscillated meanFrac 0.72 / 0.95 / 0.70 / 0.96 / 0.98 / 0.48 on FIXED eval
    # seeds -- policy variation, not map noise. Saving the last round therefore ships an
    # arbitrary snapshot of an oscillation, and a different seed would ship a different
    # module. Selecting on the eval that is already being computed costs nothing and makes
    # the deliverable the best policy actually observed rather than the most recent one.
    #
    # Selection is on eval_deaths (solveScore breaks ties), NOT on oracle_gap: matching the
    # oracle is not the objective, since its fill_target is a swept policy knob.
    import copy as _copy

    best = {"score": None, "state": None, "metrics": None, "round": None}

    def _maybe_keep(m, rd):
        score = (m["eval_deaths"], -(m["solveScore"] if np.isfinite(m["solveScore"]) else 0))
        if best["score"] is None or score < best["score"]:
            best.update(score=score, state=_copy.deepcopy(net.state_dict()),
                        metrics=dict(m), round=rd)
            return True
        return False

    for rd in range(rounds):
        frac = min(1.0, rd / max(1, int(eps_frac * rounds)))
        epsilon = eps_start + frac * (eps_end - eps_start)

        module = LearnedConsumer(SLOT_RESOURCE[slot], net=net,
                                 explore_eps=epsilon, rng=rng)
        # a fresh map per round; otherwise the module overfits one layout's encounter rate
        _, trans = collect(slot, module, seed=seed + rd, sim_len=sim_len,
                           eval_len=eval_len, reward_fn=reward_fn,
                           gamma=gamma, n_step=n_step)

        # CAP THE PER-ROUND CONTRIBUTION.
        #
        # The consumer is called once per consume, so a round's transition count is set by
        # the policy being evaluated -- and the two degenerate policies are not symmetric. A
        # fill-nothing policy must drink constantly: round 50 of v3 emitted 13,882
        # transitions against a typical ~2,500. That round alone then dominates the buffer
        # and trains the next round toward itself, which is a positive feedback loop the
        # reward cannot damp. Observed result: bistable flipping between meanFrac 0.157 and
        # 0.984, worsening as training continued.
        #
        # Capping makes every round contribute equally regardless of how chatty its policy
        # is, so buffer composition reflects rounds rather than call frequency. Sampled
        # without replacement to keep the round's own distribution intact.
        n_kept = len(trans)
        if cap_per_round and len(trans) > cap_per_round:
            pick = rng.choice(len(trans), size=cap_per_round, replace=False)
            trans = [trans[int(i)] for i in pick]
            n_kept = cap_per_round
        buffer.extend(trans)

        if len(buffer) >= warmup:
            for _ in range(updates_per_round):
                rl.learn_step(net, target, optimiser, buffer.sample(batch_size),
                              gamma, double=True)
                updates += 1
                if updates % target_sync == 0:
                    rl.sync_target(target, net)

        if eval_every and (rd + 1) % eval_every == 0:
            net.eval()
            m = evaluate(slot, net, sim_len=sim_len, eval_len=eval_len)
            net.train()
            m.update({"round": rd + 1, "buffer": len(buffer), "new_trans": n_kept,
                      "epsilon": round(epsilon, 3)})
            history.append(m)
            kept = _maybe_keep(m, rd + 1)
            print(
                f"rd {rd+1:>4}  deaths {m['eval_deaths']:>4}  solve {m['solveScore']:.3f}  "
                f"comfort {m['mean_comfort_eval']:.3f}  calls {m['n_calls']:>5}  "
                f"meanFrac {m['mean_frac']:.3f}  oracleGap {m['oracle_gap']:.3f}  "
                f"buf {len(buffer):>7}  {time.time()-t0:.0f}s{'  <- best' if kept else ''}"
            )

    net.eval()
    last = evaluate(slot, net, sim_len=sim_len, eval_len=eval_len)
    _maybe_keep(last, rounds)

    print(f"\nlast[{slot}]:  deaths {last['eval_deaths']}  solve {last['solveScore']:.3f}  "
          f"meanFrac {last['mean_frac']:.3f}")

    # ship the best observed policy, not the most recent one
    if best["state"] is not None and best["round"] != rounds:
        net.load_state_dict(best["state"])
        net.eval()
    final = best["metrics"] or last
    print(f"best[{slot}]:  deaths {final['eval_deaths']}  solve {final['solveScore']:.3f}  "
          f"comfort {final['mean_comfort_eval']:.3f}  calls {final['n_calls']}  "
          f"meanFrac {final['mean_frac']:.3f}   (round {best['round']} of {rounds})")

    fracs_seen = [h["mean_frac"] for h in history if np.isfinite(h["mean_frac"])]
    if len(fracs_seen) > 2:
        spread = max(fracs_seen) - min(fracs_seen)
        if spread > 0.25:
            print(f"  WARNING: meanFrac ranged {min(fracs_seen):.3f}-{max(fracs_seen):.3f} "
                  f"(spread {spread:.3f}) across a FIXED eval set. The policy is "
                  f"oscillating, not converging — best-checkpoint selection is masking "
                  f"instability, not curing it. Lower --death-penalty or lr before "
                  f"treating this module as settled.")

    if final["n_calls"] < 50:
        print(f"  WARNING: only {final['n_calls']} eval calls. This head is data-starved; "
              f"raise --rounds or --sim-len before reading anything into the numbers.")

    train_config = {
        "slot": slot, "rounds": rounds, "sim_len": sim_len, "eval_len": eval_len,
        "n_hidden": n_hidden, "lr": lr, "gamma": gamma, "n_step": n_step,
        "batch_size": batch_size, "buffer_size": buffer_size,
        "target_sync": target_sync, "updates_per_round": updates_per_round,
        "warmup": warmup, "eps_start": eps_start, "eps_end": eps_end,
        "eps_frac": eps_frac, "window": window, "n_bins": N_BINS,
        "cap_per_round": cap_per_round,
        "eval_config": SLOT_CONFIG[slot], "eval_seeds": list(EVAL_SEEDS),
        "rig": "insim_rig_v1 (iterated batch off-policy)",
        "reward_kind": reward,
        "death_penalty": death_penalty, "tick_scale": tick_scale,
        "reward": (f"ticks-bought/{tick_scale} + comfort over {window} "
                   f"- {death_penalty} on death" if reward == "survival"
                   else f"discounted comfort over {window} ticks (SUPERSEDED)"),
    }

    if save:
        path = save_module(
            net, kind=slot, tag=tag, arch=arch, obs_fields=OBS_FIELDS,
            metrics={"final": final, "last": last, "best_round": best["round"],
                     "history": history},
            train_config=train_config, train_seed=seed,
            notes=f"In-sim RL, oracles in other slots. Config: {SLOT_CONFIG[slot]}",
        )
        print(f"saved -> {path}")

    return net, final, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", required=True, choices=["eat", "drink"])
    p.add_argument("--tag", default="v1")
    p.add_argument("--rounds", type=int, default=None,
                   help="default 60 for drink, 120 for eat (data-starved)")
    p.add_argument("--sim-len", type=int, default=7000)
    p.add_argument("--eval-len", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward", default="survival", choices=["survival", "comfort"],
                   help="'comfort' reproduces the v1 failure; see P2 in BUILDNOTES")
    p.add_argument("--death-penalty", type=float, default=20.0)
    p.add_argument("--cap-per-round", type=int, default=2500,
                   help="0 disables; see the buffer-flooding note in train()")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    rounds = a.rounds if a.rounds is not None else (120 if a.slot == "eat" else 60)
    train(slot=a.slot, tag=a.tag, rounds=rounds, sim_len=a.sim_len,
          eval_len=a.eval_len, seed=a.seed, reward=a.reward,
          death_penalty=a.death_penalty, cap_per_round=a.cap_per_round,
          save=not a.no_save)


if __name__ == "__main__":
    main()
