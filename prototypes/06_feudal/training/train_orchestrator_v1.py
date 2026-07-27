"""
train_orchestrator_v1.py  —  learned arbitration between two competing drives.

    python training/train_orchestrator_v1.py --tag o0 --seed 0
    python tests/orchestrator_sensitivity_v1.py --tags o0

The module that carries the homeostatic content: deciding *when* to abandon a hunt for the
missing resource and detour to a known one, and when to stop consuming and move on.

CALIBRATED FIRST (P4.0), and it passed decisively — unlike every other module in this
prototype:

    solveScore      monotone on the epsilon axis, dynamic range 0.478
    p05_satiation   monotone, dynamic range 0.940
    (explorer 0.362, consumer 0.044)

The arbitration axis also shows a clean two-sided structure with an interior optimum: at
crit_scale 0 the interrupt never fires and hydration deaths rise 29 -> 45; at 3.0 it fires
constantly and water_visit_pct hits 80.9% as the agent camps on the known drive.

WHY THIS MODULE HAS A BETTER PRIOR THAN THE EXPLORER. Its observation (h, s, known flags, tile
levels) is continuous and directly informative about the decision it makes. There is no
perceptual aliasing: distinct situations produce distinct inputs. The explorer failed because
physically different cells collapsed onto identical observations, and that failure mode
structurally does not apply here.

TWO DESIGN DECISIONS CARRIED FROM EARLIER FAILURES:

  four-way head, exploration delegated   A nine-way head would have the orchestrator emit
                                         EXPLORE_k directly, doing the explorer's job too —
                                         confounding arbitration with a task already shown
                                         unlearnable. It learns WHEN to explore, not how.
  survival reward, not comfort           Comfort is flat across the tolerance band and the
                                         interesting decisions all happen inside it (P2).

Target: match or beat the oracle's 0.4776 / 35 deaths. Note the calibration hint that
crit_scale=1.5 reaches 0.5926 — the hand-tuned threshold may not be optimal, so there may be
headroom above the baseline rather than only parity to achieve.
"""

from __future__ import annotations

import argparse
import copy
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
from model_modules.learned_modules.orchestrator_learned_v1 import (  # noqa: E402
    N_ACT, OBS_FIELDS, LearnedOrchestrator, default_arch,
)
from training.rigs.insim_rig_v1 import (  # noqa: E402
    harvest, orchestrator_reward, rollout, use_module,
)
from training import rl_core_v1 as rl  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
EVAL_SEEDS = (10_001, 10_002, 10_003, 10_004, 10_005, 10_006, 10_007, 10_008)
ORACLE_BASELINE = 0.4776
ORACLE_DEATHS = 35

EXPLORER_KW = {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
               "follow_p": 0.95, "reverse_on_drop_p": 0.75}


def _oracle_params(orch_kw=None):
    return {
        "orchestrator": orch_kw if orch_kw is not None else {},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": dict(EXPLORER_KW),
    }


def _spec():
    """Learned orchestrator, oracle everywhere else — the standing training protocol."""
    return ModuleSpec("learned", "oracle", "oracle", "oracle", explorer="smell_momentum")


def collect(module, seed, sim_len, eval_len, reward_fn, gamma, n_step):
    module.start_recording()
    with use_module(orchestrator=module):
        run = rollout(_spec(), _oracle_params(), seed=seed,
                      sim_len=sim_len, eval_len=eval_len)
    module.stop_recording()
    # dispatched every tick, so the join is one-to-one and align_records asserts it
    trans = harvest(run, module, "orchestrator", reward_fn,
                    n_act=N_ACT, gamma=gamma, n_step=n_step)
    return run, trans


def evaluate(net, sim_len, eval_len, seeds=EVAL_SEEDS):
    D = T = 0
    causes = {}
    action_mix = np.zeros(N_ACT, dtype=float)
    for seed in seeds:
        module = LearnedOrchestrator(net=net, rng=np.random.default_rng(seed))
        module.start_recording()
        with use_module(orchestrator=module):
            run = rollout(_spec(), _oracle_params(), seed=seed,
                          sim_len=sim_len, eval_len=eval_len)
        module.stop_recording()
        D += int(run["death_count_eval"])
        T += int(run["n_timeouts_eval"])
        for _, a in module.records:
            action_mix[a] += 1
        eb = int(run["eval_boundary"])
        for e in run.get("death_events", []):
            if e["t"] >= eb:
                causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    mix = action_mix / max(1.0, action_mix.sum())
    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "causes": causes,
        "mix_water": float(mix[0]), "mix_food": float(mix[1]),
        "mix_consume": float(mix[2]), "mix_explore": float(mix[3]),
    }


def train(
    tag: str = "o0",
    rounds: int = 60,
    sim_len: int = 7000,
    eval_len: int = 5000,
    seed: int = 0,
    n_hidden: int = 64,
    lr: float = 5e-4,
    gamma: float = 0.99,
    n_step: int = 10,
    batch_size: int = 256,
    buffer_size: int = 300_000,
    target_sync: int = 250,
    updates_per_round: int = 500,
    warmup: int = 5000,
    cap_per_round: int = 7000,
    eps_start: float = 0.5,
    eps_end: float = 0.02,
    eps_frac: float = 0.5,
    step_reward: float = 0.01,
    death_penalty: float = 1.0,
    eval_every: int = 10,
    save: bool = True,
):
    prepare_worker()
    rng = rl.seed_everything(seed)

    arch = default_arch(n_hidden=n_hidden, kind="mlp")
    net, target, optimiser = make_trainable(arch, lr)
    buffer = rl.ReplayBuffer(buffer_size, rng=rng)
    reward_fn = orchestrator_reward(step_reward=step_reward, death_penalty=death_penalty)

    best = {"score": None, "state": None, "metrics": None, "round": None}

    def _maybe_keep(m, rd):
        s = (m["solveScore"] if np.isfinite(m["solveScore"]) else -1, -m["eval_deaths"])
        if best["score"] is None or s > best["score"]:
            best.update(score=s, state=copy.deepcopy(net.state_dict()),
                        metrics=dict(m), round=rd)
            return True
        return False

    history = []
    updates = 0
    t0 = time.time()

    for rd in range(rounds):
        frac = min(1.0, rd / max(1, int(eps_frac * rounds)))
        epsilon = eps_start + frac * (eps_end - eps_start)

        module = LearnedOrchestrator(net=net, rng=rng, explore_eps=epsilon)
        _, trans = collect(module, seed=seed + rd, sim_len=sim_len, eval_len=eval_len,
                           reward_fn=reward_fn, gamma=gamma, n_step=n_step)

        # every tick is a transition, so a round contributes ~sim_len samples regardless of
        # policy quality — the call-frequency bias that mattered for the explorer does not
        # apply here, and the cap is a memory guard rather than a debiasing measure
        if cap_per_round and len(trans) > cap_per_round:
            pick = rng.choice(len(trans), size=cap_per_round, replace=False)
            trans = [trans[int(i)] for i in pick]
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
            m = evaluate(net, sim_len=sim_len, eval_len=eval_len)
            net.train()
            m.update({"round": rd + 1, "buffer": len(buffer), "epsilon": round(epsilon, 3)})
            history.append(m)
            kept = _maybe_keep(m, rd + 1)
            print(
                f"rd {rd+1:>4}  solve {m['solveScore']:.3f}  deaths {m['eval_deaths']:>4}  "
                f"mix W/F/C/E {m['mix_water']:.2f}/{m['mix_food']:.2f}/"
                f"{m['mix_consume']:.2f}/{m['mix_explore']:.2f}  "
                f"eps {epsilon:.2f}  buf {len(buffer):>7}  "
                f"{time.time()-t0:.0f}s{'  <- best' if kept else ''}"
            )

    net.eval()
    last = evaluate(net, sim_len=sim_len, eval_len=eval_len)
    _maybe_keep(last, rounds)
    if best["state"] is not None and best["round"] != rounds:
        net.load_state_dict(best["state"])
        net.eval()
    final = best["metrics"] or last

    print(f"\nlast: solve {last['solveScore']:.3f}  deaths {last['eval_deaths']}")
    print(f"best: solve {final['solveScore']:.3f}  deaths {final['eval_deaths']}  "
          f"(round {best['round']} of {rounds})")
    print(f"      causes: {final['causes']}")
    print(f"      action mix  GO_WATER {final['mix_water']:.3f}  GO_FOOD {final['mix_food']:.3f}"
          f"  CONSUME {final['mix_consume']:.3f}  EXPLORE {final['mix_explore']:.3f}")
    verdict = ("ABOVE oracle" if final["solveScore"] > ORACLE_BASELINE
               else "below oracle")
    print(f"      oracle {ORACLE_BASELINE:.4f} ({ORACLE_DEATHS} deaths) -> {verdict}")

    if final["mix_explore"] > 0.9:
        print("  NOTE: >90% EXPLORE. The policy has collapsed to a single action — it never "
              "learned to arbitrate, it learned to always delegate.")
    elif final["mix_consume"] > 0.5:
        print("  NOTE: >50% CONSUME. Likely camping on a resource tile rather than commuting.")

    solves = [h["solveScore"] for h in history if np.isfinite(h["solveScore"])]
    if len(solves) > 2 and (max(solves) - min(solves)) > 0.15:
        print(f"  WARNING: solveScore ranged {min(solves):.3f}-{max(solves):.3f} on a FIXED "
              f"eval set. Unstable; best-checkpoint masks this, it does not cure it.")

    train_config = {
        "rounds": rounds, "sim_len": sim_len, "eval_len": eval_len, "n_hidden": n_hidden,
        "lr": lr, "gamma": gamma, "n_step": n_step, "batch_size": batch_size,
        "buffer_size": buffer_size, "target_sync": target_sync,
        "updates_per_round": updates_per_round, "warmup": warmup,
        "cap_per_round": cap_per_round, "eps_start": eps_start, "eps_end": eps_end,
        "step_reward": step_reward, "death_penalty": death_penalty,
        "eval_seeds": list(EVAL_SEEDS), "n_abstract_actions": N_ACT,
        "rig": "insim_rig_v1 (iterated batch off-policy)",
        "reward": f"survival: +{step_reward}/tick alive, -{death_penalty} on death",
        "note": "4-way abstract head; exploration delegated to the explorer module",
        "oracle_baseline": ORACLE_BASELINE,
    }

    if save:
        path = save_module(net, kind="orchestrator", tag=tag, arch=arch,
                           obs_fields=OBS_FIELDS,
                           metrics={"final": final, "last": last,
                                    "best_round": best["round"], "history": history},
                           train_config=train_config, train_seed=seed,
                           notes="Learned arbitration. Exploration delegated, not learned.")
        print(f"saved -> {path}")

    return net, final, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="o0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--sim-len", type=int, default=7000)
    p.add_argument("--eval-len", type=int, default=5000)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    train(tag=a.tag, seed=a.seed, rounds=a.rounds, sim_len=a.sim_len,
          eval_len=a.eval_len, death_penalty=a.death_penalty, save=not a.no_save)


if __name__ == "__main__":
    main()
