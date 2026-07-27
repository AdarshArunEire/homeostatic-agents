"""
train_explorer_v1.py  —  the thesis run.

    python training/train_explorer_v1.py --tag e0 --seed 0
    python training/train_explorer_v1.py --tag e1 --seed 1
    python training/train_explorer_v1.py --tag e2 --seed 2

> Does a learned, memory-carrying explorer beat reactive chemotaxis (0.48) and move toward
> the god ceiling (1.00), at smell 3 with the dead band intact?

Standing config, unchanged so the comparison holds: smell 3, band (9,11), nondoomed eval
with leeway 10, decay 0.7, h_fill 1.6 / s_fill 1.3, crit 0.7. The agent starts cold and must
find water then food; the dead band is L-2r = 3-5 hexes.

Ladder to beat / reach:  random 0.17  <  momentum 0.33  <  smell_momentum 0.48  <<  god 1.00

EVERYTHING LEARNED THE HARD WAY IS BUILT IN FROM THE FIRST RUN, not bolted on:

  best-checkpoint on FIXED eval seeds  — the pathfinder shipped a collapsed policy (arrival
                                        1.000 -> 0.734) because it saved final-round
                                        weights; seed 2 went from FAIL to a perfect gate
                                        purely by selecting episode 2500 instead of 4000
  fixed eval seeds                     — resampling the world between checkpoints measures
                                        the world, not the policy
  capped per-round contribution        — a poor explorer wanders and emits far more
                                        decisions than a competent one, so an uncapped
                                        buffer is biased toward failure BY CONSTRUCTION.
                                        This bias was wrongly claimed for the consumer; for
                                        the explorer it is real.
  3-seed protocol                      — 1 seed in 3 shipped a broken pathfinder. A single
                                        explorer run landing at 0.46 or 0.52 would be a
                                        verdict on luck.
  discovery counter                    — separates "memory does not help" from "the reward
                                        never arrived", which a solveScore alone cannot.

Read the verdict from `tests/explorer_sensitivity_v1.py` across all three seeds, never from
one training curve.
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
from model_modules.learned_modules.explorer_learned_v1 import (  # noqa: E402
    N_ACT, OBS_FIELDS, LearnedExplorer, default_arch,
)
from training.rigs.insim_rig_v1 import (  # noqa: E402
    count_discoveries, discovery_terminal, explorer_reward, harvest, rollout, use_module,
)
from training import rl_core_v1 as rl  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
# EIGHT eval seeds, not four.
#
# With four, best-checkpoint selection was picking the upper tail of a noisy estimate rather
# than a better policy: g0's round 70 scored 0.286 on these seeds and 0.1037 on held-out
# seeds 0-7, a 2.75x regression. Selecting the max of 8 noisy samples from a 4-seed mean is
# selection ON the noise. Doubling the seeds halves its variance and makes the selection
# defensible. Still disjoint from the sensitivity harness's seeds 0-7, so certification
# stays honest.
EVAL_SEEDS = (10_001, 10_002, 10_003, 10_004, 10_005, 10_006, 10_007, 10_008)
REACTIVE_FLOOR = 0.48
GOD_CEILING = 1.00


def _oracle_params(explorer_kwargs=None):
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": explorer_kwargs if explorer_kwargs is not None else {
            "persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
            "follow_p": 0.95, "reverse_on_drop_p": 0.75,
        },
    }


def _spec(impl="learned"):
    """Learned explorer, oracle everywhere else — the standing training protocol."""
    return ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer=impl)


def collect(module, seed, sim_len, eval_len, reward_fn, gamma, n_step):
    module.start_recording()
    with use_module(explorer=module):
        run = rollout(_spec(), _oracle_params({}), seed=seed,
                      sim_len=sim_len, eval_len=eval_len)
    module.stop_recording()
    # discovery is TERMINAL, matching the event the reward pays out on. Without this the
    # +10 transition bootstraps a value from after a travel-and-drink sequence in a
    # different region — see discovery_terminal() and P3.2.
    trans = harvest(run, module, "explorer", reward_fn,
                    n_act=N_ACT, gamma=gamma, n_step=n_step,
                    terminal_fn=discovery_terminal())
    return run, trans


def evaluate(net, sim_len, eval_len, seeds=EVAL_SEEDS):
    """Frozen greedy rollouts on the fixed eval seeds, pooled. solveScore is the verdict."""
    D = T = 0
    discoveries = 0
    calls = 0
    causes = {}

    for seed in seeds:
        module = LearnedExplorer(net=net, rng=np.random.default_rng(seed))
        with use_module(explorer=module):
            run = rollout(_spec(), _oracle_params({}), seed=seed,
                          sim_len=sim_len, eval_len=eval_len)
        D += int(run["death_count_eval"])
        T += int(run["n_timeouts_eval"])
        discoveries += count_discoveries(run)
        calls += module.n_calls
        eb = int(run["eval_boundary"])
        for e in run.get("death_events", []):
            if e["t"] >= eb:
                causes[e["cause"]] = causes.get(e["cause"], 0) + 1

    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "discoveries": discoveries,
        "explorer_calls": calls,
        "causes": causes,
    }


def train(
    tag: str = "e0",
    rounds: int = 80,
    sim_len: int = 7000,
    eval_len: int = 5000,
    seed: int = 0,
    n_hidden: int = 128,
    lr: float = 5e-4,
    gamma: float = 0.99,
    # 10-step returns, carried from the 03b headliner. The dead band is 3-5 hexes and an
    # explore run before a discovery is far longer, so credit has to travel further than
    # the 5 steps v1 used.
    n_step: int = 10,
    batch_size: int = 128,
    pos_frac: float = 0.25,
    buffer_size: int = 300_000,
    target_sync: int = 250,
    updates_per_round: int = 500,
    warmup: int = 2000,
    cap_per_round: int = 3000,
    discover_reward: float = 10.0,
    step_cost: float = 0.02,
    death_penalty: float = 10.0,
    eval_every: int = 10,
    save: bool = True,
):
    prepare_worker()
    rng = rl.seed_everything(seed)

    arch = default_arch(n_hidden=n_hidden, kind="noisy")
    net, target, optimiser = make_trainable(arch, lr)
    reward_fn = explorer_reward(discover_reward=discover_reward, step_cost=step_cost,
                                death_penalty=death_penalty)

    # TWO BUFFERS, stratified batches.
    #
    # Uniform replay cannot train this task. Discovery is ~0.17% of transitions, so a
    # uniform 128-batch drawn from a 231k buffer contains 0.06 rewarded samples in
    # expectation — the optimiser almost never sees the event it is supposed to learn.
    # Splitting rewarded from unrewarded and drawing a fixed fraction of each guarantees
    # every gradient step contains the signal.
    #
    # This changes SAMPLING, not the task, the reward or the observation. The agent still
    # has to find resources through a 3-5 hex scentless band from a cold start with smell 3,
    # and nothing about what it can perceive has changed. Reporting it plainly because
    # "we fixed it by sampling" is a claim that must be auditable.
    buf_pos = rl.ReplayBuffer(buffer_size // 4, rng=rng)
    buf_neg = rl.ReplayBuffer(buffer_size, rng=rng)

    def sample_batch(n):
        n_p = min(len(buf_pos), int(round(n * pos_frac)))
        n_n = n - n_p
        out = []
        if n_p:
            out += buf_pos.sample(n_p)
        if n_n and len(buf_neg):
            out += buf_neg.sample(n_n)
        return out

    best = {"score": None, "state": None, "metrics": None, "round": None}

    def _maybe_keep(m, rd):
        score = (m["solveScore"] if np.isfinite(m["solveScore"]) else -1, -m["eval_deaths"])
        if best["score"] is None or score > best["score"]:
            best.update(score=score, state=copy.deepcopy(net.state_dict()),
                        metrics=dict(m), round=rd)
            return True
        return False

    history = []
    updates = 0
    t0 = time.time()

    for rd in range(rounds):
        # NoisyNet carries exploration internally; no epsilon schedule. 03b convicted it as
        # the winning mechanism, and injecting undirected epsilon noise into the one module
        # being trained to be DIRECTED would work against the thing under test.
        module = LearnedExplorer(net=net, rng=rng)
        run, trans = collect(module, seed=seed + rd, sim_len=sim_len, eval_len=eval_len,
                             reward_fn=reward_fn, gamma=gamma, n_step=n_step)
        train_disc = count_discoveries(run)

        # STRATIFIED CAP — never subsample the signal away.
        #
        # v1 capped uniformly and that was self-defeating. Discovery is ~0.17% of
        # transitions (~25 rewarded out of ~15,000 explorer calls), so a uniform cut to
        # 3,000 kept only ~5 rewarded transitions per round. With a 231k buffer holding a
        # few hundred positives, a 128-sample batch contained 0.06 rewarded transitions in
        # expectation: nearly every gradient step saw step cost and nothing else.
        #
        # The cap exists to stop a chatty bad policy dominating the buffer, and keeping all
        # positives does not undermine that — they are rare by construction. Only the
        # zero-reward majority is thinned.
        pos = [t for t in trans if t[2] > 0.0]
        neg = [t for t in trans if t[2] <= 0.0]
        room = max(0, cap_per_round - len(pos)) if cap_per_round else len(neg)
        if cap_per_round and len(neg) > room:
            pick = rng.choice(len(neg), size=room, replace=False)
            neg = [neg[int(i)] for i in pick]
        n_pos = len(pos)
        n_kept = n_pos + len(neg)
        buf_pos.extend(pos)
        buf_neg.extend(neg)
        total_buf = len(buf_pos) + len(buf_neg)

        if total_buf >= warmup and len(buf_pos) > 0:
            for _ in range(updates_per_round):
                batch = sample_batch(batch_size)
                if not batch:
                    break
                rl.learn_step(net, target, optimiser, batch, gamma, double=True)
                if hasattr(net, "reset_noise"):
                    net.reset_noise()
                updates += 1
                if updates % target_sync == 0:
                    rl.sync_target(target, net)

        if eval_every and (rd + 1) % eval_every == 0:
            net.eval()
            m = evaluate(net, sim_len=sim_len, eval_len=eval_len)
            net.train()
            m.update({"round": rd + 1, "buffer": total_buf, "new_trans": n_kept,
                      "new_pos": n_pos, "buf_pos": len(buf_pos),
                      "train_discoveries": train_disc})
            history.append(m)
            kept = _maybe_keep(m, rd + 1)
            # discoveries per life, NOT raw: death clears memory, so a worse explorer
            # rediscovers more. On the calibration ladder random logged 136 discoveries to
            # smell_momentum's 97 while dying 121 times to 35 — raw count ranks them
            # backwards, per-life ranks them correctly (1.12 vs 2.77).
            m["disc_per_life"] = m["discoveries"] / max(1, m["eval_deaths"])
            print(
                f"rd {rd+1:>4}  solve {m['solveScore']:.3f}  deaths {m['eval_deaths']:>4}  "
                f"disc/life {m['disc_per_life']:>5.2f}  trainDisc {train_disc:>4}  "
                f"newPos {n_pos:>4}  bufPos {len(buf_pos):>6}/{total_buf:<7} "
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
          f"disc {final['discoveries']}  (round {best['round']} of {rounds})")
    print(f"      causes: {final['causes']}")
    print(f"      reactive floor {REACTIVE_FLOOR:.2f} | god ceiling {GOD_CEILING:.2f} "
          f"-> {'ABOVE floor' if final['solveScore'] > REACTIVE_FLOOR else 'below floor'}")

    # The diagnostic that separates the two ways of failing.
    trend = [h["train_discoveries"] for h in history]
    if trend and max(trend) - min(trend) < 0.1 * max(1, np.mean(trend)):
        print("  NOTE: training discoveries are flat across rounds. A null result here is "
              "'the reward never arrived', NOT 'memory does not help' — the same "
              "distinction Proto 04's Go-Explore probe was built to make.")

    solves = [h["solveScore"] for h in history if np.isfinite(h["solveScore"])]
    if len(solves) > 2 and (max(solves) - min(solves)) > 0.15:
        print(f"  WARNING: solveScore ranged {min(solves):.3f}-{max(solves):.3f} on a FIXED "
              f"eval set. Unstable; best-checkpoint masks this, it does not cure it.")

    train_config = {
        "rounds": rounds, "sim_len": sim_len, "eval_len": eval_len, "n_hidden": n_hidden,
        "lr": lr, "gamma": gamma, "n_step": n_step, "batch_size": batch_size,
        "buffer_size": buffer_size, "target_sync": target_sync, "pos_frac": pos_frac,
        "sampling": "stratified: all rewarded transitions kept, pos_frac per batch",
        "updates_per_round": updates_per_round, "warmup": warmup,
        "cap_per_round": cap_per_round, "discover_reward": discover_reward,
        "step_cost": step_cost, "death_penalty": death_penalty,
        "eval_seeds": list(EVAL_SEEDS), "smell_radius": 3, "band": [9, 11],
        "rig": "insim_rig_v1 (iterated batch off-policy)",
        "reward": "sparse on discovery (explore->GO_* transition), no smell shaping",
        "reactive_floor": REACTIVE_FLOOR, "god_ceiling": GOD_CEILING,
    }

    if save:
        path = save_module(
            net, kind="explorer", tag=tag, arch=arch, obs_fields=OBS_FIELDS,
            metrics={"final": final, "last": last, "best_round": best["round"],
                     "history": history},
            train_config=train_config, train_seed=seed,
            notes="Thesis run. Smell 3, dead band intact, cold start, both resources.",
        )
        print(f"saved -> {path}")

    return net, final, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="e0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--sim-len", type=int, default=7000)
    p.add_argument("--eval-len", type=int, default=5000)
    p.add_argument("--discover-reward", type=float, default=10.0)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    train(tag=a.tag, seed=a.seed, rounds=a.rounds, sim_len=a.sim_len,
          eval_len=a.eval_len, discover_reward=a.discover_reward, save=not a.no_save)


if __name__ == "__main__":
    main()
