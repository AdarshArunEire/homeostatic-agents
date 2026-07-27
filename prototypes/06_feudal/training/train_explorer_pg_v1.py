"""
train_explorer_pg_v1.py  —  policy gradient explorer (REINFORCE with a value baseline).

    python training/train_explorer_pg_v1.py --tag p0 --seed 0
    python tests/explorer_sensitivity_v1.py --tags p0 --policy

WHY THIS METHOD, not a preference for it. Three runs of DQN converged on ~0.10 held-out,
below the random floor of 0.1712, and the diagnosis is complete:

  1. The explorer observation carries NO positional memory — legality, last 3 actions, 3
     smell readings, need flags. Coverage, revisit-avoidance and frontier-seeking are not
     representable. The best policy in this class is REACTIVE.
  2. The best reactive policy here is a correlated random walk with chemotaxis. That is
     `smell_momentum`, the 0.4776 baseline, and optimal foraging theory says that family is
     the right one for sparse targets.
  3. That family is defined by PROBABILITIES (persist_p 0.75, follow_p 0.95,
     reverse_on_drop_p 0.75). A correlated random walk without the randomness is a straight
     line or a loop.
  4. DQN returns a greedy deterministic map from an aliased observation. It cannot express
     (3). The temperature probe confirmed it: performance rose monotonically toward the
     UNIFORM limit and asymptoted at random — the best available use of the Q-function was
     to ignore it.

A policy network learns "persist with probability p" as a parameter. This is the only method
tried so far whose output class contains the target policy.

Two structural advantages beyond that, both relevant to failures already recorded:

  no bootstrap    Monte-Carlo returns have no `next_state` value to be wrong about, so the
                  P3.2 corruption (discovery bootstrapping into a post-travel state) cannot
                  occur here even in principle.
  entropy bonus   directly rewards keeping the policy stochastic, which is the property the
                  task requires and which argmax destroyed.

The known weakness is variance, and at 0.17% reward density that is exactly where it bites.
Mitigations: value baseline, reward normalisation, and more rollouts per update than the
DQN loop used.
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model_modules.checkpoint_v1 import prepare_worker, save_module  # noqa: E402
from model_modules.contract_v1 import ModuleSpec  # noqa: E402
from model_modules.learned_modules.explorer_learned_v1 import (  # noqa: E402
    N_ACT, N_INPUT, OBS_FIELDS, LearnedExplorer,
)
from model_modules.learned_modules.policy_nets_v1 import (  # noqa: E402
    PolicyNet, discounted_returns,
)
from training.rigs.insim_rig_v1 import (  # noqa: E402
    align_records, count_discoveries, discovery_terminal, explorer_reward, rollout,
    use_module,
)
from training import rl_core_v1 as rl  # noqa: E402

H_FILL, S_FILL = 1.6, 1.3
EVAL_SEEDS = (10_001, 10_002, 10_003, 10_004, 10_005, 10_006, 10_007, 10_008)
REACTIVE_FLOOR = 0.4776
RANDOM_FLOOR = 0.1712


def _oracle_params():
    return {
        "orchestrator": {"h_fill": H_FILL, "s_fill": S_FILL, "h_crit": 0.7, "s_crit": 0.7},
        "pathfinder": {},
        "eat": {"fill_target": S_FILL},
        "drink": {"fill_target": H_FILL},
        "explorer": {},
    }


def _spec():
    return ModuleSpec("oracle", "oracle", "oracle", "oracle", explorer="learned")


def collect_episode(net, seed, sim_len, eval_len, reward_fn, terminal_fn, gamma):
    """
    One rollout -> (obs, actions, legal_masks, returns).

    Returns are Monte-Carlo and reset at terminals, so a discovery ends the credit chain
    exactly where the sub-task ends.
    """
    module = LearnedExplorer(net=net, rng=np.random.default_rng(seed), policy_mode=True)
    module.start_recording()
    with use_module(explorer=module):
        run = rollout(_spec(), _oracle_params(), seed=seed,
                      sim_len=sim_len, eval_len=eval_len)
    module.stop_recording()

    ticks = align_records(run, module.records, "explorer")
    obs = np.stack([r[0] for r in module.records])
    acts = np.array([r[1] for r in module.records], dtype=np.int64)
    # legality is the first N_ACT entries of the encoding — recovered rather than re-derived
    legal = obs[:, :N_ACT] > 0.5

    rewards, dones = [], []
    for j in range(len(ticks)):
        t = int(ticks[j])
        t_next = int(ticks[j + 1]) if j + 1 < len(ticks) else t + 1
        rewards.append(reward_fn(run, t, t_next))
        dones.append(terminal_fn(run, t, t_next))
    dones[-1] = True  # truncation at rollout end closes the last return

    rets = discounted_returns(rewards, dones, gamma)
    return run, obs, acts, legal, rets, count_discoveries(run)


def evaluate(net, sim_len, eval_len, seeds=EVAL_SEEDS):
    D = T = 0
    disc = 0
    for seed in seeds:
        module = LearnedExplorer(net=net, rng=np.random.default_rng(seed),
                                 policy_mode=True)
        with use_module(explorer=module):
            run = rollout(_spec(), _oracle_params(), seed=seed,
                          sim_len=sim_len, eval_len=eval_len)
        D += int(run["death_count_eval"])
        T += int(run["n_timeouts_eval"])
        disc += count_discoveries(run)
    return {
        "eval_deaths": int(D),
        "solveScore": (T / (T + D)) if (T + D) > 0 else float("nan"),
        "discoveries": disc,
    }


def train(
    tag: str = "p0",
    rounds: int = 80,
    episodes_per_round: int = 3,
    sim_len: int = 7000,
    eval_len: int = 5000,
    seed: int = 0,
    n_hidden: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_beta: float = 0.02,
    value_coef: float = 0.5,
    epochs_per_round: int = 4,
    minibatch: int = 512,
    max_grad_norm: float = 0.5,
    discover_reward: float = 10.0,
    step_cost: float = 0.02,
    death_penalty: float = 10.0,
    eval_every: int = 10,
    save: bool = True,
):
    prepare_worker()
    rng = rl.seed_everything(seed)

    arch = {"kind": "policy", "n_input": N_INPUT, "n_hidden": n_hidden, "n_act": N_ACT}
    net = PolicyNet(N_INPUT, n_hidden, N_ACT)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    reward_fn = explorer_reward(discover_reward=discover_reward, step_cost=step_cost,
                                death_penalty=death_penalty)
    terminal_fn = discovery_terminal()

    best = {"score": None, "state": None, "metrics": None, "round": None}

    def _maybe_keep(m, rd):
        s = (m["solveScore"] if np.isfinite(m["solveScore"]) else -1, -m["eval_deaths"])
        if best["score"] is None or s > best["score"]:
            best.update(score=s, state=copy.deepcopy(net.state_dict()),
                        metrics=dict(m), round=rd)
            return True
        return False

    history = []
    t0 = time.time()

    for rd in range(rounds):
        O, A, L, R = [], [], [], []
        train_disc = 0
        for e in range(episodes_per_round):
            _, o, a, lg, ret, d = collect_episode(
                net, seed=seed + rd * episodes_per_round + e, sim_len=sim_len,
                eval_len=eval_len, reward_fn=reward_fn, terminal_fn=terminal_fn,
                gamma=gamma)
            O.append(o); A.append(a); L.append(lg); R.append(ret)
            train_disc += d

        obs = torch.as_tensor(np.concatenate(O), dtype=torch.float32)
        acts = torch.as_tensor(np.concatenate(A), dtype=torch.long)
        legal = torch.as_tensor(np.concatenate(L), dtype=torch.bool)
        rets = torch.as_tensor(np.concatenate(R), dtype=torch.float32)
        # normalise returns across the round: the 0.17% reward density makes the raw scale
        # jump by orders of magnitude between rounds, and the gradient inherits that
        rets = (rets - rets.mean()) / (rets.std() + 1e-6)

        n = len(obs)
        for _ in range(epochs_per_round):
            perm = torch.randperm(n)
            for i in range(0, n, minibatch):
                idx = perm[i:i + minibatch]
                logits, values = net(obs[idx])
                logits = logits.masked_fill(~legal[idx], -1e9)
                dist = torch.distributions.Categorical(logits=logits)

                adv = (rets[idx] - values).detach()
                pg_loss = -(dist.log_prob(acts[idx]) * adv).mean()
                v_loss = F.mse_loss(values, rets[idx])
                # entropy bonus keeps the policy stochastic — the property the task needs
                # and the one argmax destroyed
                ent = dist.entropy().mean()
                loss = pg_loss + value_coef * v_loss - entropy_beta * ent

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimiser.step()

        if eval_every and (rd + 1) % eval_every == 0:
            net.eval()
            m = evaluate(net, sim_len=sim_len, eval_len=eval_len)
            net.train()
            with torch.no_grad():
                lg = net.logits(obs[:2048]).masked_fill(~legal[:2048], -1e9)
                ent_now = float(torch.distributions.Categorical(logits=lg).entropy().mean())
            m.update({"round": rd + 1, "train_discoveries": train_disc,
                      "entropy": ent_now, "n_trans": n})
            history.append(m)
            kept = _maybe_keep(m, rd + 1)
            print(
                f"rd {rd+1:>4}  solve {m['solveScore']:.3f}  deaths {m['eval_deaths']:>4}  "
                f"disc {m['discoveries']:>5}  trainDisc {train_disc:>4}  "
                f"H {ent_now:.3f}/{np.log(N_ACT):.3f}  n {n:>6}  "
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
    print(f"      random {RANDOM_FLOOR:.4f} | reactive floor {REACTIVE_FLOOR:.4f} -> "
          f"{'ABOVE floor' if final['solveScore'] > REACTIVE_FLOOR else ('above random' if final['solveScore'] > RANDOM_FLOOR else 'below random')}")

    ents = [h["entropy"] for h in history]
    if ents and ents[-1] < 0.15 * np.log(N_ACT):
        print(f"  NOTE: entropy collapsed to {ents[-1]:.3f} (max {np.log(N_ACT):.3f}). The "
              f"policy has gone near-deterministic, which is the DQN failure reappearing. "
              f"Raise --entropy-beta.")

    train_config = {
        "rounds": rounds, "episodes_per_round": episodes_per_round, "sim_len": sim_len,
        "eval_len": eval_len, "n_hidden": n_hidden, "lr": lr, "gamma": gamma,
        "entropy_beta": entropy_beta, "value_coef": value_coef,
        "epochs_per_round": epochs_per_round, "minibatch": minibatch,
        "discover_reward": discover_reward, "step_cost": step_cost,
        "death_penalty": death_penalty, "eval_seeds": list(EVAL_SEEDS),
        "smell_radius": 3, "band": [9, 11], "policy_mode": True,
        "algo": "REINFORCE + value baseline + entropy bonus, MC returns (no bootstrap)",
        "reward": "sparse on discovery, terminal on discovery, no smell shaping",
    }

    if save:
        path = save_module(net, kind="explorer", tag=tag, arch=arch,
                           obs_fields=OBS_FIELDS,
                           metrics={"final": final, "last": last,
                                    "best_round": best["round"], "history": history},
                           train_config=train_config, train_seed=seed,
                           notes="Policy gradient. Load with policy_mode=True.")
        print(f"saved -> {path}")

    return net, final, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="p0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--episodes-per-round", type=int, default=3)
    p.add_argument("--entropy-beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    train(tag=a.tag, seed=a.seed, rounds=a.rounds,
          episodes_per_round=a.episodes_per_round, entropy_beta=a.entropy_beta,
          lr=a.lr, save=not a.no_save)


if __name__ == "__main__":
    main()
