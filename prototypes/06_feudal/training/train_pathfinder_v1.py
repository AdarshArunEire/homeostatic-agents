"""
train_pathfinder_v1.py  —  sparse-reward + HER pathfinder, trained out of sim.

    python training/train_pathfinder_v1.py --tag v1 --episodes 4000

Run from `prototypes/06_feudal/` (the same cwd sim_instance_v3 assumes), or from anywhere
— the bootstrap below puts the prototype dir on sys.path either way.

WHY HER RATHER THAN SHAPING. Arrival reward is sparse, so most early episodes return
nothing. Hindsight fixes that without touching the reward function: an episode that failed
to reach the goal it was given DID reach the tile it ended on, so relabelling the goal
turns every failure into a success for some other goal. The agent learns "how do I get to
an arbitrary displacement" from trajectories that never once succeeded at their actual
task. That is the honest version of making this problem easy — the difficulty is removed
by better use of the data, not by writing the answer into the reward.

The result this produces is a PLUMBING result, and should be labelled as one. The
interesting claim is not "a pathfinder can be learned"; it is that the same algorithm
class that solved 0/40 monolithically in Proto 04 solves this trivially once the
decomposition hands it an observed goal. Proto 04 is the control that gives the number
meaning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- bootstrap: make `import hex_world_cached` / `model_modules.*` work as a script ---
_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import numpy as np  # noqa: E402

from model_modules.checkpoint_v1 import prepare_worker, save_module  # noqa: E402
from model_modules.learned_modules.nets_v1 import make_trainable  # noqa: E402
from model_modules.learned_modules.pathfinder_learned_v1 import (  # noqa: E402
    OBS_FIELDS,
    default_arch,
    encode_to_goal,
)
from training.rigs.pathfinder_rig_v1 import PathfinderRig, oracle_step_count  # noqa: E402
from training import rl_core_v1 as rl  # noqa: E402


# --- episode recording ----------------------------------------------------------

def rollout(rig, net, rng, epsilon: float):
    """
    One episode against the rig's own goal. Records positions/actions/masks rather than
    finished transitions, because HER needs to rebuild transitions against other goals
    from the same trajectory.
    """
    rig.reset()
    positions = [rig.pos]
    masks = [rig.legal_mask()]
    actions = []
    arrived = False

    for _ in range(rig.step_cap):
        obs = rig.obs()
        mask = masks[-1]
        if rng.random() < epsilon:
            legal = np.flatnonzero(mask)
            a = int(legal[rng.integers(len(legal))])
        else:
            a = rl.select_action(net, obs, mask)

        _, _, done, info = rig.step(a)

        actions.append(a)
        positions.append(rig.pos)
        masks.append(rig.legal_mask())

        if info["arrived"]:
            arrived = True
        if done:
            break

    return positions, actions, masks, arrived


def transitions_for_goal(rig, positions, actions, masks, goal, acc):
    """
    Rebuild n-step transitions for an arbitrary goal from a recorded trajectory.

    Terminates the sequence at the first arrival: everything after the agent reaches
    `goal` is off-task for that goal and would pollute the buffer with post-success
    wandering.
    """
    out = []
    g = tuple(goal)
    acc.clear()

    for i, a in enumerate(actions):
        p, p_next = positions[i], positions[i + 1]
        hit = p_next == g
        reward = rig.step_cost + (rig.arrive_reward if hit else 0.0)
        out.extend(
            acc.push(
                encode_to_goal((g[0] - p[0], g[1] - p[1]), scale=rig.radius),
                a,
                reward,
                encode_to_goal((g[0] - p_next[0], g[1] - p_next[1]), scale=rig.radius),
                hit,
                masks[i + 1],
            )
        )
        if hit:
            break
    else:
        # ran out of clock rather than arriving — truncation, not terminal
        out.extend(acc.flush())

    return out


def her_goals(positions, i, k, rng):
    """`future` strategy: sample k achieved states from after step i."""
    hi = len(positions) - 1
    if hi <= i:
        return []
    js = rng.integers(i + 1, hi + 1, size=k)
    return [positions[int(j)] for j in js]


# --- evaluation -----------------------------------------------------------------

def make_eval_set(rig, n_pairs: int, seed: int = 12345):
    """
    A FIXED set of (start, goal) pairs, drawn once from its own generator.

    Resampling the eval set at every checkpoint would put sampling noise on top of
    learning progress and make the curve unreadable — the same reason 03b moved to
    per-seed solve-rate rather than aggregates. Held separate from the training rng so
    changing training hyperparameters does not silently change the test set.
    """
    ev_rng = np.random.default_rng(seed)
    saved = rig.rng
    rig.rng = ev_rng
    pairs = []
    for _ in range(n_pairs):
        rig.reset()
        pairs.append((rig.pos, rig.goal))
    rig.rng = saved
    return pairs


def evaluate(rig, net, eval_set):
    """
    Greedy eval on the fixed pair set.

    Scores against the OPTIMUM, not against 'did it arrive'. A pathfinder that arrives by
    a scenic route still costs path_efficiency in the composed agent, which is the metric
    Proto 04 used to separate DRQN's scrappy crossings from FF's clean ones.
    """
    arrived = 0
    effs = []
    for start, goal in eval_set:
        rig.reset(pos=start, goal=goal)
        opt = oracle_step_count(rig, start, goal)
        info = {"arrived": False, "t": rig.step_cap}
        for _ in range(rig.step_cap):
            a = rl.select_action(net, rig.obs(), rig.legal_mask())
            _, _, done, info = rig.step(a)
            if done:
                break
        if info["arrived"]:
            arrived += 1
            effs.append(opt / max(info["t"], 1))
    return {
        "arrival_rate": arrived / len(eval_set),
        "path_efficiency": float(np.mean(effs)) if effs else 0.0,
    }


# --- train ----------------------------------------------------------------------

def train(
    tag: str = "v1",
    radius: int = 20,
    episodes: int = 4000,
    seed: int = 0,
    n_hidden: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 3,
    batch_size: int = 128,
    buffer_size: int = 200_000,
    her_k: int = 4,
    target_sync: int = 500,
    updates_per_episode: int = 8,
    warmup: int = 2000,
    eps_start: float = 0.5,
    eps_end: float = 0.05,
    eps_frac: float = 0.4,
    eval_every: int = 500,
    eval_pairs: int = 200,
    step_cost: float = -0.01,
    arrive_reward: float = 1.0,
    save: bool = True,
):
    prepare_worker()
    rng = rl.seed_everything(seed)

    rig = PathfinderRig(
        radius=radius, rng=rng, step_cost=step_cost, arrive_reward=arrive_reward
    )
    arch = default_arch(n_hidden=n_hidden, kind="mlp")
    net, target, optimiser = make_trainable(arch, lr)

    buffer = rl.ReplayBuffer(buffer_size, rng=rng)
    acc = rl.NStepAccumulator(n_step=n_step, gamma=gamma)

    eval_set = make_eval_set(rig, eval_pairs)
    final_eval_set = make_eval_set(rig, max(eval_pairs, 500), seed=54321)

    history = []
    t0 = time.time()
    updates = 0

    # Best-checkpoint selection.
    #
    # Added after a 3-seed reproducibility check: seed 2 peaked at arrival 1.000 /
    # path_eff 1.000 around episode 2500-3000, then decayed to 0.734 by episode 4000 and
    # shipped the decayed weights. Its long-band optimality came out 0.7685 against seed
    # 1's 0.9954, with errors that were BACKWARDS rather than sideways (excess steps per
    # move 0.4167 vs 0.0046 — ~90x the real cost). Seed 0 landed in between.
    #
    # So the capability is there in every seed; what varies is where training happens to
    # stop. Selecting on the fixed eval set already being computed turns a coin flip into
    # the best policy actually observed.
    import copy as _copy

    best = {"score": None, "state": None, "metrics": None, "episode": None}

    def _maybe_keep(m, ep):
        score = (m["arrival_rate"], m["path_efficiency"])
        if best["score"] is None or score > best["score"]:
            best.update(score=score, state=_copy.deepcopy(net.state_dict()),
                        metrics=dict(m), episode=ep)
            return True
        return False

    for ep in range(episodes):
        frac = min(1.0, ep / max(1, int(eps_frac * episodes)))
        epsilon = eps_start + frac * (eps_end - eps_start)

        positions, actions, masks, _ = rollout(rig, net, rng, epsilon)

        # original goal
        buffer.extend(transitions_for_goal(rig, positions, actions, masks, rig.goal, acc))

        # hindsight goals
        for i in range(len(actions)):
            for g in her_goals(positions, i, her_k, rng):
                if g == positions[i]:
                    continue  # zero displacement: already satisfied, teaches nothing
                buffer.extend(
                    transitions_for_goal(
                        rig, positions[i:], actions[i:], masks[i:], g, acc
                    )
                )

        if len(buffer) >= warmup:
            for _ in range(updates_per_episode):
                loss = rl.learn_step(
                    net, target, optimiser, buffer.sample(batch_size), gamma, double=True
                )
                updates += 1
                if updates % target_sync == 0:
                    rl.sync_target(target, net)

        if eval_every and (ep + 1) % eval_every == 0:
            net.eval()
            m = evaluate(rig, net, eval_set)
            net.train()
            m.update({"episode": ep + 1, "buffer": len(buffer), "epsilon": round(epsilon, 3)})
            history.append(m)
            kept = _maybe_keep(m, ep + 1)
            print(
                f"ep {ep+1:>6}  arrival {m['arrival_rate']:.3f}  "
                f"path_eff {m['path_efficiency']:.3f}  buf {len(buffer):>7}  "
                f"eps {epsilon:.2f}  {time.time()-t0:.0f}s{'  <- best' if kept else ''}"
            )

    net.eval()
    last = evaluate(rig, net, final_eval_set)
    _maybe_keep(evaluate(rig, net, eval_set), episodes)
    print(f"\nlast: arrival {last['arrival_rate']:.3f}  path_eff {last['path_efficiency']:.3f}")

    # ship the best observed policy, not the most recent one
    if best["state"] is not None and best["episode"] != episodes:
        net.load_state_dict(best["state"])
        net.eval()
    final = evaluate(rig, net, final_eval_set)
    print(f"best: arrival {final['arrival_rate']:.3f}  path_eff {final['path_efficiency']:.3f}"
          f"   (episode {best['episode']} of {episodes})")

    arrivals = [h["arrival_rate"] for h in history]
    if len(arrivals) > 2:
        spread = max(arrivals) - min(arrivals)
        if spread > 0.15:
            print(f"  WARNING: arrival ranged {min(arrivals):.3f}-{max(arrivals):.3f} "
                  f"(spread {spread:.3f}) on a FIXED eval set. Training is unstable; "
                  f"best-checkpoint selection is masking that, not curing it. A 3-seed "
                  f"check found one seed in three collapsing late (arrival 1.000 -> 0.734).")

    train_config = {
        "radius": radius, "episodes": episodes, "n_hidden": n_hidden, "lr": lr,
        "gamma": gamma, "n_step": n_step, "batch_size": batch_size,
        "buffer_size": buffer_size, "her_k": her_k, "target_sync": target_sync,
        "updates_per_episode": updates_per_episode, "warmup": warmup,
        "eps_start": eps_start, "eps_end": eps_end, "eps_frac": eps_frac,
        "step_cost": step_cost, "arrive_reward": arrive_reward,
        # part of the ENCODING, not a hyperparameter: inference must reuse it or the
        # net sees differently-normalised inputs than it trained on
        "scale": radius,
        "rig": "pathfinder_rig_v1",
        "reward": "sparse arrival + step cost, no shaping",
    }

    if save:
        path = save_module(
            net,
            kind="pathfinder",
            tag=tag,
            arch=arch,
            obs_fields=OBS_FIELDS,
            metrics={"final": final, "last": last, "best_episode": best["episode"],
                     "history": history},
            train_config=train_config,
            train_seed=seed,
            notes="Standalone rig (no sim). Distillation-free: sparse reward + HER.",
        )
        print(f"saved -> {path}")

    return net, final, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v1")
    p.add_argument("--radius", type=int, default=20)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--her-k", type=int, default=4)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    train(
        tag=a.tag, radius=a.radius, episodes=a.episodes, seed=a.seed,
        n_hidden=a.n_hidden, her_k=a.her_k, save=not a.no_save,
    )


if __name__ == "__main__":
    main()
