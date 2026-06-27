import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --- local imports ---
import hex_world_procedural_senses as hex_world

from sweep_fn_4 import (
    build_config,
    make_sweep_configs,
    sweep_parallel,
)

from sim_instance_v1 import sim_instance

def main():

    #import drqn as dqn
    #print(hasattr(dqn, "noisy_drqn_make_target"))  
    #print(hasattr(dqn, "noisy_drqn_make_model"))
    #print(hasattr(dqn, "noisy_drqn_DQN"))

    PROTOTYPE_NAME = "04_generalisation"
    EXPERIMENT_NAME = PROTOTYPE_NAME + "__" + "DRQN_vs_FF_matched_n10_v2"

    SEEDS = range(40)
    BASE_ENV = dict(radius=20)
    BASE_SIM = dict(sim_len=500_000, eval_len=20_000)

    # shared across BOTH arms — the controlled constants
    BASE_AGENT = dict(
        beta=0.1,
        novelty_rewards=True,
        epsilon_start=0.3,
        sigma_0=0.5,
        n_step=10,
        replay_warmup=200,
        batch_size=64,
        drqn_burn_in=5,
        drqn_learn_len=20,
        learn_every=20,
        update_ticks=500,
        over_w=0.02,
        under_w=0.5,
        senses=("smell", "vision"),
        midpoint_probe=True,
    )

    # --- ARM 1: DRQN (buffer in EPISODES) ---
    grid_drqn = {
        "agent.model_type":         ["noisy_drqn_DQN"],
        "agent.n_step":             [10],
        "agent.batch_size":         [64],
        "agent.replay_archive_len": [1300],   # episodes ≈ 50k transitions
    }

    # --- ARM 2: feedforward control (buffer in TRANSITIONS) ---
    grid_ff = {
        "agent.model_type":         ["noisy_DQN"],
        "agent.n_step":             [10],
        "agent.batch_size":         [64],
        "agent.replay_archive_len": [50000],  # transitions
    }

    configs = []
    for grid in (grid_drqn, grid_ff):
        configs += make_sweep_configs(
            base_env=BASE_ENV, base_sim=BASE_SIM,
            base_agent=BASE_AGENT, sweep_grid=grid,
        )

    print("config names:")
    for cfg in configs:
        print(" ", cfg["name"], cfg["overrides"])

    sweep_parallel(
        experiment_name=EXPERIMENT_NAME,
        configs=configs,
        seeds=SEEDS,
        train_fn=sim_instance,
        resume=True,
        PROTOTYPE_NAME=PROTOTYPE_NAME,
        max_workers=4,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()