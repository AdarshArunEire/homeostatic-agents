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
import hex_world_cached as hex_world

from sweep_fn_v5 import (
    build_config,
    make_sweep_configs,
    sweep_parallel,
)

from sim_instance_v2 import sim_instance

def main():

    #import drqn as dqn
    #print(hasattr(dqn, "noisy_drqn_make_target"))  
    #print(hasattr(dqn, "noisy_drqn_make_model"))
    #print(hasattr(dqn, "noisy_drqn_DQN"))

    PROTOTYPE_NAME = "05_generalisation"
    EXPERIMENT_NAME = PROTOTYPE_NAME + "__" + "self_imitation_v1"

    N = 24

    SEEDS = range(N)
    BASE_ENV = dict(radius=20)
    BASE_SIM = dict(sim_len=1000000, eval_len=20000)

    BASE_AGENT = dict(
        beta=0.1, novelty_rewards=True, epsilon_start=0.3, sigma_0=0.5,
        n_step=10, replay_warmup=500, batch_size=512, learn_every=20,
        update_ticks=500, over_w=0.02, under_w=0.5,
        model_type="noisy_DQN", replay_archive_len=50000,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2,
        curriculum_ramp_frac=0.6, life_cap=1000, terminals_per_map=1,
        senses=("smell", "vision"), ez_zeta=2
    )

    grid = {
        "agent.sil_frac": [0.00, 0.25, 0.50],
    }   


    configs = []
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

import subprocess

if __name__ == "__main__":
    mp.freeze_support()
    main()
    subprocess.run(["shutdown", "/s", "/t", "120"])
