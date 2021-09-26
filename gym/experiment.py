# Imports

## General
import sys
import argparse
import importlib
import time
import datetime
import random

## ML & RL
import numpy as np
import torch as th
import wandb

## Decision Transformer
from rl.mfrl import MFRL



def main(configs, seed):
    print('Start Decision Transformer experiment...')
    print('\n')
    env_name = configs['experiment']['env_name']
    data_type = configs['data']['data_type']


    group_name = f"gym-ammi-{env_name}-{data_type}"
    # exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"
    now = datetime.datetime.now()
    # exp_prefix = f"{group_name}-{now.year}/{now.month}/{now.day}-->{now.hour}:{now.minute}:{now.second}"
    exp_prefix = f"{group_name}-seed:{seed}"


    print('=' * 50)
    print(f'Starting new experiment')
    print(f"\t Environment: {env_name}")
    print(f"\t Data type: {data_type}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)

    configs['seed'] = seed

    if configs['experiment']['WandB']:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='dt-gym-ammi',
            # project='rand',
            config=configs
        )

    experiment = MFRL(configs, seed)

    agent = experiment.learn()

    # th.save(agent, f'./saved_agents/dt-agent-{env_name}.pth.tar')
    
    print('\n')
    print('...End Decision Transformer experiment')
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=int)
    # parser.add_argument('-cfg_path', type=str)

    args = parser.parse_args()

    sys.path.append("./config")
    config = importlib.import_module(args.cfg)
    seed = args.seed

    main(config.configurations, seed)
