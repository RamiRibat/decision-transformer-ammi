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

def main(configs):
    print('Start Decision Transformer experiment...')
    print('\n')
    env_name = configs['experiment']['env_name']
    data_type = configs['data']['data_type']


    group_name = f"gym-experiment-{env_name}-{data_type}"
    # exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"
    now = datetime.datetime.now()
    exp_prefix = f"{group_name}-{now.year}/{now.month}/{now.day}-->{now.hour}:{now.minute}:{now.second}"

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {data_type}')
    print('=' * 50)
    
    # if configs['experiment']['WandB']:
    #     wandb.init(
    #         name=exp_prefix,
    #         group=group_name,
    #         project='decision-transformer-gym',
    #         config=configs
    #     )

    experiment = MFRL(configs)

    experiment.learn()

    print('\n')
    print('...End Decision Transformer experiment')
    pass



# python -m fusion.run train -alg mbpo -cfg gym_w2d_at_s1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    parser.add_argument('-cfg_path', type=str)

    args = parser.parse_args()

    sys.path.append("/home/rami/AMMI/Git/decision-transformer-ammi/gym/config")
    config = importlib.import_module(args.cfg)
    print('configurations: ', config.configurations)

    main(config.configurations)