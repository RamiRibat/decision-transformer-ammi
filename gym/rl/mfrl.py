import time

import numpy as np
import torch as th
import wandb

from .orl import ORL
from data.data_handler import Data
from decision_transformer.agents.dt import DecisionTransformer



class MFRL(ORL):
    """
    Model-Free Reinforcement Learning (MFRL) module
        1. Set and build basic components of the MFRL experiment
        2. Handle the agent learning loop
    """
    def __init__(self, config, seed):
        super(MFRL, self).__init__(config, seed)
        print('Initialize MFRL!')
        self.config = config
        self.device = config['experiment']['device']
        self.seed = seed
        self._build()


    def _build(self):
        super(MFRL, self)._build()
        self._set_data_handler()
        self._set_agent()


    def _set_data_handler(self):
        self.data = Data(self.state_dim, self.act_dim, self.config, self.seed)


    def _set_agent(self):
        self.agent = DecisionTransformer(self.state_dim,
                                         self.act_dim,
                                         self.config,
                                         self.seed).to(self.device)


    def learn(self, print_logs=True):
        N = self.config['learning']['nIter'] # Number of learning iterations
        NT = self.config['learning']['iter_steps'] # Number of warmup learning iterations
        Ni = self.config['learning']['niIter'] # Number of training steps/iteration
        EE = self.config['evaluation']['eval_episodes'] # Number of episodes

        logs = dict()
        gif = True
        best_ret = 0.0

        print('Start Learning!')
        start_time = time.time()
        for n in range(N):
            if print_logs:
                print('=' * 80)
                print(f' [ Learning ] Iteration: {n} ')

            # learn
            train_start = time.time()
            trainLosses = self.train_agent(NT, print_logs)
            logs['time/training'] = time.time() - train_start

            # evaluate
            eval_start = time.time()
            eval_logs = self.evaluate_agent(EE, gif, n, print_logs)

            for k, v in eval_logs.items():
                logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(trainLosses)
            logs['training/train_loss_std'] = np.std(trainLosses)

            # log
            if print_logs:
                for k, v in logs.items():
                    print(f'{k}: {v}')

            # WandB
            if self.config['experiment']['WandB']:
                wandb.log(logs)

        return self.agent
