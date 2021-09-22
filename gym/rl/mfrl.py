# Imports
# General
import time

# ML & RL
import numpy as np
import torch as th
import wandb

# DT-AMMI
from .orl import ORL
from data.data_handler import Data
from decision_transformer.agents.dt import DecisionTransformer




class MFRL(ORL):
    def __init__(self, config):
        super(MFRL, self).__init__(config)
        print('Initialize MFRL!')
        self.config = config
        self.device = config['experiment']['device']
        self._build()


    def _build(self):
        super(MFRL, self)._build()
        self._set_data_handler()
        self._set_agent()


    def _set_data_handler(self):
        self.data = Data(self.state_dim, self.act_dim, self.config)


    def _set_agent(self):
        self.agent = DecisionTransformer(self.state_dim,
                                         self.act_dim,
                                         self.config['experiment']['max_env_len'],
                                         self.config['learning']['niIter'],
                                         self.config['learning']['iter_steps'],
                                         self.config['agent']).to(self.device)


    def _learn(self, print_logs=True):
        N = self.config['learning']['nIter'] # Number of learning iterations
        NT = self.config['learning']['iter_steps'] #
        Ni = self.config['learning']['niIter']
        EE = self.config['evaluation']['eval_episodes']
        batch_size = self.config['data']['batch_size']

        logs = dict()
        print('Start Learning!')
        start_time = time.time()
        for n in range(N):
            print(f' [ Learning ] Iter: {n} ')
            # learn
            train_start = time.time()
            trainLosses = self.train_agent(NT, batch_size)
            logs['time/training'] = time.time() - train_start

            # evaluate
            eval_start = time.time()
            eval_logs = self.evaluate_agent(EE)
            for k, v in eval_logs.items():
                logs[f'evaluation/{k}'] = v
            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(trainLosses)
            logs['training/train_loss_std'] = np.std(trainLosses)
            # log
            if print_logs:
                print('=' * 80)
                print(f'Iteration {n}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            # WandB
            if self.config['experiment']['WandB']:
                wandb.log(logs)




    def learn(self, print_logs=True):
        N = self.config['learning']['nIter'] # Number of learning iterations
        NT = self.config['learning']['iter_steps'] #
        Ni = self.config['learning']['niIter']
        EE = self.config['evaluation']['eval_episodes']

        env_targets = self.config['experiment']['env_targets']

        logs = dict()
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
            # eval_logs = dict()
            eval_start = time.time()
            # for target in env_targets:
            eval_logs = self.evaluate_agent(EE, print_logs)

            for k, v in eval_logs.items():
                logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(trainLosses)
            logs['training/train_loss_std'] = np.std(trainLosses)

            # log
            if print_logs:
                # print('=' * 80)
                # print(f'Iteration {n}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

            # WandB
            if self.config['experiment']['WandB']:
                wandb.log(logs)

        return self.agentagent
