# Imports
# General


# ML & RL
import gym
import numpy as np
import torch as th






class ORL:
    def __init__(self, config):
        print('Initialize ORL!')
        self.config = config


    def _build(self):
        self._set_env()


    def _set_env(self):
        name = self.config['experiment']['env_name']
        # seed = self.config['experiment']['seed']
        evaluate = self.config['evaluation']['evaluate']


        # self.train_env, self.dt_from_xml = create_env(name)


        self.train_env = gym.make(name) #MakeEnv(self.environment)
        # self.train_env.seed(seed)
        # self.train_env.action_space.seed(seed)
        # self.train_env.observation_space.seed(seed)


        if evaluate:
            # self.eval_env, self.dt_from_xml = create_env(name)
            self.eval_env = gym.make(name) #MakeEnv(self.environment)
            # self.eval_env.seed(seed)
            # self.eval_env.action_space.seed(seed)
            # self.eval_env.observation_space.seed(seed)
        else:
            self.eval_env = None

        self.state_dim = self.train_env.observation_space.shape[0]
        self.act_dim = self.train_env.action_space.shape[0]
        self.rew_dim = 1
        self.act_upper_lim = self.train_env.action_space.high
        self.act_lower_lim = self.train_env.action_space.low


    def init_learning(self):
        pass


    def interact(self):
        pass


    def evaluate(self):
        pass
