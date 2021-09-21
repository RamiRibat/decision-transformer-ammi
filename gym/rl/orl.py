# Imports
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
        env_name = self.config['experiment']['env_name']
        if env_name == 'hopper':
            name = 'Hopper-v3'
        elif env_name == 'halfcheetah':
            name = 'HalfCheetah-v3'
        elif env_name == 'walker2d':
            name = 'Walker2d-v3'

        # name = 'Hopper-v3'# self.config['experiment']['env_name']

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


    def train_agent(self, NT, batch_size):
        print('train Agent!')
        # NT = self.config['learning']['iter_steps']
        self.agent.train()

        Losses = []
        for nt in range(NT):
            print(f' [ Agent Training ] Step: {nt} ', end='\r')
            loss = self.agent.train_model(self.data, batch_size)
            Losses.append(loss)
            if self.agent.scheduler: self.agent.scheduler.step()

        return Losses
            


    def evaluate_agent(self, EE):
        env_targets = self.config['experiment']['env_targets']
        device = self.config['experiment']['device']
        mode = self.config['experiment']['mode']
        scale = self.config['experiment']['scale']
        E = self.config['experiment']['max_env_len']

        self.agent.eval()

        eval_logs = dict()
        for target_rew in env_targets:
            returns, lengths = [], []

            for ee in range(EE):
                print(f' [ Agent Evaluation ] Episode: {ee} ', end='\r')
                with th.no_grad():
                    ret, length = self.agent.evaluate_model(self.eval_env,
                    device, mode, scale, E, target_return=target_rew)
                returns.append(ret)
                lengths.append(length)

            eval_logs.update({
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths)})

        return eval_logs