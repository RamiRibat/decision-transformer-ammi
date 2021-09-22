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

        if self.config['evaluation']['evaluate']:
            # self.eval_env, self.dt_from_xml = create_env(name)
            self.eval_env = gym.make(name) #MakeEnv(self.environment)
            # self.eval_env.seed(seed)
            # self.eval_env.action_space.seed(seed)
            # self.eval_env.observation_space.seed(seed)
        else:
            self.eval_env = None

        self.state_dim = self.eval_env.observation_space.shape[0]
        self.act_dim = self.eval_env.action_space.shape[0]
        self.rew_dim = 1
        self.act_upper_lim = self.eval_env.action_space.high
        self.act_lower_lim = self.eval_env.action_space.low


    def train_agent(self, NT, print_logs=True):
        self.agent.train()

        Losses = []
        for nt in range(NT):
            if print_logs:
                print(f' [ Agent Training ] Step: {nt}   ', end='\r')
            loss = self.agent.train_model(self.data)
            Losses.append(loss)
            if self.agent.scheduler: self.agent.scheduler.step()

        return Losses



    def evaluate_agent(self, EE, print_logs=True):
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
                if print_logs:
                    print(f' [ Agent Evaluation ] Target: {target_rew}, Episode: {ee}   ', end='\r')
                with th.no_grad():
                    ret, length = self.agent.evaluate_model(self.eval_env,
                    device, mode, scale, E, target_return=target_rew/scale)
                returns.append(ret)
                lengths.append(length)

            eval_logs.update({
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                # f'target_{target_rew}_length_std': np.std(lengths)
                })

        return eval_logs
