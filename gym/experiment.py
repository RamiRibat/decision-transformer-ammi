import argparse


import numpy as np
import torch as th


from decision_transformer.agents.dt import DecisionTransformer
from decision_transformer.agents.traj_gptx.gpt2 import GPT2
from decision_transformer.training.trans_trainer import TransforemerTrainer





def experiment(name, settings):
    # env, env_eval = 

    state_dim, action_dim = env.observation_space[0], env.action_space[0]

    dt_agent = DecisionTransformer(s_dim=state_dim, a_dim=action_dim,
                                   settings=settings['agent'])

    trainer = TransforemerTrainer(agent=dt_agent,
                                  loss=lambda a_hat, a: th.mean((a_hat - a)**2),
                                  settings=settings['training']
                                  )

    for ep in range(settings['nEpisodes']):
        trainer.train(nEne=settings['nEnvSteps'],
                      Ep=ep+1)
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    experiment(name='gym_cheetah_exp', settings=vars(args))