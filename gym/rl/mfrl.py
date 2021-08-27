# Imports
# General


# ML & RL
import numpy as np
import torch as th

# DT-AMMI
from .orl import ORL
from decision_transformer.agents.dt import DecisionTransformer




class MFRL(ORL):
    def __init__(self, config):
        print('Initialize MFRL!')
        super(ORL, self).__init__(config)
        self.config = config
        pass


    def _build(self):
        super(ORL, self)._build()
        self._set_agent()
        pass


    def _set_agent(self):
        self.agent = DecisionTransformer(self.state_dim,
                                         self.act_dim,
                                         self.config['agent'])
        pass


    def learn(self):
        # N = self.config['Algorithm']['Learning']['nEpochs'] # Number of learning epochs
		# NT = self.config['Algorithm']['Learning']['epSteps'] # 
		# Ni = self.config['Algorithm']['Learning']['iEpochs'] # Number of initial epochs b4 training
		# E = self.config['Algorithm']['Learning']['envSteps'] # Number of env interactions
		
        # o, R, el, t = self.initialize_learning(None, NT, Ni)
        # for n in range(Ni+1, N+1):

        pass