from pathlib import Path
import sys

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from . import Swarm, BoidsSwarm


# Implements swarm with cultural evolution
class CA_Swarm(Swarm):

    def __init__(self, repulsion_o, repulsion_w, heading_change_rate=1, P_m=1, D_m=1, influence_r=100):
        super().__init__(repulsion_o, repulsion_w, heading_change_rate, P_m, D_m)
        
        self.has_init_params = False
        self.influence_r = influence_r

    def _init_params(self):
        # assume we have influence/resistance parameters
        # need to initialize these for all agents
        # need to initialize a belief space
        self.has_init_params = True
        
        # Each agent has an individual belief space - this encodes the behavioural parameters
        # TODO is there a better way to encode? how about into a smaller dimensional space ??? (or larger)
        # initialise with the parameters
        # self.bs_C = self._C # cohesion
        # self.bs_A = self._A # alignment
        # self.bs_S = self._S # separation

        # parameters: these should determine where robots decide to drop boxes
        # - near other boxes?
        # - types of boxes?
        # - in particular zones (relative or wrt world frame)
        # -- in response to environmental parameters, to other robots, signals
        
        # likelihood of dropping box
        # preference for spatial zone~


        # Influence/update factors (how influential or resistant an agent is in social exchange)
        self.influence_F = np.random.uniform(0,1,self.number_of_agents) # TODO intialise - how? random for now
        self.update_F = np.random.uniform(0,1,self.number_of_agents)

    def update_hook(self):
        if not self.has_init_params:
            self._init_params()

        if self.counter > 0:
            self.evolve()

    # TODO asynchronous evo ?
    # This is called after the main step function (step forward in swarm behaviour)
    def evolve(self):
        # Influence : each agent applies their influence to other agents' belief spaces
        # first check which agents are in proximity (should be computed in self.agent_dist)
        in_range = (self.agent_dist < self.influence_r)
        inf_F = np.multiply(self.influence_F,in_range)
        # print(inf_F[1],'\n')

        # Update : each agent updates their behavioural parameters from their belief space


        return
