from pathlib import Path
import sys

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from . import Warehouse


# Implements swarm with cultural evolution
class CA(Warehouse):

    PHASE_SOCIAL_LEARNING = 0
    PHASE_UPDATE_BEHAVIOUR = 1
    PHASE_EXECUTE_BEHAVIOUR = 2

    def __init__(self, width, height, number_of_boxes, box_radius, swarm,
		init_object_positions=Warehouse.RANDOM_OBJ_POS, box_type_ratio=[1], influence_r=100):
        super().__init__(width, height, number_of_boxes, box_radius, swarm,
		    init_object_positions=Warehouse.RANDOM_OBJ_POS, box_type_ratio=[1])
        
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


        # Influence/update factors (how influential or resistant an agent is in social exchange)
        self.influence_F = np.random.uniform(0,1,self.swarm.number_of_agents) # TODO intialise - how? random for now
        self.update_F = np.random.uniform(0,1,self.swarm.number_of_agents)

    def update_hook(self):
        if not self.has_init_params:
            self._init_params()

    def select_phase(self):
        p = np.random.uniform(0,3,self.swarm.number_of_agents)
        phase = np.floor(p)
        s = np.argwhere(phase==self.PHASE_SOCIAL_LEARNING).flatten()
        u = np.argwhere(phase==self.PHASE_UPDATE_BEHAVIOUR).flatten()
        e = np.argwhere(phase==self.PHASE_EXECUTE_BEHAVIOUR).flatten()
        return s,u,e

    def step(self, heading_bias=False, box_attraction=False):     
        self.rob_d = self.swarm.iterate(
			self.rob_c, 
			self.box_c, 
			self.box_radius,
			self.box_is_free, 
			self.map, 
			heading_bias,
			box_attraction) # the robots move using the random walk function which generates a new deviation (rob_d)
		
		# handles logic to move boxes with robots/drop boxes
        t = self.counter%10
        self.rob_c_prev[t] = self.rob_c # Save a record of centre coordinates before update
        self.rob_c = self.rob_c + self.rob_d # robots centres change as they move
        active_boxes = self.box_is_free == 0 # boxes which are on a robot
        self.box_d = np.array((active_boxes,active_boxes)).T*self.rob_d[self.robot_carrier] # move the boxes by the amount equal to the robot carrying them 
        self.box_c = self.box_c + self.box_d
		
        s,u,e = self.select_phase()   
        self.socialize(s)
        self.update(u)
        self.execute_pickup_dropoff(e)

        self.swarm.compute_metrics()
        self.counter += 1
        self.swarm.counter = self.counter

    def socialize(self, agent_ids):
        return

    # TODO asynchronous evo ?
    # This is called after the main step function (step forward in swarm behaviour)
    def update(self, agent_ids):
        # Influence : each agent applies their influence to other agents' belief spaces
        # first check which agents are in proximity (should be computed in self.agent_dist)
        in_range = (self.swarm.agent_dist < self.influence_r)
        inf_F = np.multiply(self.influence_F,in_range)
        # print(inf_F[1],'\n')

        # Update : each agent updates their behavioural parameters from their belief space

        
        return

        