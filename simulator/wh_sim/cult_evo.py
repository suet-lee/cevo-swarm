from pathlib import Path
import sys
import random
from itertools import combinations

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
		    init_object_positions=Warehouse.RANDOM_OBJ_POS, box_type_ratio=box_type_ratio)
        
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
        #self.influence_F = np.random.uniform(0,1,self.swarm.number_of_agents) # TODO intialise - how? random for now
        #self.update_F = np.random.uniform(0,1,self.swarm.number_of_agents)

    def update_hook(self):
        if not self.has_init_params:
            self._init_params()

    def select_phase(self):
        p = np.random.uniform(0,3,self.swarm.number_of_agents) # change it to control the prob
        phase = np.floor(p)
        s = np.argwhere(phase==self.PHASE_SOCIAL_LEARNING).flatten()
        u = np.argwhere(phase==self.PHASE_UPDATE_BEHAVIOUR).flatten()
        e = np.argwhere(phase==self.PHASE_EXECUTE_BEHAVIOUR).flatten()

        if len(s) % 2 != 0:
            s = s[:-1]
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
		
        self.swarm.compute_metrics()
        
        s,u,e = self.select_phase()   
        self.socialize(s)
        self.update(u)
        self.execute_pickup_dropoff(e)

        self.counter += 1
        self.swarm.counter = self.counter


    def socialize(self, agent_ids):
        used = set()
        influence_prob = 0.5  # apply influence per parameter with this chance

        for id1, id2 in combinations(agent_ids, 2):
            if id1 in used or id2 in used:
                continue

            dist = self.swarm.agent_dist[id1][id2]
            if dist >= self.influence_r:
                continue

            # Get influence rates
            rate1 = self.swarm.influence_rate[id1]
            rate2 = self.swarm.influence_rate[id2]

            if rate1 == rate2:
                continue  # no update if influence is identical

            # Determine influencee and influencer
            if rate1 > rate2:
                influencer, influencee = id1, id2
            else:
                influencer, influencee = id2, id1

            weight = abs(rate1 - rate2)
            rev_weight = 1.0 - weight

            print(
                f"Agents {influencer} (more influential) & {influencee} interacting — weight: {weight:.2f}, dist: {dist:.2f}")
            used.update([id1, id2])

            # Each param: behaviour → BS_ version
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                source_array = getattr(self, attr) # Behaviour param
                target_array = getattr(self, f'BS_{attr}')  # belief space param

                param_size = self.no_ap if attr in ['P_m', 'D_m'] else len(self.no_box_t)

                start_inf = influencer * param_size
                start_infce = influencee * param_size

                for i in range(param_size):
                    # if random.random() < influence_prob:
                    #     v_inf = source_array[start_inf + i]
                    #     v_infce = source_array[start_infce + i]
                    #
                    #     # Update behaviorally shifted version
                    #     target_array[start_infce + i] = v_infce + weight * (v_inf - v_infce)
                    #     target_array[start_inf + i] = v_inf - rev_weight * (v_inf - v_infce)

                    if random.random() < weight:
                        target_array[start_infce + i] =  source_array[start_inf + i]
                    if random.random() < rev_weight:
                        target_array[start_inf + i] = source_array[start_infce + i]

                # After the update, store the modified target_array back to self.BS_
                setattr(self, f'BS_{attr}', target_array)


    # TODO asynchronous evo ?
    # This is called after the main step function (step forward in swarm behaviour)
    def update(self, agent_ids):

        for id in agent_ids:
            # Each param: behaviour → BS_ version
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                target_array = getattr(self, attr)  # Behaviour param
                source_array= getattr(self, f'BS_{attr}')  # belief space param

                param_size = self.no_ap if attr in ['P_m', 'D_m'] else len(self.no_box_t)

                start_index = id * param_size

                for i in range(param_size):
                    if random.random() < self.swarm.resistance_rate[id]:
                        target_array[start_index + i] = source_array[start_index + i]

                # After the update, store the modified target_array back to self.BS_
                setattr(self, attr, target_array)





    #def compute_analytic (self):

