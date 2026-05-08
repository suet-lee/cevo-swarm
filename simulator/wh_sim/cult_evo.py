from pathlib import Path
import sys
import random
from itertools import combinations
from scipy.spatial.distance import cdist

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from . import Warehouse


# Implements swarm with cultural evolution
class CA(Warehouse):

    PHASE_SOCIAL_LEARNING = 0
    PHASE_UPDATE_BEHAVIOUR = 1
    PHASE_EXECUTE_BEHAVIOUR = 2

    def __init__(self, width, height, number_of_boxes, box_radius, swarm,
		init_object_positions=Warehouse.RANDOM_OBJ_POS, 
        update_rate=50, evaluate_rate=50,
        influence_r=100):
        super().__init__(width, height, number_of_boxes, box_radius, swarm,
		    init_object_positions=init_object_positions)
        
        self.influence_r = influence_r
        self.update_rate = update_rate
        self.evaluate_rate = evaluate_rate
        self.global_state_prev = None
        self.global_env_novelty = 0

        self.social_transmission_log =[]
        # self.verbose = True
        self.no_agents = swarm.number_of_agents

    # def update_hook(self):
        #

    # TODO avoid repetition from warehouse class
    def execute_pickup_dropoff(self, robots):
        self.swarm.pickup_box(self, robots)
        drop = self.swarm.dropoff_box(self, robots)
		
        if len(drop):
            # rob_n = self.robot_carrier[drop] # robot IDs to drop boxes
            valid_drop = []
            rob_n = []
            for d in drop:
                box_d = cdist([self.box_c[d]],self.box_c).flatten()
                count = len(np.argwhere(box_d<10).flatten())
                if count < 3: # Only allow 2 boxes in diameter 10 around robot
                    valid_drop.append(d)
                    rob_n.append(self.robot_carrier[d])

            self.box_is_free[valid_drop] = 1 # mark boxes as free again
            self.swarm.agent_has_box[rob_n] = 0 # mark robots as free again
            self.swarm.agent_box_id[rob_n] = -1

    # TODO avoid repetition from warehouse class (post hook)
    def iterate(self, heading_bias=False, box_attraction=False):     
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
		
        self.swarm.compute_metrics(self)
        
        # Update from BS for all robots
        if self.counter % self.update_rate == 0:
            self.update()

        # Evaluate state of the world
        if self.counter % self.evaluate_rate == 0:
            self.evaluate() # Update evaluated state of the world
            self.global_state_prev = self.box_c
        
        self.socialize()
        self.execute_pickup_dropoff()

        self.counter += 1
        self.swarm.counter = self.counter

    #TODO finish
    def compute_global_env_novelty(self):
        if self.global_state_prev is None:
            return 1
        
        # compare: self.global_state_prev to self.box_c (the new global state)
        return 1

    # Evaluate the novelty in the global state
    def evaluate(self):
        self.global_env_novelty = self.compute_global_env_novelty()
        # self.swarm.compute_local_env_novelty() #TODO can be combined


    #TODO evaluate wrt global state + in phase only
    def _compute_fitness(self, id):
        # Evaluates absolute difference between computed novelty and preferred novelty
        # Evaluation function: y= 1-|x_pref - x_obs|
        # E.g. x_pref=0.5, x_obs=1.0, y=0.5
        # E.g. x_pref=1.0, x_obs=1.0, y=1.0
        # E.g. x_pref=0.0, x_obs=1.0, y=0.0
        return 1 - abs(self.global_env_novelty-self.swarm.novelty_seeking[id])
        # return 1 - abs(self.swarm.novelty_env[id]-self.swarm.novelty_seeking[id]) # local novelty

    def _compute_agent_difference(self, id1, id2):
        return abs(self.swarm.novelty_env[id1]-self.swarm.novelty_env[id2])

    def socialize(self):
        self.social_transmission_log = []        
        ag_in_range = self.swarm.agent_dist < self.swarm.camera_sensor_range_V[0]

        for id1, id2 in combinations(range(self.no_agents), 2):
            if not ag_in_range[id1][id2]:
                continue
            
            self.social_transmission_log.append([id1, id2])
            ag_diff = self._compute_agent_difference(id1, id2)

            fit1 = self._compute_fitness(id1)
            fit2 = self._compute_fitness(id2)
            
            BS1 = self.swarm.BS[id1]
            BS2 = self.swarm.BS[id2]

            if ag_diff > self.swarm.tolerance:
                fit1 = 1-fit1
                fit2 = 1-fit2

            BS1.update_store(BS2.store,fit2)
            BS2.update_store(BS1.store,fit1)
            BS1.update_from_bank()
            BS2.update_from_bank()

    def _gen_input_metrics(self, rid):
        return [
            self.swarm.closest_ap[rid],
            self.swarm.closest_ap_dist[rid],
            self.swarm.agents_in_range[rid],
            self.swarm.box_in_range[rid],
            self.swarm.agent_has_box[rid]
        ]

    # TODO asynchronous evo ?
    def update(self):
        # noise_strength = 0.01  # Small amount of stochasticity #TODO remove redundancy

        for id in range(self.no_agents):
            # Generate parameters from BS (norm interpretation)
            metrics = self._gen_input_metrics(id)
            new_params = self.swarm.BS[id].generate_norm(metrics)
            start_index = id * self.swarm.no_ap

            # Each param: behaviour → BS_ version
            for attr in ["P_m","D_m","SC","r0"]:
                try:
                    target_array = getattr(self.swarm, attr)  # Get the stored behaviour param
                    source_array= getattr(new_params, attr)  # Get the generated belief space param

                    for i in range(self.swarm.no_ap):
                        v_new = source_array[start_index + i]
                        target_array[start_index + i] = v_new

                    # After the update, store the modified target_array back to self.BS_
                    setattr(self.swarm, attr, target_array)
                except Exception as e:
                    pass


