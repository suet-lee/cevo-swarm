from pathlib import Path
import sys
import random
from itertools import combinations
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from . import Warehouse


# Implements swarm with cultural evolution
class CA(Warehouse):

    def __init__(self, width, height, number_of_boxes, box_radius, swarm,
		init_object_positions=Warehouse.RANDOM_OBJ_POS, 
        update_rate=50, evaluate_rate=50,
        influence_r=100):
        super().__init__(width, height, number_of_boxes, box_radius, swarm,
		    init_object_positions=init_object_positions)
        
        self.influence_r = influence_r
        self.update_rate = update_rate
        self.evaluate_rate = evaluate_rate
        self.global_state_prev = self._convert_to_pixel_grid(self.box_c)
        self.global_env_novelty = 0
        self.novelty_log = []

        self.social_transmission_log =[]
        # self.verbose = True
        self.no_agents = swarm.number_of_agents

    # def update_hook(self):
        #

    # TODO avoid repetition from warehouse class
    def execute_pickup_dropoff(self):
        self.swarm.pickup_box(self)
        drop = self.swarm.dropoff_box(self)
		
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
        
        # Socialize and execute happen every iteration
        self.socialize()
        self.execute_pickup_dropoff()

        self.counter += 1
        self.swarm.counter = self.counter

    def _compute_ring_metrics(self, ap, max_r, box_c):
        # Distance from ap
        dist_from_ap = cdist([ap],box_c)
        ap_box_idx = np.argwhere(dist_from_ap.flatten() < max_r).flatten()
        
        # Mean distance from ap
        ap_m = np.mean(dist_from_ap.flatten()[ap_box_idx])
        
        # Var: dist from ap - ap_m
        dists = (dist_from_ap.flatten() - ap_m)[ap_box_idx]
        
        # normalise
        dists_n = dists/(dists.max()-dists.min())
        ap_var = np.var(dists_n)

        return ap_m, ap_var 

    #TODO unused
    def _compute_box_distribution(self, res, box_c):
        no_cells = int(self.width/res)
        grid = np.zeros([no_cells,no_cells])
        for c in box_c:
            idx0 = int(np.floor(c[0]/res))
            idx1 = int(np.floor(c[1]/res))
            grid[idx0,idx1] += 1
        
        return grid

    #TODO old implementation - remove or improve
    def compute_global_env_novelty0(self):
        if self.global_state_prev is None:
            self.global_state_prev = {'ap_m':[0]*len(self.ap), 'ap_var':[0]*len(self.ap)}
            return 1
        
        if len(self.ap) == 1:
            # empirical values for scaling
            max_diff_var = 0.1 # 0.07

            ap = self.ap[0]
            max_r = min( abs(self.width-ap[0]), ap[0], abs(self.height-ap[1]), ap[1] )
            ap_m, ap_var = self._compute_ring_metrics(ap, max_r, self.box_c)
            old_ap_m = self.global_state_prev['ap_m'][0]
            old_ap_var = self.global_state_prev['ap_var'][0]
            novelty = 0.5*abs(ap_m-old_ap_m)/max_r + 0.5*min(abs(ap_var-old_ap_var)/max_diff_var,1)
            self.global_state_prev = {'ap_m':[ap_m], 'ap_var':[ap_var]} # novelty in mean ring radius, and ring variance
        else:
            max_diff_var = 0.1

            # Assume two AP - find distance between them
            # Half the distance is the assumed maximum radius for any ring
            dist = np.linalg.norm(self.ap[0]-self.ap[1])
            max_r = dist/2
            ap_m0, ap_var0 = self._compute_ring_metrics(self.ap[0], max_r, self.box_c)
            ap_m1, ap_var1 = self._compute_ring_metrics(self.ap[1], max_r, self.box_c)
            
            old_ap_m0 = self.global_state_prev['ap_m'][0]
            old_ap_var0 = self.global_state_prev['ap_var'][0]
            old_ap_m1 = self.global_state_prev['ap_m'][1]
            old_ap_var1 = self.global_state_prev['ap_var'][1]
            novelty = 0.25*abs(ap_m0-old_ap_m0)/max_r + 0.25*abs(ap_var0-old_ap_var0)/max_diff_var \
                + 0.25*abs(ap_m1-old_ap_m1)/max_r + 0.25*abs(ap_var1-old_ap_var1)/max_diff_var

            self.global_state_prev = {'ap_m':[ap_m0,ap_m1], 'ap_var':[ap_var0,ap_var1]}

        #TODO not used
        # Box distribution by grid - count number of boxes in each cell
        # res = 25
        
        scalef = 10
        return min(novelty*scalef,1)
        
    def _convert_to_pixel_grid(self, box_c):
        img = np.zeros([500,500])
        for i in box_c:
            idx0 = int(np.floor(i[0]))
            idx1 = int(np.floor(i[1]))
            for x in range(idx0-5,idx0+5):
                for y in range(idx1-5,idx1+5):
                    img[x,y] = 1    
        return img.astype(int)

    def compute_global_env_novelty(self):        
        # Compute structural similarity
        new_gs = self._convert_to_pixel_grid(self.box_c)
        score, diff = ssim(self.global_state_prev, new_gs, full=True, data_range=1.0)
        score = (max(0.6,score)-0.6)/0.4
        # score of 1 means exactly the same
        return 1-score

    # Evaluate the novelty in the world
    def evaluate(self):
        self.global_env_novelty = self.compute_global_env_novelty()
        print(self.global_env_novelty) #TODO debugging
        self.novelty_log.append(self.global_env_novelty)
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
            
            bs1 = self.swarm.BS[id1]
            bs2 = self.swarm.BS[id2]

            if ag_diff > self.swarm.tolerance:
                fit1 = 1-fit1
                fit2 = 1-fit2

            bs1.update_bank(bs2.store,fit2)
            bs2.update_bank(bs1.store,fit1)
            bs1.update_from_bank()
            bs2.update_from_bank()

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
            if self.counter%100==0 and id==0: #TODO for debugging
                print(new_params)
            start_index = id * self.swarm.no_ap

            # Each param: behaviour → BS_ version
            for attr in ["P_m","D_m","SC","r0"]:
                try:
                    target_array = getattr(self.swarm, attr)  # Get the stored behaviour param
                    source_array = new_params[attr]  # Get the generated belief space param
                    
                    for i in range(self.swarm.no_ap):
                        v_new = source_array[i]
                        target_array[start_index + i] = v_new

                    # After the update, store the modified target_array back to self.BS_
                    setattr(self.swarm, attr, target_array)
                except Exception as e:
                    print(e)
                

