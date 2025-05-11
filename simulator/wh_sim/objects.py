import numpy as np
import math
from scipy.spatial.distance import cdist
import random
from collections import deque

class Robot:

    # max_v: max speed, assume robot moves at max speed if healthy
    # camera_sensor: assume camera range is 360deg (may be multiple cameras)
    def __init__(self, radius, max_v, camera_sensor_range, lifter_state=1): 
        self.radius = radius
        self.max_v = max_v
        self.camera_sensor_range = camera_sensor_range
        self.lifter_state = lifter_state

class Swarm:
    
    def __init__(self, repulsion_o, repulsion_w, heading_change_rate=1):
        self.agents = [] # turn this into a dictionary to make it accessible later for heterogeneous swarms?
        self.number_of_agents = 0
        self.repulsion_o = repulsion_o # repulsion margin between agents-objects
        self.repulsion_w = repulsion_w # repulsion distance between agents-walls
        self.heading_change_rate = heading_change_rate
        self.counter = 0
        self.F_heading = None
        self.agent_dist = None

    def add_agents(self, agent_obj, number):
        self.agents.append((agent_obj, number))

    def generate(self):
        self.robot_r = np.zeros(0)
        self.robot_v = np.zeros(0)
        self.camera_sensor_range_V = np.zeros(0)
        self.robot_lifter = np.zeros(0) # 0: unavailable, 1: available
        self.robot_heading = np.zeros(0)
        # self.robot_state = np.zeros(0) # storage, retrieval or idle # @TODO

        total_agents = 0
        for ag in self.agents:
            ag_obj = ag[0]
            num = ag[1]
            total_agents += num
            self.robot_r = np.append(self.robot_r, np.full(num, ag_obj.radius))
            self.robot_v = np.append(self.robot_v, np.full(num, ag_obj.max_v))
            self.camera_sensor_range_V = np.append(self.camera_sensor_range_V, np.full(num, ag_obj.camera_sensor_range))
            self.robot_lifter = np.append(self.robot_lifter, np.full(num, ag_obj.lifter_state))
        
        self.number_of_agents = total_agents
        # TODO combine agent_has_box and agent_box_id
        self.agent_has_box = np.zeros(self.number_of_agents) # agents start with no box
        self.agent_box_id = np.zeros(self.number_of_agents)*(-1) # record of box id that agent is carrying
        self.heading = 0.0314*np.random.randint(-100, 100, self.number_of_agents) # initial heading for all robots is randomly chosen
        self.computed_heading = self.heading # this is computed heading after force calculations are completed
        self.computed_heading_prev = {} # stores previous computed heading

    def init_params(self,cfg):
        # box interaction probabilities
        culture = cfg.get('culture')
        self.no_ap = len(cfg.get('ap'))
        self.no_box_t = len(cfg.get('box_type_ratio'))
        self.influence_rate = []
        self.resistance_rate = []

        # Behavioural parameters : used in their behaviour
        self.P_m = np.array([]) # pickup probability parameter
        self.D_m = np.array([]) # dropoff probability parameter
        self.SC = np.array([]) # amplification factor threshold for stone count
        self.r0 = np.array([]) # wall template radius from aggregation point (i.e. nest site)

        # Belief space parameters
        self.BS_P_m = np.array([])  # pickup probability parameter
        self.BS_D_m = np.array([])  # dropoff probability parameter
        self.BS_SC = np.array([])  # amplification factor threshold for stone count
        self.BS_r0 = np.array([])  # wall template radius from aggregation point (i.e. nest site)

        for subc in culture:
            no_agents = math.floor(self.number_of_agents*subc['ratio'])
            # probably a better way to unpack variables by list.pop() or next()
            P_m_vec = []
            D_m_vec = []
            SC_vec = []
            r0_vec = []
            for it in range(self.no_ap):
                P_m_vec.append(subc['params'][2*it])
            for it in range(self.no_ap):
                D_m_vec.append(subc['params'][2*it+1])
            for it in range(self.no_box_t):
                SC_vec.append(subc['params'][2*it+2*self.no_ap])
            for it in range(self.no_box_t):
                r0_vec.append(subc['params'][2*it+2*self.no_ap+1])
            
            P_m = np.tile(P_m_vec,(1,no_agents)).flatten()
            D_m = np.tile(D_m_vec,(1,no_agents)).flatten()
            SC = np.tile(SC_vec,(1,no_agents)).flatten()
            r0 = np.tile(r0_vec,(1,no_agents)).flatten()
            self.P_m = np.concatenate((self.P_m,P_m))
            self.D_m = np.concatenate((self.D_m,D_m))
            self.SC = np.concatenate((self.SC,SC))
            self.r0 = np.concatenate((self.r0,r0))

            if subc.get("use_fixed_rates", True):
                inf_rate = subc.get("influence_rate", 0.5)
                res_rate = subc.get("resistance_rate", 0.5)
                self.influence_rate.extend([inf_rate] * no_agents)
                self.resistance_rate.extend([res_rate] * no_agents)
            else:
                inf_range = subc.get("influence_range", (0.4, 0.9))
                res_range = subc.get("resistance_range", (0.2, 0.8))
                for _ in range(no_agents):
                    self.influence_rate.append(random.uniform(*inf_range))
                    self.resistance_rate.append(random.uniform(*res_range))

        self.influence_rate = np.array(self.influence_rate)
        self.resistance_rate = np.array(self.resistance_rate)

        # initialise the belief space
        self.BS_P_m = self.P_m
        self.BS_D_m = self.D_m
        self.BS_SC = self.SC
        self.BS_r0 = self.r0

        # fixed parameters: in future these could also be evolved in the belief space
        self.G_max = 0.55
        self.G_min = 0.01
        self.F_max = 0.55
        self.F_min = 0.01
        self.base_pickup_p = 0.1
        self.base_dropoff_p = 0.1
        self.tau = 0.025
        self.mem_size = 50

        # init computed metrics
        self.box_in_range = np.zeros(self.number_of_agents)
        self.box_in_range_mem = np.zeros((self.number_of_agents,self.mem_size))
        self.box_t_in_range = np.zeros(self.number_of_agents)
        self.box_t_in_range_mem = np.zeros((self.number_of_agents,self.mem_size))
       
        self.novelty_behav = np.zeros(self.number_of_agents)
        self.novelty_behav_mem = []
        for _ in range(self.number_of_agents):
            self.novelty_behav_mem.append({
                attr: deque(maxlen=self.mem_size) for attr in ['P_m', 'D_m', 'SC', 'r0']
            })
        self.novelty_env = np.zeros(self.number_of_agents)
        
    # @TODO allow for multiple behaviours, heterogeneous swarm
    def iterate(self, *args, **kwargs):
        self.update_hook() # allow for updates to the swarm
        return self.step(*args, **kwargs)

    # rob_c: robot center coordinates
    # box_c: box center coordinates
    def step(self, rob_c, box_c, box_r, is_box_in_transit, map, heading_bias=False, box_attraction=False):
        self.F_heading = self._generate_heading_force(heading_bias)

        # Compute euclidean (cdist) distance between agents and other agents
        self.agent_dist = cdist(rob_c, rob_c)
        F_box, F_agent = self._generate_interobject_force(box_c, box_r, rob_c, is_box_in_transit, box_attraction)

        # Compute distance to wall segments
        self.wall_dist = cdist(rob_c, map.wall_divisions)

        # Force on agent due to proximity to walls calculated elsewhere
        F_wall_avoidance = self._generate_wall_avoidance_force(rob_c, map)

        # Movement vectors summed
        F_agent += F_wall_avoidance + self.F_heading #+ F_box.T ## TODO remove F_box force to allow robots to move through boxes
        F_x = F_agent.T[0] # Repulsion vector in x
        F_y = F_agent.T[1] # in y 
        
        # New movement due to vectors
        t = self.counter%10
        self.computed_heading_prev[t] = self.computed_heading.tolist()
        # new heading due to vectors: this is actually the heading of the repelling force
        self.computed_heading = np.arctan2(F_y, F_x)
        move_x = np.multiply(self.robot_v, np.cos(self.computed_heading)) # Movement in x 
        move_y = np.multiply(self.robot_v, np.sin(self.computed_heading)) # Movement in y 
        
        # Total change in movement of agent (robot deviation)
        rob_d = -np.array([[move_x[n], move_y[n]] for n in range(0, self.number_of_agents)]) # Negative to avoid collisions
        return rob_d

    def update_hook(self):
        # Allow for updates to variables of swarm
        return

    def set_agent_box_state(self, agent_index, state):
        self.agent_has_box[agent_index] = state
        return True

    # amplification for pickup
    def _G(self,p,rob_id,SC):
        no_in_range = self.box_in_range[rob_id]
        if no_in_range < SC:
            p_ = self.G_max*p
        else:
            p_ = self.G_min*p
            
        return min(p_,1)

    # amplification for dropoff
    def _F(self,p,rob_id,SC): 
        p_ = p
        no_in_range = self.box_in_range[rob_id]
        box_in_range = self.box_in_range[rob_id]
        amp_max = np.argwhere(box_in_range > SC)
        amp_min = np.argwhere(box_in_range <= SC)
        p_[amp_max] *= self.F_max
        p_[amp_min] *= self.F_min

        return np.minimum(p_,np.ones(len(p)))

    def pickup_box(self, warehouse, robots=None):
        if robots is None:
            robots = list(range(warehouse.number_of_agents))

        dist_rob_to_box = cdist(warehouse.box_c, warehouse.rob_c) # calculates the euclidean distance from every robot to every box (centres)
        is_closest_rob_in_range = np.min(dist_rob_to_box, 1) < warehouse.box_range # if the minimum distance box-robot is less than the pick up sensory range, then qu_close_box = 1
        closest_rob_id = np.argmin(dist_rob_to_box, 1)	
        boxes_to_pickup = warehouse.is_box_free()*is_closest_rob_in_range
        to_pickup = np.argwhere(boxes_to_pickup==1)
        
		# needs to be a loop (rather than vectorised) in case two robots are close to the same box
        for box_id in to_pickup:
            closest_r = closest_rob_id[box_id][0]
            if closest_r not in robots:
                continue
            
            is_robot_carrying_box = warehouse.is_robot_carrying_box(closest_r)
            # Check if robot is already carrying a box
            if is_robot_carrying_box == 1:
                continue
            
            # check pickup probability for closest ap
            d_ap = cdist(warehouse.ap, warehouse.box_c[box_id])
            closest_ap = np.argmin(d_ap,0)
            d = np.min(d_ap,0)
            box_type = warehouse.box_types[box_id]
            # if points out of range, probability of pickup is fixed at base rate
            if d > self.camera_sensor_range_V[closest_r]:
                p = self.base_pickup_p
            else:
                d_ = d#(d*2/self.camera_sensor_range_V[closest_r]).flatten()
                ap = warehouse.ap[closest_ap]
                idx = closest_r*len(warehouse.ap)+closest_ap
                param_idx = closest_r*warehouse.number_of_box_types + box_type
                SC = self.SC[param_idx]*warehouse.number_of_boxes # SC is in range [0,1]
                r0 = self.r0[param_idx]*min(warehouse.width,warehouse.height)
                p = self.P_m[idx]*( 1 - 1/(1+self.tau*(d_-r0)*(d_-r0)) ).flatten()
                p = self._G(p, closest_r, SC)
            
            pickup = np.random.binomial(1,p)
            if pickup and warehouse.swarm.set_agent_box_state(closest_r, 1):
                warehouse.box_is_free[box_id] = 0 # change box state to 0 (not free, on a robot)
                warehouse.box_c[box_id] = warehouse.rob_c[closest_r] # change the box centre so it is aligned with its robot carrier's centre
                warehouse.robot_carrier[box_id] = closest_r # set the robot_carrier for box b to that robot ID
                self.agent_box_id[closest_r] = box_id # set box id

    def dropoff_box(self, warehouse, robots=None):
        if robots is None:
            robots = list(range(warehouse.number_of_agents))

        # first get the robots who are carrying boxes AND in the robots array
        rob_id = np.intersect1d(np.argwhere(self.agent_has_box == 1),robots)
        # get the boxes they are carrying
        # active box coordinates
        active_box_id = self.agent_box_id[rob_id].astype(int) #np.argwhere(warehouse.box_is_free == 0).flatten()
        active_c = warehouse.box_c[active_box_id]
        if len(active_c) == 0:
            return []
        
        # rob_id = warehouse.robot_carrier[active_box_id] # vector of carriers
        d = cdist(np.tile(warehouse.ap, (len(active_c),1)), active_c)
        d = d[:len(warehouse.ap)]
        # get the closest aggregation point
        d1 = np.min(d,axis=0)
        ap_idx = np.argmin(d,axis=0)
        # TODO cleanup - remove the scaling factor from code
        d2 = d1#d1*2/self.camera_sensor_range_V[rob_id] # scale down by factor cam_range/2
        
        idx = rob_id*len(warehouse.ap)+ap_idx
        box_types = warehouse.box_types[active_box_id]
        param_idx = rob_id*self.no_box_t + box_types
        SC = self.SC[param_idx]*warehouse.number_of_boxes # SC is in range [0,1]
        r0 = self.r0[param_idx]*min(warehouse.width,warehouse.height)
        p = self.D_m[idx]/(1+self.tau*(d2-r0)*(d2-r0))
        p = self._F(p,rob_id,SC)
        
        # if aggregation points are out of range, probability of dropoff is fixed at base rate
        in_range = (d1 <= self.camera_sensor_range_V[rob_id])
        p = p*in_range + self.base_dropoff_p*(1-in_range)
        drop = np.random.binomial(1,p).flatten()
        drop_idx = np.argwhere(drop==1).flatten()
        drop_box_id = active_box_id[drop_idx]
        return drop_box_id
		
    ## Avoidance behaviour for avoiding the warehouse walls ##		
    def _generate_wall_avoidance_force(self, rob_c, map): # input the warehouse map 
        ## distance from agents to walls ##
        # distance from the vertical walls to your agent (horizontal distance between x coordinates)
        difference_in_x = np.array([map.planeh-rob_c[n][1] for n in range(self.number_of_agents)])
        # distance from the horizontal walls to your agent (vertical distance between y coordinates)
        difference_in_y = np.array([map.planev-rob_c[n][0] for n in range(self.number_of_agents)])
        
        # x coordinates of the agent's centre coordinate
        agentsx = rob_c.T[0]
        # y coordinates  
        agentsy = rob_c.T[1]

        ## Are the agents within the limits of the warehouse? 
        x_lower_wall_limit = agentsx[:, np.newaxis] >= map.limh.T[0] # limh is for horizontal walls. x_lower is the bottom of the square
        x_upper_wall_limit = agentsx[:, np.newaxis] <= map.limh.T[1] # x_upper is the top bar of the warehouse square 
        # Interaction combines the lower and upper limit information to give a TRUE or FALSE value to the agents depending on if it is IN/OUT the warehouse boundaries 
        interaction = x_upper_wall_limit*x_lower_wall_limit
            
        # Fy is repulsion vector on the agent in y direction due to proximity to the horziontal walls 
        # This equation was designed to be very high when the agent is close to the wall and close to 0 otherwise
        # repulsion = np.minimum(self.repulsion_w, self.camera_sensor_range_V[0]) # @TODO figure out how wall avoidance works: what is planeh ?
        repulsion = self.repulsion_w
        Fy = np.exp(-2*abs(difference_in_x) + repulsion)
        # The repulsion vector is zero if the interaction is FALSE meaning that the agent is safely within the warehouse boundary
        Fy = Fy*difference_in_x*interaction	

        # Same as x boundaries but now in y
        y_lower_wall_limit = agentsy[:, np.newaxis] >= map.limv.T[0] # limv is vertical walls 
        y_upper_wall_limit = agentsy[:, np.newaxis] <= map.limv.T[1]
        interaction = y_lower_wall_limit*y_upper_wall_limit
        Fx = np.exp(-2*abs(difference_in_y) + repulsion)
        Fx = Fx*difference_in_y*interaction
        
        # For each agent the repulsion in x and y is the sum of the repulsion vectors from each wall
        Fx = np.sum(Fx, axis=1)
        Fy = np.sum(Fy, axis=1)
        # Combine to one vector variable
        F = np.array([[Fx[n], Fy[n]] for n in range(self.number_of_agents)])
        return F

    def _generate_heading_force(self, heading_bias=False):
        if self.counter % self.heading_change_rate == 0:
            # Add noise to the heading 
            noise = 0.01*np.random.randint(-50, 50, (self.number_of_agents))
            self.heading += noise

        # Force for movement according to new chosen heading 
        heading_x = 1*np.cos(self.heading) # move in x 
        heading_y = 1*np.sin(self.heading) # move in y

        if heading_bias:
            carriers = self.agent_has_box == 1 
            heading_x = heading_x + carriers*heading_bias # bias on heading if carrying a box

        return -np.array(list(zip(heading_x, heading_y)))
         
    # Computes repulsion forces: a negative force means comes out as attraction
    def _generate_interobject_force(self, box_c, box_r, rob_c, is_box_in_transit, box_attraction=False):
        margin = self.repulsion_o#np.minimum(self.repulsion_d, self.camera_sensor_range_V) @TODO allow for collision behaviour
        # TODO allow for a vector of robot_r (heterogeneous agents)
        self.too_close = self.agent_dist < 2*self.robot_r[0] + margin # TRUE if agent is too close to another agent (enable collision avoidance)
        
        # Compute euclidean (cdist) distance between boxes and agents
        self.box_dist = cdist(box_c, rob_c) # distance between all the boxes and all the agents
        self.too_close_boxes = self.box_dist < 2*box_r + margin # TRUE if agent is too close to a box (enable collision avoidance). Does not avoid box if agent does not have a box but this is considered later in the code (not_free*F_box)

        proximity_to_robots = rob_c[:, :, np.newaxis] - rob_c.T[np.newaxis, :, :] # Compute vectors between agents
        proximity_to_boxes = box_c[:, :, np.newaxis] - rob_c.T[np.newaxis, :, :] # Computer vectors between agents and boxes 
        
        F_box = self.too_close_boxes[:,np.newaxis,:]*proximity_to_boxes # Find which box vectors exhibit forces on the agents due to proximity 
        F_box = np.sum(F_box, axis=0) # Sum the vectors due to boxes on the agents
        
        not_free = self.agent_has_box == 1
        F_box_occupied = [not_free,not_free]*F_box 	# Only be repelled by boxes if you already have a box 
        
        if box_attraction:
            # Compute box attraction between free robots and free boxes
            too_close_free_boxes = (self.box_dist < self.camera_sensor_range_V) & np.transpose(np.tile((is_box_in_transit == 0), (self.number_of_agents, 1))) # attraction force from free boxes
            F_box_free = too_close_free_boxes[:,np.newaxis,:]*proximity_to_boxes
            F_box_free = np.sum(F_box_free, axis=0)
            free_agents = (self.agent_has_box == 1)
            noise = 0.0005*np.random.randint(0, 200, (2, self.number_of_agents))
            noise = np.add(np.ones((2, self.number_of_agents)), noise)
            F_box_free = [free_agents,free_agents]*F_box_free*noise
            F_box_total = F_box_occupied - F_box_free
        else:
            F_box_total = F_box_occupied
            
        F_agent = self.too_close[:,np.newaxis,:]*proximity_to_robots # Calc repulsion vector on agent due to proximity to other agents
        F_agent = np.sum(F_agent, axis =0).T # Sum the repulsion vectors
        
        return (F_box_total, F_agent)

    def compute_metrics(self, warehouse):
        # Global
        # self.number_of_agents
        # self.number_of_boxes
        # self.number_of_zones

        # Local
        # self.agent_dist # number of agents in range
        # self.wall_dist # walls in range
        in_range = self.box_dist < self.camera_sensor_range_V[0]
        self.box_in_range = sum(in_range) # boxes in range
        tile_box_t = np.tile(warehouse.box_types,(self.number_of_agents,1))+1 # add 1 to box type ID
        bt_in_range = in_range.astype(int)*tile_box_t.T
        for idx, it in enumerate(bt_in_range.T):
            self.box_t_in_range[idx] = sum(np.unique(it))
        
        # Novelty metrics
        self.compute_novelty_behaviour(warehouse)
        self.compute_novelty_environment(warehouse)

    # TODO vectorize ~
    def compute_novelty_behaviour(self, warehouse):
        # Step 1: Observe and add neighbors' behaviors to memory
        for agent_id in range(self.number_of_agents):
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                source_array = getattr(self, attr)
                param_size = self.no_ap if attr in ['P_m', 'D_m'] else self.no_box_t

                for neighbor_id in range(self.number_of_agents):
                    if neighbor_id == agent_id:
                        continue
                    if self.agent_dist[agent_id][neighbor_id] < warehouse.influence_r:
                        neighbor_start = neighbor_id * param_size
                        neighbor_values = source_array[neighbor_start:neighbor_start + param_size]
                        self.novelty_behav_mem[agent_id][attr].append(tuple(neighbor_values))

        # Step 2: Compute novelty from memory only (concatenated behavior vector)
        self.novelty_behav = [0.0] * self.number_of_agents
        for agent_id in range(self.number_of_agents):
            agent_vector = []
            memory_vectors = []

            # Build full behavior vector for current agent and memory
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                source_array = getattr(self, attr)
                param_size = self.no_ap if attr in ['P_m', 'D_m'] else self.no_box_t
                agent_start = agent_id * param_size
                agent_vector.extend(source_array[agent_start:agent_start + param_size])

                # Collect all past neighbor behavior vectors for this attribute
                for i, past in enumerate(self.novelty_behav_mem[agent_id][attr]):
                    # Make sure memory_vectors[i] exists and is extendable
                    if len(memory_vectors) <= i:
                        memory_vectors.append(list(past))
                    else:
                        memory_vectors[i].extend(past)

            # Now compute Euclidean distance from agent_vector to each memory_vector
            total_diff = 0.0
            for past_vector in memory_vectors:
                diff = sum((a - b) ** 2 for a, b in zip(agent_vector, past_vector)) ** 0.5
                total_diff += diff

            comparisons = len(memory_vectors)
            self.novelty_behav[agent_id] = total_diff / comparisons if comparisons > 0 else 0.0

        # Normalize novelty if needed
        # if max(self.novelty_behav) > 0:
        #     max_val = max(self.novelty_behav)
        #     self.novelty_behav = [n / max_val for n in self.novelty_behav]

        return self.novelty_behav

    def compute_novelty_environment(self,warehouse):
        time_idx = self.counter%self.mem_size # compute env perception and store in idx
        # env perception is a function of number of boxes and types of boxes (to simplify things)
        self.box_in_range_mem[:,time_idx] = self.box_in_range
        self.box_t_in_range_mem[:,time_idx] = self.box_t_in_range
        
        # compare first half of memory to last half
        if time_idx + self.mem_size/2 > self.mem_size:
            end_idx = time_idx-int(self.mem_size/2)
            old_b = np.sum(self.box_in_range_mem[:,time_idx:],axis=1)+np.sum(self.box_in_range_mem[:,:end_idx],axis=1)
            new_b = np.sum(self.box_in_range_mem[:,end_idx:time_idx],axis=1)
            # box type
            old_bt = np.sum(self.box_t_in_range_mem[:,time_idx:],axis=1)+np.sum(self.box_t_in_range_mem[:,:end_idx],axis=1)
            new_bt = np.sum(self.box_t_in_range_mem[:,end_idx:time_idx],axis=1)
        else:
            end_idx = time_idx + int(self.mem_size/2)
            old_b = np.sum(self.box_in_range_mem[:,time_idx:end_idx],axis=1)
            new_b = np.sum(self.box_in_range_mem[:,:time_idx],axis=1)+np.sum(self.box_in_range_mem[:,end_idx:],axis=1)
            # box type
            old_bt = np.sum(self.box_t_in_range_mem[:,time_idx:end_idx],axis=1)
            new_bt = np.sum(self.box_t_in_range_mem[:,:time_idx],axis=1)+np.sum(self.box_t_in_range_mem[:,end_idx:],axis=1)
        
        nov = abs(new_b-old_b)/self.mem_size#/warehouse.number_of_boxes
        amp_f = 1 + np.log(abs(old_bt-new_bt)/self.mem_size+1)/2
        
        self.novelty_env = np.minimum(np.ones(self.number_of_agents),amp_f*nov/20)
        print(self.novelty_env,"\n")
