from pathlib import Path
import sys

dir_root = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import random
import threading
import os
from os.path import dirname, realpath
import datetime
import time
import json
from scipy.spatial.distance import cdist
from itertools import combinations

from . import Swarm, CA, Warehouse, Robot

class Simulator:

    def __init__(self, config,
        verbose=False,              # verbosity
        random_seed=None):

        self.cfg = config
        self.verbose = verbose
        self.exit_threads = False

        if random_seed is None:
            self.random_seed = random.randint(0,100000000)
        else:
            self.random_seed = random_seed

        np.random.seed(int(self.random_seed))

        try:
            self.swarm = self.build_swarm(self.cfg)
        except Exception as e:
            raise e
        
        # CA evo
        self.warehouse = CA(
            self.cfg.get('warehouse', 'width'),
            self.cfg.get('warehouse', 'height'), 
            self.cfg.get('warehouse', 'number_of_boxes'), 
            self.cfg.get('warehouse', 'box_radius'), 
            self.swarm,
            self.cfg.get('warehouse', 'object_position'),
            self.cfg.get('update_rate'),
            self.cfg.get('evaluate_rate'),
            self.cfg.get('influence_r'))    
        
        self.warehouse.generate_ap(self.cfg)
        self.warehouse.verbose = self.verbose
        self.export_data = self.cfg.get('export_data')
        self.export_steps = self.cfg.get('export_steps')
        self.export_training_data = False

        self._init_log()

    def _init_log(self):
        self.data = {}
        self.CA_data = {}
        self.belief_bank_log = {} # Stores contents of the belief bank
        self.belief_bank_metrics_log = {} # Stores contents of the belief bank
        self.belief_space_log = {} # Stores contents of the belief space
        self.training_data = []
        steps = int(self.cfg.get('time_limit')/self.export_steps)
        self.export_ts = list(range(steps, self.cfg.get('time_limit')+1, steps))

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range')
        )
        
        swarm = Swarm(
            repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
            repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
            heading_change_rate=cfg.get('heading_change_rate')
        )

        swarm.add_agents(robot_obj, cfg.get('warehouse', 'number_of_agents'))
        swarm.generate()  
        swarm.init_params(cfg)      
        return swarm

    # iterate method called once per timestep
    def iterate(self):
        self.warehouse.iterate(self.cfg.get('heading_bias'), self.cfg.get('box_attraction'))
        counter = self.warehouse.counter

        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter%100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(counter)

    def exit_sim(self, counter):
        if counter > self.cfg.get('time_limit'):
            print("Exiting...")
            self.exit_threads = True

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        while self.warehouse.counter <= self.cfg.get('time_limit'):
            self.iterate()
            if self.warehouse.counter%1000 == 0:
                print("Time elapsed... ",self.warehouse.counter)
            if self.export_data:
                self.log_CA_data()
                # self.log_BS_data()
                if self.warehouse.counter in self.export_ts:
                    self.log_data()
                if self.export_training_data:
                    self.log_training_data()
        
        if self.verbose:
            print("\n")

    def log_data(self):
        if 'box_c' not in self.data:
            self.data['box_c'] = {}
        if 'rob_c' not in self.data:
            self.data['rob_c'] = {}
        
        self.data['box_c'][self.warehouse.counter] = self.warehouse.box_c.tolist()
        self.data['rob_c'][self.warehouse.counter] = self.warehouse.rob_c.tolist()

    def log_CA_data(self):
        if 'P_m' not in self.CA_data:
            self.CA_data['P_m'] = {}
        if 'D_m' not in self.CA_data:
            self.CA_data['D_m'] = {}
        if 'SC' not in self.CA_data:
            self.CA_data['SC'] = {}
        if 'r0' not in self.CA_data:
            self.CA_data['r0'] = {}
        if 'social_transmission_log' not in self.CA_data:
            self.CA_data['social_transmission_log'] = {}
        
        if self.warehouse.counter%self.cfg.get("update_rate") == 0:
            self.CA_data['P_m'][self.warehouse.counter] = self.swarm.P_m.tolist()
            self.CA_data['D_m'][self.warehouse.counter] = self.swarm.D_m.tolist()
            self.CA_data['SC'][self.warehouse.counter] = self.swarm.SC.tolist()
            self.CA_data['r0'][self.warehouse.counter] = self.swarm.r0.tolist()
        
        self.CA_data['social_transmission_log'][self.warehouse.counter] = self.warehouse.social_transmission_log

    #TODO replace with entropy~
    def compute_BS_metrics(self, BS_arr):
        if len(BS_arr) == 0:
            return -1,-1,-1
        if len(BS_arr) == 1:
            return 0, 0, 1

        dists = cdist(BS_arr, BS_arr)
        vals = []
        for i1,i2 in combinations(range(len(dists)),2):
            vals.append(float(dists[i1][i2]))

        vals_arr = np.array(vals)
        mean = vals_arr.mean()
        sim_1 = (abs(vals_arr - 1) < 0.1).sum()
        var = vals_arr.var()
        return float(mean), float(var), float(sim_1)

    def log_BS_data(self):
        # Log belief bank contents and store
        bank_metrics_log = {}
        bank_log = []
        space_log = []
        for id, bs in self.swarm.BS.items():
            m,v,s = self.compute_BS_metrics(bs.belief_bank)
            bank_metrics_log[id] = {"m":m,"v":v,"s":s}
            bank_log.append(bs.belief_bank)
            space_log.append(bs.store)
        
        if self.warehouse.counter % 1000:
            self.belief_bank_log[self.warehouse.counter] = bank_log #TODO this is too much data to save...
        if self.warehouse.counter % 100:
            self.belief_bank_metrics_log[self.warehouse.counter] = bank_log
            self.belief_space_log[self.warehouse.counter] = space_log

    def log_training_data(self):

        step = self.warehouse.counter

        for id in range(self.warehouse.no_agents):
            # ===== INPUT (5 metrics) =====
            metrics = self.warehouse._gen_input_metrics(id)

            # ===== OUTPUT (8 values) =====
            start = id * 2
            end = start + 2

            output = np.concatenate([
                self.swarm.P_m[start:end],
                self.swarm.D_m[start:end],
                self.swarm.SC[start:end],
                self.swarm.r0[start:end]
            ]).tolist()

            sample = {
                "step": step,
                "agent_id": id,
                "input": list(metrics),
                "output": output
            }

            self.training_data.append(sample)

    def save_training_data(self, filename="training_data_2.json"):

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filename, "w") as f:
            json.dump(self.training_data, f, indent=2, default=convert_numpy)
        print(f"Training data saved to {filename}")
       
