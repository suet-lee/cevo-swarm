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

from . import Swarm, CA, Warehouse, Robot

class Simulator:

    def __init__(self, config,
        verbose=False,              # verbosity
        random_seed=None):

        self.cfg = config
        self.verbose = verbose
        # self.state_changes = 0 # intended to log changes in the system from normal (if faults are injected mid-runtime for example)
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
            self.cfg.get('box_type_ratio'),
            self.cfg.get('phase_ratio'),
            self.cfg.get('influence_r'))     

        self.warehouse.generate_ap(self.cfg)   
        self.export_data = self.cfg.get('export_data')
        self.export_steps = self.cfg.get('export_steps')
        self._init_log()

    def _init_log(self):
        self.data = {}
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
            if self.export_data and self.warehouse.counter in self.export_ts:
                self.log_data()
        
        if self.verbose:
            print("\n")

    def log_data(self):
        self.data[self.warehouse.counter] = self.warehouse.box_c.tolist()

