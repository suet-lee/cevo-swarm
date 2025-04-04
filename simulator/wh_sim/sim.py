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

from . import Swarm, BoidsSwarm, CA_Swarm, Warehouse, Robot

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

        self.warehouse = Warehouse(
            self.cfg.get('warehouse', 'width'),
            self.cfg.get('warehouse', 'height'), 
            self.cfg.get('warehouse', 'number_of_boxes'), 
            self.cfg.get('warehouse', 'box_radius'), 
            self.swarm,
            self.cfg.get('warehouse', 'object_position'))            

    def build_swarm(self, cfg):
        robot_obj = Robot(
            cfg.get('robot', 'radius'), 
            cfg.get('robot', 'max_v'),
            camera_sensor_range=cfg.get('robot', 'camera_sensor_range')
        )
        
        behaviour = cfg.get('swarm_behaviour')
        
        # boids flocking behaviour
        if behaviour == 1:
            swarm = BoidsSwarm(
                repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
                repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
                heading_change_rate=cfg.get('heading_change_rate')
            )
        # CA evo swarm
        elif behaviour == 2:
            swarm = CA_Swarm(
                repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
                repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
                heading_change_rate=cfg.get('heading_change_rate'),
                influence_r=cfg.get('influence_r')
            )
        # default random walk
        else:
            swarm = Swarm(
                repulsion_o=cfg.get('warehouse', 'repulsion_object'), 
                repulsion_w=cfg.get('warehouse', 'repulsion_wall'),
                heading_change_rate=cfg.get('heading_change_rate')
            )

        swarm.add_agents(robot_obj, cfg.get('warehouse', 'number_of_agents'))
        swarm.generate()        
        # fault_types = cfg.get('faults')
        # count = 0
        # fault_count = []
        # for i, fault in enumerate(fault_types):
        #     end_count = self.fault_count[i]+count
        #     fault_count.append(end_count)
        #     faulty_agents_range = range(count, end_count)
        #     fault_cfg = self.generate_fault_type(fault, faulty_agents_range)
        #     swarm.add_fault(**fault_cfg)    
        #     count = end_count

        # swarm.fault_count = fault_count
        return swarm

    # def generate_fault_type(self, fault, faulty_agents_range):
    #     fault_type = fault['type']
        
    #     if fault_type == FaultySwarm.ALTER_AGENT_SPEED:
    #         speed = fault['cfg']['speed_at_fault']
    #         lookup = []
    #         for i in faulty_agents_range:
    #             lookup.append((0, i, speed))
    #         return {'ftype': FaultySwarm.ALTER_AGENT_SPEED, 'lookup': lookup}

    #     if fault_type == FaultySwarm.FAILED_BOX_PICKUP:
    #         lookup = {}
    #         for i in faulty_agents_range:
    #             lookup[i] = 0
    #         return {'ftype': FaultySwarm.FAILED_BOX_PICKUP, 'lookup': lookup}

    #     if fault_type == FaultySwarm.FAILED_BOX_DROPOFF:
    #         lookup = {}
    #         for i in faulty_agents_range:
    #             lookup[i] = 0
    #         return {'ftype': FaultySwarm.FAILED_BOX_DROPOFF, 'lookup': lookup}

    #     if fault_type == FaultySwarm.REDUCED_CAMERA_RANGE:
    #         r_range = fault['cfg']['reduced_range']
    #         lookup = {}
    #         for i in faulty_agents_range:
    #             lookup[i] = r_range
    #         return {'ftype': FaultySwarm.REDUCED_CAMERA_RANGE, 'lookup': lookup}
   
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
        
        if self.verbose:
            print("\n")

