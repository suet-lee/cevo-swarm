from simulator.wh_sim import *
from simulator.lib import Config
from simulator import CFG_FILES
import time
import numpy as np

###### Experiment parameters ######

ex_id = 'e_1'
verbose = False    

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

def run_ex(seed=None):
    sim = Simulator(cfg_obj,verbose=verbose,random_seed=seed)
    sim.run()

###### Run experiment ######

t0 = time.time()
seed = np.random.randint(0,10000000)
run_ex(seed)
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
