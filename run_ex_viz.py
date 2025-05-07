from simulator.wh_sim import *
from simulator.lib import Config
from simulator import CFG_FILES, MODEL_ROOT, STATS_ROOT
import time


ex_id = 'e_1'
iterations = 200
verbose = False    
batch_id = 'heading_bias'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

def gen_random_seed(iteration):
    P1 = 33331
    P2 = 73
    a = 1
    b = int(ex_id.split("_")[1])
    c = iteration
    return (a*P1 + b)*P2 + c

def run_ex():
    sim = VizSim(cfg_obj,verbose=verbose)
    sim.run()

###### Run experiment ######

t0 = time.time()
run_ex()
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
