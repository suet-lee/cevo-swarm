from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES
import time
import numpy as np

###### Experiment parameters ######

ex_id = 'e_1'
verbose = False
export_box_c = True

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######



###### Run experiment ######

t0 = time.time()
seed = np.random.randint(0,10000000)
st = SaveTo()
sim = Simulator(cfg_obj,verbose=verbose,random_seed=seed)
sim.run()
if export_box_c:
    data = sim.data
    st.export_data(ex_id,data)
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
