from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES
import time
import numpy as np

###### Experiment parameters ######

ex_id = 'e_1'
verbose = False
export_data = True

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
#sim.save_training_data()
if export_data:    
    # for key in ["P_m", "D_m", "SC", "r0"]:
    #     data = sim.CA_data[key]
    #     st.export_data(ex_id, data, key, transpose=True)
    
    # for key in ["social_transmission_log":
    #     data = sim.CA_data[key]
    #     records = [{"timestep": i, key: v} for i, v in data.items()]
    #     st.export_data(ex_id, records, key)

    dn = st.export_data(ex_id,sim.data['box_c'], "boxes")
    st.export_data(ex_id,sim.data['rob_c'], "robots")
    st.export_metadata(dn, cfg_obj.get_attributes())
    
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
