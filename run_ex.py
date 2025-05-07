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
    
    for key in ["P_m", "D_m", "SC", "r0", "BS_P_m", "BS_D_m", "BS_SC", "BS_r0", "social_transmission","Self_updates"]:
        values = sim.CA_data[key]  # Use CA_dat instead of CA_data
        records = [{"timestep": i, key: v} for i, v in enumerate(values)]
        df = pd.DataFrame(records)
        st.export_data2(ex_id, df, key)
    
    dn,fn = st.export_data(ex_id,data)
    st.export_metadata(dn,fn,
    {
        'box_type_ratio':cfg_obj.get('box_type_ratio'),
        'ap':cfg_obj.get('ap')
    })
    
t1 = time.time()
dt = t1-t0
print("Time taken: %s"%str(dt), '\n')
