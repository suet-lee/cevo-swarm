from simulator.wh_sim import *

nn = nnModel(output_size=2,
            hidden_nodes_per_layer=[3],
            weights=[0.5,1.9,0.7,0.2,0.3])

out = nn.compute([5,2])
print(out)