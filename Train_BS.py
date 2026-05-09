from simulator.wh_sim import *

model = nnModel(
    input_size=5,
    hidden_nodes_per_layer=[7, 10],
    output_size=8
)

training_data = model.load_training_data("training_data.json")


model.train(
    training_data,
    epochs=200,
    learning_rate=0.01
)

dir_path = os.path.dirname(os.path.realpath(__file__))
model.save_weights(dir_path+"/simulator/wh_sim/models/weights0.txt")