from simulator.wh_sim import *
import os

model = nnModel(
    input_size=5,
    hidden_nodes_per_layer=[7, 10],
    output_size=8
)

training_data = nnModel.load_training_data(
    "training_data_2.json"
)

model.train_model(
    training_data,
    epochs=200,
    learning_rate=0.01
)

dir_path = os.path.dirname(os.path.realpath(__file__))

model.save_weights_txt(
    dir_path + "/simulator/wh_sim/models/weights2.txt"
)