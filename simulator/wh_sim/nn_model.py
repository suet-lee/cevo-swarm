import math

class nnModel:

    def __init__(self,
                 input_size=None,
                 output_size=None,
                 hidden_nodes_per_layer=None,
                 weights=None):
        
        self.input_size = input_size or [] #TODO remove redundant
        self.output_size = output_size or []
        self.hidden_nodes_per_layer = hidden_nodes_per_layer or []
        self.weights = weights or []

    def compute(self, input_vec):
        return self._compute(input_vec)

    def _compute(self, input0):
        wid = 0

        # First layer
        weight_d = self.hidden_nodes_per_layer[0]
        weights_slice = self.weights[wid:wid + weight_d]
        wid += weight_d
        input_layer = self.computeLayer(input0, self.hidden_nodes_per_layer[0], weights_slice)

        # Hidden layers
        for i in range(1, len(self.hidden_nodes_per_layer)):
            weight_d = len(input_layer) * self.hidden_nodes_per_layer[i]
            weights_slice = self.weights[wid:wid + weight_d]
            wid += weight_d
            input_layer = self.computeLayer(input_layer, self.hidden_nodes_per_layer[i], weights_slice)
    
        # Output layer
        weight_d = len(input_layer) * self.output_size
        weights_slice = self.weights[wid:wid + weight_d]
        return self.computeLayer(input_layer, self.output_size, weights_slice)

    @staticmethod
    def sigmoid(x):
        return x / (1 + abs(x))

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def activate(x):
        return nnModel.sigmoid(x) #TODO think sigmoid or tanh

    @staticmethod
    def computeLayer(input_data, output_size0, weights):
        wid = 0
        output = []
        
        for _ in range(output_size0):
            node_value = 0
            for ipt in input_data:
                w = weights[wid]
                node_value += w * ipt
            wid += 1

            node_value = nnModel.activate(node_value)
            output.append(node_value)

        return output