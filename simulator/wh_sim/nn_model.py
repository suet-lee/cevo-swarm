import math
import json
import random


class nnModel:

    def __init__(self,
                 input_size=None,
                 output_size=None,
                 hidden_nodes_per_layer=None,
                 weights=None):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_nodes_per_layer = hidden_nodes_per_layer or []

        self.layers = [self.input_size] + self.hidden_nodes_per_layer + [self.output_size]

        expected_weights = sum(
            self.layers[i] * self.layers[i + 1]
            for i in range(len(self.layers) - 1)
        )

        if weights is None or len(weights) != expected_weights:
            print(f"Generating random weights ({expected_weights})")
            self.weights = self.random_weights()
        else:
            self.weights = weights

    def compute(self, input_vec):
        return self._compute(input_vec)

    def _compute(self, input0):
        wid = 0
        input_layer = input0

        for layer_index in range(len(self.layers) - 1):
            in_size = self.layers[layer_index]
            out_size = self.layers[layer_index + 1]

            weight_d = in_size * out_size
            weights_slice = self.weights[wid:wid + weight_d]
            wid += weight_d

            input_layer = self.computeLayer(
                input_layer,
                out_size,
                weights_slice
            )

        return input_layer

    @staticmethod
    def sigmoid(x):
        return x / (1 + abs(x))

    @staticmethod
    def sigmoid_derivative_from_output(y):
        return (1 - abs(y)) ** 2

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def activate(x):
        return nnModel.sigmoid(x)

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

            output.append(nnModel.activate(node_value))

        return output

    def random_weights(self):
        total = sum(
            self.layers[i] * self.layers[i + 1]
            for i in range(len(self.layers) - 1)
        )

        return [random.uniform(-1, 1) for _ in range(total)]

    def forward_with_cache(self, input_vec):
        activations = [input_vec]
        weighted_sums = []

        wid = 0
        current = input_vec

        for layer_index in range(len(self.layers) - 1):
            in_size = self.layers[layer_index]
            out_size = self.layers[layer_index + 1]

            layer_weights = self.weights[wid:wid + in_size * out_size]
            wid += in_size * out_size

            next_layer = []
            sums = []
            k = 0

            for _ in range(out_size):
                z = 0

                for i in range(in_size):
                    z += current[i] * layer_weights[k]
                    k += 1

                sums.append(z)
                next_layer.append(self.activate(z))

            weighted_sums.append(sums)
            activations.append(next_layer)
            current = next_layer

        return activations, weighted_sums

    def train(self, training_data, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0

            for sample in training_data:
                x = sample["input"]
                target = sample["output"]

                activations, _ = self.forward_with_cache(x)
                prediction = activations[-1]

                total_loss += sum(
                    (target[i] - prediction[i]) ** 2
                    for i in range(self.output_size)
                )

                deltas = [None] * (len(self.layers) - 1)

                # Output layer delta
                deltas[-1] = [
                    (prediction[i] - target[i]) *
                    self.sigmoid_derivative_from_output(prediction[i])
                    for i in range(self.output_size)
                ]

                # Hidden layer deltas
                for layer in reversed(range(len(deltas) - 1)):
                    current_size = self.layers[layer + 1]
                    next_size = self.layers[layer + 2]

                    next_weights_start = sum(
                        self.layers[i] * self.layers[i + 1]
                        for i in range(layer + 1)
                    )

                    delta = []

                    for i in range(current_size):
                        error = 0

                        for j in range(next_size):
                            w_index = next_weights_start + j * current_size + i
                            error += deltas[layer + 1][j] * self.weights[w_index]

                        a = activations[layer + 1][i]
                        delta.append(
                            error * self.sigmoid_derivative_from_output(a)
                        )

                    deltas[layer] = delta

                # Update weights
                wid = 0

                for layer in range(len(self.layers) - 1):
                    in_size = self.layers[layer]
                    out_size = self.layers[layer + 1]

                    for j in range(out_size):
                        for i in range(in_size):
                            grad = activations[layer][i] * deltas[layer][j]
                            self.weights[wid] -= learning_rate * grad
                            wid += 1

            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.6f}")

    def save_weights(self, filename="weights_trained.txt"):
        with open(filename, "w") as f:
            f.write(str(self.weights))

    @staticmethod
    def load_training_data(filename="training_data.json"):
        with open(filename, "r") as f:
            return json.load(f)