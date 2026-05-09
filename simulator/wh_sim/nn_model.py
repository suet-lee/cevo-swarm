import json
import torch
import torch.nn as nn
import torch.optim as optim
import ast

class nnModel(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_nodes_per_layer):

        super().__init__()

        layers = []

        prev = input_size

        for hidden_size in hidden_nodes_per_layer:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.Tanh())
            prev = hidden_size

        layers.append(nn.Linear(prev, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        return self.net(x)

    def compute(self, input_vec):

        with torch.no_grad():
            output = self.forward(input_vec)

        return output.numpy().tolist()

    @staticmethod
    def normalize_input(x):

        return [
            x[0],
            x[1] / 1000.0,
            x[2] / 10.0,
            x[3] / 50.0,
            x[4]
        ]

    def train_model(self,
                    training_data,
                    epochs=200,
                    learning_rate=0.001):

        X = torch.tensor(
            [nnModel.normalize_input(sample["input"]) for sample in training_data],
            dtype=torch.float32
        )

        Y = torch.tensor(
            [sample["output"] for sample in training_data],
            dtype=torch.float32
        )

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

        for epoch in range(epochs):

            optimizer.zero_grad()

            predictions = self.forward(X)

            loss = criterion(predictions, Y)

            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss = {loss.item():.6f}")

    def save_weights_txt(self, filename="weights0.txt"):

        weights = []

        for param in self.parameters():
            weights.extend(
                param.detach().numpy().flatten().tolist()
            )

        with open(filename, "w") as f:
            f.write(str(weights))

        print(f"Weights saved to {filename}")

    def set_weights(self, weights):

        idx = 0

        for param in self.parameters():
            shape = param.data.shape
            size = param.numel()

            values = weights[idx:idx + size]

            tensor = torch.tensor(
                values,
                dtype=torch.float32
            ).view(shape)

            param.data = tensor

            idx += size

    def get_weights(self):
        weights = []

        for param in self.parameters():
            weights.extend(
                param.detach().cpu().numpy().flatten().tolist()
            )

        return weights

    def load_weights_txt(self, filename="weights0.txt"):

        with open(filename, "r") as f:
            weights = ast.literal_eval(f.read())

        idx = 0

        for param in self.parameters():
            shape = param.data.shape

            size = param.numel()

            values = weights[idx:idx + size]

            tensor = torch.tensor(values, dtype=torch.float32)
            tensor = tensor.view(shape)

            param.data = tensor

            idx += size

        print(f"Weights loaded from {filename}")

        print(f"Model loaded from {filename}")

    @staticmethod
    def load_training_data(filename="training_data.json"):

        with open(filename, "r") as f:
            return json.load(f)