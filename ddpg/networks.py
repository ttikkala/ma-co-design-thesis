import torch
from torch import nn

""" 
    This script contains some basic networks for Q- and policy-functions.
"""


class QvalueNetwork(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        input_size,
        init_w=3e-3,
        activation=torch.nn.functional.relu
    ):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size

        self.activation = activation

        inp_size = input_size
        self._layers = []
        for i, out_size in enumerate(self.hidden_sizes):
            layer = nn.Linear(inp_size, out_size)
            self._layers.append(layer)
            self.__setattr__("fc{}".format(i), layer)
            inp_size = out_size

        self.last_layer = nn.Linear(inp_size, 1)

    def forward(self, state, action):
        input = torch.cat((state, action), dim=-1)

        for layer in self._layers:
            output = layer(input)
            output = self.activation(output)
            input = output

        output = self.last_layer(input)

        return output


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        activation=torch.nn.functional.relu
    ):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.input_size = input_size

        self.activation = activation

        inp_size = input_size
        self._layers = []
        for i, out_size in enumerate(self.hidden_sizes):
            layer = nn.Linear(inp_size, out_size)
            self._layers.append(layer)
            self.__setattr__("fc{}".format(i), layer)
            inp_size = out_size

        self.last_layer = nn.Linear(inp_size, output_size)

    def forward(self, state):
        input = state

        for layer in self._layers:
            output = layer(input)
            output = self.activation(output)
            input = output

        output = self.last_layer(input)
        output = torch.tanh(output)
        return output
