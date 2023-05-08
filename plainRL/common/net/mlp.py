import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_sizes: Sequence[int] = [128, 128],
                 activation: nn.Module = nn.ReLU,
                 ):
        """
        Multilayer Perceptron
        :param input_dim: dimension of the input vector.
        :param output_dim: dimension of the output vector.
        :param hidden_sizes: shape of MLP passed in as a list, not including input_dim and output_dim.
        :param activation: which activation to use after each layer, can be both the same activation for all layers if passed in nn.Module, or different activation for different Modules if passed in a list. Default to nn.ReLU.
        :param linear_layer: use this module as linear layer. Default to nn.Linear.
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_dim, hidden_size))
            prev_dim = hidden_size
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = activation()

    def forward(self, x):
        """
        :param x: Input Layer
        :return: Output Layer
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


if __name__ == '__main__':
    mlp = MLP(input_dim=16, output_dim=4)
    print(mlp.parameters)
