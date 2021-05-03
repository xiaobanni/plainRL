"""
@Project : Imitation-Learning
@File    : model.py
@Author  : XiaoBanni
@Date    : 2021-04-28 21:54
@Desc    : 
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Multilayer Perceptron
        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """

        :param x: Input Layer
        :return: Output Layer
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == '__main__':
    mlp = MLP(input_dim=16, output_dim=4)
    print(mlp.parameters)
