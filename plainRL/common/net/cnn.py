import torch as th
from torch import nn
from typing import Tuple


class NatureAtariNet(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param image_shape: Shape of the image input (C, H, W).
    :param features_dim: Number of features extracted.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (4, 84, 84),
        features_dim: int = 512,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 32,
                      kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.zeros(1, *image_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, action_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
