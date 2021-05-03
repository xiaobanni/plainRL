"""
@Project : Imitation-Learning
@File    : agent
@Author  : XiaoBanni
@Date    : 2021-05-02 18:50
@Desc    : 
"""

from torch import nn, optim
from model import AdvantageActorCritic, Discriminator


class GAIL:
    def __init__(self, input_dim, output_dim, cfg):
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.model = AdvantageActorCritic(input_dim, output_dim, cfg.a2c_hidden_dim).to(cfg.device)
        self.discriminator = Discriminator(input_dim + output_dim, cfg.discriminator_hidden_dim).to(cfg.device)
        # torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
        # Creates a criterion that measures the Binary Cross Entropy between the target and the output
        self.discriminator_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=cfg.lr)
        self.loss = 0
        self.discriminator_loss = 0

    def compute_gae(self, cfg, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + cfg.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + cfg.gamma * cfg.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
