"""
@Project : Imitation-Learning-and-Reinforcement-Learning
@File    : agent.py
@Author  : XiaoBanni
@Date    : 2021-07-04 23:13
@Desc    : 
"""

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from model import ValueNetwork, PolicyNetwork
from RL.replay_buffer import ReplayBuffer


class DDPG:
    def __init__(self, cfg):
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        self.value_net = ValueNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_policy_net = PolicyNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.soft_tau = cfg.soft_tau

        self.value_lr = cfg.value_lr
        self.policy_lr = cfg.policy_lr
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        # mean squared error
        self.value_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    def update(self, cfg):
        state, action, reward, next_state, done = self.replay_buffer.sample(cfg.batch_size)
        # print(np.shape(state), np.shape(action), np.shape(reward), np.shape(next_state), np.shape(done))
        # (128, 3) (128, 1) (128,) (128, 3) (128,)
        state = torch.FloatTensor(state).to(cfg.device)
        action = torch.FloatTensor(action).to(cfg.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(cfg.device)
        next_state = torch.FloatTensor(
            next_state).to(cfg.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(cfg.device)

        self.value_net(state, self.policy_net(state))

        # Actor Loss
        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        TD_target = reward + (1.0 - done) * self.gamma * target_value
        TD_target = torch.clamp(TD_target, -np.inf, np.inf)

        value = self.value_net(state, action)
        # Critic Loss
        value_loss = self.value_criterion(value, TD_target.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update target network
        for target_param, param in zip(self.target_value_net.parameters(),
                                       self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(),
                                       self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
