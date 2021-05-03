"""
@Project ：Imitation-Learning
@File    ：agent.py
@Author  ：XiaoBanni
@Date    ：2021-04-28 21:06
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from model import MLP
from replay_buffer import ReplayBuffer
from Common.utils import to_tensor_float


class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        """

        :param state_dim: About Task
        :param action_dim: About Task
        :param cfg: Config, About DQN setting
        """
        self.device = cfg.device
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.frame_idx = 0  # Decay count for epsilon
        self.epsilon = lambda frame_idx: \
            cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.q_value_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_value_net.parameters(), lr=cfg.lr)
        self.loss = 0
        self.replay_buffer = ReplayBuffer(cfg.capacity)

    def choose_action(self, state):
        # Select actions using e—greedy principle
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                # Although Q(s,a) is written in the pseudocode of the original paper,
                # it is actually the value of Q(s) output |A| dimension
                state = torch.tensor([state], device=self.device, dtype=torch.float)
                q_value = self.q_value_net(state)
                # output = torch.max(input, dim)
                # dim is the dimension 0/1 of the max function index,
                # 0 is the maximum value of each column,
                # 1 is the maximum value of each row
                # The function will return two tensors,
                # the first tensor is the maximum value of each row;
                # the second tensor is the index of the maximum value of each row.

                # .item(): only one element tensors can be converted to Python scalars
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Randomly sample transitions from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        state_batch = to_tensor_float(state_batch)
        # tensor([1, 2, 3, 4]).unsqueeze(1)  -> tensor([[1],[2],[3],[4]]) 
        action_batch = torch.tensor(
            action_batch, device=self.device).unsqueeze(1)
        reward_batch = to_tensor_float(reward_batch)
        next_state_batch = to_tensor_float(next_state_batch)
        done_batch = to_tensor_float(done_batch)

        # Calculate Q(s,a) at time t
        # q_t=Q(s_t,a_t)

        # Use index to index the value of a specific position in a dimension
        # a=torch.Tensor([[1,2],[3,4]]),
        # a.gather(1,torch.LongTensor([[0],[1]]))=torch.Tensor([[1],[4]])

        # index action_batch is obtained from the replay buffer
        q_value = self.q_value_net(state_batch).gather(
            dim=1, index=action_batch)  # shape: [32,1]
        # Calculate Q(s,a) at time t+1
        # q_{t+1}=max_a Q(s_t+1,a)

        # .detach():
        # Return a new Variable, which is separated from the current calculation graph,
        # but still points to the storage location of the original variable.
        # The difference is that requires grad is false.
        # The obtained Variable never needs to calculate its gradient and does not have grad.
        #
        # Even if it re-sets its requirements grad to true later,
        # it will not have a gradient grad
        next_q_value = self.target_net(next_state_batch).max(1)[0].detach()
        # For the termination state, the corresponding expected_q_value is equal to reward
        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)  # shape: 32
        # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        # reduce = False，return loss in vector form
        # reduce = True， return loss in scalar form
        # size_average = True，return loss.mean()
        # size_average = False，return loss.sum()
        self.loss = nn.MSELoss()(q_value, expected_q_value.unsqueeze(1))
        # Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        self.optimizer.zero_grad()
        self.loss.backward()
        # Performs a single optimization step (parameter update).
        self.optimizer.step()

    def save(self, path):
        # Returns a dictionary containing a whole state of the module.
        # Both parameters and persistent buffers (e.g. running averages) are included.
        # Keys are corresponding parameter and buffer names.
        torch.save(self.target_net.state_dict(), path + "dqn_checkpoint.pth")

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + "dqn_checkpoint.pth"))
