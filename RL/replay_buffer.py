"""
@Project : Imitation-Learning
@File    : replay_buffer.py
@Author  : XiaoBanni
@Date    : 2021-04-28 22:50
@Desc    :  Replay Buffer
"""

import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """
        RB = ReplayBuffer()
        RB_len = len(RB)
        :return:
        """
        return len(self.buffer)
