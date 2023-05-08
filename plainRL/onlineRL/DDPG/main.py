"""
@Project : Imitation-Learning-and-Reinforcement-Learning
@File    : main.py
@Author  : XiaoBanni
@Date    : 2021-07-04 17:45
@Desc    : 
"""

import os
import datetime
import gymnasium as gym
import torch
import numpy as np
from Common.utils import save_results, get_env_version, get_env_information, get_smooth_rewards
from Common.plot import plot_rewards
from RL.DDPG.agent import DDPG
from RL.exploration import OUNoise

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

curr_dir = os.path.dirname(__file__)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        # .clip(): Equivalent to but faster than ``np.minimum(a_max, np.maximum(a, a_min))``.
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class DDPGConfig:
    def __init__(self, env="Pendulum-v0", train_eps=100):
        self.algo = "DDPG"
        self.env = env
        self.result_path = curr_dir + os.sep + "results" + os.sep \
            + self.env + os.sep + curr_time + os.sep
        self.value_lr = 1e-3
        self.policy_lr = 1e-4
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.state_dim = None
        self.action_dim = None
        self.batch_size = 128
        self.hidden_dim = 256
        self.train_eps = train_eps
        self.replay_buffer_size = 1000000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def set_dim(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim


def train(cfg, env, agent):
    ou_noise = OUNoise(env.action_space)
    print("=====Start training!=====")
    rewards = []
    frame_id = 0
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            frame_id += 1
            env.render()
            action = agent.policy_net.get_action(state, cfg.device)
            action = ou_noise.get_action(action, frame_id)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # print(action.shape, reward.shape)
            # (1,)()
            agent.replay_buffer.push(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > cfg.batch_size:
                agent.update(cfg)
            state = next_state
            if done:
                print('Episode:{}/{}, Reword:{}'.format(i_episode +
                      1, cfg.train_eps, total_reward))
                rewards.append(total_reward)
    print("=====Finish training!=====")
    return rewards, get_smooth_rewards(rewards, smooth_rate=0.9)


def main():
    get_env_version()
    cfg = DDPGConfig(env="Pendulum-v0", train_eps=100)
    get_env_information(cfg.env)
    env = NormalizedActions(gym.make(cfg.env))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cfg.set_dim(state_dim, action_dim)
    agent = DDPG(cfg)
    rewards, smooth_rewards = train(cfg, env, agent)
    os.makedirs(cfg.result_path)
    save_results(rewards, smooth_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, smooth_rewards, tag='train',
                 env=cfg.env, algo=cfg.algo, path=cfg.result_path)


if __name__ == '__main__':
    main()
