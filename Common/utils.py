"""
@Project : Imitation-Learning
@File    : Utils
@Author  : XiaoBanni
@Date    : 2021-04-29 14:37
@Desc    : 
"""

import os
import sys
import torch
import numpy as np
import gym


def to_tensor_float(data, device):
    data = torch.tensor(
        data, device=device, dtype=torch.float)
    return data


def save_results(rewards, smooth_rewards, tag='train', path='.' + os.sep + 'results' + os.sep):
    np.save(path + 'rewards_' + tag + '.npy', rewards)
    np.save(path + 'smooth_rewards_' + tag + '.npy', smooth_rewards)
    print("=====Saved results!=====")


def get_env_version():
    """
    Output version information of main python package
    :return:
    """
    print("sys.version        = " + sys.version)
    print("gym.__version__    = " + gym.__version__)
    print("torch.__version    =" + torch.__version__)
    print("torch.version.cuda =" + torch.version.cuda)


def get_env_information(env_name="CartPole-v0"):
    env = gym.make(env_name)
    print("env_name = {}".format(env_name))
    print('Observation space = {}'.format(env.observation_space))
    print('Action space = {}'.format(env.action_space))


def get_smooth_rewards(rewards, smooth_rate=0.9):
    smooth_rewards = []
    for i in rewards:
        if smooth_rewards:
            smooth_rewards.append(smooth_rate * smooth_rewards[-1] + 0.1 * i)
        else:
            smooth_rewards.append(i)
    return smooth_rewards
