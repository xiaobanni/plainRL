"""
@Project : Imitation-Learning
@File    : make_envs
@Author  : XiaoBanni
@Date    : 2021-05-01 22:08
@Desc    : Create multiple environments for synchronized A2C training
"""

from plainRL.common.multiprocessing_env import SubprocVecEnv
import gymnasium as gym


def make_env(env_name):
    """
    You can use make_env()() to use function _thunk()
    :return:
    """

    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def get_envs(env_name, nums_envs=16):
    """

    :param env_name:
    :param nums_envs: Number of environments used for synchronization
    :return:
    """
    envs = [make_env(env_name) for i in range(nums_envs)]
    envs = SubprocVecEnv(envs)
    return envs


if __name__ == '__main__':
    envs = get_envs("Pendulum-v0")
    states = envs.reset()
    print(envs.observation_space.shape)
    print(envs.action_space.shape)
    print(states)
