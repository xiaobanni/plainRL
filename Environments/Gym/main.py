"""
@Project : Imitation-Learning
@File    : main.py
@Author  : XiaoBanni
@Date    : 2021-04-28 21:32
@Desc    : A brief introduction to Gym, for more https://gym.openai.com/docs/
"""

import gym

from gym import envs


def get_all_env():
    print(envs.registry.all())


def run():
    env = gym.make('CartPole-v0')
    # env = gym.make('MountainCar-v0')
    i = 0
    for i_episode in range(20):
        total_reward = 0
        # WARN:
        # You are calling 'step()' even though this environment has already returned done = True.
        # You should always call 'reset()' once you receive 'done = True' --
        # any further steps are undefined behavior.

        # So if you get 'done = True' without manually 'reset()',
        # you will fall into an infinite loop
        env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            i += 1
            # print(i, observation, reward, done, info)
            total_reward += reward
            if done:
                print("Episode finished after {} timesteps with reward {}".format(t + 1, total_reward))
                break
    env.close()


if __name__ == '__main__':
    get_all_env()
    run()
