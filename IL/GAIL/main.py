"""
@Project : Imitation-Learning
@File    : main.py
@Author  : XiaoBanni
@Date    : 2021-05-02 18:32
@Desc    : The implementation structure is similar to A2C.main.py, 
            you can get more annotations from A2C.main.py
"""
import os
import datetime
import gym
import numpy as np
import torch
from Common.utils import get_env_version, get_env_information
from model import expert_reward
from agent import GAIL
from Common.make_envs import get_envs
from Common.plot import plot_rewards
from ppo import ppo_update

curr_dir = os.path.dirname(__file__)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def env_test(env_name, model, device="cuda", vis=False):
    env = gym.make(env_name)
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        if vis:
            env.render()
        total_reward += reward
    env.close()
    return total_reward


class GAILConfig:
    def __init__(self, env="Pendulum-v0", train_frames=100000):
        self.algo = "GAIL"
        self.env = env
        self.nums_envs = 16
        self.result_path = curr_dir + os.sep + "results" + os.sep \
                           + self.env + os.sep + curr_time + os.sep
        self.a2c_hidden_dim = 256
        self.discriminator_hidden_dim = 128
        self.gamma = 0.99
        self.tau = 0.95  # Exponential weighted value in GAE loss
        self.lr = 3e-3
        self.train_frames = train_frames
        self.gae_steps = 20
        self.mini_batch_size = 5
        self.ppo_epochs = 4
        self.ppo_update_frequency = 3
        self.threshold_reward = -200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg, envs, agent):
    print("=====Load expert date!=====")
    try:
        expert_trajectory = np.load("expert_traj.npy")
    except FileNotFoundError:
        print("No expert data found.")
        assert False
    print("=====Start training!=====")
    state = envs.reset()  # A list of nums_envs * env_state_dimension size
    next_state = None
    frame_idx = 0
    ppo_update_count = 0
    test_rewards = []
    smooth_test_rewards = []
    early_stop = False
    while frame_idx < cfg.train_frames and not early_stop:
        ppo_update_count += 1
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(cfg.gae_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            # value is given by critic
            dist, value = agent.model(state)
            action = dist.sample()
            # GAIL needs to allow the agent to interact with the environment,
            # but cannot get rewards from the environment.
            # Instead, get rewards from Discriminator
            next_state, _, done, _ = envs.step(action.cpu().numpy())
            reward = expert_reward(cfg, agent, state, action.cpu().numpy())

            # Normal.log_prob
            # log_prob(value) is the logarithm of the probability of
            # calculating value in the defined normal distribution (mu, std)

            # >>> a=Normal(10.0,1.0)
            # >>> b=a.sample()
            # >>> b
            # tensor(9.0796)
            # >>> a.log_prob(b)
            # tensor(-1.3425)

            # In a normal distribution,
            # the greater the variance, the greater the entropy
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if frame_idx % 200 == 0:
                test_reward = np.mean([env_test(cfg.env, agent.model) for _ in range(10)])
                test_rewards.append(test_reward)
                if smooth_test_rewards:
                    smooth_test_rewards.append(0.9 * smooth_test_rewards[-1] + 0.1 * test_rewards[-1])
                else:
                    smooth_test_rewards.append(test_rewards[-1])
                print('Frame_idx:{}/{}, Reword:{}'.format(frame_idx, cfg.train_frames, test_rewards[-1]))
                if test_reward > cfg.threshold_reward:
                    early_stop = True

        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = agent.model(next_state)
        # values are given by critic
        # rewards are given by discriminator
        returns = agent.compute_gae(cfg, next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        if ppo_update_count % cfg.ppo_update_frequency == 0:
            # PPO improves the Actor part,
            # which is to improve the strategy parameter update
            ppo_update(agent, cfg, states, actions, log_probs, returns, advantage)

        # Training discriminator

        # np.random.randint(start,end,sample_total_num)
        expert_state_action = expert_trajectory[
                              np.random.randint(0, expert_trajectory.shape[0], 2 * cfg.gae_steps * cfg.nums_envs), :]
        expert_state_action = torch.FloatTensor(expert_state_action).to(cfg.device)
        state_action = torch.cat([states, actions], 1)
        fake = agent.discriminator(state_action)
        real = agent.discriminator(expert_state_action)
        agent.optimizer_discriminator.zero_grad()
        agent.discriminator_loss = \
            agent.discriminator_criterion(fake, torch.ones((states.shape[0], 1)).to(cfg.device)) + \
            agent.discriminator_criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(cfg.device))
        agent.discriminator_loss.backward()
        agent.optimizer_discriminator.step()

    env_test(cfg.env, agent.model, vis=True)
    return test_rewards, smooth_test_rewards


def main():
    get_env_version()
    cfg = GAILConfig(env="Pendulum-v0")
    get_env_information(cfg.env)
    env = gym.make(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = GAIL(state_dim, action_dim, cfg)
    envs = get_envs(cfg.env, cfg.nums_envs)
    rewards, smooth_rewards = train(cfg, envs, agent)
    os.makedirs(cfg.result_path)
    plot_rewards(rewards, smooth_rewards, env=cfg.env, algo=cfg.algo, save=True, path=cfg.result_path,
                 xlabel_name="Each 200 steps")
    envs.close()
    env.close()


if __name__ == '__main__':
    main()
