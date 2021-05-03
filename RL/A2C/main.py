"""
@Project : Imitation-Learning
@File    : main
@Author  : XiaoBanni
@Date    : 2021-05-01 17:54
@Desc    :
        Pay Attention!!!
        If the error is that the page size is insufficient,
        it may be due to insufficient memory space.
        Please try to clear the memory space
        or reduce the number of parallel environments
"""

import os
import datetime
import gym
import numpy as np
import torch
from agent import A2C
from Common.utils import get_env_version, get_env_information
from Common.plot import plot_rewards
from Common.make_envs import get_envs

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


class A2CConfig:
    def __init__(self, env="CartPole-v0", train_frames=60000):
        self.algo = "A2C"
        self.env = env
        self.result_path = curr_dir + os.sep + "results" + os.sep \
                           + self.env + os.sep + curr_time + os.sep
        self.gamma = 0.99
        self.lr = 3e-4
        self.train_frames = train_frames
        self.TD_step_length = 5
        self.hidden_dim = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg, envs, agent):
    print("=====Start training!=====")
    # Get state from n environments
    n_state = envs.reset()
    n_next_state = None
    frame_idx = 0
    test_rewards = []
    smooth_test_rewards = []

    while frame_idx < cfg.train_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []  # done=True -> 0 , done=False -> 1
        entropy = 0  # Entropy regular term, entropy can do more exploration

        for _ in range(cfg.TD_step_length):
            n_state = torch.FloatTensor(n_state).to(cfg.device)
            n_dist, n_value = agent.model(n_state)  # Call function: forward, pi(a|s), V(s) 
            n_action = n_dist.sample()

            # .cpu()
            # The data processing equipment from other equipment (such as .cuda() to the cpu)
            # will not change the variable type,
            # and it will still be a Tensor variable after conversion.

            # Each return value is a list

            # When the `done = True` of a certain environment,
            # there will be a function to call the `reset()` of the environment
            n_next_state, n_reward, n_done, _ = envs.step(n_action.cpu().numpy())

            # .log_prob()
            # takes the log_e of the probability (of some actions).

            # >>> action_logits = torch.rand(5)
            # >>> action_probs = F.softmax(action_logits, dim=-1)
            # >>> action_probs
            # tensor([0.1974, 0.1992, 0.1940, 0.1799, 0.2295])
            # >>> dist = Categorical(action_probs)
            # >>> action = dist.sample()
            # >>> action
            # tensor(2)
            # >>> print(dist.log_prob(action), torch.log(action_probs[action]))
            # tensor(-1.6401) tensor(-1.6401)
            n_log_prob = n_dist.log_prob(n_action)

            # entropy(): \sum -p*log(p)
            # >>> dist.entropy()
            # tensor(1.6062)
            entropy += n_dist.entropy().mean()
            log_probs.append(n_log_prob)
            values.append(n_value)
            rewards.append(torch.FloatTensor(n_reward).unsqueeze(1).to(cfg.device))
            # type(n_done): <class 'numpy.ndarray'>
            masks.append(torch.FloatTensor(1 - n_done).unsqueeze(1).to(cfg.device))

            n_state = n_next_state  # Just for clarity
            frame_idx += 1

            # test
            if frame_idx % 200 == 0:
                test_rewards.append(np.mean([env_test(cfg.env, agent.model) for _ in range(10)]))
                if smooth_test_rewards:
                    smooth_test_rewards.append(0.9 * smooth_test_rewards[-1] + 0.1 * test_rewards[-1])
                else:
                    smooth_test_rewards.append(test_rewards[-1])
                print('Frame_idx:{}/{}, Reword:{}'.format(frame_idx, cfg.train_frames, test_rewards[-1]))

        n_next_state = torch.FloatTensor(n_next_state).to(cfg.device)
        _, n_next_value = agent.model(n_next_state)  # V(s_{t+TD_step_length})
        returns = agent.compute_returns(n_next_value, rewards, masks)  # r_t+gamma_(v_{t+1})

        # .cat():
        # torch.cat(tensors, dim=0, *, out=None) → Tensor
        # Concatenates the given sequence of seq tensors in the given dimension.
        # All tensors must either have the same shape (except in the concatenating dimension)
        # or be empty.

        # From five 16-dimensional tensors to one 80-dimensional tensor
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()  # Does not participate in gradient operations
        values = torch.cat(values)
        # baseline: Q(s,a) - V(s)
        advantage = returns - values

        # loss = - E(V(s)) = - sum{ pi(a|s)*Q(s,a) }
        actor_loss = -(log_probs * advantage.detach()).mean()
        # loss = 1/2[v(t)-y_t]^2
        critic_loss = advantage.pow(2).mean()
        agent.loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        agent.optimizer.zero_grad()
        agent.loss.backward()
        agent.optimizer.step()

    env_test(cfg.env, agent.model, vis=True)
    return test_rewards, smooth_test_rewards


def main():
    get_env_version()
    cfg = A2CConfig(env="CartPole-v0", train_frames=400)
    get_env_information(cfg.env)
    env = gym.make(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2C(state_dim, action_dim, cfg)
    envs = get_envs(env_name=cfg.env)
    rewards, smooth_rewards = train(cfg, envs, agent)
    os.makedirs(cfg.result_path)
    # In fact, a step/frame contains nums_envs environmental interactions
    plot_rewards(rewards, smooth_rewards, env=cfg.env, algo=cfg.algo, save=True, path=cfg.result_path,
                 xlabel_name="Each 200 steps")
    envs.close()
    env.close()


if __name__ == '__main__':
    main()
