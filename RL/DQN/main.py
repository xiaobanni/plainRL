import os
import datetime
import gym
import torch
from agent import DQN
from Common.utils import save_results, get_env_version, get_env_information
from Common.plot import plot_rewards

curr_dir = os.path.dirname(__file__)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class DQNConfig:
    def __init__(self, env="CartPole-v0", train_eps=300):
        self.algo = "DQN"
        self.env = env
        self.result_path = curr_dir + os.sep + "results" + os.sep \
                           + self.env + os.sep + curr_time + os.sep
        self.gamma = 0.95
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001
        self.capacity = 10000  # Replay buffer capacity
        self.batch_size = 32
        self.train_eps = train_eps  # Number of episodes trained
        self.target_update = 2
        self.test_eps = 20  # Number of episodes tested
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 256


def train(cfg, env, agent):
    print("=====Start training!=====")
    rewards = []
    smooth_rewards = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        total_reword = 0
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reword += reward
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.q_value_net.state_dict())
        print('Episode:{}/{}, Reword:{}'.format(i_episode + 1, cfg.train_eps, total_reword))
        rewards.append(total_reword)
        # Smooth rewards curve
        if smooth_rewards:
            smooth_rewards.append(0.9 * smooth_rewards[-1] + 0.1 * total_reword)
        else:
            smooth_rewards.append(total_reword)
    print("=====Finish training!=====")
    return rewards, smooth_rewards


def main():
    get_env_version()
    cfg = DQNConfig(env="CartPole-v0", train_eps=100)
    # cfg = DQNConfig(env="MountainCar-v0", train_eps=1000)
    get_env_information(env_name=cfg.env)
    env = gym.make(cfg.env)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, cfg)
    rewards, smooth_rewards = train(cfg, env, agent)
    os.makedirs(cfg.result_path)
    agent.save(path=cfg.result_path)
    save_results(rewards, smooth_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, smooth_rewards, tag='train', env=cfg.env, algo=cfg.algo, path=cfg.result_path)


if __name__ == '__main__':
    main()
