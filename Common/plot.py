import matplotlib.pyplot as plt
import seaborn as sns


def plot_rewards(rewards, smooth_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./',
                 xlabel_name="epsiodes"):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo, env))
    plt.xlabel(xlabel_name)
    plt.plot(rewards, label='rewards')
    plt.plot(smooth_rewards, label='smooth rewards')
    plt.legend()
    if save:
        plt.savefig(path + "rewards_curve_{}".format(tag))
    plt.show()
