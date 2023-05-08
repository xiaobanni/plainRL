"""
@Project : Imitation-Learning
@File    : agent
@Author  : XiaoBanni
@Date    : 2021-04-30 22:52
@Desc    : 
"""

from model import AdvantageActorCriticAgent
from torch import optim


class A2C:
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.model = AdvantageActorCriticAgent(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss = 0

    def compute_returns(self, next_value, rewards, masks):
        """
        This function calculates the state value function V(s_t)
        after n-step bootstrapping
        For more information: https://zhuanlan.zhihu.com/p/29486661

        Value network:
        V(s_t)=E(r_t+gamma*V(s_{t+1}))
        In one step TD process:
        y_t=r_t+gamma*V(s_{t+1})
        Here we use multi-step TD, the step length is TD step length.

        At the same time, we need to note that each next input is a list containing n tensors,
        and n is the number of environments that are executed simultaneously

        :param next_value: V(s_{t+TD_step_length})
        :param rewards: The real rewards obtained during the n-step bootstrapping process
        :param masks: Whether the state of the environment is done, done=True -> 0 , done=False -> 1
        :return: the state value function V(s_t) after n-step bootstrapping,
                Note that the value of each step of the bootstrapping is recorded,
                and the order of the elements in the list is
                from 1 step bootstrapping to TD step length step bootstrapping
        """
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def update(self):
        """
        On policy update, Implemented in the train() function
        :return:
        """
        pass
