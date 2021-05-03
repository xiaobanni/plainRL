"""
@Project : Imitation-Learning
@File    : model
@Author  : XiaoBanni
@Date    : 2021-04-30 22:57
@Desc    : 
"""

from torch import nn
from torch.distributions import Categorical


class AdvantageActorCriticAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(AdvantageActorCriticAgent, self).__init__()
        # Strategy pi, giving the state distribution of a_t under s_t
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # >>> a=torch.Tensor([[3,2,1],[2,4,6]])
            # >>> torch.nn.functional.softmax(a,dim=0)
            # tensor([[0.7311, 0.1192, 0.0067],
            #         [0.2689, 0.8808, 0.9933]])
            # >>> torch.nn.functional.softmax(a,dim=1)
            # tensor([[0.6652, 0.2447, 0.0900],
            #         [0.0159, 0.1173, 0.8668]])
            nn.Softmax(dim=1)
        )
        # Value network v(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        probs = self.actor(x)  # pi(a|s)
        value = self.critic(x)  # V(s)
        dist = Categorical(probs)  # Distribution
        return dist, value
