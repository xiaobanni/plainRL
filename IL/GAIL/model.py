"""
@Project : Imitation-Learning
@File    : model
@Author  : XiaoBanni
@Date    : 2021-05-02 18:38
@Desc    : 
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        # The weight is initialized to a normal distribution
        nn.init.normal_(m.weight, mean=0., std=0.1)
        # The bias is initialized to a constant
        nn.init.constant_(m.bias, 0.1)


def expert_reward(cfg, agent, state, action):
    """
    The discriminator scores state-action pairs
    :param cfg:
    :param agent:
    :param state:
    :param action:
    :return:
    """
    # .cpu()
    # Returns a copy of this object in CPU memory.

    # numpy cannot read CUDA tensor and
    # needs to convert it to CPU tensor
    # >>> a=torch.Tensor([1,2,3,4]).to("cuda")
    # >>> a.numpy()
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # TypeError: can't convert cuda:0 device type tensor to numpy. Use
    #   Tensor.cpu() to copy the tensor to host memory first.
    state = state.cpu().numpy()
    state_action = torch.FloatTensor(np.concatenate([state, action], 1)).to(cfg.device)
    return -np.log(agent.discriminator(state_action).cpu().data.numpy())


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # >>> m=nn.Linear(10,1)
        # >>> m
        # Linear(in_features=10, out_features=1, bias=True)
        # >>> m.weight
        # Parameter containing:
        # tensor([[ 0.0154,  0.2331, -0.1645, -0.1201, -0.0739,  0.0450,  0.0548, -0.2106,
        #          -0.0479,  0.0109]], requires_grad=True)
        # >>> m.bias
        # Parameter containing:
        # tensor([-0.2276], requires_grad=True)
        # >>> m.weight.data.mul_(0.1)
        # tensor([[ 0.0015,  0.0233, -0.0164, -0.0120, -0.0074,  0.0045,  0.0055, -0.0211,
        #          -0.0048,  0.0011]])
        # >>> m.weight.data.mul_(0.1)
        # tensor([[ 0.0002,  0.0023, -0.0016, -0.0012, -0.0007,  0.0004,  0.0005, -0.0021,
        #          -0.0005,  0.0001]])
        # >>> m.bias.data.mul_(0)
        # tensor([-0.])
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        prob = torch.sigmoid(self.linear3(x))
        return prob


class AdvantageActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, std=0.0):
        super(AdvantageActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # torch.nn.Parameter():
        # A kind of Tensor that is to be considered a module parameter.
        #
        # Parameters are Tensor subclasses,
        # that have a very special property when used with Module s -
        # when they’re assigned as Module attributes
        # they are automatically added to the list of its parameters,
        # and will appear e.g. in parameters() iterator.
        # Assigning a Tensor doesn’t have such effect.
        # This is because one might want to cache some temporary state,
        # like last hidden state of the RNN, in the model.
        # If there was no such class as Parameter,
        # these temporaries would get registered too.
        #
        # >>> torch.Tensor([1,2,3]).requires_grad
        # False
        # >>> torch.nn.Parameter(torch.Tensor([1,2,3])).requires_grad
        # True

        # torch.ones():
        # Returns a tensor filled with the scalar value 1,
        # with the shape defined by the variable argument size.
        self.log_std = nn.Parameter(torch.ones(1, output_dim) * std)
        # .apply(fn)
        # Applies fn recursively to every submodule (as returned by .children()) as well as self.
        # Typical use includes initializing the parameters of a model (see also torch.nn.init).
        self.apply(init_weights)

    def forward(self, x):
        mu = self.actor(x)  # Mean
        std = self.log_std.exp().expand_as(mu)  # Standard deviation
        value = self.critic(x)
        # .expand_as()
        # Expand this tensor to the same size as other. 
        # self.expand_as(other) is equivalent to self.expand(other.size()).
        # torch.distributions.normal.Normal(loc, scale, validate_args=None)
        # Creates a normal (also called Gaussian) distribution parameterized by loc and scale.
        # >>> m = Normal(10.0, 1.0)
        # >>> [m.sample() for _ in range(5)]
        # [tensor(10.1613), tensor(11.9001), tensor(9.9001), tensor(10.3850), tensor(8.3563)]
        # >>> m = Normal(10.0, 10.0)
        # >>> [m.sample() for _ in range(5)]
        # [tensor(31.3052), tensor(3.3538), tensor(14.1421), tensor(10.1337), tensor(11.1297)]

        # >>> m = Normal(torch.Tensor([10.0,20.0]), torch.Tensor([1.0,10.0]))
        # >>> m.sample()
        # tensor([10.0209, 32.5655])
        dist = Normal(mu, std)  # Normal distribution
        return dist, value
