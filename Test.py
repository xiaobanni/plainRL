"""
@Project : Imitation-Learning
@File    : Test.pu
@Author  : XiaoBanni
@Date    : 2021-04-29 13:54
@Desc    : 
"""

import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np


def start():
    try:
        a = np.load("xxx.npy")
    except FileNotFoundError as e:
        # print(e.errno)
        assert False
    print("hello world!")


start()
