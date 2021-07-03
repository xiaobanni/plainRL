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
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)
ax.plot(x, y1, y2)
fig.show()
