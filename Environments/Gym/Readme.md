## Observations

The environment’s `step` function returns exactly what we need. In fact, `step` returns four values. These are:

- `observation` (**object**): an environment-specific object representing your observation of the environment. For
  example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
- `reward` (**float**): amount of reward achieved by the previous action. The scale varies between environments, but the
  goal is always to increase your total reward.
- `done` (**boolean**): whether it’s time to `reset` the environment again. Most (but not all) tasks are divided up into
  well-defined episodes, and `done` being `True` indicates the episode has terminated. (For example, perhaps the pole
  tipped too far, or you lost your last life.)
- `info` (**dict**): diagnostic information useful for debugging. It can sometimes be useful for learning (for example,
  it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of
  your agent are not allowed to use this for learning.

This is just an implementation of the classic “agent-environment loop”. Each timestep, the agent chooses an `action`,
and the environment returns an `observation` and a `reward`.

## Spaces

In the examples above, we’ve been sampling random actions from the environment’s action space. But what actually are
those actions? Every environment comes with an `action_space` and an `observation_space`. These attributes are of
type [`Space`](https://github.com/openai/gym/blob/master/gym/core.py), and they describe the format of valid actions and
observations:

```
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
```

The [`Discrete`](https://github.com/openai/gym/blob/master/gym/spaces/discrete.py) space allows a fixed range of
non-negative numbers, so in this case valid `action`s are either 0 or 1.
The [`Box`](https://github.com/openai/gym/blob/master/gym/spaces/box.py) space represents an `n`-dimensional box, so
valid `observations` will be an array of 4 numbers. We can also check the `Box`’s bounds:

```
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
```

This introspection can be helpful to write generic code that works for many different environments. `Box` and `Discrete`
are the most common `Space`s. You can sample from a `Space` or check that something belongs to it:

```
from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8
```

## [环境详解](https://github.com/openai/gym/wiki)

### [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)

start state/final state：

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210430162456.png" alt="image-20210430162456690" style="zoom:40%;" /><img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210430162749.png" alt="image-20210430162749033" style="zoom:40%;" />

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

[For more details](https://github.com/openai/gym/wiki/CartPole-v0)

### [MountainCar v0](https://github.com/openai/gym/wiki/MountainCar-v0)

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210430210021.png" alt="image-20210430210021480" style="zoom:50%;" />

**Reward**

-1 for each time step, until the goal position of 0.5 is reached. As with MountainCarContinuous v0, there is no penalty for climbing the left hill, which upon reached acts as a wall.

**Starting State**

Random position from -0.6 to -0.4 with no velocity.

**Episode Termination**

The episode ends when you reach 0.5 position, or if 200 iterations are reached.

**Solved Requirements**

MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.

### [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)

[wiki](https://github.com/openai/gym/wiki/Pendulum-v0)

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210503203628.png" alt="image-20210503203628631" style="zoom:50%;" />

The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright.

**Observation**

Type: Box(3)

| Num  | Observation | Min  | Max  |
| ---- | ----------- | ---- | ---- |
| 0    | cos(theta)  | -1.0 | 1.0  |
| 1    | sin(theta)  | -1.0 | 1.0  |
| 2    | theta dot   | -8.0 | 8.0  |

**Actions**

Type: Box(1)

| Num  | Action       | Min  | Max  |
| ---- | ------------ | ---- | ---- |
| 0    | Joint effort | -2.0 | 2.0  |

**Reward**

The precise equation for reward:

```
-(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
```

Theta is normalized between -pi and pi. Therefore, the lowest reward is `-(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044`, and the highest reward is `0`. In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity, and the least effort.

**Starting State**

Random angle from -pi to pi, and random velocity between -1 and 1

**Episode Termination**

There is no specified termination. Adding a maximum number of steps might be a good idea.

NOTE: Your environment object could be wrapped by the TimeLimit wrapper, if created using the "gym.make" method. In that case it will terminate after 200 steps.

### Pong-v0

Maximize your score in the Atari 2600 game Pong. In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3) Each action is repeatedly performed for a duration of $k$ frames, where $k$ is uniformly sampled from ${2, 3, 4}$.

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210430210150.png" alt="image-20210430210150757" style="zoom:50%;" />

### [Hopper-v2](http://gym.openai.com/envs/Hopper-v2/)

![image-20210510155643492](https://banni.oss-cn-beijing.aliyuncs.com/img/20210510155643.png)

[环境介绍](https://blog.paperspace.com/physics-control-tasks-with-deep-reinforcement-learning/)

