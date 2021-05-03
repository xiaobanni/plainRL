# Imitation-Learning

Install virtual environment

```
conda create -n IL python=3.7
conda activate IL
```

Install gym

```
pip install gym
pip install gym[atari]
```

Install pytorch 1.8.1 CUDA10.2

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Install other packages

```
conda install matplotlib
conda install seaborn
```

More about [Gym](https://gym.openai.com/)

```python
import gym

env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample()  # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

Contains implementations:

* RL
  * DQN
  * A2C

* IL
  * GAIL
    
