"""
@Project : Imitation-Learning
@File    : ppo
@Author  : XiaoBanni
@Date    : 2021-05-03 13:04
@Desc    : 
"""
import numpy as np
import torch


def ppo_iter(cfg, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // cfg.mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, cfg.mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(agent, cfg, states, actions, log_probs, returns, advantages,
               clip_param=0.2):
    for _ in range(cfg.ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(cfg, states, actions,
                                                                         log_probs,
                                                                         returns, advantages):
            dist, value = agent.model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            agent.loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            agent.optimizer.zero_grad()
            agent.loss.backward()
            agent.optimizer.step()
