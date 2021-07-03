==核心思想：Actor在给定状态下给定一个动作分布，执行动作时在该分布中采样，Actor的目标是让$ \boldsymbol{g}(s, a ; \boldsymbol{\theta})=[\underbrace{Q_{\pi}(s, a)-V_{\pi}(s)}_{\text {优势函数 }}] \cdot \nabla_
{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})$越大；Critic的目标是让TD误差越小。==

==Actor的优化目标是$\pi_\theta(a|s)$，这也是策略梯度的由来。==





#### Advantage Actor-Critic (A2C)

$$ \boldsymbol{g}(s, a ; \boldsymbol{\theta})=[\underbrace{Q_{\pi}(s, a)-V_{\pi}(s)}_{\text {优势函数 }}] \cdot \nabla_
{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) . $$ 公式中的 $Q_{\pi}-V_{\pi}$ 被称作==优势函数== (Advantage Function)
。因此，基于上面公式得到的 Actor-Critic 方法被称为 Advantage Actor-Critic，缩写 $\mathrm{A} 2 \mathrm{C}_{\circ}$

A2C 属于 Actor-Critic 方法。有一个策略网络 $\pi(a \mid s ; \boldsymbol{\theta}),$ 相当于演员，用于控制智能 体运动。还有一个价值网络 $v(s ; \boldsymbol{w})
$，相当于评委，他的评分可以帮助策略网络 (演员）改 进技术。两个神经网络的结构与上一节中的完全相同，但是本节和上一节用不同的方法 训练两个神经网络。

> 朴素的价值网络是$q(s,a;\boldsymbol w)$，带基线的价值网络是$v(s;\boldsymbol{w})$。



<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210318203632.png" alt="image-20210318203632210" style="zoom:67%;" />

- 如果 $\widehat{y}_{t}>v\left(s_{t} ; \boldsymbol{w}\right),$ 说明动作 $a_{t}$ 很好，使得奖励 $r_{t}$ 超出预期，或者新的状态 $s_
  {t+1}$比预期好；这种情况下应该更新 $\theta,$ 使得 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$ 变大。
- 如果 $\widehat{y}_{t}<v\left(s_{t} ; \boldsymbol{w}\right),$ 说明动作 $a_{t}$ 不好 $,$ 导致奖励 $r_{t}$ 不及预期，或者新的状态 $s_{t+1}$
  比预期差；这种情况下应该更新 $\theta,$ 使得 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$ 减小。

==综上所述， $\delta_{t}$ 中虽然不包含动作 $a_{t},$ 但是 $\delta_{t}$ 可以间接反映出动作 $a_{t}$ 的好坏，可以帮助策略网络（演员）改进演技。==

**训练过程**

设当前策略网络参数是 $\theta_{\mathrm{now}},$ 价值网络参数是 $\boldsymbol{w}_{\mathrm{now}} $。执行下面的步骤，将参数更新成 $\theta_{\text {new }}$ 和
$w_{\text {new }}$ :

1. 观测到当前状态 $s_{t},$ 根据策略网络做决策： $a_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\mathrm{now}}\right),$
   并让智能体执行动作 $a_{t \circ}$
2. 从环境中观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。
3. 让价值网络打分： $$ \widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right) \quad \text { 和 } \quad \widehat{v}_
   {t+1}=v\left(s_{t+1} ; \boldsymbol{w}_{\text {now }}\right)
   $$
4. 计算 TD 目标和 TD 误差： $$ \widehat{y}_{t}=r_{t}+\gamma \cdot \widehat{v}_{t+1} \quad \text { 和 } \quad \delta_
   {t}=\widehat{v}_{t}-\widehat{y}_{t} $$
5. 更新价值网络： $$ \boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot
   \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right)
   $$
6. 更新策略网络:
   $$ \boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}-\beta \cdot \delta_{t} \cdot
   \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)
   $$
   **注** 此处训练策略网络和价值网络的方法属于同策略 (On-policy)，要求行为策略 (Behavior Policy) 与目标策略 (Target Policy) 相同，都是最新的策略网络 $\pi\left(a \mid
   s ; \boldsymbol{\theta}_{\mathrm{now}}\right)$ 。不能使用经验回放，因为经验回放数组中的数据是用旧的策略网络 $\pi\left(a \mid s ;
   \boldsymbol{\theta}_{\mathrm{old}}\right)$ 获取的，不能在当前重复利用。

可以使用目标网络改进训练。

<h1>A2C: Synchronous Advantage Actor Critic</h1>
<h3><a href="https://blog.openai.com/baselines-acktr-a2c/#a2canda3c">OpenAI Blog:</a></h3>
<p>The Asynchronous Advantage Actor Critic method (A3C) has been very influential since the paper was published. The algorithm combines a few key ideas:</p>

<ul>
    <li>An updating scheme that operates on fixed-length segments of experience (say, 20 timesteps) and uses these segments to compute estimators of the returns and advantage function.</li>
    <li>Architectures that share layers between the policy and value function.</li>
    <li>Asynchronous updates.</li>
</ul>

<p>After reading the paper, AI researchers wondered whether the asynchrony led to improved performance (e.g. “perhaps the added noise would provide some regularization or exploration?“), or if it was just an implementation detail that allowed for faster training with a CPU-based implementation.</p>

<p>As an alternative to the asynchronous implementation, researchers found you can write a synchronous, deterministic implementation that waits for each actor to finish its segment of experience before performing an update, averaging over all of the actors. One advantage of this method is that it can more effectively use of GPUs, which perform best with large batch sizes. This algorithm is naturally called A2C, short for advantage actor critic. (This term has been used in several papers.)</p>

For more information: [深度强化学习 -- 进击的 Actor-Critic（A2C 和A3C）](https://zhuanlan.zhihu.com/p/148492887)

A2C 和 A3C 这两种算法，创新的引入了并行架构，即整个agent 由一个 Global Network 和多个并行独立的 worker 构成，每个 worker 都包括一套 Actor-Critic 网络。而各个 worker 都会独立的跟自己的环境去交互，得到独立的采样经验，而这些经验之间也是相互独立的，这样就打破了经验之间的耦合，起到跟 Experience Replay 相当的效果。因此通常 A2C和A3C 是不需要使用 Replay Buffer 的，这种结构本身就可以替代了。

**并行架构的优势**：

- 打破经验之间的耦合，起到类似于经验回放的作用，但又避免了 Repaly Buffer 内存占用过大的问题；
- 多个 worker 独立探索一个环境的副本，相当于同时去探索环境的不同部分，并且使用不同的策略也可以极大提升多样性（只限于 A3C），充分发挥探索的优势
- 充分利用计算资源，可以在多核 CPU 上实现与 GPU 训练像媲美的效果；
- 有效缩短训练时间，训练时间与并行进程的数量呈现近似线性的关系；
- 由于舍弃了Repaly Buffer，就可以使用 on policy 类的算法，而不用局限于 off policy （如q learning）

在这种并行架构下，同一个episode中，每个 agent 都会采样得到不同的经验，就会计算出不同的梯度。如何处理这些不同的梯度，以及如何把多个 agent 的参数整合到统一的 Global Network 中，就可以有同步和异步两种不同的方式，因此衍生出了两种不同的算法。



A2C 也会构建多个进程，包括多个并行的 worker，与独立的环境进行交互，收集独立的经验。

![img](https://pic3.zhimg.com/80/v2-4b12338a6794b042ad9c83c9bd5937fe_1440w.jpg)

但是这些 worker 是同步的，即每轮训练中，Global network 都会等待每个 worker 各自完成当前的 episode，然后把这些 worker 上传的梯度进行汇总并求平均，得到一个统一的梯度并用其更新主网络的参数，最后用这个参数同时更新所有的 worker。 相当于在 A3C 基础上加入了一个同步的环节。

A2C 跟 A3C 的一个显著区别就是，在任何时刻，不同 worker 使用的其实是同一套策略，它们是完全同步的，更新的时机也是同步的。由于各 worker彼此相同，其实 A2C 就相当于只有两个网络，其中一个 Global network 负责参数更新，另一个负责跟环境交互收集经验，只不过它利用了并行的多个环境，可以收集到去耦合的多组独立经验。

A2C 和 A3C 的区别用下面的图就可以清晰说明了，A2C 比 A3C 多的就是一个中间过程，一个同步的控制，等所有 worker 都跑完再一起同步，用的是平均梯度或累加梯度。因此这些worker们每时每刻使用的都是同样的策略，而 A3C 中不同的 worker 使用的策略可能都不相同。

![img](https://pic4.zhimg.com/80/v2-464c52e3f5592674e12caab51bb3cb77_1440w.jpg)

相比于A3C，A2C有一些显著的优势：

- A3C 中各个 agent 都是异步独立更新，每个 agent 使用不同的策略，可能会导致 global policy 的累计更新效果并不是最优（optimal）的。而 A2C 通过同步更新解决了这种不一致（
  inconsistency）的问题；
- A2C 的同步更新会让训练更加协调一致，从而潜在的加快收敛；
- 经过实践证明，A2C 对 GPU 的利用率更高，对于大 batch size 效果更好，而且在相同任务中比 A3C 取得更好的性能。

#### For More Reference

[Actor-Critic算法小结](https://zhuanlan.zhihu.com/p/29486661)

[Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)

