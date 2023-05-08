确定策略梯度 (Deterministic Policy Gradient, DPG)) 是最常用的连续控制方法。==DPG是一种Actor-Critic 方法==。

策略网络 控制智能体做运动，它基于状态 s 做出动作a。价值网络不控制智能体，只是基于状态 s 给动作 a 打分，从而指导策略网络做出改进。

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210703161511.png" alt="image-20210703161511377" style="zoom: 67%;" />

**为什么叫==确定==策略梯度：**

在离散动作空间问题中，策略网络 $π(a|s; θ)$ 是一个概率质量函数，它输出的是概率值。连续策略控制中的确定策略网络$μ(s; θ)$ 的输出是d维的向量 a，作为动作。两种策略网络一个是随机的，一个是确定性的：



**价值网络** $q(s, \boldsymbol{a} ; \boldsymbol{w})$ 是对动作价值函数 $Q_{\pi}(s, \boldsymbol{a})$ 的近似。价值网络的输人是状态 $s$ 和动作 $\boldsymbol{a}$，输出的价值$\widehat{q}=q(s, \boldsymbol{a} ; \boldsymbol{w})$ 是个实数。



<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210703171530.png" alt="image-20210703171530004" style="zoom: 67%;" />

==确定策略网络属于Off-policy方法，即行为策略可以不同于目标策略。~~因为Q(s,a)在确定策略的条件下实际上评估的就是V(s)，和Q-Learning类似。~~==



**用行为策略收集经验**

目标策略即确定策略网络$\boldsymbol{\mu}\left(s ; \boldsymbol{\theta}_{\mathrm{now}}\right)$, 其中 $\boldsymbol{\theta}_{\mathrm{now}}$ 是策略网络最新的参数。行为策略可以是任意的, 比如
$$
\boldsymbol{a}=\boldsymbol{\mu}\left(s ; \boldsymbol{\theta}_{\mathrm{old}}\right)+\boldsymbol{\epsilon}
$$
即往动作中加入噪声 $\epsilon \in \mathbb{R}^{d}$ 。



**训练策略网络**

给定状态 $s$，策略网络输出一个动作 $\boldsymbol{a}=\boldsymbol{\mu}(s ; \boldsymbol{\theta})$, 然后价值网络会给 $\boldsymbol{a}$ 打一个分数： $\widehat{q}=q(s, \boldsymbol{a} ; \boldsymbol{w})$ 。

参数 $\boldsymbol{\theta}$ 影响 $\boldsymbol{a}$, 从而影响 $\widehat{q}$ 。 分数 $\widehat{q}$ 可以反映出 $\boldsymbol{\theta}$ 的好坏程度。训练策略网络的目标就是改进参数 $\boldsymbol{\theta}$, 使 $\widehat{q}$ 变得更大。

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210703173352.png" alt="image-20210703173352194" style="zoom: 67%;" />
$$
\hat{q}=q(s,\mu(s;\theta);w)
$$
目标函数定义为打分的期望：
$$
J(\boldsymbol{\theta})=\mathbb{E}_{S}[q(S, \boldsymbol{\mu}(S ; \boldsymbol{\theta}) ; \boldsymbol{w})]
$$
策略网络的学习可以建模成这样一个最大化问题：
$$
\max_\theta J(\theta)
$$
可以用梯度上升来增大 $J(\boldsymbol{\theta})$ 。每次用随机变量 $S$ 的一个观测值（记作 $s_{j}$ ）来计算梯度：
$$
\boldsymbol{g}_{j} \triangleq \nabla_{\boldsymbol{\theta}} q\left(s_{j}, \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) ; \boldsymbol{w}\right)
$$
它是 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计。 $\boldsymbol{g}_{j}$ 叫做确定策略梯度 (Deterministic Policy Gradient), 缩写 $\mathrm{DPG}_{\circ}$

可以用链式法则求出梯度$\boldsymbol{g}_j$。
$$
\frac{\partial q}{\partial \theta}=\frac{\partial a}{\partial \theta} \cdot \frac{\partial q}{\partial a}
$$

$$
\nabla_{\boldsymbol{\theta}} q\left(s_{j}, \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) ; \boldsymbol{w}\right)=\nabla_{\boldsymbol{\theta}} \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) \cdot \nabla_{\boldsymbol{a}} q\left(s_{j}, \widehat{\boldsymbol{a}}_{j} ; \boldsymbol{w}\right), \quad \text { 其中 } \widehat{\boldsymbol{a}}_{j}=\boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right)
$$

由此我们得到更新 $\boldsymbol{\theta}$ 的算法。每次从经验回放数组里随机抽取一个状态, 记作 $s_{j}$ 。

计算 $\widehat{\boldsymbol{a}}_{j}=\boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right)$ 。 用梯度上升更新一次 $\boldsymbol{\theta}$ :
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \nabla_{\boldsymbol{\theta}} \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) \cdot \nabla_{\boldsymbol{a}} q\left(s_{j}, \widehat{\boldsymbol{a}}_{j} ; \boldsymbol{w}\right)
$$
**训练价值网络**

训练价值网络的目标是让价值网络 $q(s, \boldsymbol{a} ; \boldsymbol{w})$ 的预测越来越接近真实价值函数 $Q_{\pi}(s, \boldsymbol{a})$ 。

训练价值网络要用TD算法。

每次从经验回放数组中取出一个四元组 $(s_j, a_j,r_j,s_{j+1})$， 用它更新一次参数$w$。首先让价值网络做预测：

计算 $\mathrm{TD}$ 目标 $\widehat{y}_{j}=r_{j}+\gamma \cdot \widehat{q}_{j+1}$ 。 定义损失函数
$$
L(\boldsymbol{w})=\frac{1}{2}\left[q\left(s_{j}, \boldsymbol{a}_{j} ; \boldsymbol{w}\right)-\widehat{y}_{j}\right]^{2}
$$
计算梯度
$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{\left(\widehat{q}_{j}-\widehat{y}_{j}\right)}_{\mathrm{TD} \text { 误差 } \delta_{j}} \cdot \nabla_{\boldsymbol{w}} q\left(s_{j}, a_{j} ; \boldsymbol{w}\right),
$$
做一轮梯度下降更新参数 $\boldsymbol{w}$ :
$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \nabla_{\boldsymbol{w}} L(\boldsymbol{w})
$$


**对DPG的分析：**

==从策略学习的角度看待DPG==

==Important One==：

价值网络$q(s, \boldsymbol{a} ; \boldsymbol{w})$ 是对动作价值函数 $Q_{\pi}(s, \boldsymbol{a})$ 的近似，而不是最优动作价值函数 $Q_{\star}(s, \boldsymbol{a})$。

DPG 的训练流程中，==更新==价值网络用到 TD 目标：
$$
\widehat{y}_{j}=r_{j}+\gamma \cdot q\left(s_{j+1}, \boldsymbol{\mu}\left(s_{j+1} ; \boldsymbol{\theta}_{\text {now }}\right) ; \boldsymbol{w}_{\text {now }}\right)
$$
很显然, 当前的策略 $\mu\left(s ; \boldsymbol{\theta}_{\text {now }}\right)$ 会直接影响价值网络 $q$ 。策略不同, 得到的价值网络 $q$ 就不同。

虽然价值网络 $q(s, \boldsymbol{a} ; \boldsymbol{w})$ 通常是对动作价值函数 $Q_{\pi}(s, \boldsymbol{a})$ 的近似，但是我们最终的目标是让 $q(s, \boldsymbol{a} ; \boldsymbol{w})$ 趋近于最优动作价值函数 $Q_{\star}(s, \boldsymbol{a})$ 。 如果 $\pi$ 是最优策略$\pi^{\star}$，那么 $Q_{\pi}(s, \boldsymbol{a})$ 就等于 $Q_{\star}(s, \boldsymbol{a})$ 。训练 $\mathrm{DPG}$ 的目的是让$\boldsymbol{\mu}(s ; \boldsymbol{\theta})$ 趋近于最优策略 $\pi^{\star}$，那么理想情况下，$q(s, \boldsymbol{a} ; \boldsymbol{w})$ 最终趋近于 $Q_{\star}(s, \boldsymbol{a})$ 。

==Important Two==：

$\mathrm{DPG}$ 的训练中有行为策略 $\boldsymbol{\mu}\left(s ; \boldsymbol{\theta}_{\mathrm{old}}\right)+\boldsymbol{\epsilon}$ 和目标策略 $\boldsymbol{\mu}\left(s ; \boldsymbol{\theta}_{\mathrm{now}}\right)$ 。价值网络 $q(s, \boldsymbol{a} ; \boldsymbol{w})$近似动作价值函数 $Q_{\pi}(s, \boldsymbol{a})$ 。此处的 $\pi$ 指的是目标策略，而不是行为策略。

我们用TD算法训练价值网络，TD算法的目的在于鼓励价值网络的预测趋近于 TD 目标。理想情况下，
$$
q\left(s_{j}, \boldsymbol{a}_{j} ; \boldsymbol{w}\right)=\underbrace{r_{j}+\gamma \cdot Q\left(s_{j+1}, \boldsymbol{\mu}\left(s_{j+1} ; \boldsymbol{\theta}_{\text {now }}\right) ; \boldsymbol{w}_{\text {now }}\right)}_{\text {TD } \text { 目标 }}, \quad \forall\left(s_{j}, \boldsymbol{a}_{j}, r_{j}, s_{j+1}\right) .
$$
在收集经验的过程中, 行为策略决定了如何基于 $s_{j}$ 生成 $a_{j}$, 然而这不重要。上面的公式只希望==等式左边去拟合等式右边, 而不在乎 $\boldsymbol{a}_{j}$ 是如何生成的。==



> ==**Q学习**==是对最优动作值函数$Q_\star(s,a)$的近似。行为策略和目标策略不一致。
> $$
> \underbrace{Q_\star\left(s_{t}, a_{t} ; \boldsymbol{w}\right)}_{\text {预则 } \hat{q}_{t}} \approx \underbrace{r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q_\star\left(s_{t+1}, a ; \boldsymbol{w}\right)}_{\mathrm{TD}} \text { 目标 } \hat{y}_{t}
> $$
> $Q_\star(s_t,a_t)$表示在状态$s_t$采取行动$a_t$后，执行最优策略的期望回报。
>
> ==价值网络【希望TD误差小】和行为策略无关。==
>
> 行为策略：
> $$
> a_{t}=\left\{\begin{array}{ll}
> \operatorname{argmax}_{a} Q\left(s_{t}, a ; \boldsymbol{w}\right), & \text { 以概率 }(1-\epsilon) ; \\
> \text { 均匀抽取 } \mathcal{A} \text { 中的一个动作, } & \text { 以概率 } \epsilon .
> \end{array}\right.
> $$
> 价值网络决定目标策略：
> $$
> a_{t}=\underset{a}{\operatorname{argmax}} Q\left(s_{t}, a ; \boldsymbol{w}\right)
> $$
> ==同策略是指用相同的行为策略和目标策略，异策略是指用不同的行为策略和目标策略。【过于抽象，还是无法区分什么情况下算法实现使用同策略，什么情况下算法实现使用异策略】==
>
> 
>
> ==**SARSA**==是对动作价值函数$Q_\pi(s,a)$的近似
> $$
> V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A_{t} \sim \pi\left(\cdot \mid s_{t} ; \theta\right)}\left[Q_{\pi}\left(s_{t}, A_{t}\right)\right]
> $$
> 如果一个策略很好, 那么对于所有的状态 $S$, 状态价值 $V_{\pi}(S)$ 的均值应当很大。因此我们定义目标函数:
> $$
> J(\theta)=\mathbb{E}_{S}\left[V_{\pi}(S)\right]
> $$
> 所以策略学习可以描述为这样一个优化问题:
> $$
> \max _{\theta} J(\theta)
> $$
> 我们使用梯度上升算法更新 $\theta$, 是的 $J(\theta)$ 增加, 更新公式为:
> $$
> \theta \leftarrow \theta+\alpha \cdot \nabla_{\theta} J(\theta)
> $$
> 其中 $\alpha$ 是学习率。实现时，我们可以使用 Actor-Critic $^{[10]}$ 算法对 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 做近似。
> $$
> \frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]\right]
> $$
> 价值网络：$V_\pi(s_t)=E(r_t+\gamma*V_\pi(s_{t+1}))$
>
> Critic评估和当前策略有关，使用TD算法进行更新，如果使用回放数组，$\{s_t,a_t,r_t,s_{t+1}\}$不是由当前的$\pi$采样得到的，评估的就不是$V_\pi$。
>
> 策略网络：
>
> 策略梯度$\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) \approx\left[r_{t}+\gamma \cdot V_{\pi}\left(s_{t+1}\right)-V_{\pi}\left(s_{t}\right)\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$



==从价值学习的角度看待DPG==

可以把 DPG 看做对最优动作价值函数 $Q_{\star}(s, \boldsymbol{a})$ 的另一种近似方式, 用于连续控制问题。我们希望学到策略网络 $\boldsymbol{\mu}(s ; \boldsymbol{\theta})$ 和价值网络 $q(s, a ; \boldsymbol{w})$, 使得
$$
q(s, \boldsymbol{\mu}(s ; \boldsymbol{\theta}) ; \boldsymbol{w}) \approx \max _{\boldsymbol{a} \in \mathcal{A}} Q_{\star}(s, \boldsymbol{a}), \quad \forall s \in \mathcal{S}
$$
我们可以把 $\boldsymbol{\mu}$ 和 $q$ 看做是 $Q_{\star}$ 的近似分解，而这种分解的目的在于方便做决策：
$$
\begin{aligned}
\boldsymbol{a}_{t} &=\boldsymbol{\mu}\left(s_{t} ; \boldsymbol{\theta}\right) \\
& \approx \underset{a \in \mathcal{A}}{\operatorname{argmax}} Q_{\star}\left(s_{t}, \boldsymbol{a}\right) .
\end{aligned}
$$


|                | A2C                                                          | DPG                                                          |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Critic目标** | $Q_{\pi}(s, \boldsymbol{a})$                                 | $Q_{\pi}(s, \boldsymbol{a})$                                 |
| **Critic更新** | $$L(\boldsymbol{w})=\frac{1}{2}\left[q\left(s_{j}, \boldsymbol{a}_{j} ; \boldsymbol{w}\right)-\widehat{y}_{j}\right]^{2}\\\widehat{y}_{j}=r_{j}+\gamma \cdot q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right)$$ | $$L(\boldsymbol{w})=\frac{1}{2}\left[q\left(s_{j}, \boldsymbol{a}_{j} ; \boldsymbol{w}\right)-\widehat{y}_{j}\right]^{2}\\\widehat{y}_{j}=r_{j}+\gamma \cdot q\left(s_{j+1}, \boldsymbol{\mu}\left(s_{j+1} ; \boldsymbol{\theta}\right) ; \boldsymbol{w}\right)$$ |
| **Actor目标**  | $$\max _{\boldsymbol{\theta}}\left\{J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{S}\left[V_{\pi}(S)\right]\right\}$$ | $$\max _{\boldsymbol{\theta}}\left\{J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{S}[q(S, \boldsymbol{\mu}(S ; \boldsymbol{\theta}) ; \boldsymbol{w})]\right\}$$ |
| **Actor更新**  | $$Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})$$ | $$\nabla_{\boldsymbol{\theta}} q\left(s_{j}, \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) ; \boldsymbol{w}\right)=\nabla_{\boldsymbol{\theta}} \boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right) \cdot \nabla_{\boldsymbol{a}} q\left(s_{j}, \widehat{\boldsymbol{a}}_{j} ; \boldsymbol{w}\right), \quad\\\widehat{\boldsymbol{a}}_{j}=\boldsymbol{\mu}\left(s_{j} ; \boldsymbol{\theta}\right)$$ |



为什么A2C和DPG都是Actor-Critic架构，一个是On-Policy，一个是Off-Policy？

A2C和DPG的Critic都是对动作值函数$Q_\pi(s,a)$的近似，如何做评价他们近似的好不好呢？都是使用的TD方法。

A2C希望$q(s_j,a_j;w)$尽可能接近TD目标$\widehat{y}_{j}=r_{j}+\gamma \cdot q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right)$，~~其中这里的$a_{j+1}$是当前策略$\pi$生成的，如果使用回放数组，那么这里的$a_{j+1}$就不是当前策略$\pi$生成的了，这样就不是对动作值函数的近似~~。

DPG希望$q(s_j,a_j;w)$尽可能接近TD目标$\widehat{y}_{j}=r_{j}+\gamma \cdot q\left(s_{j+1}, \boldsymbol{\mu}\left(s_{j+1} ; \boldsymbol{\theta}\right) ; \boldsymbol{w}\right)$，其中这里的

