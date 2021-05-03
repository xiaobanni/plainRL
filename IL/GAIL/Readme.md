#### 生成对抗模仿学习

生成判别模仿学习 (Generative Adversarial Imitation Learning，缩写 GAIL) 需要让智能体与环境交互，但是无法从环境获得奖励。GAIL还需要收集人类专家的决策记录（即很多条轨迹）。GAIL
的目标是学习一个策略网络，使得判别器无法区分一条轨迹是策略网络的决策还是人类专家的决策。

**批量随机梯度 (Mini-Batch SGD)**：上述训练生成器和判别器的方式其实是随机梯度下降 (SGD)，每次只用一个样本。实践中，不妨每次用一个批量 (Batch) 的样本，比如用 b =16 个，那么会计算出 b 个梯度。用
b 个梯度的平均去更新生成器和判别器。

**GAIL的训练**

训练的目的是让生成器（即策略网络）生成的轨迹与数据集中的轨迹（即被模仿对象的轨迹 $)$ 一样好。在训练结束的时候，判别器无法区分生成的轨迹与数据集里的轨迹。
**训练生成器：** 设 $\theta_{\mathrm{now}}$ 是当前策略网络的参数。用策略网络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\mathrm{now}}\right)$
控制智能体与环境交互, 得到一条轨迹： $$ \tau=\left[s_{1}, a_{1}, s_{2}, a_{2}, \cdots, s_{n}, a_{n}\right]
$$ 判别器可以评价 $\left(s_{t}, a_{t}\right)$ 有多真实; $D\left(s_{t}, a_{t} ; \phi\right)$ 越大，说明 $\left(s_{t}, a_{t}\right)$
在判别器的眼里越真实。把 $$ u_{t}=\ln D\left(s_{t}, a_{t} ; \phi\right)
$$ 作为第 $t$ 步的回报; $u_{t}$ 越大，则说明 $\left(s_{t}, a_{t}\right)$ 越真实。我们有这样一条轨迹： $$ s_{1}, a_{1}, u_{1}, \quad s_{2}, a_{2},
u_{2}, \quad \cdots, \quad s_{n}, a_{n}, u_{n} $$ 于是可以用 $\mathrm{TRPO}$ 来更新策略网络。设当前策略网络的参数为 $\theta_{\mathrm{now}
\circ}$ 定义目标函数： $$ \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\mathrm{now}}\right) \triangleq
\frac{1}{n} \sum_{t=1}^{n} \frac{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)}{\pi\left(a_{t} \mid s_{t} ;
\boldsymbol{\theta}_{\mathrm{now}}\right)} \cdot u_{t} $$ 求解下面的带约束的最大化问题，得到新的参数： $$ \boldsymbol{\theta}_{\text {new
}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_
{\text {now }}\right) ; \quad \text { s.t. } \operatorname{dist}\left(\boldsymbol{\theta}_{\text {now }},
\boldsymbol{\theta}\right) \leq \Delta . \tag{17.2} $$ 此处的 dist 衡量 $\theta_{\mathrm{now}}$ 与 $\boldsymbol{\theta}$ 的区别，
$\Delta$ 是一个需要调的超参数。TRPO 的详细解释见第 $9.1$ 节。

**训练判别器**： 训练判别器的目的是让它能区分真的轨迹与生成的轨迹。从训练数据 集中均匀抽样一条轨迹，记作 $$ \tau^{\text {real
}}=\left[s_{1}^{\text {real }}, a_{1}^{\text {real }}, \cdots, s_{m}^{\text {real }}, a_{m}^{\text {real }}\right]
$$ 用策略网络控制智能体与环境交互，得到一条轨迹，记作 $$
\tau^{\mathrm{fake}}=\left[s_{1}^{\mathrm{fake}}, a_{1}^{\mathrm{fake}}, \cdots, s_{n}^{\mathrm{fake}}, a_{n}^{\mathrm{fake}}\right]
. $$ 公式中的 $m$ 、 $n$ 分别是两条轨迹的长度。 训练判别器的时候, 要鼓励判别器做出准确的判断。我们希望判别器知道 $\left(s_{t}^{\mathrm{real}}, a_{t}^{\text {real
}}\right)$ 是真的，所以应该鼓励 $D\left(s_{t}^{\text {real }}, a_{t}^{\text {real }} ; \phi\right)$ 尽量大。我们希望判别器知道 $\left(s_
{t}^{\text {fake }}, a_{t}^{\text {fake }}\right)$ 是假 的，所以应该鼓励 $D\left(s_{t}^{\text {fake }}, a_{t}^{\mathrm{fake}} ;
\phi\right)$ 尽量小。定义损失函数 $$ F\left(\tau^{\text {real }}, \tau^{\text {fake }} ; \phi\right)=\underbrace{\frac{1}{m} \sum_
{t=1}^{m} \ln \left[1-D\left(s_{t}^{\text {real }}, a_{t}^{\text {real }} ; \phi\right)\right]}_{D \text { 的输出越大, 这一项越小
}}+\underbrace{\frac{1}{n} \sum_{t=1}^{n} \ln D\left(s_{t}^{\text {fake }}, a_{t}^{\text {fake }} ; \phi\right)}_{D
\text { 的输出越小, 这一项越小 }} . $$ 我们希望损失函数尽量小，也就是说判别器能区分开真假轨迹。可以做梯度下降来更新 参数 $\phi:$ $$ \phi \leftarrow \phi-\eta \cdot
\nabla_{\phi} F\left(\tau^{\text {real }}, \tau^{\text {fake }} ; \phi\right) \tag{17.3} $$ 这样可以让损失函数减小，让判别器更能区分开真假轨迹。

**训练流程**： 每一轮训练更新一个生成器 $，$ 更新一次判别器。训练重复以下步骤，直到收玫。设当前生成器和判别器的参数分别为 $\theta_{\text {now }}$ 和 $\phi_{\text {now }}$ 。

1. 从训练数据集中均匀抽样一条轨迹，记作 $$ \tau^{\text {real
   }}=\left[s_{1}^{\text {real }}, a_{1}^{\text {real }}, \cdots, s_{m}^{\text {real }}, a_{m}^{\text {real }}\right]
   $$
2. 用策略网络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right)$ 控制智能体与环境交互, 得到一条轨迹，记作

$$
\tau^{\mathrm{fake}}=\left[s_{1}^{\mathrm{fake}}, a_{1}^{\mathrm{fake}}, \cdots, s_{n}^{\mathrm{fake}}, a_{n}^{\mathrm{fake}}\right]
. $$

3. 用判别器评价策略网络的决策是否真实： $$ u_{t}=\ln D\left(s_{t}^{\text {fake }}, a_{t}^{\text {fake }} ; \phi_{\text {now }}\right),
   \quad \forall t=1, \cdots, n $$
4. 把 $\tau^{\mathrm{fake}}$ 和 $u_{1}, \cdots, u_{n}$ 作为输入，用公式 $(17.2)$ 更新策略网络参数，得到 $\theta_{\text {new }}$ 。
5. 把 $\tau^{\text {real }}$ 和 $\tau^{\text {fake }}$ 作为输入，用公式 $(17.3)$ 更新判别器参数，得到 $\phi_{\text {new }}$ 。



在实际的GAIL代码编写中，需要将$D_w$的输出clip到(0,1)内。对于Policy Net，其实可以使用各种强化学习方法，实际使用的$Q(s,a)$ 也可以用各种优势函数来计算。例如，使用Actor-Critic结构的PPO算法，优势函数就可以用$D_w$计算得到的$C(a,a)$和Critic中得到的$V$值计算GAE损失，来更新Policy部分中的Actor和Critic网络。

