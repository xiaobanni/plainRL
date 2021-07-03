==核心思想：使用Q函数近似值函数，执行时选择Q函数最大的动作执行。==



### 训练过程

使用目标网络的话，Q 学习算法的用下面的方式实现。每次随机从经验回放数组中取一个四元组，记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。设 $\mathrm{DQN}$
和目标网络当前的参数分别为 $\boldsymbol{w}_{\text {now }}$ 和$\boldsymbol{w}_{\text {now }}^{-}$ ，执行下面的步骤对参数做一次更新：

1. 对 DQN 做正向传播，得到： $$ \widehat{q}_{j}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)
   $$
2. 对目标网络做正向传播，得到 $$ \widehat{q}_{j+1}=\max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now
   }}^{-}\right)
   $$
3. 计算 TD 目标和 TD 误差： $$ \widehat{y_{j}}=r_{j}+\gamma \cdot \widehat{q_{j+1}} \quad \text { 和 } \quad \delta_
   {j}=\widehat{q}_{j}-\widehat{y}_{j} $$
4. 对 $\mathrm{DQN}$ 做反向传播，得到梯度 $\nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\mathrm{now}}\right)$ 。
5. 做梯度下降更新 DQN 的参数： $$ \boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{j}
   \cdot \nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)
   $$
6. 设 $\tau \in(0,1)$ 是需要手动调的超参数。做加权平均更新目标网络的参数：

$$ \boldsymbol{w}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{w}_{\text {new }}+(1-\tau) \cdot \boldsymbol{w}_
{\text {now }}^{-} $$

<img src="https://banni.oss-cn-beijing.aliyuncs.com/img/20210318175642.png" alt="image-20210318175641935" />

如图 $6.4($ 左) 所示，原始的 Q 学习算法用 $\mathrm{DQN}$ 计算 $\widehat{y},$ 然后拿 $\widehat{y}$ 更新 DQN 自己，造成自举。如图 $6.4($ 右 $)$
所示，可以改用目标网络计算 $\widehat{y},$ 这样就避免了用 DQN 的估计更新 DQN 自己，降低自举造成的危害。然而这种方法并不可能完全避免自举，原因是目标网络的参数仍然与 DQN 相关。

![image-20210430125205788](https://banni.oss-cn-beijing.aliyuncs.com/img/20210430125213.png)