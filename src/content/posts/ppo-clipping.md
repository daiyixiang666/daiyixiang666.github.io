---
title: "PPO 的 clipping，到底限制了什么？"
description: "用正负两个 advantage 的小例子，看看被截断的是哪个目标，以及它没有保证什么。"
date: 2026-11-01
category: "强化学习"
cover: "/covers/ppo.svg"
coverAlt: "PPO 概率比率与截断目标的折线示意图"
sample: true
---

看到 PPO 中的 `clip`，一个自然的理解是「把策略更新限制在一个范围内」。这个说法有直觉上的帮助，但还不够准确。

**PPO-Clip 截断的是代理目标中的概率比率；它并没有直接把所有动作概率硬性约束在区间内。**

## 先看概率比率

对旧策略采样得到的状态和动作，定义：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

若比率为 $1.2$，意味着新策略对这个动作给出的概率，是旧策略的 $1.2$ 倍。

PPO-Clip 最大化：

$$
L^{\mathrm{clip}}(\theta)=\mathbb E_t\left[\min\left(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

## Advantage 为正时

假设 $\hat A_t=2$，$\epsilon=0.2$。这个样本希望动作概率提高。

| 比率 $r_t$ | 原目标 $r_t\hat A_t$ | 截断后参与比较的值 | 最终取最小值 |
| --- | --- | --- | --- |
| 1.0 | 2.0 | 2.0 | 2.0 |
| 1.2 | 2.4 | 2.4 | 2.4 |
| 1.5 | 3.0 | 2.4 | 2.4 |

超过 $1.2$ 后，这个样本的代理目标不再因为比率继续增大而提高。

## Advantage 为负时

假设 $\hat A_t=-2$。我们希望降低这个动作的概率。

当 $r_t=0.5$ 时，原目标为 $-1.0$，截断后的值为 $-1.6$；取最小值得到 $-1.6$。继续降低比率，不能继续改善这个样本的目标。

但如果 $r_t=1.5$，动作概率朝错误的方向上升，原目标为 $-3.0$。此时取最小值仍然保留 $-3.0$，让目标对这种变化产生惩罚。

## 对应到代码

```python
# All tensors below have shape [B].
ratio = (new_log_prob - old_log_prob).exp()
surrogate = ratio * advantage
clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
policy_loss = -torch.minimum(surrogate, clipped).mean()
```

负号把最大化代理目标变成最小化 loss。`old_log_prob` 和用于策略更新的 `advantage` 应作为固定数据，不沿旧策略或 advantage 估计过程反向传播。

由于神经网络参数被多个状态、动作和样本共享，即使某个样本已经进入平坦区，它的概率比率也可能因其他样本的更新继续变化。因此，clipping 不能保证比率始终落在区间内，也不能保证 KL 散度不超过某个固定阈值。

继续阅读：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。
