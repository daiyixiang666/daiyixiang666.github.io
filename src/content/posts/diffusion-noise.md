---
title: "扩散模型为什么要学着预测噪声？"
description: "从带噪样本的构造出发，理解噪声预测目标，以及训练和采样之间的区别。"
date: 2026-11-01
category: "生成模型"
cover: "/covers/diffusion.svg"
coverAlt: "从清晰结构逐渐过渡到随机噪声的四阶段示意图"
sample: true
---

扩散模型的一个常见训练过程看起来很简单：给图片加噪声，让网络预测加进去的噪声。

但生成时，我们手里并没有那张原始图片。那么，训练中的噪声预测与生成之间，是怎样连起来的？

## 一步得到任意时刻的带噪样本

对于 DDPM 的前向过程，定义 $\alpha_t=1-\beta_t$，以及 $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。可以直接采样：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

这意味着训练时不必真的从 $x_0$ 连续加噪声 $t$ 次。采样一个时间步 $t$ 和一份噪声，就能构造对应的 $x_t$。

## 网络看见什么

网络输入带噪样本 $x_t$ 和时间步 $t$，输出与样本形状相同的噪声估计 $\epsilon_\theta(x_t,t)$。

常用的简化训练目标是：

$$
L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right].
$$

这里的预测是统计意义上的估计。给定一个带噪样本，模型通常不能精确知道这一次训练中加进去的全部噪声；平方误差目标引导它学习相应条件分布下的条件均值。

## 为什么能连接到去噪

由带噪样本的构造式，可以写出一个对原样本的估计：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

这展示了噪声预测和原样本估计之间的联系。不过，标准 DDPM 采样会使用所定义的反向转移分布，逐步从 $x_t$ 采样 $x_{t-1}$；不能把上面这个代数关系直接当成完整的采样器。

## 一段训练伪代码

```python
# x0, noise, xt, prediction: [B, C, H, W]
# alpha_bar_t is reshaped to [B, 1, 1, 1] for broadcasting.
noise = torch.randn_like(x0)
xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise
prediction = model(xt, t)
loss = torch.nn.functional.mse_loss(prediction, noise)
```

训练时有真实数据 $x_0$，所以我们知道监督目标 `noise`。采样时从噪声开始，反复调用学到的模型，按照选定的采样规则更新样本。

**训练构造监督信号，采样使用学到的反向过程。** 分清这两件事，很多看起来矛盾的地方就更容易理解了。

继续阅读：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)。
