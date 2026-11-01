---
title: "从一个不等式，理解 VAE 的 ELBO"
description: "为什么重建误差和 KL 散度会出现在同一个目标里？从边缘似然开始，走一遍推导。"
date: 2026-11-01
category: "生成模型"
cover: "/covers/latent.svg"
coverAlt: "由观测空间映射到潜在空间的蓝绿色等高线示意图"
sample: true
---

VAE 的训练目标经常写成「重建项加上 KL 正则项」。但如果只记住这个形式，很容易忘记：它首先是在处理一个难以直接计算的概率。

这篇示例从一个问题开始：**观察到数据 $x$，我们怎样提高模型给它的概率？**

## 难点藏在积分里

引入潜变量 $z$ 后，生成模型可以写成 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$。我们希望最大化的边缘对数似然是：

$$
\log p_\theta(x)=\log\int p_\theta(x,z)\,dz.
$$

积分把所有可能的 $z$ 都考虑进来。在一般的神经网络模型里，这个积分很难精确求出。

于是，我们引入一个容易采样、容易计算密度的分布 $q_\phi(z\mid x)$。它就是编码器所表示的近似后验。

## 下界从哪里来

在适当的支撑集条件下，用 $q_\phi(z\mid x)$ 改写积分，再应用 Jensen 不等式：

$$
\begin{aligned}
\log p_\theta(x)
&=\log\mathbb E_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\\
&\geq \mathbb E_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\\
&=\mathcal L(\theta,\phi;x).
\end{aligned}
$$

右边就是 ELBO，即证据下界。把联合分布展开，可以得到更熟悉的形式：

$$
\mathcal L=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}\big(q_\phi(z\mid x)\|p(z)\big).
$$

第一项鼓励采样得到的 $z$ 能够解释输入 $x$；第二项约束近似后验与先验之间的距离。两项来自同一个似然下界。

## 一个形状明确的代码例子

设一个 batch 有 $B$ 个样本，潜变量有 $D$ 维。编码器输出 `mu` 和 `logvar`，形状都是 `[B, D]`。当先验为标准正态分布、近似后验为对角高斯时：

```python
# mu, logvar: [B, D]
# Sum over latent dimensions; average over examples.
kl_per_example = -0.5 * (
    1 + logvar - mu.square() - logvar.exp()
).sum(dim=-1)
kl = kl_per_example.mean()
```

这里 `sum(dim=-1)` 消掉潜变量维度，得到 `[B]`；`mean()` 再对样本取平均，得到标量。

重建项也应先按样本计算，再对 batch 取平均。若一项按像素求和、另一项却按像素取平均，两者的相对权重就会改变。

## 下界与似然之间还差什么

一个更有解释力的恒等式是：

$$
\log p_\theta(x)-\mathcal L(\theta,\phi;x)
=D_{\mathrm{KL}}\big(q_\phi(z\mid x)\|p_\theta(z\mid x)\big).
$$

因此，当近似后验等于真实后验时，下界才会取等号。优化 ELBO 不等于在每一步都精确优化边缘似然；两者之间还有近似后验带来的间隙。

> 先问目标在近似什么，再看每一项如何实现，往往比直接背损失函数更有帮助。

继续阅读：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)。
