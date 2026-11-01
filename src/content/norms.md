---
title: "Understanding Optimizers Through Norms"
description: "From vector and matrix norms to the update rules of SGD, Adam, and Muon."
---

“Move in the direction of steepest descent” sounds like a complete instruction. But it leaves something unspecified: how are we measuring the size of a step?

This question is easier to see before looking at any particular optimizer. Start with three vector norms, extend the idea to matrices, and then use those norms to choose a parameter update.

## 1. Three ways to measure a vector

Let $x=(x_1,\ldots,x_d)^\top\in\mathbb R^d$. A norm assigns a nonnegative size to $x$. It is zero only at the zero vector, scales by $|c|$ when $x$ is multiplied by $c$, and satisfies the triangle inequality. Different norms can satisfy all three properties while measuring size differently.

The most familiar family is

$$
\|x\|_p=\left(\sum_{i=1}^{d}\lvert x_i\rvert^p\right)^{1/p},\qquad 1\leq p<\infty.
$$

Three cases will be enough here:

<div class="table-scroll" role="region" aria-label="Vector norms" tabindex="0">

| Vector norm | Definition | What a budget of one means |
| :--- | :--- | :--- |
| $L_1$: $\|x\|_1$ | $\sum_i \lvert x_i\rvert$ | The absolute values share a total budget of one. |
| $L_2$: $\|x\|_2$ | $\sqrt{\sum_i x_i^2}$ | The squared coordinates sum to at most one. |
| $L_\infty$: $\|x\|_\infty$ | $\max_i \lvert x_i\rvert$ | Every coordinate has its own bound of one. |

</div>

For the same vector $x=(3,4)^\top$,

$$
\|x\|_1=7,\qquad \|x\|_2=5,\qquad \|x\|_\infty=4.
$$

None of these numbers contradicts the others. They answer different questions about the same vector. In this article, $L_p$ and $\ell_p$ refer to these finite-dimensional vector norms.

### Draw the allowed steps

A norm's **unit ball** is the set $B_p=\{x:\|x\|_p\leq1\}$. “Ball” includes its interior; the shape need not be round.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/vector-balls-mobile.svg" /><img src="/figures/vector-balls.svg" alt="The L1 unit ball is a diamond, the L2 ball a disk, and the L-infinity ball a square, all drawn on the same coordinate scale." width="1200" height="440" /></picture>
<figcaption>Figure 1. Three definitions of a unit-sized vector. The axes use the same scale in all panels.</figcaption>
</figure>

With an $L_1$ budget, spending more on one coordinate leaves less for the others. With an $L_2$ budget, coordinates also compete, but through their squares. With an $L_\infty$ budget, each coordinate can reach its bound independently.

For example, $(1,1)$ is allowed by the $L_\infty$ unit ball, but its $L_2$ norm is $\sqrt2$ and its $L_1$ norm is $2$. Giving two methods a radius of one in different norms does **not** give them the same Euclidean movement budget.

## 2. A matrix can be measured in two ways

Now let $A\in\mathbb R^{m\times n}$. We can view it as a collection of $mn$ entries, or as a linear map that takes an input $x\in\mathbb R^n$ to an output $Ax\in\mathbb R^m$.

The first view leads naturally to the **Frobenius norm**:

$$
\|A\|_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}a_{ij}^2}
=\|\operatorname{vec}(A)\|_2.
$$

Here $\operatorname{vec}(A)$ just stacks the entries into a vector. Their order does not affect this norm. Frobenius is the Euclidean length of that vector.

The second view asks how much the matrix can amplify an input. Once we choose a vector norm for the input and output, we can define an **induced matrix norm**:

$$
\|A\|_{p\to p}
:=\sup_{x\ne0}\frac{\|Ax\|_p}{\|x\|_p}
=\sup_{\|x\|_p\leq1}\|Ax\|_p.
$$

This is the largest output size possible under a unit input budget. The input and output use the same type of vector norm, even if their dimensions differ. In the table below, $\|A\|_p$ is shorthand for $\|A\|_{p\to p}$.

<div class="table-scroll" role="region" aria-label="Matrix norms" tabindex="0">

| Matrix norm | Formula | What it measures |
| :--- | :--- | :--- |
| Induced $L_1$: $\|A\|_1$ | $\max_j\sum_i\lvert a_{ij}\rvert$ | Largest absolute column sum. |
| Induced $L_2$: $\|A\|_2$ | $\sigma_{\max}(A)$ | Largest amplification of Euclidean input length. Also called the **spectral norm**. |
| Induced $L_\infty$: $\|A\|_\infty$ | $\max_i\sum_j\lvert a_{ij}\rvert$ | Largest absolute row sum. |
| Frobenius: $\|A\|_F$ | $\sqrt{\sum_{i,j}a_{ij}^2}$ | Euclidean length of all entries taken together. |

</div>

The same subscripts mean different things depending on whether the object is a vector or a matrix. In particular,

$$
\boxed{\|A\|_2\ne\sqrt{\sum_{i,j}a_{ij}^2}\quad\text{in general}.}
$$

The expression on the right is $\|A\|_F$. Likewise, the entrywise sum $\sum_{i,j}\lvert a_{ij}\rvert$ is an entrywise $L_1$ norm, but it is **not** the induced matrix $L_1$ norm used here.

### Why columns for one and rows for infinity?

Write $a_{:j}$ for column $j$. For any input $x$,

$$
\|Ax\|_1
=\left\|\sum_jx_ja_{:j}\right\|_1
\leq\sum_j|x_j|\,\|a_{:j}\|_1
\leq\left(\max_j\sum_i\lvert a_{ij}\rvert\right)\|x\|_1.
$$

Choose $x=e_j$, the coordinate vector selecting a largest-sum column. It attains this bound, so the largest column sum is exactly the induced $L_1$ norm.

For $L_\infty$, each input coordinate satisfies $|x_j|\leq1$. Thus

$$
|(Ax)_i|\leq\sum_j\lvert a_{ij}\rvert.
$$

Pick a largest-sum row $i_*$ and set $x_j=\operatorname{sign}(a_{i_*j})$. Its contributions add with the same sign and attain the bound. A zero matrix is immediate. This gives the largest absolute row sum.

## 3. Matrix L2 versus Frobenius

The distinction becomes visible when a matrix acts on a circle. Consider

$$
A=\begin{pmatrix}1&2\\0&2\end{pmatrix}.
$$

As a Euclidean unit input moves around the circle, $Ax$ traces an ellipse.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/matrix-map-mobile.svg" /><img src="/figures/matrix-map.svg" alt="For A with rows (1,2) and (0,2), the input unit circle maps to an ellipse. A blue unit input direction maps to the ellipse's longest semiaxis, of length about 2.921." width="1100" height="490" loading="lazy" /></picture>
<figcaption>Figure 2. Blue arrows show a most-amplified unit input and its output. The gray output arrow is the other principal semiaxis; the dashed circle has radius one.</figcaption>
</figure>

The ellipse's semiaxis lengths are the **singular values** of $A$. To see why, write a singular value decomposition $A=U\Sigma V^\top$. The orthogonal factors $U$ and $V$ preserve Euclidean lengths; $\Sigma$ stretches orthogonal directions by the singular values. The largest possible stretch is therefore

$$
\|A\|_2=\sigma_1.
$$

Frobenius combines all singular values instead:

$$
\|A\|_F^2
=\operatorname{tr}(A^\top A)
=\sum_k\sigma_k^2.
$$

The trace is the sum of diagonal entries. Here $\operatorname{tr}(A^\top A)$ adds the squared length of every column, which is exactly the sum of squared entries.

So spectral norm keeps the **largest** semiaxis; Frobenius takes the square root of the **sum of squared** semiaxis lengths. For a nonzero matrix of rank $r$,

$$
\|A\|_2\leq\|A\|_F\leq\sqrt r\,\|A\|_2.
$$

The inequalities follow directly from $\sigma_1^2\leq\sum_k\sigma_k^2\leq r\sigma_1^2$. Here rank means the number of nonzero singular values.

### Compare their unit balls on a matrix slice

A $2\times2$ matrix lives in a four-dimensional space of entries. We can draw a two-dimensional slice by restricting it to $A=\operatorname{diag}(a,b)$:

$$
\|A\|_2=\max(|a|,|b|),\qquad
\|A\|_F=\sqrt{a^2+b^2}.
$$

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/matrix-balls-mobile.svg" /><img src="/figures/matrix-balls.svg" alt="On the diagonal-matrix slice diag(a,b), the spectral unit ball is a square and the Frobenius unit ball is a disk. The identity matrix at (1,1) belongs to the square but lies outside the disk." width="1000" height="460" loading="lazy" /></picture>
<figcaption>Figure 3. A slice of matrix space, not the full matrix ball. The marked point is the identity: spectral norm one, Frobenius norm square root of two.</figcaption>
</figure>

The identity preserves the Euclidean length of every input, so its induced $L_2$ norm is one. But it has two unit entries, so its Frobenius norm is $\sqrt2$. More generally, $\|I_n\|_2=1$ while $\|I_n\|_F=\sqrt n$.

## 4. Put the numbers side by side

Return to the same matrix $A=\left(\begin{smallmatrix}1&2\\0&2\end{smallmatrix}\right)$. Its absolute column sums are $1$ and $4$; its absolute row sums are $3$ and $2$. Also,

$$
A^\top A=\begin{pmatrix}1&2\\2&8\end{pmatrix},
\qquad
\lambda_{\max}(A^\top A)=\frac{9+\sqrt{65}}{2}.
$$

The largest eigenvalue of $A^\top A$ is the square of the largest singular value. This gives a concrete comparison:

<div class="table-scroll" role="region" aria-label="Numerical matrix comparison" tabindex="0">

| Quantity | Value for this $A$ |
| :--- | :--- |
| Induced matrix $L_1$ | $4$ |
| Induced matrix $L_2$ | $\sqrt{(9+\sqrt{65})/2}\approx2.921$ |
| Induced matrix $L_\infty$ | $3$ |
| Frobenius | $\sqrt{1^2+2^2+0^2+2^2}=3$ |
| Entrywise $L_1$ | $1+2+0+2=5$ |

</div>

The equality between Frobenius and induced $L_\infty$ in this example is coincidental. Their definitions are different.

In NumPy, specify which norm you mean:

```python
import numpy as np

x = np.array([3., 4.])
A = np.array([[1., 2.], [0., 2.]])

np.linalg.norm(x, ord=1)       # 7.0
np.linalg.norm(x, ord=2)       # 5.0
np.linalg.norm(x, ord=np.inf)  # 4.0

np.linalg.norm(A, ord=1)       # 4.0: largest column sum
np.linalg.norm(A, ord=2)       # 2.920809626...: spectral
np.linalg.norm(A, ord=np.inf)  # 3.0: largest row sum
np.linalg.norm(A, ord="fro")   # 3.0: Frobenius
np.linalg.norm(A.ravel(), 2)   # 3.0: vectorized L2
```

For a two-dimensional array, `np.linalg.norm(A)` defaults to Frobenius, not spectral. For a one-dimensional array, the default is vector $L_2$. The array's shape matters as well as `ord`. See the [NumPy norm reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) for the complete conventions.

## 5. Now use a norm to choose an update

Let $w\in\mathbb R^d$ be the current parameters, $\Delta$ a proposed increment, and $g=\nabla\mathcal L(w)$ the gradient of a differentiable loss. For a small increment,

$$
\mathcal L(w+\Delta)-\mathcal L(w)
\approx g^\top\Delta.
$$

The gradient predicts the first-order effect of a step. By itself, it does not specify a movement budget. If we minimize the linear expression without any restriction, then for $g\ne0$, choosing $\Delta=-t g$ sends it to $-\infty$ as $t$ grows. That does not mean the actual loss keeps decreasing.

One way to complete the local problem is to choose a norm and a radius $r>0$:

$$
\Delta^\star\in\underset{\|\Delta\|_p\leq r}{\operatorname{argmin}}\;g^\top\Delta.
$$

Keep $g$ fixed. Change only the norm. The optimal step changes:

<div class="table-scroll" role="region" aria-label="Optimal vector increments" tabindex="0">

| Step constraint | One optimal increment, for $g\ne0$ | Best first-order change |
| :--- | :--- | :--- |
| $\|\Delta\|_1\leq r$ | $-r\,\operatorname{sign}(g_k)e_k$, $k\in\arg\max_i\lvert g_i\rvert$ | $-r\|g\|_\infty$ |
| $\|\Delta\|_2\leq r$ | $-r\,g/\|g\|_2$ | $-r\|g\|_2$ |
| $\|\Delta\|_\infty\leq r$ | $-r\,\operatorname{sign}(g)$ | $-r\|g\|_1$ |

</div>

Here $e_k$ is the vector with a one at coordinate $k$ and zeros elsewhere. The sign function is applied coordinate by coordinate, with $\operatorname{sign}(0)=0$. If $g=0$, every feasible step has zero first-order change; choosing zero is sufficient.

### L1: spend on a largest-magnitude coordinate

For any feasible step,

$$
g^\top\Delta\geq-\sum_i\lvert g_i\rvert|\Delta_i|
\geq-\|g\|_\infty\sum_i|\Delta_i|
\geq-r\|g\|_\infty.
$$

Spend the whole budget on a coordinate with maximal $\lvert g_i\rvert$, moving against its sign. This attains the bound. If several coordinates tie, the optimum need not be unique: the budget can be split among those coordinates with the appropriate signs. The constraint permits sparse optimal steps; it does not force a unique sparse solution in every case.

### L2: align with the negative gradient

Cauchy–Schwarz gives

$$
g^\top\Delta\geq-\|g\|_2\|\Delta\|_2\geq-r\|g\|_2.
$$

The step $\Delta=-rg/\|g\|_2$ reaches the bound. This is the normalized negative-gradient step: direction from $g$, length from $r$.

### L-infinity: let each coordinate reach its bound

The constraints $|\Delta_i|\leq r$ are independent. If $g_i>0$, choose $\Delta_i=-r$; if $g_i<0$, choose $\Delta_i=r$. Thus one optimum is $-r\operatorname{sign}(g)$, with objective $-r\sum_i\lvert g_i\rvert$.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/steepest-steps-mobile.svg" /><img src="/figures/steepest-steps.svg" alt="For gradient (3,4) and radius one, optimal steps are (0,-1) under L1, (-0.6,-0.8) under L2, and (-1,-1) under L-infinity. Parallel contour lines touch the three unit balls at these respective points." width="1200" height="460" loading="lazy" /></picture>
<figcaption>Figure 4. Same gradient, different allowed steps. Gray lines are level sets of 3Δ₁ + 4Δ₂; each colored dashed line passes through that constraint's optimum.</figcaption>
</figure>

For $g=(3,4)^\top$ and $r=1$, the best linear changes are $-4$, $-5$, and $-7$. These numbers do not rank three methods under an identical movement budget: the three unit balls contain different steps.

The norms in the last column are called **dual norms**. They measure how large the gradient's linear effect can be on a unit ball. That is why $L_1$ pairs with $L_\infty$, while $L_2$ pairs with itself. These are local linear problems, as in the steepest-descent treatment in [Boyd and Vandenberghe, §9.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

## 6. SGD: a squared Euclidean movement cost

The $L_2$-constrained solution is $-rg/\|g\|_2$. Ordinary gradient descent instead uses $-\eta g$. The two share a direction, but the latter's length is $\eta\|g\|_2$, which changes with the gradient.

A squared movement penalty gives exactly the ordinary gradient step:

$$
\Delta^\star
=\underset{\Delta}{\operatorname{argmin}}\left\{
 g^\top\Delta+\frac{1}{2\eta}\|\Delta\|_2^2
\right\}
=-\eta g,\qquad \eta>0.
$$

Differentiate with respect to $\Delta$: $g+\Delta/\eta=0$. The positive quadratic penalty makes this the unique minimum. A hard radius and a movement penalty are two ways to control a step; their numerical parameters have different meanings.

At training step $t$, let $w_{t-1}$ be the parameters before the update and $g_t=\nabla\mathcal L_t(w_{t-1})$ the gradient of the current minibatch loss. Using that stochastic gradient in the same local problem gives **SGD without momentum or weight decay**:

$$
w_t=w_{t-1}-\eta_t g_t.
$$

The stochastic part concerns how the gradient is estimated. The Euclidean movement cost explains how that estimate is turned into a step. With a matrix parameter and gradient $G_t$, the analogous cost is $\|\Delta W\|_F^2/(2\eta_t)$, and the update is $\Delta W=-\eta_tG_t$.

This separates two choices: which information to use, and how to measure the cost of moving. Adam changes both.

## 7. Adam: a changing, coordinate-weighted L2 geometry

Gradient coordinates can have very different scales, and a single gradient can fluctuate. Adam keeps a signed average of recent gradients and a separate average of their squares. With zero initialization, its state is

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
\end{aligned}
\qquad m_0=v_0=0.
$$

The square is elementwise, and $0\leq\beta_1,\beta_2<1$. The first average can cancel positive and negative gradients; the second records their squared magnitudes. It is a second raw moment, not a variance.

Because the averages start at zero, the observed gradients initially have weights summing to less than one. Divide by those weight sums:

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

The coordinate update is

$$
\Delta_{t,i}=-\eta_t\frac{\widehat m_{t,i}}{\sqrt{\widehat v_{t,i}}+\varepsilon},
\qquad \varepsilon>0.
$$

These are the standard Adam recurrences in [Kingma and Ba, Algorithm 1](https://arxiv.org/abs/1412.6980). Here we consider Adam without weight decay.

### Read the denominator as a movement cost

Freeze the state for this step and define the positive diagonal matrix

$$
D_t=\operatorname{diag}\left(\sqrt{\widehat v_{t,1}}+\varepsilon,\ldots,
\sqrt{\widehat v_{t,d}}+\varepsilon\right).
$$

It defines a weighted Euclidean norm:

$$
\|\Delta\|_{D_t}:=\sqrt{\Delta^\top D_t\Delta}.
$$

Replace SGD's squared Euclidean cost with this squared norm, and replace the current gradient with the signed historical average:

$$
\begin{aligned}
\Delta_t
&=\underset{\Delta}{\operatorname{argmin}}\left\{
\widehat m_t^\top\Delta+\frac{1}{2\eta_t}\|\Delta\|_{D_t}^2\right\}\\
&=-\eta_tD_t^{-1}\widehat m_t.
\end{aligned}
$$

The derivative is $\widehat m_t+D_t\Delta/\eta_t$. Setting it to zero recovers Adam exactly. A larger diagonal entry of $D_t$ makes movement in that coordinate more expensive.

For example, with frozen $\widehat m_t=(3,4)^\top$, $D_t=\operatorname{diag}(9,4)$, and $\eta_t=0.1$,

$$
\Delta_t=-0.1\begin{pmatrix}3/9\\4/4\end{pmatrix}
=\begin{pmatrix}-1/30\\-0.1\end{pmatrix}.
$$

The step is not parallel to $-(3,4)^\top$: the two coordinates pay different movement costs.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/adam-geometry-mobile.svg" /><img src="/figures/adam-geometry.svg" alt="For a fixed linear signal (3,4), a Euclidean unit ball is a circle and the weighted ball with D equal to diag(9,4) is an ellipse. Their optimal arrows point in different directions." width="1080" height="480" loading="lazy" /></picture>
<figcaption>Figure 5. Same linear signal, different movement costs. Arrows minimize the linear objective over each pictured unit ball. These balls visualize cost contours; Adam itself uses the quadratic penalty above, rather than imposing a fixed unit radius.</figcaption>
</figure>

### Why this is not simply a sign update

On the first step, bias correction gives $\widehat m_1=g_1$ and $\widehat v_1=g_1^{\odot2}$. Thus, for a coordinate whose gradient magnitude is much larger than $\varepsilon$, the first update is approximately $-\eta_1\operatorname{sign}(g_{1,i})$.

Later, the numerator and denominator summarize gradient histories. In general the numerator is not the current gradient, and the denominator is not its absolute value. Adam is therefore not simply the $L_\infty$-constrained rule from Section 5.

The weighted-norm interpretation is exact **with the historical state held fixed**. It does not say that $D_t$ is the true loss Hessian, or that Adam is steepest descent on the current loss under a fixed norm. Momentum can even point against the current gradient.

## 8. Muon: from a spectral budget to a matrix update

Adam measures each coordinate separately. A matrix parameter also has an input-output structure. For a linear layer $y=Wx$, holding $x$ fixed gives

$$
\Delta y=(\Delta W)x,\qquad
\|\Delta y\|_2\leq\|\Delta W\|_2\|x\|_2.
$$

A spectral-norm budget on $\Delta W$ controls the largest output change per unit input. This motivates a matrix version of the local problem.

### Keep the linear objective; change the budget

Let $G=\nabla_W\mathcal L$ have the same shape as $W$. The first-order pairing is

$$
\langle G,\Delta W\rangle_F
=\sum_{i,j}G_{ij}(\Delta W)_{ij}
=\operatorname{tr}(G^\top\Delta W).
$$

The Frobenius inner product adds up the first-order effects of the entries. Its subscript does not force us to use Frobenius to constrain the update. Compare these two problems:

$$
\begin{array}{ll}
\displaystyle\min_{\|\Delta W\|_F\leq r}\langle G,\Delta W\rangle_F
&\displaystyle\Longrightarrow\quad\Delta W^\star=-r\frac{G}{\|G\|_F},\\[1em]
\displaystyle\min_{\|\Delta W\|_2\leq r}\langle G,\Delta W\rangle_F
&\displaystyle\Longrightarrow\quad\Delta W^\star=-rUV^\top.
\end{array}
$$

Here $G\ne0$, and $G=U\Sigma V^\top$ is a compact SVD retaining only its nonzero singular values. The first result is the vector $L_2$ argument applied to all entries. The second needs a matrix argument.

Write $G=\sum_{i=1}^{q}\sigma_i u_iv_i^\top$, where $q=\operatorname{rank}(G)$. For any spectral-feasible increment,

$$
\langle G,\Delta W\rangle_F
=\sum_i\sigma_i u_i^\top\Delta Wv_i
\geq-r\sum_i\sigma_i.
$$

Each $u_i,v_i$ is a unit vector, and $\|\Delta W\|_2\leq r$, so each scalar $u_i^\top\Delta Wv_i$ is at least $-r$. The choice $\Delta W=-rUV^\top$ maps each $v_i$ to $-ru_i$, satisfying the budget and attaining all those lower bounds simultaneously.

Thus $-rUV^\top$ is an optimal increment. The best linear change is $-r\|G\|_*$, where $\|G\|_*:=\sum_i\sigma_i$ is the **nuclear norm**, the dual of the spectral norm under this pairing. If $G=0$, choose zero; for a rank-deficient $G$, the optimum need not be unique.

### What changed in the update?

The gradient is $U\Sigma V^\top$. The ideal spectral-budget direction is $UV^\top$: it keeps the singular directions and replaces each nonzero singular value by one. This is a matrix operation, not an elementwise sign.

For $G=\operatorname{diag}(3,1)$ and radius one,

$$
\Delta W_F=-\frac{\operatorname{diag}(3,1)}{\sqrt{10}},
\qquad
\Delta W_2=-I_2.
$$

The Frobenius budget shares a total squared-entry budget; the spectral budget permits a unit change along both orthogonal singular directions.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/matrix-steps-mobile.svg" /><img src="/figures/matrix-steps.svg" alt="On the diagonal update slice diag(a,b), gradient diag(3,1) gives the Frobenius optimum (-3,-1) divided by square root of ten, and the spectral optimum (-1,-1)." width="1080" height="480" loading="lazy" /></picture>
<figcaption>Figure 6. Matrix updates on the diagonal slice ΔW = diag(a,b), with G = diag(3,1). Lines are level sets of 3a + b. The spectral-feasible update has Frobenius length square root of two, so the two optima do not use an identical Euclidean budget.</figcaption>
</figure>

### From the exact direction to practical Muon

The $UV^\top$ direction is the geometric core connecting this calculation to Muon. In practice, Muon orthogonalizes a momentum-based update and uses a short Newton–Schulz iteration to approximate that operation; see [Keller Jordan's original description](https://kellerjordan.github.io/posts/muon/).

A schematic momentum version is

$$
\begin{aligned}
B_t&=\mu B_{t-1}+(1-\mu)G_t,\qquad B_0=0,\\
W_t&=(1-\eta_t\lambda)W_{t-1}
-\eta_t s_{m,n}\operatorname{Ortho}(B_t).
\end{aligned}
$$

Here $0\leq\mu<1$ controls momentum, $\lambda\geq0$ is decoupled weight decay, and $s_{m,n}$ denotes an implementation-dependent shape scaling. Ideally, $\operatorname{Ortho}(B_t)$ returns the compact-SVD product $UV^\top$ for $B_t$; a finite numerical iteration need not return that product exactly.

Implementations may also use a Nesterov-style combination in place of $B_t$. Momentum, shape scaling, decay, and the approximation method are additional algorithm choices, not consequences forced by the spectral-ball minimization. The local problem explains a direction, rather than specifying every detail of a training recipe. Nor does orthogonalizing an update make the updated weight matrix itself orthogonal.

## 9. Put SGD, Adam, and Muon in the same picture

The common structure is a linear signal plus a rule for how much movement is allowed or how expensive it is:

<div class="table-scroll" role="region" aria-label="Optimizer geometry comparison" tabindex="0">

| Method or idealized core | Linear signal | Movement rule | Resulting increment |
| :--- | :--- | :--- | :--- |
| Plain SGD | Current stochastic gradient $g_t$ | Squared Euclidean cost $\|\Delta\|_2^2/(2\eta_t)$ | $-\eta_tg_t$ |
| Adam, with state fixed | Signed historical average $\widehat m_t$ | Squared weighted norm $\|\Delta\|_{D_t}^2/(2\eta_t)$ | $-\eta_tD_t^{-1}\widehat m_t$ |
| Ideal Muon core | Matrix momentum signal $B_t=U\Sigma V^\top$ | Spectral budget $\|\Delta W\|_2\leq r$ | $-rUV^\top$ |

</div>

For matrix SGD, replace the Euclidean vector cost by the Frobenius cost. The Muon row isolates the exact spectral direction before numerical approximation and the other recipe choices above. The table compares mathematical structures; it does not rank training performance or equate the methods' step sizes.

Norms make the difference in movement rules explicit: Euclidean cost for SGD, a changing coordinate-weighted cost for Adam, and a matrix spectral budget for Muon's ideal direction. The information feeding those rules matters too: the current gradient and a historical average are not interchangeable.

Finally, minimizing a linear prediction is a local calculation. A direction $d$ with $g^\top d<0$ decreases the same differentiable loss for sufficiently small positive steps, but a large step can still increase it. A minibatch gradient need not be a descent direction for the full-data loss, and historical momentum need not be a descent direction even for the current batch. Geometry explains a proposed step; it does not replace checking its finite-step behavior.
