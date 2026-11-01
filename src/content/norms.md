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

## 6. Define the descent problem first

The training objective is to minimize a loss $\mathcal L(\Theta)$ over parameters $\Theta$. At the current parameters $\Theta_{t-1}$, we choose an increment $\Delta$ and set $\Theta_t=\Theta_{t-1}+\Delta$.

For a differentiable loss, the current gradient $G_t=\nabla\mathcal L(\Theta_{t-1})$ gives the local model

$$
\mathcal L(\Theta_{t-1}+\Delta)
\approx\mathcal L(\Theta_{t-1})+\langle G_t,\Delta\rangle.
$$

The pairing is a dot product for vectors and the Frobenius inner product for matrices. Since the first term does not depend on $\Delta$, the local descent problem is to make $\langle G_t,\Delta\rangle$ small while controlling the size of the move.

### Complete the problem with a movement rule

Let $\rho_t(\Delta)$ be our chosen norm of the increment. Two ways to complete the local problem are

$$
\boxed{\min_{\rho_t(\Delta)\leq r_t}\langle G_t,\Delta\rangle}
$$

and

$$
\boxed{\min_\Delta\left\{\langle G_t,\Delta\rangle
+\frac{1}{2\eta_t}\rho_t(\Delta)^2\right\}},
\qquad \eta_t>0.
$$

The first imposes a hard radius $r_t$; the second makes large moves increasingly expensive. They need not produce the same step length. Choosing a norm specifies the geometry, while the radius or penalty coefficient determines how that geometry is used to size a step.

We can also use a historical gradient estimate instead of the current gradient. Write the chosen linear signal as $S_t$. This gives a common template:

$$
\min_\Delta\bigl\{\langle S_t,\Delta\rangle
+\text{movement cost or constraint}\bigr\}.
$$

To compare geometries, first **hold $S_t$ fixed and change only the norm**. After solving those local problems, we can specify how an optimizer constructs $S_t$ and its movement rule. A norm alone does not determine a complete optimizer.

## 7. Change the norm, compare the resulting steps

The three choices below answer the same question: given a linear signal, which increment gives the greatest predicted improvement for the permitted movement?

### Choice 1: Euclidean or Frobenius movement

For a vector signal $s$, use $\rho(\Delta)=\|\Delta\|_2$. Under a hard radius, Section 5 gave the normalized step $-rs/\|s\|_2$ for $s\ne0$. With a squared movement cost, we instead solve

$$
\Delta^\star
=\underset{\Delta}{\operatorname{argmin}}\left\{
 s^\top\Delta+\frac{1}{2\eta}\|\Delta\|_2^2
\right\}.
$$

Differentiation gives $s+\Delta/\eta=0$, hence

$$
\boxed{\Delta^\star=-\eta s.}
$$

Every coordinate pays the same squared-movement price. For a matrix signal $S$, the corresponding norm is Frobenius:

$$
\underset{\Delta}{\operatorname{argmin}}\left\{
\langle S,\Delta\rangle_F+\frac{1}{2\eta}\|\Delta\|_F^2\right\}
=-\eta S.
$$

This is the geometry obtained by treating all parameter entries as one Euclidean vector. The hard-radius version has a fixed norm $r$; the penalized step has norm $\eta\|s\|_2$, or $\eta\|S\|_F$ for a matrix. Their directions agree, but their lengths follow different rules.

### Choice 2: a coordinate-weighted L2 norm

Now change the movement norm. Let $D_t$ be a positive diagonal matrix, held fixed during the local minimization, and define

$$
\rho_t(\Delta)=\|\Delta\|_{D_t}
:=\sqrt{\Delta^\top D_t\Delta}.
$$

The Euclidean ball becomes an ellipsoid. A larger diagonal entry of $D_t$ makes movement in that coordinate more expensive. Keep the same kind of linear objective and quadratic penalty:

$$
\Delta_t
=\underset{\Delta}{\operatorname{argmin}}\left\{
 s_t^\top\Delta+\frac{1}{2\eta_t}\Delta^\top D_t\Delta\right\}.
$$

Its derivative is $s_t+D_t\Delta/\eta_t$, so

$$
\boxed{\Delta_t=-\eta_tD_t^{-1}s_t.}
$$

We have not yet said how $D_t$ or $s_t$ should be chosen. This result only says what update follows from a given positive diagonal movement cost and a given linear signal.

For $s_t=(3,4)^\top$, $D_t=\operatorname{diag}(9,4)$, and $\eta_t=0.1$,

$$
\Delta_t=-0.1\begin{pmatrix}3/9\\4/4\end{pmatrix}
=\begin{pmatrix}-1/30\\-0.1\end{pmatrix}.
$$

The Euclidean choice with the same signal and penalty coefficient would give $(-0.3,-0.4)^\top$. Changing the norm changes both the direction and the amount assigned to each coordinate.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/adam-geometry-mobile.svg" /><img src="/figures/adam-geometry.svg" alt="For a fixed linear signal (3,4), a Euclidean unit ball is a circle and the weighted ball with D equal to diag(9,4) is an ellipse. Their optimal arrows point in different directions." width="1080" height="480" loading="lazy" /></picture>
<figcaption>Figure 5. Same linear signal, different movement norms. Arrows minimize the linear objective over each pictured unit ball. These balls visualize cost contours; the penalized update above does not impose a fixed unit radius.</figcaption>
</figure>

### Choice 3: the matrix spectral norm

A matrix can also be measured by its action on inputs. For a linear layer $y=Wx$, holding $x$ fixed gives

$$
\Delta y=(\Delta W)x,\qquad
\|\Delta y\|_2\leq\|\Delta W\|_2\|x\|_2.
$$

This suggests a different movement rule: bound the largest output change per unit input, using $\rho(\Delta W)=\|\Delta W\|_2$. To isolate its effect, keep a fixed matrix signal $G$ and compare two hard-radius problems:

$$
\begin{array}{ll}
\displaystyle\min_{\|\Delta W\|_F\leq r}\langle G,\Delta W\rangle_F
&\displaystyle\Longrightarrow\quad\Delta W^\star=-r\frac{G}{\|G\|_F},\\[1em]
\displaystyle\min_{\|\Delta W\|_2\leq r}\langle G,\Delta W\rangle_F
&\displaystyle\Longrightarrow\quad\Delta W^\star=-rUV^\top.
\end{array}
$$

Here $G\ne0$, and $G=U\Sigma V^\top$ is a compact SVD retaining only its nonzero singular values. The first result is the vector $L_2$ argument applied to all entries. For the second, write $G=\sum_{i=1}^{q}\sigma_i u_iv_i^\top$, where $q=\operatorname{rank}(G)$.

Under the spectral constraint, each $v_i$ and $u_i$ has unit Euclidean length, so $u_i^\top\Delta Wv_i\geq-r$. Therefore

$$
\langle G,\Delta W\rangle_F
=\sum_i\sigma_i u_i^\top\Delta Wv_i
\geq-r\sum_i\sigma_i.
$$

The choice $\Delta W=-rUV^\top$ has spectral norm $r$ and maps every $v_i$ to $-ru_i$. It attains the bound, proving the claimed optimum. The best linear value is $-r\|G\|_*$, where $\|G\|_*:=\sum_i\sigma_i$ is the nuclear norm, dual to the spectral norm under the Frobenius pairing. If $G=0$, choose zero; for rank-deficient $G$, the optimum need not be unique.

The update retains the singular directions but replaces every nonzero singular value by the same magnitude $r$. In contrast, the Frobenius solution scales all the original singular values by a common factor and preserves their relative sizes.

For $G=\operatorname{diag}(3,1)$ and radius one,

$$
\Delta W_F=-\frac{\operatorname{diag}(3,1)}{\sqrt{10}},
\qquad
\Delta W_2=-I_2.
$$

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/matrix-steps-mobile.svg" /><img src="/figures/matrix-steps.svg" alt="On the diagonal update slice diag(a,b), gradient diag(3,1) gives the Frobenius optimum (-3,-1) divided by square root of ten, and the spectral optimum (-1,-1)." width="1080" height="480" loading="lazy" /></picture>
<figcaption>Figure 6. Same matrix signal and radius, different norms. On the diagonal slice ΔW = diag(a,b), the Frobenius budget shares a squared-entry budget, while the spectral budget permits a unit change in each singular direction. Their Euclidean movement budgets are different.</figcaption>
</figure>

The inner product has stayed the same in both problems:

$$
\langle G,\Delta W\rangle_F
=\sum_{i,j}G_{ij}(\Delta W)_{ij}
=\operatorname{tr}(G^\top\Delta W).
$$

It describes the first-order effect on loss. The norm specifies which moves we compare. A Frobenius inner product does not require a Frobenius movement budget.

## 8. From those norm choices to SGD, Adam, and Muon

We now have three update rules derived from three movement geometries. To identify actual optimizers, we still need to specify their linear signals, how their norm or state is chosen, and how a step is sized.

### Euclidean cost + a stochastic gradient: SGD

<div class="section-alias" id="6-sgd-a-squared-euclidean-movement-cost"></div>

At step $t$, take $s_t=g_t=\nabla\mathcal L_t(w_{t-1})$, the gradient of the current minibatch loss. Use the squared Euclidean movement cost from Choice 1:

$$
\Delta_t=-\eta_tg_t,\qquad
w_t=w_{t-1}+\Delta_t.
$$

This is **SGD without momentum or weight decay**. For matrix parameters, use the Frobenius norm and obtain $\Delta W_t=-\eta_tG_t$. The stochastic part concerns how we estimate the gradient; the Euclidean or Frobenius cost explains how that estimate becomes an increment.

### Weighted L2 cost + historical statistics: Adam

<div class="section-alias" id="7-adam-a-changing-coordinate-weighted-l2-geometry"></div>

Choice 2 gave $\Delta_t=-\eta_tD_t^{-1}s_t$ for any fixed positive diagonal $D_t$. Adam makes a particular choice of both $s_t$ and $D_t$ from gradient history.

Starting at $m_0=v_0=0$, define

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
\end{aligned}
\qquad 0\leq\beta_1,\beta_2<1.
$$

The first average is signed; the second uses elementwise squares and records a second raw moment, not a variance. Correct the zero-initialization weight sums:

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

Now choose the linear signal and movement metric:

$$
\begin{aligned}
s_t&=\widehat m_t,\\
D_t&=\operatorname{diag}\left(\sqrt{\widehat v_{t,1}}+\varepsilon,\ldots,
\sqrt{\widehat v_{t,d}}+\varepsilon\right),\qquad\varepsilon>0.
\end{aligned}
$$

Substitute them into the weighted-norm solution from Choice 2. Coordinate by coordinate, it becomes

$$
\boxed{\Delta_{t,i}=-\eta_t
\frac{\widehat m_{t,i}}{\sqrt{\widehat v_{t,i}}+\varepsilon}.}
$$

This recovers Adam without weight decay; the historical statistics are the standard choices in [Kingma and Ba, Algorithm 1](https://arxiv.org/abs/1412.6980). The norm interpretation is exact with those statistics fixed during the local minimization. It does not imply that $D_t$ is the loss Hessian or that Adam uses a fixed norm throughout training.

Adam is also not simply the $L_\infty$ sign rule from Section 5. Its first update is approximately sign-shaped when each nonzero gradient is large relative to $\varepsilon$, but later its numerator and denominator summarize histories. The signed average can even point against the current gradient.

### Spectral budget + matrix momentum: Muon's core

<div class="section-alias" id="8-muon-from-a-spectral-budget-to-a-matrix-update"></div>

Choice 3 gave $-rUV^\top$ for a fixed matrix signal with compact SVD $U\Sigma V^\top$. Apply that operation to a matrix momentum signal:

$$
B_t=\mu B_{t-1}+(1-\mu)G_t,\qquad
B_0=0,\quad0\leq\mu<1.
$$

If $B_t=U_t\Sigma_tV_t^\top$, the ideal spectral-budget solution is

$$
\boxed{\Delta W_t=-r_tU_tV_t^\top.}
$$

This is the geometric connection to Muon: the momentum matrix's singular directions are retained, while its nonzero singular values are replaced by one before applying a step scale. It is a matrix operation, not an elementwise sign.

Practical Muon uses a short Newton–Schulz iteration to approximate this operation on a momentum-based update; see [Keller Jordan's original description](https://kellerjordan.github.io/posts/muon/). A schematic momentum version is

$$
W_t=(1-\eta_t\lambda)W_{t-1}
-\eta_t s_{m,n}\operatorname{Ortho}(B_t).
$$

Here $\lambda\geq0$ is decoupled weight decay, $s_{m,n}$ is an implementation-dependent shape scaling, and ideally $\operatorname{Ortho}(B_t)=U_tV_t^\top$. In that idealization, the movement radius is $r_t=\eta_t s_{m,n}$ for a positive shape scale. A finite numerical iteration need not return the exact SVD product.

Implementations may use a Nesterov-style combination instead of $B_t$. Momentum, shape scaling, decay, and numerical approximation are additional choices beyond the norm argument. The spectral problem explains the core direction; it does not uniquely specify every Muon implementation. Orthogonalizing an update also does not make the updated weight matrix itself orthogonal.

### The same descent template, with different choices

<div class="section-alias" id="9-put-sgd-adam-and-muon-in-the-same-picture"></div>

<div class="table-scroll" role="region" aria-label="Optimizer geometry comparison" tabindex="0">

| Norm and movement rule | Linear signal | Derived increment | Optimizer connection |
| :--- | :--- | :--- | :--- |
| Euclidean or Frobenius, squared cost | Current stochastic gradient $g_t$ | $-\eta_tg_t$ | Plain SGD |
| Coordinate-weighted $L_2$, squared cost | Historical average $\widehat m_t$, with Adam's $D_t$ | $-\eta_tD_t^{-1}\widehat m_t$ | Adam, with state fixed |
| Matrix spectral norm, hard budget | Matrix momentum $B_t=U_t\Sigma_tV_t^\top$ | $-r_tU_tV_t^\top$ | Ideal Muon core |

</div>

The logic runs from **a local descent problem, to a choice of movement norm, to the resulting increment, and then to an optimizer's choices of signal and step size**. The three methods differ in their geometry, but also in how they construct the information that geometry acts on. The table compares those choices; it does not rank training performance or equate step sizes across different norms.

Finally, this is a local calculation. A direction $d$ with $g^\top d<0$ decreases the same differentiable loss for sufficiently small positive steps, but a large step can still increase it. A minibatch gradient need not decrease the full-data loss, and historical momentum need not decrease the current batch loss. The norm argument explains a proposed update, while its finite-step behavior still needs to be checked.
