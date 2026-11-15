---
title: "When the Reward Model Lies"
description: "False positives, false negatives, and what they do to GRPO, Flow-GRPO and DiffusionNFT, from a two-action bandit to generating handwritten digits."
---

Every RL fine-tuning pipeline trusts a reward model that is sometimes wrong. A verifier passes a wrong proof. A classifier calls a 5 a 3. A preference model misses a correct answer and scores it zero. The natural worry is that training will go to the wrong place.

The answer depends on two things: what kind of mistakes the reward model makes, and which algorithm consumes the reward. This article starts from the smallest model where the question can be answered exactly. It then adds the pieces that real systems have: KL regularisation, group-normalised advantages, reward-weighted (negative-aware) updates, many classes, and errors that depend on the sample. At the end, 78 runs on a flow model drawing MNIST digits test the predictions. Most hold. The ones that don't are the interesting part: Section 7 traces each miss to a specific simplification in the theory and corrects it.

## 1. The two-action model

The policy has two outputs: $a=1$ (correct) and $a=0$ (wrong). Write $p=\pi_\theta(a=1)=\sigma(\theta)$. The true reward is $r(a)=\mathbb 1[a=1]$, but we only observe a noisy binary reward $R$ with

$$
P(R=1\mid a=0)=\mathrm{FP},\qquad P(R=0\mid a=1)=\mathrm{FN}.
$$

A **false positive** gives full marks to a wrong answer. A **false negative** gives zero to a correct one. The errors are independent of everything except whether the answer is correct, so the reward model is equally unreliable on every wrong output. Section 5 drops this assumption.

Everything in the next sections follows from one line:

$$
\mathbb E[R\mid a]=\mathrm{FP}+\underbrace{(1-\mathrm{FP}-\mathrm{FN})}_{\Delta}\,r(a).
$$

**The noisy reward is an affine function of the true reward.** The intercept is FP and the slope is the *signal gap* $\Delta=1-\mathrm{FP}-\mathrm{FN}$.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-affine-mobile.svg" /><img src="/figures/rn-affine.svg" alt="Expected observed reward for wrong and correct answers. Clean reward goes from 0 to 1. FP=0.3 raises the left end to 0.3, FN=0.3 lowers the right end to 0.7, and both together give 0.3 to 0.7. With FP=FN=0.6 the line slopes downward." width="1000" height="560" /></picture>
<figcaption>Noise does not reorder the two actions while Δ &gt; 0. It only compresses the gap between them. When FP + FN &gt; 1, the gap changes sign.</figcaption>
</figure>

So there are three regimes:

<div class="table-scroll" role="region" aria-label="Regimes" tabindex="0">

| Condition | What the reward model prefers |
| :--- | :--- |
| $\mathrm{FP}+\mathrm{FN}<1$ | correct answers, with the gap shrunk to $\Delta$ |
| $\mathrm{FP}+\mathrm{FN}=1$ | nothing: $R$ is independent of $a$ |
| $\mathrm{FP}+\mathrm{FN}>1$ | wrong answers |

</div>

A reward model with 40% FP and 40% FN is terrible, but it still points the right way.

## 2. Plain policy gradient: same destination, slower

Without regularisation, the objective is $J(\theta)=\mathbb E_{a\sim\pi_\theta}[R]$. Using the affine relation,

$$
J_{\text{noisy}}(\theta)=\mathrm{FP}+\Delta\,J_{\text{true}}(\theta),
\qquad
\nabla_\theta J_{\text{noisy}}=\Delta\,\nabla_\theta J_{\text{true}}.
$$

The constant FP disappears from the gradient, because a baseline removes it anyway. The expected update is the clean update multiplied by $\Delta$. For the logit, $\dot\theta=\eta\,\Delta\,p(1-p)$.

The consequences are:

* **Same fixed point.** If $\Delta>0$, both objectives are maximised by $p=1$.
* **Slower by a factor of $\Delta$ in expectation.** Halving the gap is like halving the learning rate.
* **Noisier.** A single reward has variance $m(1-m)$ with $m=\mathrm{FP}+\Delta p$. The clean variance is $p(1-p)$, which goes to zero as the policy improves. The noisy variance does not: at $p=1$ it is still $\mathrm{FN}(1-\mathrm{FN})$.

In expectation the update depends only on $\Delta$, so FP and FN are interchangeable here. A 30% FP rate costs exactly as much as a 30% FN rate.

## 3. GRPO and the group standard deviation

GRPO does not use $R$ directly. For a group of $G$ samples it uses

$$
A_i=\frac{R_i-\operatorname{mean}(R_{1:G})}{\operatorname{std}(R_{1:G})}.
$$

Dividing by the group std rescales every update by roughly $1/s$, where $s=\sqrt{m(1-m)}$. This partly undoes the $\Delta$ slowdown: when noise compresses the rewards, the normalisation stretches them back. It does not undo it completely, because $s$ measures the total spread of $R$, noise included, not just the part caused by the action.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-curves-mobile.svg" /><img src="/figures/rn-curves.svg" alt="Simulated learning curves of P(correct) from 0.1 under clean reward, FP=0.3, FN=0.3 and both, for three update rules. All runs converge to 1. The FP=FN=0.3 curve is slowest in every panel. Under the reward-weighted rule, FP=0.3 is clearly slower than FN=0.3." width="1300" height="400" loading="lazy" /></picture>
<figcaption>Monte Carlo simulation of the two-action model (64 runs, G = 8, starting from p = 0.1). Every run reaches the correct answer, only at different speeds. Left: group-mean baseline. Middle: GRPO. Right: reward-weighted regression toward π<sub>old</sub>·r, which is how I first (wrongly) modelled DiffusionNFT; see Section 7.1.</figcaption>
</figure>

The right panel of the next figure measures how many GRPO steps it takes to reach $p=0.95$ over a grid of FP and FN. The contour lines follow the anti-diagonals: what matters is mostly FP + FN, which is $\Delta$ again.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-heat-mobile.svg" /><img src="/figures/rn-heat.svg" alt="Three heat maps over FP and FN from 0 to 0.45. The signal gap decreases along anti-diagonals. The odds factor log((1-FN)/FP) depends mostly on FP. Simulated GRPO steps to reach 0.95 grow along anti-diagonals, like the signal gap." width="1400" height="420" loading="lazy" /></picture>
<figcaption>Left: the signal gap Δ, which sets policy-gradient speed. Middle: the log odds factor that sets the speed of a reward-weighted update. Right: simulated GRPO, whose cost follows Δ.</figcaption>
</figure>

## 4. Where the destination does move

Plain policy gradient always ends at $p=1$, so noise can only slow it down. Two common ingredients change that.

### 4a. KL regularisation

With a reverse-KL penalty to a reference policy, the objective $\mathbb E_\pi[R]-\beta\,\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$ has the closed-form optimum

$$
\pi^*(a)\propto\pi_{\text{ref}}(a)\exp\!\big(\mathbb E[R\mid a]/\beta\big)
\;\Longrightarrow\;
\operatorname{logit}p^*=\operatorname{logit}p_{\text{ref}}+\frac{\Delta}{\beta}.
$$

**With KL, reward noise is equivalent to a larger KL coefficient, $\beta_{\text{eff}}=\beta/\Delta$.** The end point moves toward the reference policy. It still depends only on $\Delta$, so FP and FN remain interchangeable for a raw reward.

GRPO's normalisation breaks the symmetry. In the logit, the expected dynamics are

$$
\dot\theta=p(1-p)\Big[\frac{\Delta}{s(p)}-\beta(\theta-\theta_{\text{ref}})\Big],
\qquad s(p)=\sqrt{m(1-m)},\; m=\mathrm{FP}+\Delta p.
$$

Now look at $s$ as $p\to1$:

* If $\mathrm{FN}=0$, then $m\to1$ and $s\to0$. The reward push $\Delta/s$ grows without bound and eventually beats the linear KL pull. Below a critical $\beta$ the policy saturates at $p=1$ as if there were no KL at all. This happens with a clean reward too.
* If $\mathrm{FN}>0$, then $s\ge\sqrt{\mathrm{FN}(1-\mathrm{FN})}$ near convergence. The push stays bounded, and the KL term wins at a finite $p^*<1$.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-kl-mobile.svg" /><img src="/figures/rn-kl.svg" alt="P(correct) where training settles against the KL coefficient on a log scale. Raw reward with any noise of the same Delta gives a smooth decreasing curve. GRPO with clean reward and with FP=0.2 stays at 1 until beta is around 0.6 and then drops sharply. GRPO with FN=0.2 decreases smoothly." width="1000" height="590" loading="lazy" /></picture>
<figcaption>End point of the expected KL-regularised dynamics, starting from p<sub>ref</sub> = 0.1. Under GRPO normalisation, false negatives act as a regulariser: they keep the group std away from zero, so the KL term can hold the policy.</figcaption>
</figure>

This is the first place where the two error types behave differently, and the difference comes entirely from the std in the denominator.

### 4b. Reward-weighted and negative-aware updates (DiffusionNFT)

DiffusionNFT does not use a policy gradient. It treats a reward $r\in[0,1]$ as the probability that a sample is "good" and trains the model toward the positive distribution

$$
\pi^+(x)\propto\pi_{\text{old}}(x)\,r(x),
$$

while explicitly pushing away from $\pi^-(x)\propto\pi_{\text{old}}(x)(1-r(x))$. It does this with implicit velocities $v^\pm=(1\mp\beta)v_{\text{old}}\pm\beta v_\theta$, trained by flow matching on the forward process. Plug the noisy binary reward in as $r$:

$$
\frac{\pi^+(1)}{\pi^+(0)}=\frac{\pi_{\text{old}}(1)}{\pi_{\text{old}}(0)}\cdot\frac{1-\mathrm{FN}}{\mathrm{FP}}.
$$

**A full step multiplies the odds by $(1-\mathrm{FN})/\mathrm{FP}$, a ratio rather than a difference.** That is very asymmetric:

<div class="table-scroll" role="region" aria-label="Odds factor" tabindex="0">

| FP | FN | $\Delta$ (policy gradient) | $\frac{1-\mathrm{FN}}{\mathrm{FP}}$ (reward-weighted) |
| ---: | ---: | ---: | ---: |
| 0 | 0.5 | 0.5 | $\infty$ |
| 0.5 | 0 | 0.5 | 2 |
| 0.1 | 0.1 | 0.8 | 9 |
| 0.3 | 0.3 | 0.4 | 2.3 |

</div>

For a likelihood-ratio style update, false negatives only throw away some good samples. False positives contaminate the positive set. The right panel of the learning-curve figure shows this: FP = 0.3 is visibly slower than FN = 0.3. The middle heat map shows the cost depending mostly on FP.

This was my prediction before running anything. It is wrong, and not only because of the details. Section 7 shows that the odds-ratio picture misdescribes what the NFT loss actually optimises.

## 5. Beyond two actions: when noise can be exploited

So far the reward model was equally wrong on every wrong answer. Real reward models are not. Two generalisations matter.

**Class-dependent errors.** With $K$ outputs and $q_c=\mathbb E[R\mid c]$, plain policy gradient converges to $\arg\max_c q_c$, and the KL optimum is $\pi^*(c)\propto\pi_{\text{ref}}(c)e^{q_c/\beta}$. Suppose the target is 3 and the classifier calls 8s a 3 with probability $\rho$:

* If $\rho<1-\mathrm{FN}$, the policy still ends at 3s. With KL, 8s keep extra mass, more than any other wrong digit.
* If $\rho>1-\mathrm{FN}$, the ranking flips and the policy converges to drawing 8s. A 20% FN rate on true 3s plus a 90% FP rate on 8s is enough.

**Sample-dependent errors.** For a generative model, the "action" is an image, and the error rate is a function $\mathrm{FP}(x)$, not a constant. Then $\mathbb E[R\mid x]$ is no longer affine in the true reward, and its maximiser can be different. Within the set of wrong images, the policy can move toward the ones the reward model misreads most. This is reward hacking, and none of the fixed-point results above protect against it. They all relied on noise being independent of $x$ given correctness.

The two-action model answers "does noise move the optimum?" with *no* (plain PG), *toward the reference* (KL), or *depends on FN* (GRPO + KL). A rich policy adds a fourth answer: *yes, toward whatever the reward model gets wrong*.

## 6. The MNIST experiment

To test all of this, I trained a small unconditional rectified-flow UNet on MNIST (50 epochs). Its samples are close to uniform over digits, with 14–17% 3s, near the toy model's $p_0=0.1$. I then RL fine-tuned it to draw 3s for 300 iterations, with 128 samples per iteration split into groups of 16. A CNN oracle (98.8% test accuracy) defines the true reward $\mathbb 1[\text{oracle says }3]$. The training reward is a corrupted version of it. Every configuration ran with two seeds on a single laptop RTX 4060, 78 runs in total.

I compare three algorithms:

* **Flow-GRPO**: SDE sampling with per-step Gaussian log-probabilities, group-normalised advantages, PPO clipping, and a KL to the base model.
* **DiffusionNFT**: the negative-aware forward-process loss with normalised rewards, $r=\tfrac12+\tfrac12\operatorname{clip}(A/2,-1,1)$.
* **DiffusionNFT-raw**: the same loss with the binary reward used directly as the optimality probability.

Code, logs and a per-run table are in [`experiments/reward_noise_mnist`](https://github.com/daiyixiang666/daiyixiang666.github.io/tree/main/experiments/reward_noise_mnist).

### Image-independent noise: same destination, slower, with one exception

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-curves-mobile.svg" /><img src="/figures/rn-mnist-curves.svg" alt="Share of 3s over 300 RL iterations for Flow-GRPO, DiffusionNFT with normalised reward, and DiffusionNFT with raw reward, under clean reward, FP=0.3, FN=0.3 and both. Most curves rise toward 1, and FP=FN=0.3 is slowest. For raw-reward DiffusionNFT with FN=0.3, the curve plateaus near 0.6 and then collapses to about 0.05 after iteration 240." width="1300" height="400" loading="lazy" /></picture>
<figcaption>Share of samples the oracle labels 3. Thick lines average two seeds; thin lines are individual seeds. Compare with the toy simulation in Section 3.</figcaption>
</figure>

For Flow-GRPO the toy model's prediction holds almost exactly. FP = 0.3 and FN = 0.3 have the same $\Delta$, and they cost about the same: the share of 3s reaches 0.9 at iterations 180 and 190 on average, against 140 for clean. Both end above 0.96. With FP = FN = 0.3 ($\Delta=0.4$), the curve is still climbing at iteration 300 (0.71 and 0.83). Normalised DiffusionNFT behaves the same way, only a little slower.

**Raw-reward DiffusionNFT is the exception, and it fails in the direction opposite to Section 4b.** With FP = 0.3 it is only mildly slower than clean (0.97 at the end). With FN = 0.3 it climbs to about 0.6, stays there for about 100 iterations, and then collapses in both seeds, to 0.07 and 0.04 by iteration 300. The sample grids show what it collapses *to*:

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-grids-mobile.svg" /><img src="/figures/rn-mnist-grids.svg" alt="Four 8-by-8 sample grids. Flow-GRPO with clean reward produces varied 3s. Flow-GRPO when 8s are rewarded more than 3s produces scribbly 8s. Flow-GRPO with flipped reward produces many digits but few 3s. Raw-reward DiffusionNFT with FN=0.3 produces almost entirely blank black squares with a few small 3s." width="1300" height="360" loading="lazy" /></picture>
<figcaption>Final samples (seed 0). The collapsed run produces blank canvases, which the oracle happens to label "1".</figcaption>
</figure>

The model does not drift to another digit. It erases its output. At the plateau around iteration 240, about 40% of samples are already blank, and that is where the non-3 mass went. So raw NFT first removed wrong digits by erasing them, then erased the 3s too. KL slows this down but does not prevent it: with KL, the end points are 0.49 and 0.67.

Section 7 derives where this comes from: a term in the NFT update that amplifies the model's own fitting error whenever the average raw reward is below one half.

### Where the destination moves

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-destination-mobile.svg" /><img src="/figures/rn-mnist-destination.svg" alt="Final share of 3s (solid) and 8s (hatched) for five reward settings and three algorithms. Clean: almost all 3s. FP=FN=0.45: 3s between 0.19 and 0.37. FP=FN=0.6: 3s fall to 0.04 to 0.13, below the base model's 0.15. 8s rewarded at 0.7: almost all 3s. 8s rewarded at 0.9 with FN=0.3: 83 to 98 percent 8s and few 3s." width="1100" height="430" loading="lazy" /></picture>
<figcaption>End of training, no KL. The dashed line is the base model's share of 3s.</figcaption>
</figure>

The sign and ranking predictions all hold, for all three algorithms:

* **FP = FN = 0.6** ($\Delta<0$): the models learn to *avoid* 3s. The share falls from 0.15 to between 0.04 and 0.13.
* **FP = FN = 0.45** ($\Delta=0.1$): weak drift in the right direction, ending between 0.19 and 0.37 after 300 iterations.
* **8s rewarded at 0.7**, below the 1.0 of a true 3: every algorithm still ends on 3s (0.93–0.996).
* **8s rewarded at 0.9 with FN = 0.3**: now $q_8=0.9>q_3=0.7$, and every algorithm converges to 8s (83–98%). The noisy reward settles around 0.87, which is higher than the 0.7 that drawing perfect 3s would earn. The policy is optimising exactly what it was given.

### KL: noise pulls the end point back, but GRPO's FN effect is only partly visible

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-kl-mobile.svg" /><img src="/figures/rn-mnist-kl.svg" alt="Final share of 3s with and without KL for each algorithm and four reward settings. Without KL almost every setting ends near 1. With KL, clean ends around 0.9 to 0.97, FP=0.3 and FN=0.3 end lower, and the 8-confusion setting ends lower too. Raw-reward DiffusionNFT with FN=0.3 is at 0.2 without KL and 0.5 to 0.67 with KL." width="1300" height="370" loading="lazy" /></picture>
<figcaption>Gray: no KL. Purple: with KL (Flow-GRPO β = 0.04, NFT β = 0.1). Two seeds each.</figcaption>
</figure>

With KL, noise moves every end point back toward the base model, as $\beta_{\text{eff}}=\beta/\Delta$ predicts. For Flow-GRPO the ordering is clean 0.95 > FP-only 0.86 > FN-only 0.80, so FN does hold the policy furthest from saturation. But Section 4a also predicted that clean and FP-only would run away to 100% despite the KL, and that did not happen at this $\beta$ within 300 iterations. Either this $\beta$ is above the critical value, or the runaway needs longer. With the KL, the 8-confusion setting keeps 6–13% 8s, against 1–2% for clean, which is the $\pi_{\text{ref}}(c)e^{q_c/\beta}$ prediction.

### The weak reward model was not hacked

I expected the reward from an MLP trained on 300 labels (82% test accuracy) to be exploited, with its reward rising while the oracle's stalled. Instead both rose together, ending at 0.95–0.997 on the oracle for every algorithm. The likely reason is that this weak model is a good detector of 3s in particular: it says "3" for 78% of real 3s and at most 9% of any other digit. Starting from a model that only draws plausible digits, the easiest way to earn its reward is to draw real 3s. Hacking would need images off the data manifold that fool the MLP, and 300 small steps did not find any. This is a statement about this reward model and budget, not a general one.

### Scorecard

<div class="table-scroll" role="region" aria-label="Scorecard" tabindex="0">

| Prediction | Result |
| :--- | :--- |
| FP and FN with the same $\Delta$ cost the same under policy gradient | holds (Flow-GRPO, NFT-normalised) |
| FP hurts reward-weighted (raw NFT) updates more than FN | **wrong**: FN makes raw NFT collapse to blank images (Section 7.1–7.2) |
| $\Delta<0$ learns the wrong answer | holds |
| A class rewarded more than the target takes over | holds: converges to 8s |
| KL + noise moves the end point toward the reference | holds |
| Under GRPO + KL, FN keeps the policy furthest from saturation | ordering holds; the predicted runaway for clean / FP-only does not appear (Section 7.3) |
| A weak learned reward gets hacked | does not happen with this reward model (Section 7.4) |

</div>

## 7. Why theory and experiment disagreed

Four predictions missed, in different ways. Each miss traces back to a specific simplification in Sections 2–4. Fixing those simplifications explains the data and makes new predictions that have not been tested yet (Section 8).

<div class="table-scroll" role="region" aria-label="Mismatches" tabindex="0">

| Mismatch | Simplification that caused it | Corrected theory |
| :--- | :--- | :--- |
| Raw NFT: FN, not FP, is harmful | treated NFT as "move to $\pi_{\text{old}}\cdot r$" | the NFT optimum is a covariance, so it is symmetric in FP/FN (7.1) |
| Raw NFT collapses to blank images | assumed the old model fits its own samples exactly | a self-consistency term with coefficient $(2\bar r-1)/\beta$ amplifies errors when $\bar r<\tfrac12$ (7.2) |
| No GRPO + KL runaway | group size $G\to\infty$ | for finite $G$, the push is bounded and the end point is finite (7.3) |
| Noise slows Flow-GRPO less than $\Delta$ | plain SGD dynamics | Adam and clipping rescale the gradient (7.5) |
| Weak reward not hacked | "exploitable in principle" read as "will be exploited" | hacking needs reachable false positives (7.4) |

</div>

### 7.1 NFT optimises a covariance, not a likelihood ratio

Section 4b assumed that one NFT step moves the model to $\pi^+\propto\pi_{\text{old}}\,r$, which multiplies the odds by $(1-\mathrm{FN})/\mathrm{FP}$. Instead, minimise the actual loss pointwise. At a noisy point $x_t$, with flow-matching target $u$, the per-sample loss is

$$
\ell(v)=r\,\big\|(1-\beta)v_{\text{old}}+\beta v-u\big\|^2+(1-r)\,\big\|(1+\beta)v_{\text{old}}-\beta v-u\big\|^2 .
$$

Setting the expected gradient to zero gives $\beta v=\mathbb E[(2r-1)u\mid x_t]+(1+\beta-2\bar r)\,v_{\text{old}}$, where $\bar r=\mathbb E[r\mid x_t]$. Now write $\mathbb E[u\mid x_t]=v_{\text{old}}+\varepsilon$, where $\varepsilon(x_t)$ is how far the old model is from the velocity of its *own* samples. Then

$$
\boxed{\;v^*=v_{\text{old}}+\frac{2}{\beta}\operatorname{Cov}(r,u\mid x_t)+\frac{2\bar r-1}{\beta}\,\varepsilon\;}
$$

The first correction term is the useful one. A covariance is unchanged by adding a constant to $r$ and scales linearly with a multiplier. Under image-independent noise $r=\mathrm{FP}+\Delta\cdot\mathbb 1[\text{correct}]+\text{noise}$, so

$$
\operatorname{Cov}(r,u\mid x_t)=\Delta\cdot\operatorname{Cov}(\mathbb 1[\text{correct}],u\mid x_t).
$$

**At the population level, NFT behaves like a policy gradient: it is slowed by a factor of $\Delta$, and FP and FN are interchangeable.** The odds-ratio table in Section 4b describes reward-weighted regression, which keeps only the positive branch. NFT also has a negative branch, and the two together cancel the constant FP term exactly as a baseline does. This matches the runs without collapse. Normalised NFT slows by 0.78 and 0.79 for FP-only and FN-only, and by 0.41 for both ($\Delta$ = 0.7, 0.7, 0.4). Raw NFT with FP = 0.3 slows by 0.64.

So the first half of the miss was a wrong derivation, not a wrong experiment.

### 7.2 The self-consistency term: why raw rewards with FN collapse

The second term multiplies $\varepsilon$, the model's own fitting error, by $c=(2\bar r-1)/\beta$. After the update, the remaining error is $(c-1)\varepsilon$, so repeated updates

* shrink the error when $0<c<2$, that is $\tfrac12<\bar r<\tfrac12+\beta$;
* **amplify** it when $c<0$, that is **when the average reward is below one half.**

The intuition is simple. When most samples have $r=0$, the loss mostly pushes *away* from the model's own outputs. Pushing away from your own samples is unlearning, and unlearning has no fixed target. Any error the model has gets reinforced instead of corrected.

Apply this to the runs ($\beta=1$). With raw rewards, $\bar r$ is the fraction of samples the reward model says yes to:

<div class="table-scroll" role="region" aria-label="Mean reward" tabindex="0">

| Noise | $\bar r$ as the share of 3s, $p$, grows | Below ½ while |
| :--- | :--- | :--- |
| clean | $p$ | $p<0.5$ (briefly, at the start) |
| FP = 0.3 | $0.3+0.7p$ | $p<0.29$ |
| FN = 0.3 | $0.7p$ | $p<0.71$ |
| FP = FN = 0.3 | $0.3+0.4p$ | $p<0.5$ |
| normalised / centred $r$ | $\approx 0.5$ for every group | never strongly below |

</div>

FN lowers the ceiling on $\bar r$, so the model has to pass $p=0.71$ before training becomes self-correcting. The FN runs plateaued at $p\approx0.6$, where $\bar r\approx0.42<\tfrac12$, and stayed in the amplifying regime long enough to fall apart. Normalised rewards pin $\bar r\approx\tfrac12$, where $c\approx0$: the error is neither corrected nor amplified, which is why that variant was stable.

To make this quantitative, sum the per-iteration amplification $\log|1-c|$ over all iterations with $c<0$. Weight each by the speed at which $\pi_{\text{old}}$ follows the model ($1-0.9$), using the logged mean reward:

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-amp-mobile.svg" /><img src="/figures/rn-mnist-amp.svg" alt="Scatter of the cumulative error-amplification index against the worst drop after the peak. All runs that never blank out have an index below 5 and a ratio near 1. Every run with an index of 10 or more drops sharply: main-sweep FN=0.3 runs at about 10 drop to near zero, follow-up FN=0.3 runs at 20 to 24 drop to zero, and FN=0.45 runs at 21 to 24 drop to about half." width="1000" height="560" loading="lazy" /></picture>
<figcaption>Cumulative amplification index Σ<sub>c&lt;0</sub> 0.1·log|1−c| against the deepest drop after the peak, for every raw-reward DiffusionNFT run with β = 1 that learned some 3s. Circles: the main sweep, used to define the index. Triangles: the follow-up sweep, run afterwards.</figcaption>
</figure>

In the main sweep, every healthy run has an index of at most 3.3, and both collapsed runs are at about 10. I defined the index after seeing those runs, so the real test is the follow-up sweep. There, all 12 raw-reward runs with clean reward, FP = 0.3, FN = 0.1 or FN = 0.2 stay below 4.7 and never collapse. All 6 runs with FN = 0.3 or 0.45 reach 20 or more and lose at least half their 3s.

The index says *whether* the amplifying regime lasts long enough to matter. It does not say how bad the result is: FN = 0.45 drops further than its index alone would suggest, then partly recovers (Section 8). It also does not explain *why the end state is a blank canvas*, because the theory says errors grow, not which direction they grow in. A plausible reason is that the negatives are many different digits and 3s, and they agree on where ink isn't, while the positives disagree on where ink is. I have not checked that.

### 7.3 The GRPO runaway was a large-group artifact

Section 4a found that with FN = 0 the group std $s\to0$, so the push $\Delta/s$ beats the KL pull. That calculation used the expected std, which is the $G\to\infty$ limit. With a finite group, a group where all samples are correct has *zero* advantage, not an infinite one. Near $p=1$, almost every group is like that. The only groups that push are the ones with a single wrong sample, which occur with probability about $G(1-p)$ and give a push of about $1/\sqrt G$. So the push is about $\sqrt G\,(1-p)$. That is the same order as the KL pull, $\beta(1-p)(\theta-\theta_{\text{ref}})$, so there is a finite end point near $\theta_{\text{ref}}+\sqrt G/\beta$.

Computing the exact expected GRPO update for $G=16$ by enumerating all group outcomes:

<div class="table-scroll" role="region" aria-label="Finite G" tabindex="0">

| $\beta$ | clean, $G{=}\infty$ | clean, $G{=}16$ | FP 0.3, $G{=}\infty$ | FP 0.3, $G{=}16$ | FN 0.3, $G{=}\infty$ | FN 0.3, $G{=}16$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 1 (runaway) | 0.996 | 1 (runaway) | 0.946 | 0.744 | 0.708 |
| 1.0 | 0.571 | 0.535 | 0.424 | 0.399 | 0.445 | 0.423 |

</div>

With $G=16$ nothing runs away, and at moderate $\beta$ the ordering becomes clean > FP > FN. That is the ordering Flow-GRPO + KL showed (0.95 > 0.86 > 0.80), where the infinite-group theory had predicted clean = FP = 1. The FN effect itself survives at finite $G$. What changes is that FP no longer saturates with clean. The bandit's $\beta$ and Flow-GRPO's per-step Gaussian KL are on different scales, so only the ordering can be compared, not the values.

### 7.4 Exploitable is not exploited

Section 5 said sample-dependent errors *allow* reward hacking. It needs one more condition to *happen*: images with high false-positive rate must be reachable by the optimiser before it finds the true target. The weak MLP says "3" for 78% of real 3s but at most 9% of any other digit. Starting from a model that only draws digit-like images, the steepest way up its reward is to draw 3s, which the oracle also accepts. The false positives that exist would require off-manifold images, and 300 small steps from a natural-image prior do not go there. To see hacking, the reward model would need high FP on *on-manifold* images (as in the 8-confusion runs), or training would need to be long and aggressive enough to leave the manifold.

### 7.5 Speed: Δ is the right direction, the optimiser compresses it

<div class="table-scroll" role="region" aria-label="Speed" tabindex="0">

| Logit-speed ratio vs clean | FP 0.3 | FN 0.3 | FP = FN = 0.3 |
| :--- | ---: | ---: | ---: |
| $\Delta$ (plain policy gradient) | 0.70 | 0.70 | 0.40 |
| GRPO, $G=16$, expected dynamics | 0.71 | 0.71 | 0.38 |
| Flow-GRPO, measured | 0.91 | 0.90 | 0.46 |
| NFT normalised, measured | 0.78 | 0.79 | 0.41 |
| NFT raw, measured | 0.64 | collapses | 0.32 |

</div>

Speed is the slope of $\operatorname{logit}p$ against iteration while $0.25<p<0.75$, averaged over two seeds. The symmetry between FP and FN holds everywhere except raw NFT with FN, which is the 7.2 instability. The sizes are less exact. Flow-GRPO loses only 10% of its speed where theory says 29%. The theory assumes the step is proportional to the expected gradient. Flow-GRPO trains with Adam and gradient clipping, and both normalise away much of the gradient's scale, so a smaller $\Delta$ costs less than it would under plain SGD. This explanation is untested. Rerunning with SGD would test it.

## 8. Testing the corrected theory

The corrected theory makes predictions the main sweep did not test. I wrote these down before running a follow-up sweep of 42 runs (DiffusionNFT without KL, 450 iterations, three seeds each, the same base model and oracle). The follow-up also logs the fraction of blank samples.

<figure>
<picture><source media="(max-width: 560px)" srcset="/figures/rn-mnist-followup-mobile.svg" /><img src="/figures/rn-mnist-followup.svg" alt="Three panels over 450 iterations. Left: share of 3s for raw-reward DiffusionNFT at FN 0.1, 0.2, 0.3 and 0.45. FN 0.1 and 0.2 rise to about 1. FN 0.3 peaks near 0.58 and falls to 0. FN 0.45 peaks near 0.46, falls to about 0.25, then recovers to 0.4. Middle: at FN=0.3, turning off the negative branch, centring the reward, or normalising it all rise to about 1. Raw reward with beta 0.3 and 0.1 drops earlier. Right: blank-sample fraction. Raw beta 1 goes to 1 after iteration 250. Beta 0.1 goes to 1 around iteration 120 and falls back by 220. Beta 0.3 peaks near 0.5. The three fixes stay at 0." width="1400" height="400" loading="lazy" /></picture>
<figcaption>Follow-up sweep. Thick lines are means over three seeds; thin lines are individual seeds.</figcaption>
</figure>

<div class="table-scroll" role="region" aria-label="Predictions and results" tabindex="0">

| Block | Run | Prediction (written before) | Result | |
| :--- | :--- | :--- | :--- | :---: |
| A | raw, FN = 0.1, 0.2 | no collapse | 0.94–1.0 at the end in all 6 runs | ✓ |
| A | raw, FN = 0.3 | collapse | 3/3 collapse to 0.00–0.05, all samples blank | ✓ |
| A | raw, FN = 0.45 | collapse, earlier than FN = 0.3 | blank-out starts at the same time (it 150–170), reaches 68–94% blank, then partly recovers to 0.34–0.51 | ✗ |
| A | all raw runs | blank images appear before the share of 3s falls | blank > 5% at it 130–170, decline at it 200–250 | ✓ |
| B | negative branch off, FN = 0.3 | no collapse | 0.94–0.99, never any blank samples | ✓ |
| C | group-centred raw $r$, FN = 0.3 | no collapse | 1.00, no blank samples | ✓ |
| D | raw, $\beta$ = 0.3 and 0.1, FN = 0.3 | collapse sooner | blank-out at it 60–80 for $\beta$ = 0.1, vs 130–170 for $\beta$ = 1; with $\beta$ = 0.3, two of three seeds reach 47% and 99% blank | ✓ (onset) |
| E | normalised $r$, 450 iterations | stays stable | 0.99 at the end, never below 0.98 after the peak | ✓ |

</div>

The two decisive results are B and C. Removing the negative branch removes the collapse, which shows the push away from false negatives is *necessary*. Re-centring the raw reward within each group, without any std scaling, also removes it, which shows that the average reward sitting below one half is the trigger, not the scale of the reward. Together they rule out the obvious alternative, that raw rewards are simply too large or too noisy. D confirms the counter-intuitive direction: a smaller $\beta$, normally read as a gentler update, blanks out about twice as early.

Two results the corrected theory gets wrong or does not cover:

* **Blank is not always permanent.** With FN = 0.45, and with $\beta$ = 0.1 or 0.3, the model blanks out and then partly comes back. With $\beta$ = 1 and FN = 0.3, blank is absorbing. The theory only says errors grow while $\bar r<\tfrac12$. A guess for the recovery: once almost every sample is blank and gets $r=0$, the negative branch pushes away from *blank* images, which drives the model back toward digits. Whether that escape works seems to depend on $\beta$ and on how much digit-shaped mass is left, and I have no model for it.
* **The FN = 0.45 timing prediction failed.** I predicted earlier collapse because $\bar r$ stays below ½ longer. Onset was the same as at FN = 0.3. The index tracks how long the amplifying regime lasts, but not how fast the error grows inside it, and that rate evidently depends on more than $\bar r$.

A second generalisation also held up only partly. For $\beta\neq1$ the stability condition is $0<c<2$, so the cumulative index should count both $c<0$ and $c>2$. It orders onset correctly (index above 5 at iteration 20 for $\beta$ = 0.1, 40 for $\beta$ = 0.3, 100–110 for $\beta$ = 1). But it does not predict severity: one $\beta$ = 0.3 seed reaches an index of 10 and stays healthy. It is a warning signal, not a threshold.

## 9. Summary

* Image-independent noise makes the reward an affine function of the truth, with slope $\Delta=1-\mathrm{FP}-\mathrm{FN}$. Policy-gradient methods reach the same answer, and on MNIST Flow-GRPO treats FP and FN almost identically.
* KL regularisation pulls noisy runs back toward the reference. With finite groups, GRPO + KL settles in the order clean > FP > FN, without the runaway the infinite-group analysis predicted.
* When the reward model ranks a wrong class above the right one, every algorithm follows it.
* DiffusionNFT's loss optimises a covariance, so at the population level it is just as symmetric in FP and FN as a policy gradient. My odds-ratio prediction was a derivation error.
* The NFT update also contains a self-consistency term, $(2\bar r-1)\varepsilon/\beta$. With raw rewards whose mean is below one half, it amplifies the model's own errors. False negatives keep the mean down, which is why they, and not false positives, made raw NFT collapse. Group-normalised rewards keep the mean at one half, which is why DiffusionNFT's default is stable. A follow-up sweep tested this: turning off the negative branch, or only centring the reward, removes the collapse, and a smaller β makes it come sooner, all as predicted. What the theory still misses is when the blank state is permanent and how fast the collapse develops.
