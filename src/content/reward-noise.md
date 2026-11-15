---
title: "When the Reward Model Lies"
description: "False positives, false negatives, and what they do to GRPO, Flow-GRPO and DiffusionNFT, from a two-action bandit to generating handwritten digits."
---

Every RL fine-tuning pipeline trusts a reward model that is sometimes wrong. A verifier passes a wrong proof. A classifier calls a 5 a 3. A preference model misses a correct answer and scores it zero. The natural worry is that training will go to the wrong place.

The answer depends on two things: what kind of mistakes the reward model makes, and which algorithm consumes the reward. This article starts from the smallest model where the question can be answered exactly. It then adds the pieces that real systems have: KL regularisation, group-normalised advantages, reward-weighted (negative-aware) updates, many classes, and errors that depend on the sample. At the end, 78 runs on a flow model drawing MNIST digits test the predictions. Most hold, one fails in an instructive way, and one failure mode I expected never appeared.

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
<figcaption>Monte Carlo simulation of the two-action model (64 runs, G = 8, starting from p = 0.1). Every run reaches the correct answer, only at different speeds. Left: group-mean baseline. Middle: GRPO. Right: an idealised reward-weighted update (Section 4b).</figcaption>
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

Two caveats keep this from being the whole story. First, the default DiffusionNFT reward is not the raw $R$. It is the group-normalised advantage mapped into $[0,1]$ by $r=\tfrac12+\tfrac12\operatorname{clip}(A/Z,-1,1)$. Since that map is affine in $R$ within a group, the odds factor becomes $\tfrac{c+d(1-\mathrm{FN})}{c+d\,\mathrm{FP}}$, which is bounded and less asymmetric. Second, the step size $\beta$ and the soft update of $\pi_{\text{old}}$ mean a real iteration takes only a fraction of the full step.

The second caveat can be made precise. Minimising the NFT loss pointwise, and using $\mathbb E[u\mid x_t]=v_{\text{old}}$ for samples drawn from $\pi_{\text{old}}$ (where $u$ is the flow-matching target), gives

$$
v^*(x_t)=v_{\text{old}}(x_t)+\frac{2}{\beta}\,\bar r(x_t)\,\big(v^+(x_t)-v_{\text{old}}(x_t)\big),
\qquad \bar r(x_t)=\mathbb E[r\mid x_t].
$$

The update points toward the positive velocity $v^+$, but its length depends on the local mean reward $\bar r(x_t)$. It lands exactly on $\pi^+$ only when $2\bar r/\beta=1$. With raw rewards, the step size therefore depends on how often the reward model says yes, so FN changes it directly. Normalised rewards keep $\bar r$ near $\tfrac12$.

Most importantly, the odds argument only describes the *population* target. It says nothing about what the negative branch does to a finite network when some negatives are good samples. **The experiment in Section 6 contradicts the prediction of this subsection for raw rewards: false negatives, not false positives, are what break it.**

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

I don't have a derivation for this. Here is the asymmetry that fits the data. In NFT, every sample with $r=0$ is actively pushed *away from*. With false positives, the negative set still contains only wrong digits, so the push is in the right direction. The mistakes only add some wrong digits to the positives, and imitating a sample is a bounded error. With false negatives, a fixed fraction of real 3s are pushed away from on every iteration. A blank canvas is the one image that is far from every negative at once. Normalised rewards centre $r$ within each group, so positives and negatives always carry equal total weight. That is consistent with the normalised variant being stable under the same noise, but it is a hypothesis, not a proof.

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
| FP hurts reward-weighted (raw NFT) updates more than FN | **wrong**: FN makes raw NFT collapse to blank images |
| $\Delta<0$ learns the wrong answer | holds |
| A class rewarded more than the target takes over | holds: converges to 8s |
| KL + noise moves the end point toward the reference | holds |
| Under GRPO + KL, FN keeps the policy furthest from saturation | ordering holds; the predicted runaway for clean / FP-only does not appear |
| A weak learned reward gets hacked | does not happen with this reward model |

</div>

## 7. Summary

* Image-independent noise makes the reward an affine function of the truth, with slope $\Delta=1-\mathrm{FP}-\mathrm{FN}$. As long as $\Delta>0$, policy-gradient methods reach the same answer, slower by about a factor of $\Delta$. On MNIST, Flow-GRPO treats FP and FN almost identically.
* KL regularisation turns noise into a larger effective $\beta=\beta/\Delta$, so the end point moves toward the reference. Class-dependent errors leave their trace in the KL optimum.
* When the reward model prefers a wrong class over the right one, every algorithm follows it. No algorithm can recover a ranking the reward does not contain.
* The population analysis of DiffusionNFT-style updates was the wrong guide. With raw rewards, false negatives feed real good samples into the negative branch, and the model escaped by erasing its output. Normalising the reward within each group, as DiffusionNFT does by default, removed the failure in these runs.
* The open question from this experiment: when exactly does pushing away from false negatives make a negative-aware method collapse, and how does that depend on the NFT step size β?
