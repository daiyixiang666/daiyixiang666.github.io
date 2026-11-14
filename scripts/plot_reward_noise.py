"""Toy figures for the reward-noise article: python3 scripts/plot_reward_noise.py.

Binary bandit: the policy outputs a=1 (correct) with probability p = sigmoid(theta).
The reward model reports R=1 with probability
    q1 = P(R=1 | a=1) = 1 - FN,      q0 = P(R=1 | a=0) = FP.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'public' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
BLUE, TEAL, GOLD, RED, GRAY = '#3852c4', '#138579', '#b17624', '#b8433a', '#83909e'
rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p / (1 - p))


import os
PREVIEW = os.environ.get('PREVIEW')


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=.14)
    if PREVIEW and not name.endswith('mobile'):
        fig.savefig(Path(PREVIEW) / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10, colors='#667080')


# ---------------------------------------------------------------- simulators
P0, TARGET = 0.1, 0.95


def run_grpo(fp, fn, G=8, lr=0.5, steps=3000, runs=64):
    """Monte-Carlo GRPO on the logit; returns mean p-trajectory and steps-to-target."""
    theta = np.full(runs, logit(P0))
    traj = np.empty((steps, runs))
    hit = np.full(runs, np.nan)
    for t in range(steps):
        p = sig(theta)
        a = rng.random((runs, G)) < p[:, None]
        q = np.where(a, 1 - fn, fp)
        R = (rng.random((runs, G)) < q).astype(float)
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)
        grad = (A * (a - p[:, None])).mean(1)          # d/dtheta log pi(a) = a - p
        theta += lr * grad
        theta = np.clip(theta, -12, 12)
        traj[t] = sig(theta)
        newly = np.isnan(hit) & (traj[t] >= TARGET)
        hit[newly] = t
    return traj.mean(1), np.nanmedian(np.where(np.isnan(hit), steps, hit))


def run_pg(fp, fn, G=8, lr=0.5, steps=3000, runs=64):
    """Group-mean baseline, no std normalisation (Dr.GRPO / RLOO-like)."""
    theta = np.full(runs, logit(P0))
    traj = np.empty((steps, runs))
    for t in range(steps):
        p = sig(theta)
        a = rng.random((runs, G)) < p[:, None]
        R = (rng.random((runs, G)) < np.where(a, 1 - fn, fp)).astype(float)
        A = R - R.mean(1, keepdims=True)
        theta = np.clip(theta + lr * 4 * (A * (a - p[:, None])).mean(1), -12, 12)
        traj[t] = sig(theta)
    return traj.mean(1)


def run_rwr(fp, fn, G=8, lam=0.3, steps=3000, runs=64):
    """Idealised NFT / reward-weighted step: pi_new ∝ pi_old * E[r|a]^lam, E[r|a] estimated
    from the group with add-half smoothing (raw binary reward used as optimality prob.)."""
    theta = np.full(runs, logit(P0))
    traj = np.empty((steps, runs))
    for t in range(steps):
        p = sig(theta)
        a = rng.random((runs, G)) < p[:, None]
        R = (rng.random((runs, G)) < np.where(a, 1 - fn, fp)).astype(float)
        n1, n0 = a.sum(1), (~a).sum(1)
        r1 = ((R * a).sum(1) + .5) / (n1 + 1)
        r0 = ((R * ~a).sum(1) + .5) / (n0 + 1)
        upd = np.where((n1 > 0) & (n0 > 0), np.log(r1 / r0), 0.0)
        theta = np.clip(theta + lam * upd, -12, 12)
        traj[t] = sig(theta)
    return traj.mean(1)


# ---------------------------------------------------------------- figure 1: affine reward
# E[R|a] = FP + (1-FP-FN) * 1[a=1]
for mobile in [False, True]:
    fig, ax = plt.subplots(figsize=(5, 3.6) if mobile else (6.4, 3.6), layout='constrained')
    style(ax)
    xs = np.array([0, 1])
    for (fp, fn, c, lab) in [(0, 0, BLUE, 'clean'), (.3, 0, GOLD, 'FP = 0.3'),
                             (0, .3, TEAL, 'FN = 0.3'), (.3, .3, RED, 'FP = FN = 0.3'),
                             (.6, .6, GRAY, 'FP = FN = 0.6 (flipped)')]:
        ax.plot(xs, [fp, 1 - fn], 'o-', color=c, lw=2, label=lab)
    ax.set_xticks([0, 1], ['wrong (a=0)', 'correct (a=1)'])
    ax.set_ylabel(r'$\mathbb{E}[R\mid a]$'); ax.set_ylim(-.05, 1.05)
    ax.set_title('Noise compresses the reward gap to 1 − FP − FN')
    ax.legend(frameon=False, fontsize=9, loc='upper left' if not mobile else 'best')
    save(fig, 'rn-affine-mobile' if mobile else 'rn-affine')

# ---------------------------------------------------------------- figure 2: learning curves
settings = [(0, 0, BLUE, 'clean'), (.3, 0, GOLD, 'FP=0.3'), (0, .3, TEAL, 'FN=0.3'),
            (.3, .3, RED, 'FP=FN=0.3')]
curves = {}
for fp, fn, *_ in settings:
    curves[('pg', fp, fn)] = run_pg(fp, fn, lr=.08, steps=800)
    curves[('grpo', fp, fn)] = run_grpo(fp, fn, lr=.08, steps=800)[0]
    curves[('rwr', fp, fn)] = run_rwr(fp, fn, lam=.05, steps=800)
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3,
                             figsize=(5, 11) if mobile else (13, 3.8), layout='constrained', sharey=True)
    for ax, key, title in zip(axes, ['pg', 'grpo', 'rwr'],
                              ['mean baseline (no std)', 'GRPO (group std)', 'reward-weighted (NFT-style)']):
        style(ax)
        for fp, fn, c, lab in settings:
            ax.plot(curves[(key, fp, fn)], color=c, lw=2, label=lab)
        ax.set_title(title); ax.set_xlabel('update step'); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('P(correct)')
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    save(fig, 'rn-curves-mobile' if mobile else 'rn-curves')

# ---------------------------------------------------------------- figure 3: speed heat maps
grid = np.linspace(0, .45, 10)
H = {k: np.zeros((len(grid), len(grid))) for k in ['delta', 'grpo', 'ratio']}
for i, fn in enumerate(grid):
    for j, fp in enumerate(grid):
        H['delta'][i, j] = 1 - fp - fn
        H['ratio'][i, j] = np.log((1 - fn) / max(fp, 1e-3))
        H['grpo'][i, j] = run_grpo(fp, fn, lr=.08, steps=3000, runs=48)[1]
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3,
                             figsize=(5, 13) if mobile else (14, 4.2), layout='constrained')
    panels = [('delta', r'signal gap $\Delta=1-$FP$-$FN', 'viridis', None),
              ('ratio', r'odds factor $\log\frac{1-\mathrm{FN}}{\mathrm{FP}}$', 'viridis', (0, 4)),
              ('grpo', 'GRPO (simulated): median steps to P=0.95', 'magma_r', None)]
    for ax, (k, title, cmap, lim) in zip(axes, panels):
        im = ax.imshow(H[k], origin='lower', cmap=cmap, extent=[0, .45, 0, .45], aspect='equal',
                       vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_xlabel('FP'); ax.set_ylabel('FN'); ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, 'rn-heat-mobile' if mobile else 'rn-heat')

# ---------------------------------------------------------------- figure 4: KL optimum
# Expected dynamics on the logit with reverse-KL to pi_ref:
#   dtheta/dt = p(1-p) * [ Delta / s  -  beta (theta - theta_ref) ]
# raw reward: s = 1      -> theta* = theta_ref + Delta / beta  (depends on Delta only)
# GRPO:       s = sqrt(m(1-m)), m = FP + Delta p   (group std; vanishes as p->1 iff FN = 0)
p_ref = 0.1


def kl_optimum(fp, fn, beta, norm, steps=40000, dt=0.05):
    th0 = logit(p_ref); th = th0; d = 1 - fp - fn
    for _ in range(steps):
        p = sig(th); m = fp + d * p
        s = np.sqrt(max(m * (1 - m), 1e-10)) if norm else 1.0
        th += dt * p * (1 - p) * (d / s - beta * (th - th0))
        if th > 25:
            return 1.0
    return sig(th)


betas = np.geomspace(0.2, 3, 40)
for mobile in [False, True]:
    fig, ax = plt.subplots(figsize=(5, 3.9) if mobile else (6.6, 3.9), layout='constrained')
    style(ax)
    for fp, fn, c, ls, lab in [(0, 0, BLUE, '-', 'clean'), (.2, 0, GOLD, '-', 'FP = 0.2'),
                               (0, .2, TEAL, '-', 'FN = 0.2')]:
        ax.plot(betas, [kl_optimum(fp, fn, b, True) for b in betas], color=c, lw=2.2, ls=ls,
                label=f'GRPO-normalised, {lab}')
    ax.plot(betas, [kl_optimum(.2, 0, b, False) for b in betas], color=GRAY, lw=1.8, ls='--',
            label='raw reward, FP = 0.2 or FN = 0.2')
    ax.set_xscale('log'); ax.set_xlabel(r'KL coefficient $\beta$')
    ax.set_ylabel('P(correct) where training settles')
    ax.set_title(rf'KL-regularised end point ($p_{{ref}}$ = {p_ref})')
    ax.set_ylim(0, 1.03); ax.legend(frameon=False, fontsize=9, loc='lower left')
    save(fig, 'rn-kl-mobile' if mobile else 'rn-kl')

print('wrote figures to', OUT)
