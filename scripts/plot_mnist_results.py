"""MNIST result figures for the reward-noise article: python3 scripts/plot_mnist_results.py

Reads experiments/reward_noise_mnist/results_summary.json and grids/ (produced on the GPU box).
"""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments' / 'reward_noise_mnist'
OUT = REPO / 'public' / 'figures'
PREVIEW = os.environ.get('PREVIEW')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
BLUE, TEAL, GOLD, RED, GRAY, PURPLE = '#3852c4', '#138579', '#b17624', '#b8433a', '#83909e', '#7a4fa3'
runs = json.loads((EXP / 'results_summary.json').read_text())['runs']
ALGOS = [('flowgrpo', 'Flow-GRPO'), ('nft', 'DiffusionNFT (normalised r)'), ('nftraw', 'DiffusionNFT (raw r)')]


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=.14)
    if PREVIEW and not name.endswith('mobile'):
        fig.savefig(Path(PREVIEW) / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10, colors='#667080')


def curve(name, key='oracle_acc', digit=None):
    L = runs[name]['log']
    it = np.array([r['it'] for r in L])
    y = np.array([r['hist'][digit] if digit is not None else r[key] for r in L])
    return it, y


def final(name, digit=None, k=5):
    L = runs[name]['log'][-k:]
    return np.mean([r['hist'][digit] if digit is not None else r['oracle_acc'] for r in L])


# ------------------------------------------------ figure 1: learning curves, image-independent noise
SETS = [('clean', BLUE, 'clean'), ('sym-0.3_0', GOLD, 'FP=0.3'), ('sym-0_0.3', TEAL, 'FN=0.3'),
        ('sym-0.3_0.3', RED, 'FP=FN=0.3')]
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3, figsize=(5, 11) if mobile else (13, 3.9),
                             layout='constrained', sharey=True)
    for ax, (algo, title) in zip(axes, ALGOS):
        style(ax)
        for nz, c, lab in SETS:
            ys = []
            for s in (0, 1):
                it, y = curve(f'{algo}_kl0_{nz}_s{s}')
                ax.plot(it, y, color=c, lw=.9, alpha=.35)
                ys.append(y)
            ax.plot(it, np.mean(ys, 0), color=c, lw=2.2, label=lab)
        ax.set_title(title); ax.set_xlabel('RL iteration'); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('share of 3s (oracle)')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    save(fig, 'rn-mnist-curves-mobile' if mobile else 'rn-mnist-curves')

# ------------------------------------------------ figure 2: where training ends (destination tests)
CASES = [('clean', 'clean'), ('sym-0.45_0.45', 'FP=FN=0.45\n(Δ=0.1)'), ('sym-0.6_0.6', 'FP=FN=0.6\n(Δ=−0.2)'),
         ('confuse-8_0.7', '8 → 3 at 0.7'), ('confuse-8_0.9_0.3', '8 → 3 at 0.9,\nFN=0.3')]
start3 = np.mean([runs[f'{a}_kl0_clean_s{s}']['log'][0]['hist'][3] for a, _ in ALGOS for s in (0, 1)])
for mobile in [False, True]:
    fig, ax = plt.subplots(figsize=(5.4, 4.6) if mobile else (10, 3.9), layout='constrained')
    style(ax)
    x = np.arange(len(CASES)); w = .26
    for j, ((algo, title), c) in enumerate(zip(ALGOS, [BLUE, TEAL, GOLD])):
        for s, mk in [(0, 'o'), (1, 's')]:
            v3 = [final(f'{algo}_kl0_{nz}_s{s}', 3) for nz, _ in CASES]
            v8 = [final(f'{algo}_kl0_{nz}_s{s}', 8) for nz, _ in CASES]
            ax.bar(x + (j - 1) * w + (s - .5) * w / 2, v3, w / 2, color=c, alpha=.9,
                   label=title if s == 0 else None)
            ax.bar(x + (j - 1) * w + (s - .5) * w / 2, v8, w / 2, bottom=v3, color=c, alpha=.3,
                   hatch='///', edgecolor='white', lw=0)
    ax.axhline(start3, color=GRAY, ls='--', lw=1)
    ax.text(len(CASES) - .5, start3 + .02, 'base model: 3s', fontsize=9, color=GRAY, ha='right')
    ax.set_xticks(x, [lab for _, lab in CASES], fontsize=9)
    ax.set_ylabel('share of samples'); ax.set_ylim(0, 1.05)
    ax.set_title('Solid: share of 3s.  Hatched on top: share of 8s.  (no KL, two seeds each)', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper center' if not mobile else 'upper right')
    save(fig, 'rn-mnist-destination-mobile' if mobile else 'rn-mnist-destination')

# ------------------------------------------------ figure 3: KL end points
KLSETS = [('clean', 'clean'), ('sym-0.3_0', 'FP=0.3'), ('sym-0_0.3', 'FN=0.3'), ('confuse-8_0.7', '8→3 at 0.7')]
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3, figsize=(5, 10) if mobile else (13, 3.6),
                             layout='constrained', sharey=True)
    for ax, (algo, title) in zip(axes, ALGOS):
        style(ax)
        for i, (nz, lab) in enumerate(KLSETS):
            for kl, c, off in [('kl0', GRAY, -.12), ('kl', PURPLE, .12)]:
                vals = [final(f'{algo}_{kl}_{nz}_s{s}') for s in (0, 1)]
                ax.scatter([i + off] * 2, vals, color=c, s=34, zorder=3,
                           label=('no KL' if kl == 'kl0' else 'with KL') if i == 0 else None)
        ax.set_xticks(range(len(KLSETS)), [l for _, l in KLSETS], fontsize=9)
        ax.set_title(title); ax.set_ylim(0, 1.03)
    axes[0].set_ylabel('final share of 3s')
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    save(fig, 'rn-mnist-kl-mobile' if mobile else 'rn-mnist-kl')

# ------------------------------------------------ figure 4: sample grids
GRIDS = [('flowgrpo_kl0_clean_s0', 'Flow-GRPO, clean'),
         ('flowgrpo_kl0_confuse-8_0.9_0.3_s1', 'Flow-GRPO, 8 rewarded > 3'),
         ('flowgrpo_kl0_confuse-8_0.9_0.3_s0', 'same, other seed:\n"8" only to the oracle'),
         ('flowgrpo_kl0_sym-0.6_0.6_s0', 'Flow-GRPO, flipped (Δ<0)'),
         ('nftraw_kl0_sym-0_0.3_s0', 'NFT raw, FN=0.3 (collapsed)')]
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 2 if mobile else 5, figsize=(6, 9.6) if mobile else (16, 3.8),
                             layout='constrained')
    flat = np.ravel(axes)
    for ax in flat:
        ax.axis('off')
    for ax, (name, title) in zip(flat, GRIDS):
        ax.imshow(mpimg.imread(EXP / 'grids' / f'{name}.png'), interpolation='nearest')
        ax.set_title(title, fontsize=10)
    save(fig, 'rn-mnist-grids-mobile' if mobile else 'rn-mnist-grids')

print('wrote MNIST figures to', OUT)

# ------------------------------------------------ figure 5: error-amplification index (Section 7.2)
# c = (2 rbar - 1) / beta with beta = 1; old policy follows at rate 1 - 0.9; one eval = 10 iterations
pts = []
for name, run in runs.items():
    if not name.startswith('nftraw_kl0'):
        continue
    L = run['log']
    R = np.array([r['noisy_reward'] for r in L]); acc = np.array([r['oracle_acc'] for r in L])
    c = 2 * R - 1
    amp = np.sum(np.where(c < 0, np.log(np.abs(1 - np.minimum(c, 0))), 0) * 0.1 * 10)
    peak, fin = acc.max(), acc[-5:].mean()
    if peak < 0.4:
        continue                      # never learned 3s (flipped / confused / no-signal runs)
    k = acc.argmax(); pts.append((amp, acc[k:].min() / peak, name))
fu_pts = []
for name, run in runs.items():
    if not name.startswith('fu_A_raw'):
        continue
    L = run['log']
    R = np.array([r['noisy_reward'] for r in L]); acc = np.array([r['oracle_acc'] for r in L])
    c = 2 * R - 1
    amp = np.sum(np.where(c < 0, np.log(np.abs(1 - np.minimum(c, 0))), 0) * 0.1 * 10)
    k = acc.argmax(); fu_pts.append((amp, acc[k:].min() / acc[k], name))
for mobile in [False, True]:
    fig, ax = plt.subplots(figsize=(5, 3.9) if mobile else (6.6, 3.7), layout='constrained')
    style(ax)
    for amp, ratio, name in pts:
        col = RED if 'sym-0_0.3' in name else BLUE
        ax.scatter(amp, ratio, color=col, s=40, zorder=3)
    for amp, ratio, name in fu_pts:
        col = RED if ('sym-0_0.3' in name or 'sym-0_0.45' in name) else BLUE
        ax.scatter(amp, ratio, facecolor='none', edgecolor=col, marker='^', s=50, lw=1.4, zorder=3)
    ax.scatter([], [], facecolor='none', edgecolor=BLUE, marker='^', label='follow-up, healthy (not used to define the index)')
    ax.scatter([], [], color=BLUE, label='main sweep, other noise');
    ax.scatter([], [], color=RED, marker='^', facecolor='none', label='follow-up FN = 0.3 / 0.45')
    ax.scatter([], [], color=RED, label='main sweep, FN = 0.3')
    ax.set_xlabel(r'cumulative amplification  $\sum_{c<0} 0.1\,\log|1-c|$')
    ax.set_ylabel('worst share of 3s after peak / peak'); ax.set_ylim(-0.03, 1.08)
    ax.set_title('Raw-reward DiffusionNFT, β = 1', fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc='center right')
    save(fig, 'rn-mnist-amp-mobile' if mobile else 'rn-mnist-amp')
print('wrote amplification figure')

# ------------------------------------------------ figure 6: follow-up sweep (Section 8)
def fu_mean(base, key):
    ys = []
    for s in (0, 1, 2):
        L = runs[f'fu_{base}_s{s}']['log']
        ys.append([r[key] for r in L])
    it = [r['it'] for r in runs[f'fu_{base}_s0']['log']]
    return np.array(it), np.array(ys)


DOSE = [('A_raw_sym-0_0.1', TEAL, 'FN = 0.1'), ('A_raw_sym-0_0.2', GOLD, 'FN = 0.2'),
        ('A_raw_sym-0_0.3', RED, 'FN = 0.3'), ('A_raw_sym-0_0.45', PURPLE, 'FN = 0.45')]
FIX = [('A_raw_sym-0_0.3', RED, 'raw (β = 1)'), ('D_raw_beta0.3_sym-0_0.3', '#d9822b', 'raw, β = 0.3'),
       ('D_raw_beta0.1_sym-0_0.3', '#6b2a24', 'raw, β = 0.1'), ('B_raw_neg0_sym-0_0.3', BLUE, 'negative branch off'),
       ('C_center_sym-0_0.3', TEAL, 'group-centred r'), ('E_norm_sym-0_0.3', GRAY, 'normalised r')]
for mobile in [False, True]:
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3, figsize=(5, 12) if mobile else (14, 4.0),
                             layout='constrained')
    for ax, sets, key, title, ylab in [
            (axes[0], DOSE, 'oracle_acc', 'Raw reward: FN dose', 'share of 3s'),
            (axes[1], FIX, 'oracle_acc', 'FN = 0.3: interventions', 'share of 3s'),
            (axes[2], FIX, 'blank_frac', 'FN = 0.3: blank images', 'fraction of blank samples')]:
        style(ax)
        for base, c, lab in sets:
            it, ys = fu_mean(base, key)
            for y in ys:
                ax.plot(it, y, color=c, lw=.8, alpha=.3)
            ax.plot(it, ys.mean(0), color=c, lw=2.1, label=lab)
        ax.set_title(title); ax.set_xlabel('RL iteration'); ax.set_ylabel(ylab); ax.set_ylim(0, 1.03)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper left')
    save(fig, 'rn-mnist-followup-mobile' if mobile else 'rn-mnist-followup')
print('wrote follow-up figure')
