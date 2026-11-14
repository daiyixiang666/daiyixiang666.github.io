"""Run the experiment grid used in the article. Resumable: finished runs are skipped.

    python sweep.py                # everything (base + classifiers + all RL runs)
    python sweep.py --only grpo    # subset by substring of run name
    python sweep.py --dry          # print commands only
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

NOISES = [
    'clean',
    'sym:0.3,0',          # FP only
    'sym:0,0.3',          # FN only
    'sym:0.3,0.3',        # both
    'sym:0.45,0.45',      # almost no signal (Delta = 0.1)
    'sym:0.6,0.6',        # flipped (Delta = -0.2): theory says it learns the wrong way
    'confuse:8,0.7',      # digit 8 often rewarded, still below target (q_8 = 0.7 < q_3 = 1)
    'confuse:8,0.9,0.3',  # digit 8 rewarded MORE than a true 3 (0.9 > 0.7): wrong fixed point
    'weak',               # weak learned reward model: instance-dependent errors
]
ALGOS = {
    'flowgrpo': [],
    'nft': [],                                  # normalised-advantage reward (DiffusionNFT default)
    'nftraw': ['--nft_reward', 'raw'],          # raw binary reward as optimality probability
}
KL = {'kl0': ['--beta', '0'], 'kl': None}       # per-algo KL value below
KL_VALUE = {'flowgrpo': '0.04', 'nft': '0.1', 'nftraw': '0.1'}
SEEDS = [0, 1]


def jobs():
    for algo, extra in ALGOS.items():
        real = 'nft' if algo.startswith('nft') else 'flowgrpo'
        for noise in NOISES:
            for seed in SEEDS:
                for kl in ['kl0', 'kl']:
                    if kl == 'kl' and noise not in ('clean', 'sym:0.3,0', 'sym:0,0.3', 'confuse:8,0.7'):
                        continue      # the KL ablation only on the configs where theory predicts a shift
                    beta = ['--beta', '0'] if kl == 'kl0' else ['--beta', KL_VALUE[algo]]
                    name = f"{algo}_{kl}_{noise.replace(':', '-').replace(',', '_')}_s{seed}"
                    yield name, [PY, str(ROOT / 'rn_mnist.py'), 'rl', '--algo', real, '--noise', noise,
                                 '--seed', str(seed), '--name', name, *beta, *extra]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default=''); p.add_argument('--dry', action='store_true')
    p.add_argument('--rl_args', default='', help='extra args forwarded to every rl run, e.g. "--lr 3e-5"')
    a = p.parse_args()
    ck = ROOT / 'ckpt'
    if not (ck / 'base.pt').exists():
        cmd = [PY, str(ROOT / 'rn_mnist.py'), 'base']
        print(' '.join(cmd)); a.dry or subprocess.run(cmd, check=True)
    if not (ck / 'oracle.pt').exists():
        cmd = [PY, str(ROOT / 'rn_mnist.py'), 'classifier']
        print(' '.join(cmd)); a.dry or subprocess.run(cmd, check=True)
    if not a.dry:
        subprocess.run([PY, str(ROOT / 'rn_mnist.py'), 'eval-base'], check=True)
    for name, cmd in jobs():
        if a.only and a.only not in name:
            continue
        if (ROOT / 'runs' / name / 'final.pt').exists():
            continue
        cmd = cmd + shlex.split(a.rl_args)
        print(' '.join(cmd), flush=True)
        if not a.dry:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
