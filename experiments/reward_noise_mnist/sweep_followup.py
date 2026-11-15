"""Follow-up: why does raw-reward DiffusionNFT collapse to blank images under false negatives?

All runs: DiffusionNFT, no KL, 450 iterations (longer than the main sweep, to catch late collapse),
seeds 0-2. Resumable; reuses ckpt/ from the main sweep (do NOT retrain base/classifiers).

    python sweep_followup.py --only A_     # one block
    python sweep_followup.py --dry

Blocks
  A  FN dose-response, raw reward:      FN in {0.1, 0.2, 0.3, 0.45}, plus FP=0.3 and clean controls
  B  negative branch off (raw reward):  --nft_neg_weight 0, FN=0.3 and clean
     -> if the collapse disappears, the negative branch fed with false negatives is the cause
  C  re-centred raw reward (no std):    --nft_reward center, FN=0.3 and clean
     -> separates "centring within the group" from "std scaling" (normalised reward does both)
  D  NFT step size at FN=0.3, raw:      --nft_beta in {0.3, 0.1}  (main sweep used 1.0)
  E  normalised reward, long horizon:   FN=0.3 and clean, to check it does not just collapse later
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = [0, 1, 2]
ITERS = ['--iters', '450']


def jobs():
    def tag(nz):
        return nz.replace(':', '-').replace(',', '_')
    for nz in ['clean', 'sym:0,0.1', 'sym:0,0.2', 'sym:0,0.3', 'sym:0,0.45', 'sym:0.3,0']:
        yield 'A', f'A_raw_{tag(nz)}', ['--noise', nz, '--nft_reward', 'raw']
    for nz in ['clean', 'sym:0,0.3']:
        yield 'B', f'B_raw_neg0_{tag(nz)}', ['--noise', nz, '--nft_reward', 'raw', '--nft_neg_weight', '0']
        yield 'C', f'C_center_{tag(nz)}', ['--noise', nz, '--nft_reward', 'center']
        yield 'E', f'E_norm_{tag(nz)}', ['--noise', nz, '--nft_reward', 'norm']
    for b in ['0.3', '0.1']:
        yield 'D', f'D_raw_beta{b}_sym-0_0.3', ['--noise', 'sym:0,0.3', '--nft_reward', 'raw', '--nft_beta', b]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default=''); p.add_argument('--dry', action='store_true')
    p.add_argument('--rl_args', default='')
    a = p.parse_args()
    assert a.dry or (ROOT / 'ckpt' / 'base.pt').exists(), 'run the main sweep first (needs ckpt/base.pt)'
    n = 0
    for block, base, extra in jobs():
        for s in SEEDS:
            name = f'fu_{base}_s{s}'
            if a.only and a.only not in name:
                continue
            if (ROOT / 'runs' / name / 'final.pt').exists():
                continue
            cmd = [PY, str(ROOT / 'rn_mnist.py'), 'rl', '--algo', 'nft', '--beta', '0', '--seed', str(s),
                   '--name', name, *ITERS, *extra, *shlex.split(a.rl_args)]
            print(' '.join(cmd), flush=True); n += 1
            if not a.dry:
                subprocess.run(cmd, check=True)
    print(n, 'runs')


if __name__ == '__main__':
    main()
