# Reward noise on MNIST: Flow-GRPO vs DiffusionNFT

Experiments for the article *When the Reward Model Lies*. An unconditional rectified-flow
model is trained on MNIST, then RL fine-tuned to draw one digit (3). It starts at about 10%,
the same starting point as the toy bandit. The reward is a corrupted version of an oracle classifier.

## Run (Windows GPU box)

```powershell
cd experiments\reward_noise_mnist
python -m venv .venv; .\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy
python rn_mnist.py base           # ~30 epochs, a few minutes on a modern GPU
python rn_mnist.py classifier     # oracle CNN (~99%) + weak MLP on 300 labels
python rn_mnist.py eval-base      # expect roughly 0.1 per digit
python sweep.py                   # all RL runs; resumable, skips finished runs
python collect.py                 # -> results_summary.json and grids/
```

Copy `results_summary.json`, `grids/` and `ckpt/classifiers.json` back to the blog repo
(the same folder). Do not commit `data/`, `ckpt/*.pt` or `runs/`.

Quick first pass (about 10% of the cost): `python sweep.py --only _s0` runs seed 0 only.

## Noise models

| `--noise` | meaning |
| --- | --- |
| `clean` | R = 1[oracle says 3] |
| `sym:FP,FN` | flip independently of the image |
| `confuse:8,RATE[,FN]` | only images of 8 get false positives, at RATE |
| `weak` | R = 1[a weak 300-label MLP says 3]: instance-dependent, can be exploited |

## Algorithms

* `flowgrpo`: SDE sampling (eta 0.7, 10 steps), group-normalised advantage (G = 16), PPO clip,
  Gaussian-mean KL to the base model.
* `nft`: DiffusionNFT negative-aware loss on the forward process with implicit positive/negative
  velocities `v± = (1∓β)v_old ± β v_θ`. The reward is either the normalised advantage mapped to
  [0, 1] (`norm`, the default) or the raw binary reward (`raw`).
