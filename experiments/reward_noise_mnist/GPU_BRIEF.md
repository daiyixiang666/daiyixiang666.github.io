# Brief for the Windows GPU session (folder: minist)

Paste everything below into the Claude session running in the `minist` folder.

---

Run the MNIST reward-noise experiments for my blog post "When the Reward Model Lies" on this GPU.

**Code:** `experiments/reward_noise_mnist/` from the blog repo
(https://github.com/daiyixiang666/daiyixiang666.github.io, branch `reward-noise`). Copy that folder
into this directory. Don't edit the algorithms. If something crashes, fix the minimum needed and
note what you changed.

**What it is:** a small rectified-flow UNet on MNIST (unconditional, about 10% per digit) is RL
fine-tuned to draw the digit 3. The reward is a deliberately corrupted oracle classifier with
false positives, false negatives, class confusion (8 rewarded as 3), or a weak 300-label MLP. Two algorithms are compared:
Flow-GRPO (SDE sampling, PPO clip, group-normalised advantage) and DiffusionNFT (negative-aware
forward-process loss, normalised or raw reward).

**Steps**
1. Create a venv, install CUDA torch + torchvision + numpy (see README.md). Check `torch.cuda.is_available()`.
2. `python rn_mnist.py base`, then `python rn_mnist.py classifier`, then `python rn_mnist.py eval-base`.
   Sanity: each digit should be roughly 0.07–0.13 in the oracle histogram, and oracle test acc should be ≥ 0.985.
   If the base looks bad (blurry, skewed histogram), train longer (`--epochs 50`) before continuing.
3. Time one run: `python sweep.py --only flowgrpo_kl0_clean_s0`. It should show oracle_acc rising
   from ~0.1. If it doesn't move in 300 iters, try `python sweep.py --only flowgrpo_kl0_clean_s0 --rl_args "--lr 3e-5"`
   (delete that run's folder first) and report what worked; then pass the same `--rl_args` to the full sweep. Same check for `nft_kl0_clean_s0`.
4. Run seed 0 of everything first: `python sweep.py --only _s0`, then the full `python sweep.py`.
5. `python collect.py`, then send back `results_summary.json`, the `grids/` folder and
   `ckpt/classifiers.json`, plus a short table: final oracle_acc, final noisy_reward, final digit-8 share,
   diversity for each run.

**What I expect** (report whether each holds):
- `sym:0.3,0` and `sym:0,0.3` both reach high 3-accuracy without KL; flowgrpo slows similarly for both; `nftraw` is slowed more by FP.
- `sym:0.6,0.6` makes 3s *rarer*.
- `confuse:8,0.9,0.3` drifts to 8s; `confuse:8,0.7` stays on 3s but keeps extra 8s under KL.
- `weak`: noisy reward rises while oracle accuracy stalls or drops (reward hacking).
- With KL, flowgrpo + FN-only noise settles below 100% while clean/FP-only saturates.
