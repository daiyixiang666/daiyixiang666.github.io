# Reward-noise MNIST results

Windows GPU run (RTX 4060 Laptop, torch 2.6.0+cu124). All 78 runs finished (3 algos x 26 configs x seeds 0, 1; default hyperparameters, `--lr 1e-5`, 300 iterations). No algorithm code was edited.

Setup notes
- Base model retrained with `--epochs 50` (30 epochs gave oracle share of digit 3 = 0.165). Final base oracle histogram: `[0.096, 0.096, 0.084, 0.136, 0.088, 0.092, 0.092, 0.085, 0.123, 0.108]` (digit 3 is 0.136, slightly above the 0.13 bound; iteration-0 eval of the RL runs shows 0.14 to 0.17).
- Oracle test acc 0.9876, weak MLP test acc 0.8215 (300 labels).
- Default lr worked for both algos (flowgrpo clean: 0.165 -> 0.996 in 300 its; nft clean: 0.17 -> 0.997), so no `--rl_args` were used.

## Do the expectations hold? (seed 0 / seed 1, mean of last 5 evals)

1. **sym:0.3,0 and sym:0,0.3 reach high 3-accuracy without KL; flowgrpo slows about equally: holds for flowgrpo and nft. `nftraw` slowed more by FP: does NOT hold, the opposite happens.**
   flowgrpo FP-only 0.985/0.990 and FN-only 0.970/0.964, with iterations to 0.9 of 190/170 vs 180/200. nft: 0.954/0.933 vs 0.960/0.963. nftraw FP-only is only mildly slower than clean (0.971/0.968; 200/230 its vs 120/150) but nftraw FN-only fails: it climbs to about 0.6 (peak 0.63/0.65) and then collapses to 0.20/0.18 (0.07/0.04 at it 300) after about iteration 240.
2. **sym:0.6,0.6 makes 3s rarer: holds.** Final oracle_acc flowgrpo 0.039/0.058, nft 0.073/0.095, nftraw 0.110/0.133, all below the starting 0.14 to 0.17.
3. **confuse:8,0.9,0.3 drifts to 8s; confuse:8,0.7 stays on 3s but keeps extra 8s under KL: holds.**
   (0.9,0.3): digit-8 share 0.83 to 0.97 and 3-accuracy 0.01 to 0.15 for all three algos, while the noisy reward is about 0.85 to 0.88. (0.7) without KL: 3-accuracy 0.93 to 0.996, 8 share 0.00 to 0.06. With KL: 8 share 0.06 to 0.16, versus 0.01 to 0.02 for clean+KL.
4. **weak: noisy reward rises while oracle accuracy stalls or drops (reward hacking): does NOT hold.** The weak MLP reward and the oracle rise together (flowgrpo 0.994/0.997, nft 0.992/0.991, nftraw 0.997/0.947 oracle at the end, noisy reward within 0.04 of it). The only visible gap is nftraw seed 1 (oracle 0.947, noisy 0.909 at its tail, still rising at it 300), which is a slower run, not hacking.
5. **With KL, flowgrpo + FN-only noise settles below 100% while clean/FP-only saturates: only partly.** Ordering is as predicted (clean 0.946/0.948 > FP-only 0.860/0.864 > FN-only 0.795/0.797), but clean and FP-only do not saturate near 100% either; all KL runs sit below that. Flowgrpo+KL clean and FN-only have plateaued by it 300; FP-only was still creeping up (about 0.87 -> 0.88 over the last evals).

Other observations: the `nftraw` FN-only collapse shows in both seeds; `nftraw_kl0_sym-0.3_0.3_s1` also stalls (0.58, peak 0.64) while seed 0 reaches 0.91; `sym:0.45,0.45` (almost no signal) stays around 0.19 to 0.37.

## Per-run table (final iteration, n=1000 samples)

Digit-8 share = fraction of samples the oracle labels 8. Diversity = mean pairwise oracle-feature L2 among samples labelled 3. The `avg5` columns average the last 5 evals (it 260 to 300).

| run | done | it | oracle_acc | noisy_reward | digit-8 share | diversity | acc (avg5) | noisyR (avg5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowgrpo_kl0_clean_s0 | yes | 300 | 0.996 | 0.996 | 0.001 | 32.59 | 0.994 | 0.994 |
| flowgrpo_kl0_clean_s1 | yes | 300 | 0.998 | 0.998 | 0.001 | 32.71 | 0.996 | 0.996 |
| flowgrpo_kl0_confuse-8_0.7_s0 | yes | 300 | 0.994 | 0.997 | 0.005 | 35.71 | 0.988 | 0.993 |
| flowgrpo_kl0_confuse-8_0.7_s1 | yes | 300 | 0.985 | 0.991 | 0.010 | 35.13 | 0.985 | 0.992 |
| flowgrpo_kl0_confuse-8_0.9_0.3_s0 | yes | 300 | 0.008 | 0.913 | 0.983 | 40.58 | 0.012 | 0.880 |
| flowgrpo_kl0_confuse-8_0.9_0.3_s1 | yes | 300 | 0.025 | 0.877 | 0.958 | 41.35 | 0.027 | 0.875 |
| flowgrpo_kl0_sym-0.3_0.3_s0 | yes | 300 | 0.758 | 0.638 | 0.041 | 43.54 | 0.712 | 0.582 |
| flowgrpo_kl0_sym-0.3_0.3_s1 | yes | 300 | 0.871 | 0.645 | 0.023 | 40.40 | 0.831 | 0.622 |
| flowgrpo_kl0_sym-0.3_0_s0 | yes | 300 | 0.988 | 0.993 | 0.001 | 34.42 | 0.985 | 0.990 |
| flowgrpo_kl0_sym-0.3_0_s1 | yes | 300 | 0.992 | 0.996 | 0.003 | 34.94 | 0.990 | 0.993 |
| flowgrpo_kl0_sym-0.45_0.45_s0 | yes | 300 | 0.210 | 0.505 | 0.149 | 48.43 | 0.205 | 0.474 |
| flowgrpo_kl0_sym-0.45_0.45_s1 | yes | 300 | 0.398 | 0.487 | 0.080 | 45.14 | 0.371 | 0.481 |
| flowgrpo_kl0_sym-0.6_0.6_s0 | yes | 300 | 0.039 | 0.641 | 0.106 | 48.75 | 0.039 | 0.589 |
| flowgrpo_kl0_sym-0.6_0.6_s1 | yes | 300 | 0.039 | 0.582 | 0.126 | 47.12 | 0.058 | 0.578 |
| flowgrpo_kl0_sym-0_0.3_s0 | yes | 300 | 0.977 | 0.729 | 0.004 | 36.48 | 0.970 | 0.678 |
| flowgrpo_kl0_sym-0_0.3_s1 | yes | 300 | 0.961 | 0.660 | 0.004 | 35.45 | 0.964 | 0.669 |
| flowgrpo_kl0_weak_s0 | yes | 300 | 0.995 | 0.998 | 0.000 | 27.57 | 0.994 | 0.997 |
| flowgrpo_kl0_weak_s1 | yes | 300 | 0.997 | 1.000 | 0.000 | 27.98 | 0.997 | 0.997 |
| flowgrpo_kl_clean_s0 | yes | 300 | 0.944 | 0.944 | 0.008 | 37.98 | 0.946 | 0.946 |
| flowgrpo_kl_clean_s1 | yes | 300 | 0.949 | 0.949 | 0.012 | 37.41 | 0.948 | 0.948 |
| flowgrpo_kl_confuse-8_0.7_s0 | yes | 300 | 0.880 | 0.933 | 0.070 | 38.92 | 0.866 | 0.921 |
| flowgrpo_kl_confuse-8_0.7_s1 | yes | 300 | 0.886 | 0.930 | 0.066 | 38.48 | 0.876 | 0.925 |
| flowgrpo_kl_sym-0.3_0_s0 | yes | 300 | 0.876 | 0.910 | 0.013 | 38.72 | 0.860 | 0.900 |
| flowgrpo_kl_sym-0.3_0_s1 | yes | 300 | 0.893 | 0.928 | 0.025 | 39.06 | 0.864 | 0.904 |
| flowgrpo_kl_sym-0_0.3_s0 | yes | 300 | 0.808 | 0.595 | 0.026 | 39.80 | 0.795 | 0.556 |
| flowgrpo_kl_sym-0_0.3_s1 | yes | 300 | 0.825 | 0.568 | 0.041 | 40.16 | 0.797 | 0.548 |
| nft_kl0_clean_s0 | yes | 300 | 0.997 | 0.997 | 0.001 | 31.95 | 0.992 | 0.992 |
| nft_kl0_clean_s1 | yes | 300 | 0.985 | 0.985 | 0.005 | 31.92 | 0.988 | 0.988 |
| nft_kl0_confuse-8_0.7_s0 | yes | 300 | 0.973 | 0.991 | 0.022 | 32.57 | 0.968 | 0.983 |
| nft_kl0_confuse-8_0.7_s1 | yes | 300 | 0.953 | 0.981 | 0.041 | 32.71 | 0.932 | 0.974 |
| nft_kl0_confuse-8_0.9_0.3_s0 | yes | 300 | 0.111 | 0.892 | 0.877 | 39.98 | 0.140 | 0.847 |
| nft_kl0_confuse-8_0.9_0.3_s1 | yes | 300 | 0.061 | 0.872 | 0.921 | 43.29 | 0.062 | 0.871 |
| nft_kl0_sym-0.3_0.3_s0 | yes | 300 | 0.749 | 0.645 | 0.043 | 39.70 | 0.688 | 0.575 |
| nft_kl0_sym-0.3_0.3_s1 | yes | 300 | 0.647 | 0.559 | 0.060 | 40.89 | 0.553 | 0.520 |
| nft_kl0_sym-0.3_0_s0 | yes | 300 | 0.970 | 0.977 | 0.008 | 34.87 | 0.954 | 0.966 |
| nft_kl0_sym-0.3_0_s1 | yes | 300 | 0.945 | 0.959 | 0.014 | 33.91 | 0.933 | 0.954 |
| nft_kl0_sym-0.45_0.45_s0 | yes | 300 | 0.199 | 0.511 | 0.098 | 45.46 | 0.190 | 0.470 |
| nft_kl0_sym-0.45_0.45_s1 | yes | 300 | 0.208 | 0.467 | 0.144 | 46.15 | 0.189 | 0.465 |
| nft_kl0_sym-0.6_0.6_s0 | yes | 300 | 0.081 | 0.624 | 0.113 | 48.79 | 0.073 | 0.581 |
| nft_kl0_sym-0.6_0.6_s1 | yes | 300 | 0.103 | 0.578 | 0.115 | 47.55 | 0.095 | 0.572 |
| nft_kl0_sym-0_0.3_s0 | yes | 300 | 0.971 | 0.727 | 0.007 | 34.36 | 0.960 | 0.672 |
| nft_kl0_sym-0_0.3_s1 | yes | 300 | 0.962 | 0.669 | 0.007 | 33.29 | 0.963 | 0.671 |
| nft_kl0_weak_s0 | yes | 300 | 0.997 | 0.996 | 0.001 | 28.54 | 0.992 | 0.992 |
| nft_kl0_weak_s1 | yes | 300 | 0.988 | 0.990 | 0.003 | 28.09 | 0.991 | 0.989 |
| nft_kl_clean_s0 | yes | 300 | 0.933 | 0.933 | 0.018 | 37.05 | 0.910 | 0.910 |
| nft_kl_clean_s1 | yes | 300 | 0.904 | 0.904 | 0.016 | 36.76 | 0.900 | 0.900 |
| nft_kl_confuse-8_0.7_s0 | yes | 300 | 0.828 | 0.912 | 0.106 | 38.43 | 0.800 | 0.878 |
| nft_kl_confuse-8_0.7_s1 | yes | 300 | 0.772 | 0.862 | 0.132 | 38.62 | 0.734 | 0.843 |
| nft_kl_sym-0.3_0_s0 | yes | 300 | 0.813 | 0.872 | 0.039 | 38.99 | 0.757 | 0.828 |
| nft_kl_sym-0.3_0_s1 | yes | 300 | 0.723 | 0.806 | 0.051 | 39.53 | 0.659 | 0.762 |
| nft_kl_sym-0_0.3_s0 | yes | 300 | 0.812 | 0.610 | 0.036 | 38.59 | 0.770 | 0.539 |
| nft_kl_sym-0_0.3_s1 | yes | 300 | 0.789 | 0.544 | 0.037 | 37.77 | 0.737 | 0.514 |
| nftraw_kl0_clean_s0 | yes | 300 | 1.000 | 1.000 | 0.000 | 19.37 | 0.999 | 0.999 |
| nftraw_kl0_clean_s1 | yes | 300 | 0.996 | 0.996 | 0.000 | 20.90 | 0.998 | 0.998 |
| nftraw_kl0_confuse-8_0.7_s0 | yes | 300 | 0.999 | 1.000 | 0.001 | 18.30 | 0.996 | 0.998 |
| nftraw_kl0_confuse-8_0.7_s1 | yes | 300 | 0.992 | 0.996 | 0.007 | 19.91 | 0.988 | 0.995 |
| nftraw_kl0_confuse-8_0.9_0.3_s0 | yes | 300 | 0.144 | 0.888 | 0.845 | 28.07 | 0.154 | 0.853 |
| nftraw_kl0_confuse-8_0.9_0.3_s1 | yes | 300 | 0.087 | 0.868 | 0.900 | 28.13 | 0.095 | 0.872 |
| nftraw_kl0_sym-0.3_0.3_s0 | yes | 300 | 0.934 | 0.720 | 0.021 | 35.39 | 0.907 | 0.660 |
| nftraw_kl0_sym-0.3_0.3_s1 | yes | 300 | 0.643 | 0.557 | 0.045 | 39.16 | 0.582 | 0.534 |
| nftraw_kl0_sym-0.3_0_s0 | yes | 300 | 0.981 | 0.985 | 0.002 | 20.84 | 0.971 | 0.978 |
| nftraw_kl0_sym-0.3_0_s1 | yes | 300 | 0.971 | 0.981 | 0.006 | 23.61 | 0.968 | 0.978 |
| nftraw_kl0_sym-0.45_0.45_s0 | yes | 300 | 0.286 | 0.523 | 0.089 | 46.50 | 0.264 | 0.479 |
| nftraw_kl0_sym-0.45_0.45_s1 | yes | 300 | 0.218 | 0.472 | 0.148 | 48.07 | 0.195 | 0.466 |
| nftraw_kl0_sym-0.6_0.6_s0 | yes | 300 | 0.136 | 0.615 | 0.122 | 39.12 | 0.110 | 0.575 |
| nftraw_kl0_sym-0.6_0.6_s1 | yes | 300 | 0.131 | 0.570 | 0.163 | 36.92 | 0.133 | 0.565 |
| nftraw_kl0_sym-0_0.3_s0 | yes | 300 | 0.068 | 0.051 | 0.001 | 29.78 | 0.205 | 0.143 |
| nftraw_kl0_sym-0_0.3_s1 | yes | 300 | 0.037 | 0.026 | 0.000 | 34.01 | 0.178 | 0.126 |
| nftraw_kl0_weak_s0 | yes | 300 | 1.000 | 1.000 | 0.000 | 19.52 | 0.997 | 0.998 |
| nftraw_kl0_weak_s1 | yes | 300 | 0.961 | 0.926 | 0.000 | 27.50 | 0.947 | 0.909 |
| nftraw_kl_clean_s0 | yes | 300 | 0.974 | 0.974 | 0.006 | 28.77 | 0.968 | 0.968 |
| nftraw_kl_clean_s1 | yes | 300 | 0.974 | 0.974 | 0.003 | 28.18 | 0.977 | 0.977 |
| nftraw_kl_confuse-8_0.7_s0 | yes | 300 | 0.909 | 0.951 | 0.056 | 29.32 | 0.897 | 0.940 |
| nftraw_kl_confuse-8_0.7_s1 | yes | 300 | 0.899 | 0.943 | 0.067 | 29.35 | 0.897 | 0.946 |
| nftraw_kl_sym-0.3_0_s0 | yes | 300 | 0.901 | 0.931 | 0.020 | 31.44 | 0.875 | 0.910 |
| nftraw_kl_sym-0.3_0_s1 | yes | 300 | 0.863 | 0.904 | 0.036 | 31.58 | 0.855 | 0.900 |
| nftraw_kl_sym-0_0.3_s0 | yes | 300 | 0.325 | 0.252 | 0.006 | 31.15 | 0.493 | 0.345 |
| nftraw_kl_sym-0_0.3_s1 | yes | 300 | 0.745 | 0.524 | 0.015 | 33.69 | 0.670 | 0.469 |

