# Follow-up: why does raw-reward DiffusionNFT collapse under false negatives?

All 42 runs finished (blocks A to E, 3 seeds each). Windows GPU run (RTX 4060 Laptop). All runs: DiffusionNFT, no KL, 450 iterations, seeds 0 to 2, reusing `ckpt/` from the main sweep. Code from commit f699232, not otherwise modified.

Definitions used in the table
- collapsed = peak oracle_acc > 0.4 and final oracle_acc < half of peak (the rule you specified; "final" = mean of the last 3 evals, it 430 to 450, because single evals use n=1000 samples).
- The rule misses transient collapses that later recover (blocks A at FN=0.45 and D), so the table also gives the maximum blank_frac (and its iteration) and the minimum oracle_acc after the peak. blank_frac = fraction of samples whose mean pixel value (0 to 1 scale) is below 0.02, i.e. blank canvases; ink = mean pixel value.
- decline onset = first iteration after the peak where oracle_acc < 0.8 x peak. "blank_frac>0.05 it" = first iteration where more than 5% of samples are blank.

## Answers

**A: Is there an FN threshold? Does higher FN start the collapse earlier? Does blank_frac rise first?**
- Threshold between FN=0.2 and 0.3 (raw reward, beta 1.0). Clean, FP=0.3, FN=0.1 and FN=0.2 never collapse (final oracle_acc 0.94 to 1.0). FN=0.3 collapses in 3/3 seeds: peaks 0.64/0.65/0.53 at it 150 to 230, final oracle_acc 0.002/0.000/0.057, final blank_frac 0.99/1.00/0.48. In seed 0 the final samples are blank canvases (ink 0.004) and the oracle labels 99% of them "1".
- Not monotonic in FN. FN=0.45 has about the same decline onset as 0.3 (it 220 to 230 vs 200 to 250), is not flagged as collapsed (peak 0.41 to 0.53, final 0.34 to 0.49), but shows a transient blank-out (max blank_frac 0.68 to 0.94 at it 280 to 300) followed by partial recovery (final blank_frac about 0).
- blank_frac does rise before oracle_acc falls: at FN=0.3 it exceeds 0.05 at it 130 to 170, the peak is at it 150 to 230 and the decline starts at it 200 to 250. In seed 0, blank_frac is 0.27 at it 180 while oracle_acc is still about 0.6; the mass moves to the "1" label while the 3s hold up for roughly 60 more iterations. A rising blank_frac alone is not sufficient: FN=0.2 shows a transient rise (max 0.13 to 0.20 at it 150 to 180) and recovers.

**B: With the negative branch off (`--nft_neg_weight 0`), does FN=0.3 still collapse? No.** Peak oracle_acc 0.999/0.987/0.990, final 0.993/0.963/0.982, blank_frac 0 throughout. Clean with the negative branch off also reaches 0.996/0.967/0.989. So with raw reward, the negative branch fed with false negatives is necessary for the collapse. Seed 1 is a little slower (final 0.96 to 0.97); the positive-only runs end with thinner strokes (ink about 0.09 vs about 0.15).

**C: Does centring alone (`--nft_reward center`, no std scaling) prevent it? Yes, in these runs.** FN=0.3: peak 0.998/1.000/0.997, final 0.996/0.997/0.995, no blank samples; clean the same (0.995 to 0.999).

**D: Does a smaller nft_beta delay or prevent it? No, it does not prevent it, and it does not delay it: the blank-out happens earlier.**
- beta 0.1: all 3 seeds peak early and low (0.42 to 0.47 at it 60 to 80), decline from it 90 to 100, hit blank_frac 0.95 to 1.00 at it 140 to 170 with oracle_acc as low as 0.004 to 0.096, and then recover (blank_frac back to 0 by about it 250 to 300 in the seeds I inspected, 0 and 1) and climb slowly to 0.36 to 0.44 by it 450. The rule flags 0/3 as collapsed only because of this late partial recovery.
- beta 0.3: mixed. Seed 0 has a mild transient (max blank_frac 0.14) and reaches 0.985 (no collapse); seed 1 reaches blank_frac 0.99 at it 180, falls to 0.05 and ends at 0.27 after recovering from blank canvases; seed 2 reaches blank_frac 0.47 at it 230, dips to 0.41 and recovers to 0.73 by it 450.
- For comparison, at beta 1.0 (block A) the blank state was absorbing in 2 of 3 seeds (final blank_frac 0.99/1.00, seed 2 0.48) and did not recover within 450 iterations. These runs show that a smaller beta makes the blank-out transient rather than permanent, not that it prevents it. Only 3 seeds, so the beta 0.3 spread (0.27 to 0.98) is large.

**E: Does normalised NFT stay stable to 450 iterations? Yes.** FN=0.3: final oracle_acc 0.987/0.991/0.990 (min after peak 0.983 to 0.984), blank_frac 0 throughout. Clean: 0.995/0.998/0.997.

Caveats: 3 seeds per config, 450 iterations; oracle_acc is measured on n=1000 samples per eval. This follow-up did not test why the negative branch with false negatives pushes toward blank canvases; the table only shows that removing the branch or centring the raw reward removes the collapse.

## Per-run table

| run | peak oracle_acc | peak it | final oracle_acc (last eval) | final oracle_acc (avg last 3) | final blank_frac | final ink | collapsed | decline onset it | blank_frac>0.05 it | max blank_frac (it) | min oracle_acc after peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fu_A_raw_clean_s0 | 1.000 | 230 | 1.000 | 1.000 | 0.000 | 0.125 | no | None | None | 0.00 (0) | 0.998 |
| fu_A_raw_clean_s1 | 1.000 | 310 | 1.000 | 1.000 | 0.000 | 0.153 | no | None | None | 0.02 (200) | 0.998 |
| fu_A_raw_clean_s2 | 1.000 | 310 | 1.000 | 1.000 | 0.000 | 0.118 | no | None | None | 0.00 (0) | 0.997 |
| fu_A_raw_sym-0.3_0_s0 | 1.000 | 440 | 1.000 | 1.000 | 0.000 | 0.100 | no | None | None | 0.00 (0) | 1.000 |
| fu_A_raw_sym-0.3_0_s1 | 1.000 | 430 | 1.000 | 1.000 | 0.000 | 0.092 | no | None | None | 0.00 (0) | 1.000 |
| fu_A_raw_sym-0.3_0_s2 | 1.000 | 380 | 1.000 | 1.000 | 0.000 | 0.110 | no | None | None | 0.00 (0) | 0.997 |
| fu_A_raw_sym-0_0.1_s0 | 1.000 | 320 | 1.000 | 1.000 | 0.000 | 0.154 | no | None | None | 0.00 (210) | 0.999 |
| fu_A_raw_sym-0_0.1_s1 | 0.997 | 420 | 0.997 | 0.992 | 0.000 | 0.217 | no | None | 160 | 0.16 (200) | 0.987 |
| fu_A_raw_sym-0_0.1_s2 | 0.995 | 400 | 0.995 | 0.992 | 0.000 | 0.192 | no | None | 310 | 0.05 (310) | 0.990 |
| fu_A_raw_sym-0_0.2_s0 | 0.998 | 400 | 0.998 | 0.998 | 0.000 | 0.226 | no | None | 190 | 0.10 (220) | 0.995 |
| fu_A_raw_sym-0_0.2_s1 | 0.961 | 420 | 0.944 | 0.944 | 0.026 | 0.253 | no | None | 140 | 0.20 (180) | 0.944 |
| fu_A_raw_sym-0_0.2_s2 | 0.959 | 450 | 0.959 | 0.946 | 0.008 | 0.243 | no | None | 200 | 0.26 (240) | 0.959 |
| fu_A_raw_sym-0_0.3_s0 | 0.635 | 200 | 0.003 | 0.002 | 0.991 | 0.004 | YES | 250 | 160 | 1.00 (410) | 0.000 |
| fu_A_raw_sym-0_0.3_s1 | 0.653 | 230 | 0.000 | 0.000 | 0.999 | 0.003 | YES | 250 | 130 | 1.00 (420) | 0.000 |
| fu_A_raw_sym-0_0.3_s2 | 0.528 | 150 | 0.051 | 0.057 | 0.477 | 0.033 | YES | 200 | 170 | 1.00 (310) | 0.002 |
| fu_A_raw_sym-0_0.45_s0 | 0.534 | 170 | 0.508 | 0.488 | 0.007 | 0.068 | no | 230 | 170 | 0.74 (300) | 0.305 |
| fu_A_raw_sym-0_0.45_s1 | 0.460 | 170 | 0.336 | 0.343 | 0.000 | 0.070 | no | 220 | 150 | 0.94 (280) | 0.202 |
| fu_A_raw_sym-0_0.45_s2 | 0.415 | 180 | 0.394 | 0.394 | 0.000 | 0.116 | no | 230 | 160 | 0.68 (280) | 0.214 |
| fu_B_raw_neg0_clean_s0 | 0.999 | 430 | 0.995 | 0.996 | 0.000 | 0.091 | no | None | None | 0.00 (0) | 0.994 |
| fu_B_raw_neg0_clean_s1 | 0.995 | 420 | 0.957 | 0.967 | 0.000 | 0.092 | no | None | None | 0.00 (0) | 0.957 |
| fu_B_raw_neg0_clean_s2 | 0.996 | 400 | 0.979 | 0.989 | 0.000 | 0.095 | no | None | None | 0.00 (0) | 0.979 |
| fu_B_raw_neg0_sym-0_0.3_s0 | 0.999 | 430 | 0.988 | 0.993 | 0.000 | 0.095 | no | None | None | 0.00 (0) | 0.988 |
| fu_B_raw_neg0_sym-0_0.3_s1 | 0.987 | 420 | 0.938 | 0.963 | 0.000 | 0.090 | no | None | None | 0.00 (0) | 0.938 |
| fu_B_raw_neg0_sym-0_0.3_s2 | 0.990 | 400 | 0.971 | 0.982 | 0.000 | 0.099 | no | None | None | 0.00 (0) | 0.969 |
| fu_C_center_clean_s0 | 0.998 | 400 | 0.994 | 0.995 | 0.000 | 0.158 | no | None | None | 0.00 (0) | 0.993 |
| fu_C_center_clean_s1 | 1.000 | 450 | 1.000 | 0.997 | 0.000 | 0.150 | no | None | None | 0.00 (0) | 1.000 |
| fu_C_center_clean_s2 | 1.000 | 400 | 1.000 | 0.999 | 0.000 | 0.153 | no | None | None | 0.00 (0) | 0.995 |
| fu_C_center_sym-0_0.3_s0 | 0.998 | 400 | 0.997 | 0.996 | 0.000 | 0.163 | no | None | None | 0.00 (0) | 0.993 |
| fu_C_center_sym-0_0.3_s1 | 1.000 | 450 | 1.000 | 0.997 | 0.000 | 0.147 | no | None | None | 0.00 (0) | 1.000 |
| fu_C_center_sym-0_0.3_s2 | 0.997 | 430 | 0.996 | 0.995 | 0.000 | 0.139 | no | None | None | 0.00 (0) | 0.992 |
| fu_D_raw_beta0.1_sym-0_0.3_s0 | 0.468 | 70 | 0.448 | 0.441 | 0.000 | 0.270 | no | 100 | 70 | 1.00 (170) | 0.027 |
| fu_D_raw_beta0.1_sym-0_0.3_s1 | 0.418 | 60 | 0.394 | 0.363 | 0.000 | 0.269 | no | 90 | 60 | 1.00 (140) | 0.004 |
| fu_D_raw_beta0.1_sym-0_0.3_s2 | 0.450 | 80 | 0.397 | 0.406 | 0.000 | 0.282 | no | 100 | 80 | 0.95 (170) | 0.096 |
| fu_D_raw_beta0.3_sym-0_0.3_s0 | 0.985 | 400 | 0.975 | 0.978 | 0.016 | 0.283 | no | None | 100 | 0.14 (220) | 0.975 |
| fu_D_raw_beta0.3_sym-0_0.3_s1 | 0.496 | 100 | 0.281 | 0.271 | 0.000 | 0.227 | no | 120 | 80 | 0.99 (180) | 0.051 |
| fu_D_raw_beta0.3_sym-0_0.3_s2 | 0.734 | 450 | 0.734 | 0.686 | 0.000 | 0.138 | no | None | 110 | 0.47 (230) | 0.734 |
| fu_E_norm_clean_s0 | 0.998 | 410 | 0.994 | 0.995 | 0.000 | 0.156 | no | None | None | 0.00 (0) | 0.992 |
| fu_E_norm_clean_s1 | 1.000 | 450 | 1.000 | 0.998 | 0.000 | 0.145 | no | None | None | 0.00 (0) | 1.000 |
| fu_E_norm_clean_s2 | 1.000 | 420 | 0.999 | 0.997 | 0.000 | 0.146 | no | None | None | 0.00 (0) | 0.994 |
| fu_E_norm_sym-0_0.3_s0 | 0.996 | 400 | 0.992 | 0.987 | 0.000 | 0.162 | no | None | None | 0.00 (0) | 0.983 |
| fu_E_norm_sym-0_0.3_s1 | 0.995 | 380 | 0.994 | 0.991 | 0.000 | 0.147 | no | None | None | 0.00 (100) | 0.984 |
| fu_E_norm_sym-0_0.3_s2 | 0.994 | 430 | 0.993 | 0.990 | 0.000 | 0.143 | no | None | None | 0.00 (0) | 0.983 |

