# Follow-up: why does raw-reward DiffusionNFT collapse under false negatives?

INTERMEDIATE: block A only (18 runs). Blocks B to E are still running and will be added in the next commit.

Windows GPU run (RTX 4060 Laptop). All runs: DiffusionNFT, no KL, 450 iterations, seeds 0 to 2, reusing `ckpt/` from the main sweep. Code from commit f699232 (not otherwise modified). Collapse = peak oracle_acc > 0.4 and final oracle_acc < half of peak. "final oracle_acc" columns: the last eval (it 450) and the mean of the last 3 evals (it 430 to 450), because single evals use n=1000 samples and are noisy. "decline onset" = first iteration after the peak where oracle_acc < 0.8 x peak. "blank_frac>0.05 it" = first iteration where more than 5% of samples have mean pixel value below 0.02 (blank canvases).

## Block A: FN dose-response, raw reward (seeds 0, 1, 2)

Does a threshold FN exist for the collapse? **Yes, between FN=0.2 and FN=0.3.** The clean and FP=0.3 controls and FN=0.1 and 0.2 never collapse: peak oracle_acc is about 1.0 for clean and FP=0.3, 1.00/0.997/0.995 for FN=0.1 and 0.998/0.961/0.959 for FN=0.2, with final values (avg of last 3 evals) of 0.99 to 1.0 for FN=0.1 and 0.94 to 1.0 for FN=0.2. FN=0.3 collapses in 3/3 seeds: peaks 0.64/0.65/0.53 at it 150 to 230, final oracle_acc 0.002/0.000/0.057 (avg of last 3 evals), and final blank_frac 0.99/1.00/0.48. In seed 0 the final samples are blank canvases (ink 0.004, blank_frac 0.99, and the oracle labels 99% of them as "1").

Does higher FN make it start earlier? **Not monotonic.** FN=0.3: decline onset it 250/250/200. FN=0.45: decline onset it 230/220/230, so only marginally earlier than 0.3, and it is not a full collapse: oracle_acc peaks at 0.41 to 0.53 (it 170 to 180) and ends at 0.34 to 0.49. The FN=0.45 runs do show a transient blank-out (seed 0: blank_frac 0.74 at it 300, oracle_acc 0.32) and then recover (blank_frac 0.01, oracle_acc 0.51 at it 450), so they oscillate and the "collapsed" flag (final < half of peak) does not fire. Treat FN=0.45 as "unstable, partly recovers", not as "no effect".

Does blank_frac rise before oracle_acc falls? **Yes.** FN=0.3: blank_frac first exceeds 0.05 at it 160/130/170, the peak oracle_acc is at it 200/230/150, and the oracle_acc decline only starts at it 250/250/200. In seed 0, blank_frac is already 0.27 at it 180 while oracle_acc is still 0.58 to 0.61 (it 150 to 240); mass moves into the blank class ("1" label: 0.26 at it 180, 0.37 at it 240) while the rate of 3s holds up for about 60 more iterations, then blank takes over. FN=0.2 also shows a transient blank_frac rise (0.13 to 0.20 at it 150 to 180, seed 1) that recovers by it 420 (0.01), so a rising blank_frac alone is not sufficient for a collapse.

## Per-run table (block A)

| run | peak oracle_acc | peak it | final oracle_acc (last eval) | final oracle_acc (avg last 3) | final blank_frac | final ink | collapsed | decline onset it | blank_frac>0.05 it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fu_A_raw_clean_s0 | 1.000 | 230 | 1.000 | 1.000 | 0.000 | 0.125 | no | None | None |
| fu_A_raw_clean_s1 | 1.000 | 310 | 1.000 | 1.000 | 0.000 | 0.153 | no | None | None |
| fu_A_raw_clean_s2 | 1.000 | 310 | 1.000 | 1.000 | 0.000 | 0.118 | no | None | None |
| fu_A_raw_sym-0.3_0_s0 | 1.000 | 440 | 1.000 | 1.000 | 0.000 | 0.100 | no | None | None |
| fu_A_raw_sym-0.3_0_s1 | 1.000 | 430 | 1.000 | 1.000 | 0.000 | 0.092 | no | None | None |
| fu_A_raw_sym-0.3_0_s2 | 1.000 | 380 | 1.000 | 1.000 | 0.000 | 0.110 | no | None | None |
| fu_A_raw_sym-0_0.1_s0 | 1.000 | 320 | 1.000 | 1.000 | 0.000 | 0.154 | no | None | None |
| fu_A_raw_sym-0_0.1_s1 | 0.997 | 420 | 0.997 | 0.992 | 0.000 | 0.217 | no | None | 160 |
| fu_A_raw_sym-0_0.1_s2 | 0.995 | 400 | 0.995 | 0.992 | 0.000 | 0.192 | no | None | 310 |
| fu_A_raw_sym-0_0.2_s0 | 0.998 | 400 | 0.998 | 0.998 | 0.000 | 0.226 | no | None | 190 |
| fu_A_raw_sym-0_0.2_s1 | 0.961 | 420 | 0.944 | 0.944 | 0.026 | 0.253 | no | None | 140 |
| fu_A_raw_sym-0_0.2_s2 | 0.959 | 450 | 0.959 | 0.946 | 0.008 | 0.243 | no | None | 200 |
| fu_A_raw_sym-0_0.3_s0 | 0.635 | 200 | 0.003 | 0.002 | 0.991 | 0.004 | YES | 250 | 160 |
| fu_A_raw_sym-0_0.3_s1 | 0.653 | 230 | 0.000 | 0.000 | 0.999 | 0.003 | YES | 250 | 130 |
| fu_A_raw_sym-0_0.3_s2 | 0.528 | 150 | 0.051 | 0.057 | 0.477 | 0.033 | YES | 200 | 170 |
| fu_A_raw_sym-0_0.45_s0 | 0.534 | 170 | 0.508 | 0.488 | 0.007 | 0.068 | no | 230 | 170 |
| fu_A_raw_sym-0_0.45_s1 | 0.460 | 170 | 0.336 | 0.343 | 0.000 | 0.070 | no | 220 | 150 |
| fu_A_raw_sym-0_0.45_s2 | 0.415 | 180 | 0.394 | 0.394 | 0.000 | 0.116 | no | 230 | 160 |

