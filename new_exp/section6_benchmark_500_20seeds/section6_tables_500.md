# Section 6 Benchmark Add-On Results, 500 OOD Cases and 20 Seeds

Best causal lambda by 500-case OOD accuracy: `4.00`.

## Accuracy and Confidence Intervals

| Model | lambda | ID acc. | OOD acc. (500) | 95% bootstrap CI | Gen. gap |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.00 | 0.854 | 0.604 | [0.549, 0.662] | 0.252 |
| causal_invariance | 0.10 | 0.853 | 0.615 | [0.558, 0.671] | 0.242 |
| causal_invariance | 0.25 | 0.852 | 0.621 | [0.565, 0.677] | 0.236 |
| causal_invariance | 0.50 | 0.852 | 0.630 | [0.573, 0.683] | 0.227 |
| causal_invariance | 0.75 | 0.852 | 0.640 | [0.584, 0.693] | 0.217 |
| causal_invariance | 1.00 | 0.852 | 0.650 | [0.594, 0.701] | 0.206 |
| causal_invariance | 1.50 | 0.851 | 0.671 | [0.621, 0.717] | 0.186 |
| causal_invariance | 2.00 | 0.850 | 0.692 | [0.642, 0.735] | 0.164 |
| causal_invariance | 3.00 | 0.847 | 0.726 | [0.683, 0.761] | 0.123 |
| causal_invariance | 4.00 | 0.843 | 0.753 | [0.723, 0.777] | 0.091 |

## Paired Statistical Tests vs Baseline

| lambda | McNemar p | Discordant | Wilcoxon p |
| --- | --- | --- | --- |
| 0.10 | 1.666e-09 | 339 | 0.2388 |
| 0.25 | 2.214e-19 | 366 | 0.01924 |
| 0.50 | 9.76e-37 | 449 | 0.002806 |
| 0.75 | 1.382e-57 | 545 | 0.002223 |
| 1.00 | 3.389e-82 | 633 | 0.001177 |
| 1.50 | 1.22e-132 | 849 | 0.0005532 |
| 2.00 | 6.535e-190 | 1059 | 0.0002717 |
| 3.00 | 1.011e-279 | 1393 | 0.0002695 |
| 4.00 | <1e-300 | 1673 | 0.0002354 |

Baseline mean OOD accuracy on the 500-case subset: `0.604`.
Best causal mean OOD accuracy on the 500-case subset: `0.753`.
