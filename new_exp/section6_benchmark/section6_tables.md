# Section 6 Benchmark Add-On Results

Best causal lambda by 200-case OOD accuracy: `3.00`.

## Accuracy and Confidence Intervals

| Model | lambda | ID acc. | OOD acc. (200) | 95% bootstrap CI | Gen. gap |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.00 | 0.865 | 0.645 | [0.563, 0.723] | 0.213 |
| causal_invariance | 0.25 | 0.868 | 0.641 | [0.574, 0.702] | 0.225 |
| causal_invariance | 0.75 | 0.867 | 0.663 | [0.602, 0.715] | 0.200 |
| causal_invariance | 1.50 | 0.867 | 0.696 | [0.643, 0.740] | 0.167 |
| causal_invariance | 3.00 | 0.863 | 0.732 | [0.695, 0.764] | 0.118 |

## Paired Statistical Tests vs Baseline

| lambda | McNemar p | Discordant | Wilcoxon p |
| --- | --- | --- | --- |
| 0.25 | 0.6271 | 38 | 0.75 |
| 0.75 | 0.01535 | 50 | 0.625 |
| 1.50 | 5.195e-09 | 79 | 0.125 |
| 3.00 | 3.447e-18 | 111 | 0.125 |

Baseline mean OOD accuracy on the 200-case subset: `0.645`.
Best causal mean OOD accuracy on the 200-case subset: `0.732`.
