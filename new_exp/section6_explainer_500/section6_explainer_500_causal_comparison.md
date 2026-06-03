# Baseline vs Causal Explanation Comparison

## Feature Reliance and Attention

| Feature | Baseline masking drop | Causal masking drop | Baseline attention | Causal attention |
| --- | --- | --- | --- | --- |
| image | 0.000 | -0.002 | 0.046 | 0.057 |
| biomarker_score | 0.216 | 0.408 | 0.154 | 0.243 |
| age_risk | 0.414 | 0.643 | 0.322 | 0.283 |
| hospital_bias | 0.429 | 0.279 | 0.410 | 0.265 |
| admin_noise | 0.000 | 0.001 | 0.030 | 0.070 |

## Retrieval

| Metric | Baseline | Causal |
| --- | --- | --- |
| retrieval evidence purity | 0.413 | 0.584 |

## Leakage

| Metric | Baseline | Causal |
| --- | --- | --- |
| Grad-CAM leakage | 0.056 | 0.016 |
