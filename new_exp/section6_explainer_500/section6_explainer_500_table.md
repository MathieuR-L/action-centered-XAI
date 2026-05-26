# Section 6 Explainer Results on 500 OOD Cases

| Method | Latency ms | Seed stability | Causal alignment | Grad-CAM leakage |
| --- | --- | --- | --- | --- |
| attention | 1.06 | 0.936 | 0.700 |  |
| integrated_gradients | 5.31 | 0.933 | 0.800 |  |
| lime | 47.58 | 0.898 | 0.800 |  |
| kernel_shap | 126.61 | 0.904 | 0.900 |  |
| modality_ablation | 3.69 | 0.931 | 0.900 |  |
| gradcam | 2.16 | 0.835 |  | 0.080 |
| retrieval_proxy | 3.41 |  |  |  |
