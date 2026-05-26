# Section 6 Explainer Results on 200 OOD Cases

| Method | Latency ms | Seed stability | Causal alignment | Grad-CAM leakage |
| --- | --- | --- | --- | --- |
| attention | 1.13 | 0.612 | 0.900 |  |
| integrated_gradients | 5.62 | 0.764 | 0.900 |  |
| lime | 48.87 | 0.743 | 0.900 |  |
| kernel_shap | 127.92 | 0.795 | 0.900 |  |
| modality_ablation | 3.72 | 0.761 | 0.900 |  |
| gradcam | 2.24 | 0.770 |  | 0.075 |
| retrieval_proxy | 1.94 |  |  |  |
