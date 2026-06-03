# Section 6 Baseline Explainer Results on 500 OOD Cases

| Method | Latency ms | Seed stability | Causal alignment | Grad-CAM leakage | Evidence purity |
| --- | --- | --- | --- | --- | --- |
| attention | 1.15 | 0.847 | 0.900 |  |  |
| integrated_gradients | 5.62 | 0.875 | 1.000 |  |  |
| lime | 48.75 | 0.826 | 0.900 |  |  |
| kernel_shap | 142.66 | 0.844 | 1.000 |  |  |
| modality_ablation | 3.98 | 0.827 | 1.000 |  |  |
| gradcam | 2.30 | 0.868 |  | 0.056 |  |
| retrieval_proxy | 2.35 |  |  |  | 0.413 |
