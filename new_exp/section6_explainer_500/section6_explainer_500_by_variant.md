# Section 6 Explainer Results on 500 OOD Cases by Model Variant

| Variant | Method | Latency ms | Seed stability | Causal alignment | Grad-CAM leakage | Evidence purity |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | attention | 1.15 | 0.847 | 0.900 |  |  |
| baseline | integrated_gradients | 5.62 | 0.875 | 1.000 |  |  |
| baseline | lime | 48.75 | 0.826 | 0.900 |  |  |
| baseline | kernel_shap | 142.66 | 0.844 | 1.000 |  |  |
| baseline | modality_ablation | 3.98 | 0.827 | 1.000 |  |  |
| baseline | gradcam | 2.30 | 0.868 |  | 0.056 |  |
| baseline | retrieval_proxy | 2.35 |  |  |  | 0.413 |
| causal_invariance | attention | 1.32 | 0.893 | 0.900 |  |  |
| causal_invariance | integrated_gradients | 6.40 | 0.882 | 1.000 |  |  |
| causal_invariance | lime | 53.24 | 0.829 | 0.900 |  |  |
| causal_invariance | kernel_shap | 163.19 | 0.851 | 1.000 |  |  |
| causal_invariance | modality_ablation | 4.49 | 0.836 | 0.900 |  |  |
| causal_invariance | gradcam | 2.55 | 0.822 |  | 0.016 |  |
| causal_invariance | retrieval_proxy | 3.01 |  |  |  | 0.584 |
