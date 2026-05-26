# New Experiment Outputs

This folder contains add-on experiments requested after review. The paper files
were not modified.

## Section 6 Benchmark Robustness

Script:

```bash
source .venv/bin/activate
python new_exp/section6_benchmark_experiments.py
```

Outputs are in `new_exp/section6_benchmark/`.

The stronger publication-oriented run requested after the initial 200-case
experiment is in `new_exp/section6_benchmark_500_20seeds/` and was run with:

```bash
source .venv/bin/activate
python new_exp/section6_benchmark_experiments.py \
  --ood-eval-cases 500 \
  --seeds 7,17,27,37,47,57,67,77,87,97,107,117,127,137,147,157,167,177,187,197 \
  --lambdas 0.10,0.25,0.50,0.75,1.00,1.50,2.00,3.00,4.00 \
  --output-dir new_exp/section6_benchmark_500_20seeds
```

Publication-oriented settings:

- Dataset: `dermamnist`
- Seeds: `20`
- Lambda grid: `0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00`
- Train limit: `2000`
- Epochs: `4`
- OOD evaluation subset: `500` cases
- Bootstrap repetitions: `5000`

Publication-oriented result:

- Best lambda in this run: `4.00`.
- Baseline 500-case OOD accuracy: `0.604`.
- Best causal 500-case OOD accuracy: `0.753`.
- Best-lambda generalization gap: `0.091`.
- McNemar p-value at lambda `4.00`: `<1e-300`.
- Wilcoxon p-value across 20 seeds at lambda `4.00`: `0.000235`.

Key publication-oriented outputs:

- `section6_tables_500.md`: manuscript-ready 500-case accuracy, CI, McNemar, and Wilcoxon tables.
- `summary_by_lambda.csv`: numeric summary by lambda.
- `mcnemar_tests.csv`: paired exact McNemar tests on pooled 500-case predictions.
- `wilcoxon_tests.csv`: paired Wilcoxon tests across 20 seeds.
- `lambda_grid_ood_accuracy.png`: OOD accuracy by lambda.
- `lambda_grid_generalization_gap.png`: generalization gap by lambda.
- `raw_runs_with_predictions.json`: raw per-seed predictions for audit.

Main settings:

- Dataset: `dermamnist`
- Seeds: `7, 17, 27, 37, 47`
- Lambda grid: `0.25, 0.75, 1.50, 3.00`
- Train limit: `2000`
- Epochs: `4`
- OOD evaluation subset: `200` cases
- Bootstrap repetitions: `5000`

Key outputs:

- `section6_tables.md`: manuscript-ready accuracy, CI, McNemar, and Wilcoxon tables.
- `summary_by_lambda.csv`: numeric summary by lambda.
- `mcnemar_tests.csv`: paired exact McNemar tests on pooled 200-case predictions.
- `wilcoxon_tests.csv`: paired Wilcoxon tests across five seeds.
- `lambda_grid_ood_accuracy.png`: OOD accuracy by lambda.
- `lambda_grid_generalization_gap.png`: generalization gap by lambda.
- `raw_runs_with_predictions.json`: raw per-seed predictions for audit.

Observed result:

- Best lambda in this run: `3.00`.
- Baseline 200-case OOD accuracy: `0.645`.
- Best causal 200-case OOD accuracy: `0.732`.
- McNemar p-value at lambda `3.00`: `3.45e-18`.
- Wilcoxon across five seeds at lambda `3.00`: `0.125`.

## Section 6 Explainer Table on 200 OOD Cases

Script:

```bash
source .venv/bin/activate
python new_exp/section6_explainer_200_cases.py
```

Outputs are in `new_exp/section6_explainer_200/`.

The stronger 500-case explainer rerun is in `new_exp/section6_explainer_500/`
and was run with:

```bash
source .venv/bin/activate
python new_exp/section6_explainer_200_cases.py \
  --explain-samples 500 \
  --output-dir new_exp/section6_explainer_500
```

Publication-oriented explainer result:

- Attention latency: `1.06` ms/case.
- KernelSHAP latency: `126.61` ms/case.
- Grad-CAM leakage shift: `0.080`.

Key publication-oriented outputs:

- `section6_explainer_500_table.md`: manuscript-ready 500-case explainer metric table.
- `explainer_500_table.csv`: numeric explainer metrics.
- `raw_explainer_500_results.json`: full benchmark payload for the 500-case run.

Main settings:

- Dataset: `dermamnist`
- Seed: `7`
- Train limit: `2000`
- Epochs: `4`
- OOD explanation subset: `200` cases

Key outputs:

- `section6_explainer_200_table.md`: manuscript-ready explainer metric table.
- `explainer_200_table.csv`: numeric explainer metrics.
- `raw_explainer_200_results.json`: full benchmark payload.

Observed result:

- Attention latency: `1.13` ms/case.
- KernelSHAP latency: `127.92` ms/case.
- Grad-CAM leakage shift: `0.075`.
