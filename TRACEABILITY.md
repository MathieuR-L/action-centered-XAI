# Traceability Matrix

This document records how the paper's main assessments are supported.

## Experiment-Backed Assessments

All method assessments in the following manuscript components are backed by the controlled benchmark in `experiments/mxai_assessment.py` and the exported results in `experiments/results/ace_experiment_results.json`:

- Abstract benchmark claims in `paper/main.tex`
- `Table 3`: measured suitability across clinical actions
- All `Clinical Suitability Assessment` boxes for:
  - KernelSHAP / SHAP proxy
  - LIME
  - Integrated Gradients
  - Grad-CAM
  - Attention
  - Retrieval-grounded explanation proxy
  - Modality ablation
- `Table 6`: controlled clinical action suitability matrix
- `Section 6`: controlled experimental assessment of ACE claims
- `Section 7`: ACE cross-dataset stress-test benchmark

## Exact Experimental Results Used in the Paper

- Baseline in-distribution accuracy: `89.8%`
- Baseline out-of-distribution accuracy: `66.1%`
- Causal-invariance in-distribution accuracy: `85.8%`
- Causal-invariance out-of-distribution accuracy: `82.9%`
- Attention latency: `2.75 ms/case`
- Grad-CAM latency: `3.76 ms/case`
- Modality ablation latency: `8.09 ms/case`
- Integrated Gradients latency: `10.60 ms/case`
- LIME latency: `111.54 ms/case`
- KernelSHAP latency: `294.16 ms/case`
- Grad-CAM leakage shift: `0.031` on the baseline model
- Retrieval evidence purity: `0.52` baseline, `0.73` after causal training

## Cross-Dataset Suite Results Used in the Paper

These support the new Section 7 benchmark discussion:

- Domains covered: `BloodMNIST`, `DermaMNIST`, `OCTMNIST`, `OrganAMNIST`, `PathMNIST`, `PneumoniaMNIST`
- Attention mean latency across domains: `3.16 ms/case`
- Attention fastest on: `6/6` datasets
- Grad-CAM mean latency: `5.57 ms/case`
- Grad-CAM mean leakage: `0.031`
- Integrated Gradients mean causal alignment: `0.867`
- LIME mean causal alignment: `0.867`
- KernelSHAP mean causal alignment: `0.883`
- Retrieval proxy mean latency: `3.85 ms/case`
- Retrieval proxy mean top-5 evidence purity: `0.492`

## Bibliography-Backed Assessments

The following claim families remain literature-grounded rather than benchmark-grounded:

- Background and historical framing of XAI and MXAI
- Published results for MedCLIP, Med-PaLM, GPT-4V, CausalCLIPSeg, MDFormer, SMMILE, MuCR, and related systems
- Regulatory framing involving FDA guidance and the EU AI Act
- Telesurgery latency literature
- Human-centered evaluation claims and clinician-trust literature
- Domain application case studies not reproduced locally
- Illustrative clinical workflows that clarify how MXAI explanations could appear in practice

These statements are cited directly in `paper/main.tex` and resolved through `paper/references.bib`.

## Deliberate Scope Limits

- The benchmark uses DermaMNIST plus controlled synthetic clinical covariates.
- It validates method properties relevant to ACE:
  - latency
  - retraining stability
  - spurious multimodal reliance
  - cross-modal leakage
  - intervention sensitivity
- It does **not** claim to reproduce:
  - full hospital deployment
  - radiologist reader studies
  - FDA validation
  - full V-RAG report generation
- large medical foundation model inference stacks
- illustrative workflow examples that are pedagogical rather than reproduced experiments

## Editorial Principle Used

Where a previous manuscript statement was stronger than what the controlled experiments supported, the manuscript was revised to match the measured evidence rather than to overclaim.
