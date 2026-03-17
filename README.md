# MXAI ACE Reproducibility Bundle

This folder is a GitHub-ready reproducibility package for the controlled experiments and manuscript artifacts used in the paper.

## What Is Included

- `experiments/mxai_assessment.py`
  Reproduces the controlled multimodal benchmark, trains the baseline and causal-invariance models, and exports the measured metrics.
- `experiments/make_paper_figures.py`
  Regenerates the paper figures from the exported JSON results.
- `experiments/results/ace_experiment_results.json`
  Frozen results used in the current manuscript.
- `paper/`
  The LaTeX manuscript, bibliography, generated PDF, and all figures needed to compile the paper.
- `TRACEABILITY.md`
  A claim-to-evidence map showing which assessments are backed by experiments and which are backed by bibliography.
- `MANIFEST.sha256`
  File hashes for the canonical manuscript and experiment artifacts bundled here.
- `run_repro.py`
  Convenience entrypoint that reruns the benchmark and regenerates figures with the paper configuration.

## Environment

The paper results were generated with:

- Python `3.12.1`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `captum==0.8.0`
- `medmnist==3.0.2`
- `scikit-learn==1.4.1.post1`
- `matplotlib==3.8.2`

An RTX 4050 Laptop GPU was used for the reported runtime numbers. CPU execution is possible, but latencies will differ.

## Install

1. Install a compatible PyTorch build for your machine.
2. Install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

If you want the exact CUDA wheel family used here, install PyTorch separately before step 2.

## Reproduce the Paper Experiments

Run the full benchmark and regenerate the paper figures:

```bash
python run_repro.py
```

This writes:

- `experiments/results/ace_experiment_results.json`
- `paper/ace_benchmark_overview.png`
- `paper/ace_spurious_signal.png`

## Compile the Paper

From the `paper/` directory:

```bash
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Audit Citation Coverage

To verify that every `\cite{...}` key used by the manuscript exists in the bibliography:

```bash
python audit_support.py
```

## Notes

- The benchmark is intentionally controlled and lightweight. It is designed to validate ACE-style assessments, not to simulate a full clinical deployment stack.
- The retrieval experiment is a retrieval-only proxy, not a full V-RAG report-generation pipeline.
- The manuscript was edited so that method assessments are either experimentally grounded by this benchmark or explicitly literature-grounded by citation.
- Example workflows that were not reproduced locally are labeled as illustrative in the manuscript rather than presented as original empirical case studies.
