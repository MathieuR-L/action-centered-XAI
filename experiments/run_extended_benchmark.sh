#!/usr/bin/env bash
set -euo pipefail

python experiments/mxai_benchmark_suite.py \
  --datasets bloodmnist dermamnist octmnist organamnist pathmnist pneumoniamnist \
  --epochs 3 \
  --batch-size 128 \
  --train-limit 1500 \
  --explain-samples 8 \
  --output-dir experiments/results

python experiments/make_suite_figures.py
