import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results" / "ace_experiment_results.json"
PAPER_DIR = ROOT / "paper"


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def save_overview_figure(results: dict) -> None:
    baseline = results["models"]["baseline"]
    causal = results["models"]["causal_invariance"]
    methods = results["experiments"]["baseline"]["methods"]

    labels = ["ID", "OOD"]
    baseline_scores = [
        baseline["id_test"]["accuracy"] * 100.0,
        baseline["ood_test"]["accuracy"] * 100.0,
    ]
    causal_scores = [
        causal["id_test"]["accuracy"] * 100.0,
        causal["ood_test"]["accuracy"] * 100.0,
    ]

    latency_methods = [
        "attention",
        "gradcam",
        "modality_ablation",
        "integrated_gradients",
        "lime",
        "kernel_shap",
    ]
    latency_labels = ["Attention", "Grad-CAM", "Ablation", "IG", "LIME", "KernelSHAP"]
    latency_values = [methods[name]["latency_ms_mean"] for name in latency_methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(labels))
    width = 0.34

    axes[0].bar(x - width / 2, baseline_scores, width=width, color="#c44e52", label="Baseline")
    axes[0].bar(x + width / 2, causal_scores, width=width, color="#55a868", label="Causal-Invariance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Robustness Under Spurious Hospital Shift")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    for idx, value in enumerate(baseline_scores):
        axes[0].text(idx - width / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(causal_scores):
        axes[0].text(idx + width / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    latency_positions = np.arange(len(latency_labels))
    axes[1].bar(latency_positions, latency_values, color=["#4c72b0", "#64b5cd", "#8172b3", "#ccb974", "#dd8452", "#937860"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(latency_positions)
    axes[1].set_xticklabels(latency_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Latency (ms/case, log scale)")
    axes[1].set_title("Explanation Runtime on the Baseline Model")
    axes[1].grid(axis="y", alpha=0.2)
    for idx, value in enumerate(latency_values):
        axes[1].text(idx, value * 1.08, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "ace_benchmark_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spurious_signal_figure(results: dict) -> None:
    baseline = results["experiments"]["baseline"]
    causal = results["experiments"]["causal_invariance"]

    feature_labels = ["Image", "Biomarker", "Age", "Hospital", "Noise"]
    importance_baseline = [baseline["feature_importance_ood"][name] for name in baseline["feature_importance_ood"]]
    importance_causal = [causal["feature_importance_ood"][name] for name in causal["feature_importance_ood"]]

    attention_baseline = baseline["methods"]["attention"]["mean_scores"]
    attention_causal = causal["methods"]["attention"]["mean_scores"]
    attention_baseline_values = [attention_baseline[name] for name in attention_baseline]
    attention_causal_values = [attention_causal[name] for name in attention_causal]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(feature_labels))
    width = 0.34

    axes[0].bar(x - width / 2, importance_baseline, width=width, color="#c44e52", label="Baseline")
    axes[0].bar(x + width / 2, importance_causal, width=width, color="#55a868", label="Causal-Invariance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_labels, rotation=15)
    axes[0].set_ylabel("OOD Accuracy Drop After Feature Masking")
    axes[0].set_title("Measured Feature Reliance")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - width / 2, attention_baseline_values, width=width, color="#4c72b0", label="Baseline")
    axes[1].bar(x + width / 2, attention_causal_values, width=width, color="#64b5cd", label="Causal-Invariance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_labels, rotation=15)
    axes[1].set_ylabel("Mean Attention Score")
    axes[1].set_title("Attention Redistribution After Causal Training")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "ace_spurious_signal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results()
    save_overview_figure(results)
    save_spurious_signal_figure(results)


if __name__ == "__main__":
    main()
