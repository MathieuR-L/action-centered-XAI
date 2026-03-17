import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results" / "ace_benchmark_suite_results.json"
PAPER_DIR = ROOT / "paper"


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def rank_positions(values: list[float], *, descending: bool) -> list[int]:
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks.tolist()


def save_suite_rank_consistency(results: dict) -> None:
    dataset_keys = list(results["datasets"].keys())
    labels = [results["datasets"][key]["dataset"]["base_dataset"].replace("MNIST", "") for key in dataset_keys]
    latency_methods = [
        ("attention", "Attention"),
        ("gradcam", "Grad-CAM"),
        ("modality_ablation", "Ablation"),
        ("integrated_gradients", "IG"),
        ("lime", "LIME"),
        ("kernel_shap", "KernelSHAP"),
        ("retrieval_proxy", "Retrieval"),
    ]
    causal_methods = [
        ("attention", "Attention"),
        ("integrated_gradients", "IG"),
        ("lime", "LIME"),
        ("kernel_shap", "KernelSHAP"),
        ("modality_ablation", "Ablation"),
    ]

    latency_rank_matrix = []
    causal_rank_matrix = []
    for key in dataset_keys:
        payload = results["datasets"][key]
        methods = payload["experiments"]["baseline"]["methods"]
        retrieval = payload["experiments"]["baseline"]["retrieval"]
        latency_values = [
            retrieval["latency_ms_mean"] if method_key == "retrieval_proxy" else methods[method_key]["latency_ms_mean"]
            for method_key, _ in latency_methods
        ]
        latency_rank_matrix.append(rank_positions(latency_values, descending=False))

        causal_values = [methods[method_key]["causal_alignment_spearman"] for method_key, _ in causal_methods]
        causal_rank_matrix.append(rank_positions(causal_values, descending=True))

    latency_rank_matrix = np.asarray(latency_rank_matrix, dtype=np.int32)
    causal_rank_matrix = np.asarray(causal_rank_matrix, dtype=np.int32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    im0 = axes[0].imshow(latency_rank_matrix, aspect="auto", cmap="YlGnBu_r", vmin=1, vmax=len(latency_methods))
    axes[0].set_xticks(np.arange(len(latency_methods)))
    axes[0].set_xticklabels([label for _, label in latency_methods], rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Latency Rank by Dataset (1 = fastest)")
    for row in range(latency_rank_matrix.shape[0]):
        for col in range(latency_rank_matrix.shape[1]):
            axes[0].text(col, row, str(latency_rank_matrix[row, col]), ha="center", va="center", fontsize=8)

    im1 = axes[1].imshow(causal_rank_matrix, aspect="auto", cmap="YlOrRd_r", vmin=1, vmax=len(causal_methods))
    axes[1].set_xticks(np.arange(len(causal_methods)))
    axes[1].set_xticklabels([label for _, label in causal_methods], rotation=25, ha="right")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Causal-Alignment Rank (1 = best)")
    for row in range(causal_rank_matrix.shape[0]):
        for col in range(causal_rank_matrix.shape[1]):
            axes[1].text(col, row, str(causal_rank_matrix[row, col]), ha="center", va="center", fontsize=8)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "ace_suite_rank_consistency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_suite_method_profile(results: dict) -> None:
    method_order = [
        ("attention", "Attention"),
        ("gradcam", "Grad-CAM"),
        ("modality_ablation", "Ablation"),
        ("integrated_gradients", "IG"),
        ("lime", "LIME"),
        ("kernel_shap", "KernelSHAP"),
        ("retrieval_proxy", "Retrieval"),
    ]
    method_means = results["aggregate"]["method_means"]
    labels = [label for _, label in method_order]
    latency = [method_means[key]["latency_ms_mean"] for key, _ in method_order]
    stability = [method_means[key].get("stability_cosine_mean", np.nan) for key, _ in method_order]
    causal_alignment = [method_means[key].get("causal_alignment_spearman", np.nan) for key, _ in method_order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    positions = np.arange(len(labels))
    colors = ["#4c72b0", "#64b5cd", "#8172b3", "#ccb974", "#dd8452", "#937860", "#55a868"]

    axes[0].bar(positions, latency, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("Latency (ms/case, log scale)")
    axes[0].set_title("Mean Runtime Across Datasets")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(positions, stability, color=colors)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Seed Stability")
    axes[1].set_title("Mean Stability Across Datasets")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(positions, causal_alignment, color=colors)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].set_ylim(-0.2, 1.05)
    axes[2].set_ylabel("Causal Alignment")
    axes[2].set_title("Mean Causal Alignment Across Datasets")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "ace_suite_method_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results()
    save_suite_rank_consistency(results)
    save_suite_method_profile(results)


if __name__ == "__main__":
    main()
