#!/usr/bin/env python
"""Section 6 robustness add-on experiments for the ACE paper.

This script intentionally lives outside the paper and original experiment
scripts. It reuses the benchmark implementation while adding the reviewer-
requested controls: more OOD cases, a lambda grid, five seeds, bootstrap
confidence intervals, McNemar tests, and Wilcoxon signed-rank tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from mxai_assessment import (  # noqa: E402
    ControlledMedMNIST,
    MultimodalAttentionNet,
    TrainingConfig,
    evaluate_model,
    make_loader,
    set_seed,
    train_model,
)


@dataclass
class RunResult:
    dataset: str
    seed: int
    lambda_value: float
    model_type: str
    id_accuracy: float
    ood_accuracy: float
    ood_accuracy_200: float
    generalization_gap: float
    ood_correct_200: list[int]
    ood_pred_200: list[int]
    ood_label_200: list[int]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def prediction_records(
    model: MultimodalAttentionNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    labels_out: list[int] = []
    correct: list[int] = []
    with torch.no_grad():
        for images, tabular, labels in loader:
            logits = model(images.to(device), tabular.to(device))
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            batch_labels = labels.numpy()
            preds.extend(batch_preds.astype(int).tolist())
            labels_out.extend(batch_labels.astype(int).tolist())
            correct.extend((batch_preds == batch_labels).astype(int).tolist())
    return preds, labels_out, correct


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, reps: int = 5000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(array, size=(reps, len(array)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(correct_a: list[int], correct_b: list[int]) -> dict[str, float]:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    baseline_only = int(np.sum(a & ~b))
    causal_only = int(np.sum(~a & b))
    discordant = baseline_only + causal_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(baseline_only, causal_only), discordant, 0.5).pvalue)
    return {
        "baseline_only_correct": baseline_only,
        "causal_only_correct": causal_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def paired_wilcoxon(values_a: list[float], values_b: list[float]) -> float:
    diffs = np.asarray(values_b, dtype=np.float64) - np.asarray(values_a, dtype=np.float64)
    if np.allclose(diffs, 0):
        return 1.0
    return float(wilcoxon(values_b, values_a, zero_method="wilcox", alternative="two-sided").pvalue)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def make_datasets(dataset_key: str, train_limit: int, ood_eval_cases: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ControlledMedMNIST(
        dataset_key,
        "train",
        transform=transform,
        ood_hospital_bias=False,
        limit=train_limit,
    )
    val_dataset = ControlledMedMNIST(dataset_key, "val", transform=transform, ood_hospital_bias=False)
    test_dataset = ControlledMedMNIST(dataset_key, "test", transform=transform, ood_hospital_bias=False)
    ood_dataset = ControlledMedMNIST(dataset_key, "test", transform=transform, ood_hospital_bias=True)
    ood_200 = Subset(ood_dataset, list(range(min(ood_eval_cases, len(ood_dataset)))))
    return train_dataset, val_dataset, test_dataset, ood_dataset, ood_200


def train_one(
    *,
    dataset_key: str,
    seed: int,
    lambda_value: float,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    train_limit: int,
    ood_eval_cases: int,
    device: torch.device,
) -> RunResult:
    set_seed(seed)
    train_dataset, val_dataset, test_dataset, ood_dataset, ood_200 = make_datasets(
        dataset_key, train_limit, ood_eval_cases
    )
    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False)
    ood_loader = make_loader(ood_dataset, batch_size, shuffle=False)
    ood_200_loader = make_loader(ood_200, batch_size, shuffle=False)

    config = TrainingConfig(epochs=epochs, batch_size=batch_size, lr=lr, causal_lambda=lambda_value)
    model = MultimodalAttentionNet(num_classes=train_dataset.num_classes).to(device)
    model, _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        config,
        seed=seed,
        causal_invariance=(model_type == "causal_invariance"),
    )
    id_metrics = evaluate_model(model, test_loader, device)
    ood_metrics = evaluate_model(model, ood_loader, device)
    preds_200, labels_200, correct_200 = prediction_records(model, ood_200_loader, device)
    ood_acc_200 = float(np.mean(correct_200))
    return RunResult(
        dataset=dataset_key,
        seed=seed,
        lambda_value=lambda_value,
        model_type=model_type,
        id_accuracy=float(id_metrics["accuracy"]),
        ood_accuracy=float(ood_metrics["accuracy"]),
        ood_accuracy_200=ood_acc_200,
        generalization_gap=float(id_metrics["accuracy"] - ood_metrics["accuracy"]),
        ood_correct_200=correct_200,
        ood_pred_200=preds_200,
        ood_label_200=labels_200,
    )


def summarize(
    results: list[RunResult],
    output_dir: Path,
    bootstrap_reps: int,
    ood_eval_cases: int,
) -> dict[str, object]:
    rng = np.random.default_rng(20260521)
    flat_rows = []
    for result in results:
        row = asdict(result)
        row.pop("ood_correct_200")
        row.pop("ood_pred_200")
        row.pop("ood_label_200")
        flat_rows.append(row)
    write_csv(output_dir / "raw_runs.csv", flat_rows)

    by_key: dict[tuple[str, float], list[RunResult]] = {}
    for result in results:
        by_key.setdefault((result.model_type, result.lambda_value), []).append(result)

    summary_rows: list[dict[str, object]] = []
    for (model_type, lambda_value), group in sorted(by_key.items()):
        ood_values = [item.ood_accuracy_200 for item in group]
        full_ood_values = [item.ood_accuracy for item in group]
        id_values = [item.id_accuracy for item in group]
        gap_values = [item.generalization_gap for item in group]
        ci_low, ci_high = bootstrap_ci(ood_values, rng, bootstrap_reps)
        summary_rows.append(
            {
                "model_type": model_type,
                "lambda": lambda_value,
                "n_seeds": len(group),
                "id_accuracy_mean": float(np.mean(id_values)),
                "ood_accuracy_full_mean": float(np.mean(full_ood_values)),
                "ood_accuracy_200_mean": float(np.mean(ood_values)),
                "ood_accuracy_200_ci_low": ci_low,
                "ood_accuracy_200_ci_high": ci_high,
                "generalization_gap_mean": float(np.mean(gap_values)),
                "ood_accuracy_200_std": float(np.std(ood_values, ddof=1)) if len(ood_values) > 1 else 0.0,
            }
        )
    write_csv(output_dir / "summary_by_lambda.csv", summary_rows)

    baseline_by_seed = {item.seed: item for item in results if item.model_type == "baseline"}
    mcnemar_rows = []
    wilcoxon_rows = []
    lambdas = sorted({item.lambda_value for item in results if item.model_type == "causal_invariance"})
    for lambda_value in lambdas:
        causal_group = [item for item in results if item.model_type == "causal_invariance" and item.lambda_value == lambda_value]
        paired_baseline = [baseline_by_seed[item.seed] for item in causal_group if item.seed in baseline_by_seed]
        causal_group = [item for item in causal_group if item.seed in baseline_by_seed]
        if not causal_group:
            continue

        pooled_base_correct: list[int] = []
        pooled_causal_correct: list[int] = []
        for base_item, causal_item in zip(paired_baseline, causal_group):
            pooled_base_correct.extend(base_item.ood_correct_200)
            pooled_causal_correct.extend(causal_item.ood_correct_200)
        test = mcnemar_exact(pooled_base_correct, pooled_causal_correct)
        test["lambda"] = lambda_value
        mcnemar_rows.append(test)

        wilcoxon_rows.append(
            {
                "lambda": lambda_value,
                "baseline_ood_accuracy_200_by_seed": ";".join(f"{item.ood_accuracy_200:.4f}" for item in paired_baseline),
                "causal_ood_accuracy_200_by_seed": ";".join(f"{item.ood_accuracy_200:.4f}" for item in causal_group),
                "p_value": paired_wilcoxon(
                    [item.ood_accuracy_200 for item in paired_baseline],
                    [item.ood_accuracy_200 for item in causal_group],
                ),
            }
        )
    write_csv(output_dir / "mcnemar_tests.csv", mcnemar_rows)
    write_csv(output_dir / "wilcoxon_tests.csv", wilcoxon_rows)

    best_row = max(
        [row for row in summary_rows if row["model_type"] == "causal_invariance"],
        key=lambda row: (row["ood_accuracy_200_mean"], -row["generalization_gap_mean"]),
    )
    baseline_row = next(row for row in summary_rows if row["model_type"] == "baseline")

    md_rows = []
    for row in summary_rows:
        md_rows.append(
            [
                row["model_type"],
                f'{row["lambda"]:.2f}',
                f'{row["id_accuracy_mean"]:.3f}',
                f'{row["ood_accuracy_200_mean"]:.3f}',
                f'[{row["ood_accuracy_200_ci_low"]:.3f}, {row["ood_accuracy_200_ci_high"]:.3f}]',
                f'{row["generalization_gap_mean"]:.3f}',
            ]
        )
    table = markdown_table(
        ["Model", "lambda", "ID acc.", f"OOD acc. ({ood_eval_cases})", "95% bootstrap CI", "Gen. gap"],
        md_rows,
    )

    tests_table = markdown_table(
        ["lambda", "McNemar p", "Discordant", "Wilcoxon p"],
        [
            [
                f'{row["lambda"]:.2f}',
                f'{row["p_value"]:.4g}',
                int(row["discordant_pairs"]),
                f'{next(w["p_value"] for w in wilcoxon_rows if math.isclose(w["lambda"], row["lambda"])):.4g}',
            ]
            for row in mcnemar_rows
        ],
    )

    report = (
        "# Section 6 Benchmark Add-On Results\n\n"
        f"Best causal lambda by {ood_eval_cases}-case OOD accuracy: `{best_row['lambda']:.2f}`.\n\n"
        "## Accuracy and Confidence Intervals\n\n"
        + table
        + "\n## Paired Statistical Tests vs Baseline\n\n"
        + tests_table
        + "\n"
        f"Baseline mean OOD accuracy on the {ood_eval_cases}-case subset: `{baseline_row['ood_accuracy_200_mean']:.3f}`.\n"
        f"Best causal mean OOD accuracy on the {ood_eval_cases}-case subset: `{best_row['ood_accuracy_200_mean']:.3f}`.\n"
    )
    (output_dir / "section6_tables.md").write_text(report, encoding="utf-8")

    payload = {
        "summary_by_lambda": summary_rows,
        "mcnemar_tests": mcnemar_rows,
        "wilcoxon_tests": wilcoxon_rows,
        "best_lambda": best_row["lambda"],
        "ood_eval_cases": ood_eval_cases,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def plot_results(summary: dict[str, object], output_dir: Path) -> None:
    rows = summary["summary_by_lambda"]
    causal = [row for row in rows if row["model_type"] == "causal_invariance"]
    baseline = next(row for row in rows if row["model_type"] == "baseline")
    xs = np.asarray([row["lambda"] for row in causal], dtype=np.float64)
    ys = np.asarray([row["ood_accuracy_200_mean"] for row in causal], dtype=np.float64)
    lows = np.asarray([row["ood_accuracy_200_ci_low"] for row in causal], dtype=np.float64)
    highs = np.asarray([row["ood_accuracy_200_ci_high"] for row in causal], dtype=np.float64)

    plt.figure(figsize=(7, 4.2))
    plt.plot(xs, ys, marker="o", label="Causal invariance")
    plt.fill_between(xs, lows, highs, alpha=0.18, label="95% bootstrap CI")
    plt.axhline(baseline["ood_accuracy_200_mean"], color="black", linestyle="--", linewidth=1.2, label="Baseline")
    plt.xlabel("Causal consistency lambda")
    plt.ylabel(f"OOD accuracy on {summary.get('ood_eval_cases', 200)} cases")
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "lambda_grid_ood_accuracy.png", dpi=220)
    plt.close()

    gaps = np.asarray([row["generalization_gap_mean"] for row in causal], dtype=np.float64)
    plt.figure(figsize=(7, 4.2))
    plt.plot(xs, gaps, marker="o", color="#8c2d04", label="Causal invariance")
    plt.axhline(baseline["generalization_gap_mean"], color="black", linestyle="--", linewidth=1.2, label="Baseline")
    plt.xlabel("Causal consistency lambda")
    plt.ylabel("Generalization gap")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "lambda_grid_generalization_gap.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 6 ACE add-on experiments.")
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--lambdas", default="0.25,0.75,1.50,3.00")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--ood-eval-cases", type=int, default=200)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--output-dir", default="new_exp/section6_benchmark")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    lambdas = parse_float_list(args.lambdas)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "dataset": args.dataset,
        "seeds": seeds,
        "lambdas": lambdas,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_limit": args.train_limit,
        "ood_eval_cases": args.ood_eval_cases,
        "bootstrap_reps": args.bootstrap_reps,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: list[RunResult] = []
    for seed in seeds:
        print(f"[section6] baseline seed={seed}")
        results.append(
            train_one(
                dataset_key=args.dataset,
                seed=seed,
                lambda_value=0.0,
                model_type="baseline",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                train_limit=args.train_limit,
                ood_eval_cases=args.ood_eval_cases,
                device=device,
            )
        )
        for lambda_value in lambdas:
            print(f"[section6] causal seed={seed} lambda={lambda_value}")
            results.append(
                train_one(
                    dataset_key=args.dataset,
                    seed=seed,
                    lambda_value=lambda_value,
                    model_type="causal_invariance",
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    train_limit=args.train_limit,
                    ood_eval_cases=args.ood_eval_cases,
                    device=device,
                )
            )

    raw_payload = [asdict(result) for result in results]
    (output_dir / "raw_runs_with_predictions.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    summary = summarize(results, output_dir, args.bootstrap_reps, args.ood_eval_cases)
    plot_results(summary, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
