#!/usr/bin/env python
"""Run the Section 6 explainer table on an OOD explanation subset.

This complements section6_benchmark_experiments.py. It keeps the original
Captum-based explanation suite but moves from the manuscript's 12-case setting
to reviewer-requested larger OOD subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from mxai_assessment import run_assessment  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def explainer_rows(payload: dict[str, object], variant: str) -> list[dict[str, object]]:
    experiments = payload["experiments"][variant]
    methods = experiments["methods"]
    rows = []
    for method_name, metrics in methods.items():
        rows.append(
            {
                "variant": variant,
                "method": method_name,
                "latency_ms_mean": metrics.get("latency_ms_mean"),
                "stability_cosine_mean": metrics.get("stability_cosine_mean"),
                "causal_alignment_spearman": metrics.get("causal_alignment_spearman"),
                "leakage_shift_mean": metrics.get("leakage_shift_mean"),
                "evidence_purity": "",
            }
        )
    rows.append(
        {
            "variant": variant,
            "method": "retrieval_proxy",
            "latency_ms_mean": experiments["retrieval"]["latency_ms_mean"],
            "stability_cosine_mean": "",
            "causal_alignment_spearman": "",
            "leakage_shift_mean": "",
            "evidence_purity": experiments["retrieval"]["evidence_purity"],
        }
    )
    return rows


def format_cell(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def write_explainer_report(path: Path, title: str, rows: list[dict[str, object]], include_variant: bool) -> None:
    md_rows = []
    for row in rows:
        values = []
        if include_variant:
            values.append(row["variant"])
        values.extend(
            [
                row["method"],
                format_cell(row["latency_ms_mean"], 2),
                format_cell(row["stability_cosine_mean"], 3),
                format_cell(row["causal_alignment_spearman"], 3),
                format_cell(row["leakage_shift_mean"], 3),
                format_cell(row["evidence_purity"], 3),
            ]
        )
        md_rows.append(values)

    headers = ["Method", "Latency ms", "Seed stability", "Causal alignment", "Grad-CAM leakage", "Evidence purity"]
    if include_variant:
        headers = ["Variant", *headers]
    report = title + "\n\n" + markdown_table(headers, md_rows)
    path.write_text(report, encoding="utf-8")
    print(report)


def causal_comparison_report(payload: dict[str, object]) -> str:
    baseline = payload["experiments"]["baseline"]
    causal = payload["experiments"]["causal_invariance"]
    feature_names = payload["dataset"]["feature_names"]

    def feature_value(block: dict[str, object], feature: str) -> float:
        return float(block["feature_importance_ood"][feature])

    def attention_value(block: dict[str, object], feature: str) -> float:
        return float(block["methods"]["attention"]["mean_scores"][feature])

    rows = []
    for feature in feature_names:
        rows.append(
            [
                feature,
                f"{feature_value(baseline, feature):.3f}",
                f"{feature_value(causal, feature):.3f}",
                f"{attention_value(baseline, feature):.3f}",
                f"{attention_value(causal, feature):.3f}",
            ]
        )

    retrieval = [
        [
            "retrieval evidence purity",
            f"{float(baseline['retrieval']['evidence_purity']):.3f}",
            f"{float(causal['retrieval']['evidence_purity']):.3f}",
        ]
    ]
    leakage = [
        [
            "Grad-CAM leakage",
            f"{float(baseline['methods']['gradcam']['leakage_shift_mean']):.3f}",
            f"{float(causal['methods']['gradcam']['leakage_shift_mean']):.3f}",
        ]
    ]
    return (
        "# Baseline vs Causal Explanation Comparison\n\n"
        "## Feature Reliance and Attention\n\n"
        + markdown_table(
            ["Feature", "Baseline masking drop", "Causal masking drop", "Baseline attention", "Causal attention"],
            rows,
        )
        + "\n## Retrieval\n\n"
        + markdown_table(["Metric", "Baseline", "Causal"], retrieval)
        + "\n## Leakage\n\n"
        + markdown_table(["Metric", "Baseline", "Causal"], leakage)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 200-case ACE explainer assessment.")
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--explain-samples", type=int, default=200)
    parser.add_argument("--causal-lambda", type=float, default=4.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output-dir", default="new_exp/section6_explainer_200")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_limit": args.train_limit,
        "explain_samples": args.explain_samples,
        "causal_lambda": args.causal_lambda,
        "include_causal_explanations": not args.baseline_only,
        "device": str(device),
        "torch_version": torch.__version__,
        "note": "Uses mxai_assessment.run_assessment with final-protocol causal lambda by default.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = run_assessment(
        dataset_key=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        explain_samples=args.explain_samples,
        train_limit=args.train_limit,
        device=device,
        include_causal_explanations=not args.baseline_only,
        causal_lambda=args.causal_lambda,
    )
    suffix = args.explain_samples
    (output_dir / f"raw_explainer_{suffix}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    baseline_rows = explainer_rows(payload, "baseline")
    legacy_rows = [{key: value for key, value in row.items() if key != "variant"} for row in baseline_rows]
    write_csv(output_dir / f"explainer_{suffix}_table.csv", legacy_rows)
    write_explainer_report(
        output_dir / f"section6_explainer_{suffix}_table.md",
        f"# Section 6 Baseline Explainer Results on {args.explain_samples} OOD Cases",
        baseline_rows,
        include_variant=False,
    )

    if not args.baseline_only:
        by_variant_rows = baseline_rows + explainer_rows(payload, "causal_invariance")
        write_csv(output_dir / f"explainer_{suffix}_by_variant.csv", by_variant_rows)
        write_explainer_report(
            output_dir / f"section6_explainer_{suffix}_by_variant.md",
            f"# Section 6 Explainer Results on {args.explain_samples} OOD Cases by Model Variant",
            by_variant_rows,
            include_variant=True,
        )
        (output_dir / f"section6_explainer_{suffix}_causal_comparison.md").write_text(
            causal_comparison_report(payload),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
