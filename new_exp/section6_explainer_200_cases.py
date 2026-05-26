#!/usr/bin/env python
"""Run the Section 6 explainer table on 200 OOD cases.

This complements section6_benchmark_experiments.py. It keeps the original
Captum-based explanation suite but moves from the manuscript's 12-case setting
to a reviewer-requested 200-case OOD subset.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 200-case ACE explainer assessment.")
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--explain-samples", type=int, default=200)
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
        "device": str(device),
        "torch_version": torch.__version__,
        "note": "Uses mxai_assessment.run_assessment with the original fixed causal lambda of 0.75.",
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
        include_causal_explanations=False,
    )
    (output_dir / "raw_explainer_200_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    methods = payload["experiments"]["baseline"]["methods"]
    rows = []
    for method_name, metrics in methods.items():
        row = {
            "method": method_name,
            "latency_ms_mean": metrics.get("latency_ms_mean"),
            "stability_cosine_mean": metrics.get("stability_cosine_mean"),
            "causal_alignment_spearman": metrics.get("causal_alignment_spearman"),
            "leakage_shift_mean": metrics.get("leakage_shift_mean"),
        }
        rows.append(row)
    rows.append(
        {
            "method": "retrieval_proxy",
            "latency_ms_mean": payload["experiments"]["baseline"]["retrieval"]["latency_ms_mean"],
            "stability_cosine_mean": "",
            "causal_alignment_spearman": "",
            "leakage_shift_mean": "",
        }
    )
    write_csv(output_dir / "explainer_200_table.csv", rows)

    md_rows = []
    for row in rows:
        md_rows.append(
            [
                row["method"],
                "" if row["latency_ms_mean"] == "" else f'{float(row["latency_ms_mean"]):.2f}',
                "" if row["stability_cosine_mean"] == "" else f'{float(row["stability_cosine_mean"]):.3f}',
                "" if row["causal_alignment_spearman"] in ("", None) else f'{float(row["causal_alignment_spearman"]):.3f}',
                "" if row["leakage_shift_mean"] in ("", None) else f'{float(row["leakage_shift_mean"]):.3f}',
            ]
        )
    report = (
        f"# Section 6 Explainer Results on {args.explain_samples} OOD Cases\n\n"
        + markdown_table(
            ["Method", "Latency ms", "Seed stability", "Causal alignment", "Grad-CAM leakage"],
            md_rows,
        )
    )
    (output_dir / "section6_explainer_200_table.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
