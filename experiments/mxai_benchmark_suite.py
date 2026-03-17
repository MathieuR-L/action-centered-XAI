import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from mxai_assessment import run_assessment


DEFAULT_DATASETS = [
    "bloodmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "pathmnist",
    "pneumoniamnist",
]


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def aggregate_results(dataset_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    aggregate = {
        "datasets_evaluated": list(dataset_payloads.keys()),
        "baseline_id_accuracy_mean": safe_mean(
            [payload["models"]["baseline"]["id_test"]["accuracy"] for payload in dataset_payloads.values()]
        ),
        "baseline_ood_accuracy_mean": safe_mean(
            [payload["models"]["baseline"]["ood_test"]["accuracy"] for payload in dataset_payloads.values()]
        ),
        "causal_id_accuracy_mean": safe_mean(
            [payload["models"]["causal_invariance"]["id_test"]["accuracy"] for payload in dataset_payloads.values()]
        ),
        "causal_ood_accuracy_mean": safe_mean(
            [payload["models"]["causal_invariance"]["ood_test"]["accuracy"] for payload in dataset_payloads.values()]
        ),
        "ood_gain_mean": safe_mean(
            [
                payload["models"]["causal_invariance"]["ood_test"]["accuracy"]
                - payload["models"]["baseline"]["ood_test"]["accuracy"]
                for payload in dataset_payloads.values()
            ]
        ),
        "generalization_gap_reduction_mean": safe_mean(
            [
                (
                    payload["models"]["baseline"]["id_test"]["accuracy"]
                    - payload["models"]["baseline"]["ood_test"]["accuracy"]
                )
                - (
                    payload["models"]["causal_invariance"]["id_test"]["accuracy"]
                    - payload["models"]["causal_invariance"]["ood_test"]["accuracy"]
                )
                for payload in dataset_payloads.values()
            ]
        ),
        "method_means": {},
        "winner_counts": {
            "fastest": {},
            "most_stable": {},
            "best_causal_alignment": {},
        },
    }

    method_names = [
        "attention",
        "integrated_gradients",
        "lime",
        "kernel_shap",
        "modality_ablation",
        "gradcam",
    ]

    for payload in dataset_payloads.values():
        methods = payload["experiments"]["baseline"]["methods"]

        fastest_method = min(method_names, key=lambda name: methods[name]["latency_ms_mean"])
        aggregate["winner_counts"]["fastest"][fastest_method] = aggregate["winner_counts"]["fastest"].get(fastest_method, 0) + 1

        stable_method = max(method_names, key=lambda name: methods[name]["stability_cosine_mean"])
        aggregate["winner_counts"]["most_stable"][stable_method] = aggregate["winner_counts"]["most_stable"].get(stable_method, 0) + 1

        causal_method = max(
            ["attention", "integrated_gradients", "lime", "kernel_shap", "modality_ablation"],
            key=lambda name: methods[name]["causal_alignment_spearman"],
        )
        aggregate["winner_counts"]["best_causal_alignment"][causal_method] = (
            aggregate["winner_counts"]["best_causal_alignment"].get(causal_method, 0) + 1
        )

    for method_name in method_names:
        per_dataset = [payload["experiments"]["baseline"]["methods"][method_name] for payload in dataset_payloads.values()]
        keys = set().union(*(entry.keys() for entry in per_dataset))
        aggregate["method_means"][method_name] = {}
        for key in keys:
            numeric_values = [entry[key] for entry in per_dataset if isinstance(entry.get(key), (int, float))]
            if numeric_values:
                aggregate["method_means"][method_name][key] = safe_mean(numeric_values)

    retrieval_entries = [payload["experiments"]["baseline"]["retrieval"] for payload in dataset_payloads.values()]
    aggregate["method_means"]["retrieval_proxy"] = {}
    retrieval_keys = set().union(*(entry.keys() for entry in retrieval_entries))
    for key in retrieval_keys:
        numeric_values = [entry[key] for entry in retrieval_entries if isinstance(entry.get(key), (int, float))]
        if numeric_values:
            aggregate["method_means"]["retrieval_proxy"][key] = safe_mean(numeric_values)

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset ACE benchmark suite.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--explain-samples", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_payloads: Dict[str, Dict[str, object]] = {}

    for offset, dataset_key in enumerate(args.datasets):
        print(f"[ACE suite] Launching {dataset_key}")
        dataset_payloads[dataset_key] = run_assessment(
            dataset_key=dataset_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + offset * 10,
            explain_samples=args.explain_samples,
            train_limit=args.train_limit,
            device=device,
            include_causal_explanations=False,
        )
        baseline_ood = dataset_payloads[dataset_key]["models"]["baseline"]["ood_test"]["accuracy"]
        causal_ood = dataset_payloads[dataset_key]["models"]["causal_invariance"]["ood_test"]["accuracy"]
        print(
            "[ACE suite] {0}: baseline OOD={1:.3f}, causal OOD={2:.3f}, gain={3:.3f}".format(
                dataset_key,
                baseline_ood,
                causal_ood,
                causal_ood - baseline_ood,
            )
        )

    payload = {
        "config": {
            "datasets": args.datasets,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "explain_samples": args.explain_samples,
            "train_limit": args.train_limit,
            "device": str(device),
        },
        "datasets": dataset_payloads,
        "aggregate": aggregate_results(dataset_payloads),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "ace_benchmark_suite_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
