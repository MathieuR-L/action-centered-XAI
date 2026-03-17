import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the ACE benchmark results and paper figures.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--explain-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run(
        [
            sys.executable,
            "experiments/mxai_assessment.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--train-limit",
            str(args.train_limit),
            "--explain-samples",
            str(args.explain_samples),
            "--seed",
            str(args.seed),
            "--output-dir",
            "experiments/results",
        ]
    )
    run([sys.executable, "experiments/make_paper_figures.py"])


if __name__ == "__main__":
    main()
