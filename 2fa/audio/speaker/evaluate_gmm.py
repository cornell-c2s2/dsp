import os
import numpy as np
import joblib
import csv
import argparse

# import shared gmm utilities
from gmm_utils import load_model, evaluate_dir

TEST_DIR = "data/gmm_test/"  # all positive should be named pos*.wav, all else negative
TARGET_MODEL = "models/target_gmm.joblib"
UBM_MODEL = "models/ubm_gmm.joblib"

# will use this as default if not provided on command line
DEFAULT_THRESHOLD = 0.24


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GMM target vs UBM on a test directory"
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output CSV filename (if provided, results will be saved)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold for LLR (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_gmm = load_model(TARGET_MODEL)
    ubm_gmm = load_model(UBM_MODEL)

    results = evaluate_dir(TEST_DIR, target_gmm, ubm_gmm, args.threshold)

    print(f"\n=== EVALUATION RESULTS ({len(results)} files) ===")
    pos_scores = [r["llr"] for r in results if r["label"] == 1]
    neg_scores = [r["llr"] for r in results if r["label"] == 0]

    if pos_scores and neg_scores:
        print(f"Avg positive LLR: {np.mean(pos_scores):.3f}")
        print(f"Avg negative LLR: {np.mean(neg_scores):.3f}")

    acc = np.mean([r["correct"] for r in results]) * 100 if results else 0.0
    print(f"Accuracy @ threshold {args.threshold:.2f}: {acc:.2f}%\n")

    for r in results:
        tag = "✅" if r["correct"] else "❌"
        print(
            f"{tag} {r['file']:<25}  LLR={r['llr']:+.3f}  Label={r['label']} Pred={r['pred']}"
        )

    if args.out:
        if results:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to {args.out}")
        else:
            print("\nNo results to save.")

    if results:
        llrs = np.array([r["llr"] for r in results])
        labels = np.array([r["label"] for r in results])

        thresholds = np.linspace(min(llrs), max(llrs), 200)
        accs = [np.mean((llrs > t) == labels) for t in thresholds]
        best_t = thresholds[np.argmax(accs)]
        best_acc = max(accs)
        print(f"{best_t:.7f}, {best_acc:.7f}")


if __name__ == "__main__":
    main()
