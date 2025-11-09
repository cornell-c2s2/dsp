import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GMM evaluation results")
    parser.add_argument(
        "--csv", "-c", required=True, help="CSV file from GMM evaluation"
    )
    parser.add_argument(
        "--outdir", "-o", default="plots", help="Directory to save plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load results
    df = pd.read_csv(args.csv)
    if not {"llr", "label", "pred", "correct"}.issubset(df.columns):
        raise ValueError("CSV must include llr, label, pred, and correct columns")

    llrs = df["llr"].values
    labels = df["label"].values
    preds = df["pred"].values

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{args.outdir}/confusion_matrix.png", dpi=300)
    plt.close()

    # --- 2. LLR Distribution ---
    plt.figure()
    plt.hist(llrs[labels == 1], bins=30, alpha=0.6, label="Positive", density=True)
    plt.hist(llrs[labels == 0], bins=30, alpha=0.6, label="Negative", density=True)
    plt.xlabel("Log-Likelihood Ratio (LLR)")
    plt.ylabel("Density")
    plt.title("LLR Distribution by Class")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(f"{args.outdir}/llr_distribution.png", dpi=300)
    plt.close()

    thresholds = np.linspace(min(llrs), max(llrs), 200)
    accs = [np.mean((llrs > t) == labels) for t in thresholds]
    best_t = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    plt.figure()
    plt.plot(thresholds, accs, label="Accuracy")
    plt.axvline(best_t, color="r", linestyle="--", label=f"Best Th={best_t:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Threshold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(f"{args.outdir}/accuracy_vs_threshold.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, llrs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(f"{args.outdir}/roc_curve.png", dpi=300)
    plt.close()

    print(f"✅ Charts saved in: {args.outdir}")
    print(f"Best threshold: {best_t:.3f}, Accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    import os

    main()
