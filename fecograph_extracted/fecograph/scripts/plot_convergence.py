# =============================================================================
# FL Convergence Curve Plotter (Paper Fig. 8 & Fig. 9)
#
# Reads one or more training_history.csv files and plots accuracy/F1/loss
# curves over communication rounds. Supports overlaying multiple experiments
# for comparison (e.g., SupCon vs SSLCon, Ditto vs FedAvg, different λ).
#
# Usage:
#   # Single experiment:
#   python scripts/plot_convergence.py --csv logs/training_history.csv
#
#   # Compare multiple experiments:
#   python scripts/plot_convergence.py \
#       --csv results/ditto_supcon/training_history.csv \
#       --csv results/ditto_sslcon/training_history.csv \
#       --labels "Ditto+SupCon" "Ditto+SSLCon"
# =============================================================================

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_metric(ax, dfs, labels, metric_col, ylabel, title):
    for df, label in zip(dfs, labels):
        if metric_col in df.columns:
            ax.plot(df["round"], df[metric_col], label=label, linewidth=1.5)
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot FL convergence curves (Paper Fig. 8/9)")
    parser.add_argument("--csv", action="append", required=True,
                        help="Path to training_history.csv (can specify multiple)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels for each CSV (same order)")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: logs/convergence.png)")
    args = parser.parse_args()

    dfs = []
    for path in args.csv:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded {path}: {len(df)} rounds")

    if args.labels and len(args.labels) == len(dfs):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(p)) or f"Exp {i+1}"
                  for i, p in enumerate(args.csv)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Test Accuracy (local + personalized)
    ax = axes[0, 0]
    for df, label in zip(dfs, labels):
        ax.plot(df["round"], df["mean_acc_local"], label=f"{label} (Global)", linewidth=1.5)
        if "mean_acc_pers" in df.columns:
            ax.plot(df["round"], df["mean_acc_pers"], label=f"{label} (Personal)",
                    linewidth=1.5, linestyle="--")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Test Accuracy over Rounds (Paper Fig. 8)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: F1 Score
    ax = axes[0, 1]
    for df, label in zip(dfs, labels):
        ax.plot(df["round"], df["mean_f1_local"], label=f"{label} (Global)", linewidth=1.5)
        if "mean_f1_pers" in df.columns:
            ax.plot(df["round"], df["mean_f1_pers"], label=f"{label} (Personal)",
                    linewidth=1.5, linestyle="--")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("F1 Score over Rounds", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Contrastive Loss
    plot_metric(axes[1, 0], dfs, labels, "loss_supcon",
                "Contrastive Loss", "Contrastive Loss (SupCon / SSLCon)")

    # Bottom-right: CE Loss
    plot_metric(axes[1, 1], dfs, labels, "loss_ce",
                "Cross-Entropy Loss", "Classification Loss (CE)")

    fig.suptitle("FeCoGraph FL Training Convergence", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if args.output is None:
        os.makedirs("logs", exist_ok=True)
        args.output = os.path.join("logs", "convergence.png")

    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Also print summary table
    print("\n" + "=" * 70)
    print("Summary (final round values):")
    print("=" * 70)
    for df, label in zip(dfs, labels):
        last = df.iloc[-1]
        print(f"  {label}:")
        print(f"    Rounds:   {int(last['round'])}")
        print(f"    Acc (G):  {last['mean_acc_local']:.4f}   F1 (G):  {last['mean_f1_local']:.4f}")
        if "mean_acc_pers" in df.columns:
            print(f"    Acc (P):  {last['mean_acc_pers']:.4f}   F1 (P):  {last['mean_f1_pers']:.4f}")
        best_acc = df["mean_acc_pers"].max() if "mean_acc_pers" in df.columns else df["mean_acc_local"].max()
        print(f"    BMTA:     {best_acc:.4f}")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
