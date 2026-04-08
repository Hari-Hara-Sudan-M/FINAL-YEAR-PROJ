# =============================================================================
# Detailed Model Evaluation (Paper Table III - per-attack F1 scores)
#
# Loads a checkpoint and evaluates on test data, printing:
#   - Overall accuracy, precision, recall, F1 (macro)
#   - Per-class precision, recall, F1, support
#   - Confusion matrix
#
# Usage:
#   python scripts/evaluate_model.py --checkpoint checkpoints/global_model_best_pers.pt
#   python scripts/evaluate_model.py --checkpoint checkpoints/global_model_final.pt --mode multiclass
# =============================================================================

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI,
    CLASSIFICATION_MODE, DATA_DIR, SEED
)
from models.model import FeCoGraphModel
from data.line_graph import csv_to_line_graph

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ATTACK_NAMES_IDS2018 = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

BINARY_NAMES = {0: "Benign", 1: "Malicious"}


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True Label",
           xlabel="Predicted Label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Confusion matrix saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Detailed model evaluation (Paper Table III)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_csv", nargs="+", default=None,
                        help="Test CSV(s) to evaluate on (default: all client test CSVs)")
    parser.add_argument("--mode", default=CLASSIFICATION_MODE,
                        choices=["binary", "multiclass"])
    parser.add_argument("--output_dir", default="logs", help="Output directory")
    args = parser.parse_args()

    num_classes = NUM_CLASSES_BINARY if args.mode == "binary" else NUM_CLASSES_MULTI
    label_col = "Label_ENCODED" if args.mode == "binary" else "Attack_ENCODED"
    name_map = BINARY_NAMES if args.mode == "binary" else ATTACK_NAMES_IDS2018

    if args.data_csv is None:
        args.data_csv = []
        for i in range(10):
            p = os.path.join(DATA_DIR, f"client_{i}_test.csv")
            if os.path.exists(p):
                args.data_csv.append(p)
        if not args.data_csv:
            print("No test CSVs found in", DATA_DIR)
            sys.exit(1)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=num_classes,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Round: {ckpt.get('round', '?')}  BMTA: {ckpt.get('bmta_pers', 'N/A')}")

    all_y_true = []
    all_y_pred = []

    for csv_path in args.data_csv:
        print(f"\nEvaluating on: {csv_path}")
        graph = csv_to_line_graph(csv_path, label_col=label_col)
        labels = graph.ndata["label"]

        with torch.no_grad():
            _, _, logits = model(graph, graph.ndata["feat"])
        preds = logits.argmax(dim=-1)

        all_y_true.append(labels.numpy())
        all_y_pred.append(preds.numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [name_map.get(l, f"Class {l}") for l in present_labels]

    # Overall metrics
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 (macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    # Per-class report (Paper Table III)
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS (Paper Table III)")
    print("=" * 70)
    report = classification_report(y_true, y_pred,
                                   labels=present_labels,
                                   target_names=target_names,
                                   zero_division=0,
                                   digits=4)
    print(report)

    # Confusion matrix
    os.makedirs(args.output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.mode}.png")
    plot_confusion_matrix(cm, target_names, cm_path)

    # Save report to text file
    report_path = os.path.join(args.output_dir, f"eval_report_{args.mode}.txt")
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Round: {ckpt.get('round', '?')}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Test files: {args.data_csv}\n\n")
        f.write(f"Accuracy:   {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision:  {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Recall:     {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"F1 (macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n\n")
        f.write(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
