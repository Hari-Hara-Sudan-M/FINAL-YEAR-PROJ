# =============================================================================
# t-SNE Visualization of Learned Embeddings (Paper Fig. 6)
#
# Loads a trained model checkpoint and plots t-SNE of encoder embeddings
# colored by class label. Reproduces the scatter plots in Fig. 6.
#
# Usage:
#   python scripts/plot_tsne.py --checkpoint checkpoints/global_model_best_pers.pt
#   python scripts/plot_tsne.py --checkpoint checkpoints/global_model_final.pt --mode multiclass
# =============================================================================

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI,
    CLASSIFICATION_MODE, DATA_DIR, SEED
)
from models.model import FeCoGraphModel
from data.line_graph import csv_to_line_graph

ATTACK_NAMES_IDS2018 = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

BINARY_NAMES = {0: "Benign", 1: "Malicious"}


def main():
    parser = argparse.ArgumentParser(description="t-SNE embedding visualization (Paper Fig. 6)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--data_csv", default=None,
                        help="CSV file to load (default: data/client_0_test.csv)")
    parser.add_argument("--mode", default=CLASSIFICATION_MODE,
                        choices=["binary", "multiclass"])
    parser.add_argument("--max_nodes", type=int, default=10000,
                        help="Max nodes to plot (for speed)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="Output image path (default: logs/tsne_<tag>.png)")
    args = parser.parse_args()

    if args.data_csv is None:
        args.data_csv = os.path.join(DATA_DIR, "client_0_test.csv")

    num_classes = NUM_CLASSES_BINARY if args.mode == "binary" else NUM_CLASSES_MULTI
    label_col = "Label_ENCODED" if args.mode == "binary" else "Attack_ENCODED"
    name_map = BINARY_NAMES if args.mode == "binary" else ATTACK_NAMES_IDS2018

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
    print(f"  Loaded from round {ckpt.get('round', '?')}")

    # Load graph
    print(f"Loading data: {args.data_csv} (label_col={label_col})")
    graph = csv_to_line_graph(args.data_csv, label_col=label_col)
    labels = graph.ndata["label"].numpy()

    # Extract embeddings
    with torch.no_grad():
        embeddings = model.encode(graph, graph.ndata["feat"]).numpy()
    print(f"  Embeddings shape: {embeddings.shape}")

    # Subsample if too large
    N = len(labels)
    if N > args.max_nodes:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(N, args.max_nodes, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        print(f"  Subsampled to {args.max_nodes} nodes")

    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                random_state=SEED, n_iter=1000, learning_rate="auto",
                init="pca")
    coords = tsne.fit_transform(embeddings)

    # Plot
    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 2))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        name = name_map.get(cls, f"Class {cls}")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=name, s=8, alpha=0.6, edgecolors="none")

    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(f"FeCoGraph Embeddings — {args.mode} (Round {ckpt.get('round', '?')})",
                 fontsize=14)
    ax.legend(loc="best", fontsize=8, markerscale=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output is None:
        tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
        os.makedirs("logs", exist_ok=True)
        args.output = os.path.join("logs", f"tsne_{tag}.png")

    fig.savefig(args.output, dpi=200)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
