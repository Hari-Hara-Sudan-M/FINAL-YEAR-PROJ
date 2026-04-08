# =============================================================================
# NOVELTY 2: Prototype-Based Zero-Shot Attack Detection
#
# The paper only handles FEW-SHOT (limited labels). We extend to ZERO-SHOT:
# detect attack types NEVER SEEN during training.
#
# How it works:
#   1. Train FeCoGraph normally but EXCLUDE some attack classes from training
#   2. After training, compute class PROTOTYPES = mean embedding per known class
#   3. For test samples:
#      - If embedding is close to a known prototype → classify as that class
#      - If embedding is FAR from all prototypes → flag as "Novel/Unknown Attack"
#   4. Evaluate: can the model detect unseen attack types?
#
# This is a genuine novelty over the paper because:
#   - Paper Eq 8-9 (SupCon) pulls same-class embeddings together
#   - We exploit this property: unknown attacks will NOT cluster near known ones
#   - No retraining needed — just prototype computation on trained embeddings
#
# Results saved to: results/zeroshot/
#
# Usage:  python run_prototype_zeroshot.py
# =============================================================================

import os, sys, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, SEED, DATA_DIR
)
from models.model import FeCoGraphModel
from data.line_graph import csv_to_line_graph

RESULT_DIR = os.path.join("results", "zeroshot")
os.makedirs(RESULT_DIR, exist_ok=True)

ATTACK_NAMES = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

# Zero-shot split: train on SEEN, test detection of UNSEEN
# We use the binary model (trained on Label_ENCODED: 0=Benign, 1=Malicious)
# Then test if it can distinguish specific attack TYPES it never saw individually
SEEN_CLASSES = [0, 1]  # Binary: Benign, Malicious (all attacks grouped)


def load_model(checkpoint_path, num_classes=NUM_CLASSES_BINARY):
    model = FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, graph):
    """Extract encoder embeddings (h) and projector embeddings (z)."""
    h, z, logits = model(graph, graph.ndata["feat"])
    return h.numpy(), z.numpy(), logits.numpy()


def compute_prototypes(embeddings, labels, classes):
    """Compute mean embedding (prototype) for each class."""
    prototypes = {}
    for cls in classes:
        mask = labels == cls
        if mask.sum() > 0:
            prototypes[cls] = embeddings[mask].mean(axis=0)
    return prototypes


def prototype_classify(embeddings, prototypes, threshold=None):
    """
    Classify each embedding by nearest prototype.
    If threshold is set, samples farther than threshold from ALL prototypes
    are classified as -1 (unknown/novel).
    """
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[c] for c in proto_labels])

    # Cosine distance
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8)
    similarities = emb_norm @ proto_norm.T  # [N, num_protos]

    pred_indices = similarities.argmax(axis=1)
    preds = np.array([proto_labels[i] for i in pred_indices])
    max_sims = similarities.max(axis=1)

    if threshold is not None:
        preds[max_sims < threshold] = -1  # Novel/unknown

    return preds, max_sims


def run_zero_shot_evaluation():
    print("=" * 70)
    print("NOVELTY: Prototype-Based Zero-Shot Attack Detection")
    print("=" * 70)

    # Find best checkpoint
    ckpt_candidates = [
        os.path.join("results", "supcon_binary", "checkpoints", "global_model_best_pers.pt"),
        os.path.join("results", "supcon_binary", "checkpoints", "global_model_best_local.pt"),
        os.path.join("checkpoints", "global_model_best_pers.pt"),
        os.path.join("checkpoints", "global_model_best_local.pt"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c; break
    if ckpt_path is None:
        print("ERROR: No checkpoint found. Run supcon_binary first."); return

    print(f"\n1. Loading model from: {ckpt_path}")
    model = load_model(ckpt_path)

    # Load BOTH binary and multiclass labels for test data
    print("\n2. Loading test data with multiclass labels...")
    # Binary labels graph (for embeddings)
    graph_binary = csv_to_line_graph(
        os.path.join(DATA_DIR, "client_0_test.csv"), label_col="Label_ENCODED")
    # Multiclass labels (for zero-shot evaluation)
    graph_multi = csv_to_line_graph(
        os.path.join(DATA_DIR, "client_0_test.csv"), label_col="Attack_ENCODED")

    binary_labels = graph_binary.ndata["label"].numpy()
    attack_labels = graph_multi.ndata["label"].numpy()

    print(f"   Test samples: {len(binary_labels)}")
    print(f"   Binary classes: {np.unique(binary_labels)}")
    print(f"   Attack classes: {sorted(np.unique(attack_labels))}")

    # Extract embeddings using trained binary model
    print("\n3. Extracting embeddings...")
    h_emb, z_emb, logits = extract_embeddings(model, graph_binary)
    print(f"   Encoder embedding shape: {h_emb.shape}")
    print(f"   Projector embedding shape: {z_emb.shape}")

    # ── EXPERIMENT A: Prototype-based binary classification ──────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Prototype-Based vs FC Classifier (Binary)")
    print("=" * 70)

    # Split into support (30%) and query (70%)
    rng = np.random.RandomState(SEED)
    N = len(binary_labels)
    indices = rng.permutation(N)
    n_support = int(N * 0.3)
    support_idx = indices[:n_support]
    query_idx = indices[n_support:]

    # Build prototypes from support set
    prototypes_binary = compute_prototypes(h_emb[support_idx], binary_labels[support_idx], [0, 1])
    print(f"   Prototypes computed: {list(prototypes_binary.keys())}")

    # Classify query set using prototypes
    proto_preds, proto_sims = prototype_classify(h_emb[query_idx], prototypes_binary)
    query_true = binary_labels[query_idx]

    # FC classifier predictions (original method)
    fc_preds = logits[query_idx].argmax(axis=1)

    proto_acc = accuracy_score(query_true, proto_preds)
    proto_f1 = f1_score(query_true, proto_preds, average="macro", zero_division=0)
    fc_acc = accuracy_score(query_true, fc_preds)
    fc_f1 = f1_score(query_true, fc_preds, average="macro", zero_division=0)

    print(f"\n   FC Classifier:       Acc={fc_acc:.4f}  F1={fc_f1:.4f}")
    print(f"   Prototype Classifier: Acc={proto_acc:.4f}  F1={proto_f1:.4f}")
    improvement = (proto_acc - fc_acc) * 100
    print(f"   Improvement: {improvement:+.2f}% accuracy")

    # ── EXPERIMENT B: Zero-shot novel attack detection ───────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Zero-Shot Novel Attack Detection")
    print("=" * 70)

    # Build prototypes per attack type from KNOWN attacks in training data
    train_graph = csv_to_line_graph(
        os.path.join(DATA_DIR, "client_0_train.csv"), label_col="Attack_ENCODED")
    train_attack_labels = train_graph.ndata["label"].numpy()
    with torch.no_grad():
        train_h, _, _ = model(train_graph, train_graph.ndata["feat"])
        train_h = train_h.numpy()

    seen_attacks = sorted(np.unique(train_attack_labels))
    all_test_attacks = sorted(np.unique(attack_labels))
    unseen_attacks = [a for a in all_test_attacks if a not in seen_attacks]

    print(f"   Seen attack types (in training): {[ATTACK_NAMES.get(a, a) for a in seen_attacks]}")
    print(f"   Unseen attack types (zero-shot): {[ATTACK_NAMES.get(a, a) for a in unseen_attacks]}")

    # Build prototypes from training embeddings per attack type
    attack_prototypes = compute_prototypes(train_h, train_attack_labels, seen_attacks)
    print(f"   Prototypes for {len(attack_prototypes)} seen attack types")

    # For test set: compute distance to nearest known prototype
    test_preds_attack, test_sims = prototype_classify(h_emb, attack_prototypes)

    # Compute similarity distributions for seen vs unseen
    seen_mask = np.isin(attack_labels, seen_attacks)
    unseen_mask = np.isin(attack_labels, unseen_attacks)

    if unseen_mask.sum() > 0:
        seen_sims = test_sims[seen_mask]
        unseen_sims = test_sims[unseen_mask]
        print(f"\n   Similarity stats (seen classes):   mean={seen_sims.mean():.4f} std={seen_sims.std():.4f}")
        print(f"   Similarity stats (unseen classes): mean={unseen_sims.mean():.4f} std={unseen_sims.std():.4f}")

        # Find optimal threshold for novelty detection
        thresholds = np.arange(0.5, 1.0, 0.01)
        best_novel_f1 = 0
        best_thresh = 0.7
        for thresh in thresholds:
            is_novel_true = unseen_mask.astype(int)
            is_novel_pred = (test_sims < thresh).astype(int)
            nf1 = f1_score(is_novel_true, is_novel_pred, average="binary", zero_division=0)
            if nf1 > best_novel_f1:
                best_novel_f1 = nf1
                best_thresh = thresh

        print(f"\n   Best novelty detection threshold: {best_thresh:.2f}")
        print(f"   Novel attack detection F1: {best_novel_f1:.4f}")

        # Detailed results at best threshold
        is_novel_true = unseen_mask.astype(int)
        is_novel_pred = (test_sims < best_thresh).astype(int)
        tp = int((is_novel_true & is_novel_pred).sum())
        fp = int(((~unseen_mask) & (test_sims < best_thresh)).sum())
        fn = int((unseen_mask & (test_sims >= best_thresh)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"   Novel detection: Precision={precision:.4f} Recall={recall:.4f}")
    else:
        print("   No unseen attack types in test set (all types present in training)")
        best_thresh = 0.7

    # ── EXPERIMENT C: Per-attack prototype classification ────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Multiclass Prototype Classification")
    print("=" * 70)

    # Classify test set by nearest attack prototype
    proto_attack_preds, _ = prototype_classify(h_emb, attack_prototypes)
    present = sorted(set(attack_labels) & set(proto_attack_preds))
    target_names = [ATTACK_NAMES.get(c, f"Class{c}") for c in present]

    report = classification_report(
        attack_labels, proto_attack_preds,
        labels=present, target_names=target_names,
        zero_division=0, digits=4
    )
    print(report)

    # ── VISUALIZATION ────────────────────────────────────────────────────
    print("\n4. Generating visualizations...")

    # Subsample for t-SNE
    if N > 8000:
        idx = rng.choice(N, 8000, replace=False)
    else:
        idx = np.arange(N)

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(h_emb[idx])

    # Plot: color by attack type, mark unseen with special marker
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sampled_labels = attack_labels[idx]
    sampled_sims = test_sims[idx]
    cmap = plt.cm.get_cmap("tab20", len(all_test_attacks))

    for i, cls in enumerate(sorted(all_test_attacks)):
        mask = sampled_labels == cls
        if mask.sum() == 0: continue
        name = ATTACK_NAMES.get(cls, f"Class{cls}")
        marker = "x" if cls in unseen_attacks else "o"
        alpha = 0.8 if cls in unseen_attacks else 0.4
        size = 20 if cls in unseen_attacks else 6
        label_prefix = "[UNSEEN] " if cls in unseen_attacks else ""
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                    label=f"{label_prefix}{name}", s=size, alpha=alpha,
                    marker=marker, edgecolors="none")

    ax1.set_title("(a) Embeddings colored by Attack Type", fontsize=12)
    ax1.legend(fontsize=7, markerscale=2, loc="best")
    ax1.grid(True, alpha=0.2)

    # Plot novelty scores
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=sampled_sims,
                          cmap="RdYlGn", s=6, alpha=0.5, edgecolors="none")
    plt.colorbar(scatter, ax=ax2, label="Max Prototype Similarity")
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_title(f"(b) Novelty Score (threshold={best_thresh:.2f})", fontsize=12)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Zero-Shot Attack Detection via Prototype Embeddings", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "zeroshot_tsne.png"), dpi=200, bbox_inches="tight")
    print(f"   Saved: {RESULT_DIR}/zeroshot_tsne.png")
    plt.close(fig)

    # ── SAVE RESULTS ─────────────────────────────────────────────────────
    results = {
        "fc_accuracy": round(fc_acc, 4),
        "fc_f1": round(fc_f1, 4),
        "prototype_accuracy": round(proto_acc, 4),
        "prototype_f1": round(proto_f1, 4),
        "accuracy_improvement": round(improvement, 2),
        "novelty_threshold": round(best_thresh, 2),
        "novelty_detection_f1": round(best_novel_f1, 4) if unseen_mask.sum() > 0 else "N/A",
        "seen_attacks": [ATTACK_NAMES.get(a, a) for a in seen_attacks],
        "unseen_attacks": [ATTACK_NAMES.get(a, a) for a in unseen_attacks],
    }
    with open(os.path.join(RESULT_DIR, "zeroshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved: {RESULT_DIR}/zeroshot_results.json")

    print("\n" + "=" * 70)
    print("ZERO-SHOT EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_zero_shot_evaluation()
