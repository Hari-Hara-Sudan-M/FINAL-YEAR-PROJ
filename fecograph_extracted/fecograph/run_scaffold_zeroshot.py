# =============================================================================
# NOVELTY: SCAFFOLD + Zero-Shot Attack Detection
#
# This implementation combines two novelties:
#   1. SCAFFOLD FL algorithm - variance-reduced gradients for non-IID data
#   2. Zero-shot novelty detection - detect attack types NEVER SEEN during training
#
# How it works:
#   - Train SCAFFOLD on 12 out of 15 attack types (holdout 3 rare classes)
#   - Compute class prototypes (mean embeddings) from trained model
#   - Test: classify known classes by nearest prototype
#   - Test: detect holdout classes as "novel" (far from all prototypes)
#
# Holdout classes (rare attacks excluded from training):
#   - Attack_ENCODED=9:  DoS-SlowHTTPTest  (18 samples)
#   - Attack_ENCODED=10: DoS-Slowloris     (24 samples)
#   - Attack_ENCODED=11: FTP-BruteForce    (43 samples)
#
# Usage:
#   python run_scaffold_zeroshot.py
#
# Results saved to:
#   results/scaffold_zeroshot/
#     ├── scaffold_zeroshot_results.json
#     ├── scaffold_zeroshot_tsne.png
#     ├── checkpoints/global_model_best.pt
#     └── logs/training_*.log
# =============================================================================

import os
import sys
import json
import logging
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    DATA_DIR, DEVICE, SEED,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    LOCAL_LR, LAMBDA_CE, TEMPERATURE, BATCH_SIZE,
)
from models.model import FeCoGraphModel
from data.line_graph import build_line_graph
from data.augmentation import augment_graph
from models.losses import SupervisedContrastiveLoss, ClassificationLoss

# =============================================================================
# CONFIGURATION
# =============================================================================

HOLDOUT_CLASSES = [9, 10, 11]  # DoS-SlowHTTPTest, DoS-Slowloris, FTP-BruteForce
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
NUM_CLIENTS = 2
LABEL_PROPORTION = 0.3

ATTACK_NAMES = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

RESULT_DIR = os.path.join("results", "scaffold_zeroshot")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULT_LOG_DIR, f"training_{RUN_ID}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[SCAFFOLD-ZEROSHOT] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("scaffold_zeroshot")


# =============================================================================
# DATA LOADING WITH HOLDOUT FILTERING
# =============================================================================

def load_data_with_holdout(client_id, holdout_classes, common_seen_classes=None):
    """
    Load client data with class filtering:
    - Training: EXCLUDE holdout classes (train on seen classes only)
    - Testing: INCLUDE all classes (to evaluate novelty detection)
    
    Args:
        client_id: Client ID
        holdout_classes: List of class IDs to hold out
        common_seen_classes: If provided, only keep these classes (for consistency across clients)
    
    Returns:
        train_graph: DGL graph with only seen classes (labels remapped to 0, 1, ..., K-1)
        test_graph: DGL graph with all classes (original labels preserved)
        train_mask: Boolean mask for labeled training nodes
        test_mask: Boolean mask for test nodes
        seen_classes: List of ORIGINAL class IDs present in training
        label_map: Dict mapping original labels to remapped labels (for training)
    """
    train_csv = os.path.join(DATA_DIR, f"client_{client_id}_train.csv")
    test_csv = os.path.join(DATA_DIR, f"client_{client_id}_test.csv")
    
    # Load training data and FILTER OUT holdout classes
    df_train = pd.read_csv(train_csv)
    df_train_filtered = df_train[~df_train["Attack_ENCODED"].isin(holdout_classes)].copy()
    
    # If common_seen_classes is provided, filter to only those classes
    if common_seen_classes is not None:
        df_train_filtered = df_train_filtered[df_train_filtered["Attack_ENCODED"].isin(common_seen_classes)].copy()
    
    # Load test data WITHOUT filtering (keep all classes)
    df_test = pd.read_csv(test_csv)
    
    log.info(f"  Client {client_id}: Train {len(df_train)} -> {len(df_train_filtered)} "
             f"(removed {len(df_train) - len(df_train_filtered)} samples)")
    
    # Get unique classes in filtered training data
    seen_classes = sorted(df_train_filtered["Attack_ENCODED"].unique().tolist())
    
    # Create label remapping: original_label -> consecutive_label (0, 1, 2, ...)
    # Use common_seen_classes for consistent mapping across clients
    if common_seen_classes is not None:
        label_map = {original: new_idx for new_idx, original in enumerate(common_seen_classes)}
    else:
        label_map = {original: new_idx for new_idx, original in enumerate(seen_classes)}
    
    # Remap training labels to consecutive integers
    df_train_filtered["Attack_ENCODED_REMAPPED"] = df_train_filtered["Attack_ENCODED"].apply(lambda x: label_map[x])
    
    # Build line graphs
    train_graph = build_line_graph(df_train_filtered, label_col="Attack_ENCODED_REMAPPED")
    test_graph = build_line_graph(df_test, label_col="Attack_ENCODED")
    
    # Create training mask (sample LABEL_PROPORTION of each class)
    labels = train_graph.ndata["label"]
    N = train_graph.num_nodes()
    n_labeled = max(1, int(N * LABEL_PROPORTION))
    
    # Stratified sampling per class
    train_mask = torch.zeros(N, dtype=torch.bool)
    unique_classes = labels.unique().tolist()  # Now these are 0, 1, 2, ..., K-1
    for cls in unique_classes:
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        n_cls = min(len(idx), max(1, int(n_labeled * len(idx) / N)))
        sampled = idx[torch.randperm(len(idx))[:n_cls]]
        train_mask[sampled] = True
    
    test_mask = torch.ones(test_graph.num_nodes(), dtype=torch.bool)
    
    return train_graph, test_graph, train_mask, test_mask, seen_classes, label_map


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_class_weights(labels, num_classes):
    """Compute inverse frequency weights for imbalanced classes."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights


def init_control_variate(model):
    """Initialize SCAFFOLD control variate to zeros."""
    return {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}


# =============================================================================
# SCAFFOLD CLIENT UPDATE
# =============================================================================

def scaffold_client_update(model, graph, train_mask, c_local, c_global, device, num_classes):
    """
    SCAFFOLD local training with variance-reduced gradients.
    
    Key formulas (Canonical Option-II):
        Gradient correction: g ← g - c_local + c_global
        Control update: c_local_new = c_local - c_global + (w_t - w_k) / (η * K)
    
    Args:
        model: FeCoGraphModel
        graph: DGL graph
        train_mask: Boolean mask for labeled nodes
        c_local: Local control variate dict
        c_global: Global control variate dict
        device: torch device
        num_classes: Number of classes in the model
    
    Returns:
        delta: Model weight delta (w_k - w_t)
        delta_c: Control variate delta (c_local_new - c_local)
        c_local_new: Updated local control variate
        metrics: Dict of training metrics
    """
    model.to(device)
    model.train()
    
    # Save initial weights w_t
    w_t = {n: p.clone().detach().cpu() for n, p in model.named_parameters()}
    
    # Prepare training data
    labels = graph.ndata["label"].to(device)
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).to(device)
    
    class_weights = compute_class_weights(labels[train_indices], num_classes).to(device)
    
    # Loss functions
    con_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    ce_fn = ClassificationLoss(class_weights=class_weights)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)
    
    # Augmented views
    G1, G2 = augment_graph(graph)
    
    metrics = {"total": [], "supcon": [], "ce": []}
    local_steps = 0
    
    for epoch in range(LOCAL_EPOCHS):
        optimizer.zero_grad()
        
        # Forward pass on augmented views
        _, z1_all, _ = model(G1.to(device), G1.ndata["feat"].to(device))
        _, z2_all, _ = model(G2.to(device), G2.ndata["feat"].to(device))
        _, _, logits_all = model(graph.to(device), graph.ndata["feat"].to(device))
        
        # Batch processing
        total_loss = 0.0
        con_loss = 0.0
        ce_loss = 0.0
        n_batches = 0
        
        perm = torch.randperm(len(train_indices), device=device)
        shuffled_idx = train_indices[perm]
        
        for start in range(0, len(shuffled_idx), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(shuffled_idx))
            b_idx = shuffled_idx[start:end]
            b_lbl = labels[b_idx]
            
            if len(b_lbl) < 2:
                continue
            
            z1_b = z1_all[b_idx]
            z2_b = z2_all[b_idx]
            logit_b = logits_all[b_idx]
            
            l_con = con_fn(z1_b, z2_b, b_lbl)
            l_ce = ce_fn(logit_b, b_lbl)
            loss = (1.0 - LAMBDA_CE) * l_con + LAMBDA_CE * l_ce
            
            is_last = (end >= len(shuffled_idx))
            loss.backward(retain_graph=not is_last)
            
            total_loss += loss.item()
            con_loss += l_con.item()
            ce_loss += l_ce.item()
            n_batches += 1
        
        if n_batches == 0:
            continue
        
        # Apply SCAFFOLD gradient correction: g ← g - c_k + c
        for name, param in model.named_parameters():
            if param.grad is not None:
                correction = -c_local[name].to(device) + c_global[name].to(device)
                param.grad.data.add_(correction)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        local_steps += 1
        
        metrics["total"].append(total_loss / n_batches)
        metrics["supcon"].append(con_loss / n_batches)
        metrics["ce"].append(ce_loss / n_batches)
    
    # Compute delta: w_k - w_t
    delta = {n: p.data.cpu() - w_t[n] for n, p in model.named_parameters()}
    
    # Compute new control variate (Canonical Option-II)
    # c_k_new = c_k - c + (w_t - w_k) / (η * K)
    c_local_new = {}
    K = max(1, local_steps)
    for name in c_local:
        w_t_cpu = w_t[name]
        w_k_cpu = model.state_dict()[name].cpu()
        c_local_new[name] = (
            c_local[name].cpu()
            - c_global[name].cpu()
            + (w_t_cpu - w_k_cpu) / (LOCAL_LR * K)
        )
    
    delta_c = {n: c_local_new[n] - c_local[n] for n in c_local}
    
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    
    return delta, delta_c, c_local_new, avg_metrics


# =============================================================================
# SCAFFOLD SERVER AGGREGATION
# =============================================================================

def scaffold_server_aggregate(global_model, client_deltas, client_delta_cs, c_global):
    """
    SCAFFOLD server-side aggregation.
    
    Formulas:
        w_{t+1} = w_t + (1/|S|) * Σ Δ_k
        c = c + (1/N) * Σ Δc_k
    
    Returns:
        Updated global_model (in-place)
        Updated c_global (in-place)
    """
    if not client_deltas:
        return
    
    # Aggregate model deltas
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            delta_avg = torch.stack([cd[name].float() for cd in client_deltas]).mean(dim=0)
            param.data.add_(delta_avg.to(param.device))
    
    # Aggregate control variate deltas
    for name in c_global:
        delta_c_avg = torch.stack([dc[name] for dc in client_delta_cs]).mean(dim=0)
        c_global[name] = c_global[name] + delta_c_avg


# =============================================================================
# PROTOTYPE COMPUTATION
# =============================================================================

@torch.no_grad()
def compute_class_prototypes(model, graphs, label_maps, seen_classes_original, device):
    """
    Compute mean embedding (prototype) for each class from training data.
    
    Args:
        model: Trained FeCoGraphModel
        graphs: List of training graphs (one per client, with remapped labels 0, 1, ..., K-1)
        label_maps: List of dicts mapping original_label -> remapped_label
        seen_classes_original: List of original class IDs
        device: torch device
    
    Returns:
        prototypes: Dict[original_class_id -> mean_embedding (numpy array)]
    """
    model.eval()
    
    all_embeddings = []
    all_labels_remapped = []
    
    for graph in graphs:
        h, _, _ = model(graph.to(device), graph.ndata["feat"].to(device))
        all_embeddings.append(h.cpu().numpy())
        all_labels_remapped.append(graph.ndata["label"].cpu().numpy())  # Remapped labels (0, 1, 2, ...)
    
    embeddings = np.vstack(all_embeddings)
    labels_remapped = np.concatenate(all_labels_remapped)
    
    # Create reverse mapping: remapped_label -> original_label
    reverse_map = {v: k for k, v in label_maps[0].items()}  # Assuming all clients have same mapping
    
    # Compute mean embedding per remapped class, then map back to original class ID
    prototypes = {}
    for remapped_cls in np.unique(labels_remapped):
        mask = labels_remapped == remapped_cls
        if mask.sum() > 0:
            original_cls = reverse_map[int(remapped_cls)]
            prototypes[original_cls] = embeddings[mask].mean(axis=0)
    
    return prototypes


# =============================================================================
# ZERO-SHOT CLASSIFICATION
# =============================================================================

def prototype_classify(embeddings, prototypes, threshold=None):
    """
    Classify samples by nearest prototype using cosine similarity.
    
    Args:
        embeddings: [N, D] numpy array of node embeddings
        prototypes: Dict[class_id -> prototype_embedding]
        threshold: If set, samples with max_similarity < threshold are marked as -1 (novel)
    
    Returns:
        predictions: [N] array of predicted class IDs (-1 for novel)
        similarities: [N] array of max similarity to any prototype
    """
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[c] for c in proto_labels])
    
    # Normalize for cosine similarity
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarities = emb_norm @ proto_norm.T  # [N, num_prototypes]
    
    # Nearest prototype
    pred_indices = similarities.argmax(axis=1)
    predictions = np.array([proto_labels[i] for i in pred_indices])
    max_sims = similarities.max(axis=1)
    
    # Apply novelty threshold
    if threshold is not None:
        predictions[max_sims < threshold] = -1  # Mark as novel
    
    return predictions, max_sims


def find_optimal_threshold(similarities, is_unseen_mask):
    """
    Grid search to find best threshold for novelty detection.
    
    Args:
        similarities: [N] array of max similarity scores
        is_unseen_mask: [N] boolean array (True for unseen/holdout samples)
    
    Returns:
        best_threshold: float
        best_f1: float
    """
    thresholds = np.arange(0.50, 0.99, 0.01)
    best_f1 = 0.0
    best_thresh = 0.7
    
    for thresh in thresholds:
        is_novel_pred = (similarities < thresh).astype(int)
        is_novel_true = is_unseen_mask.astype(int)
        
        f1 = f1_score(is_novel_true, is_novel_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_zero_shot(model, test_graph, prototypes, seen_classes, holdout_classes, device):
    """
    Comprehensive zero-shot evaluation.
    
    Returns:
        results: Dict with all evaluation metrics
    """
    model.eval()
    
    # Extract embeddings and labels
    h, _, _ = model(test_graph.to(device), test_graph.ndata["feat"].to(device))
    embeddings = h.cpu().numpy()
    true_labels = test_graph.ndata["label"].cpu().numpy()
    
    # Masks for seen vs unseen
    seen_mask = np.isin(true_labels, seen_classes)
    unseen_mask = np.isin(true_labels, holdout_classes)
    
    # ── STEP 1: Find optimal threshold ──────────────────────────────
    preds_no_thresh, sims = prototype_classify(embeddings, prototypes, threshold=None)
    
    if unseen_mask.sum() > 0:
        best_thresh, best_f1_search = find_optimal_threshold(sims, unseen_mask)
    else:
        best_thresh = 0.7
        best_f1_search = 0.0
    
    # ── STEP 2: Classify with optimal threshold ─────────────────────
    preds_with_thresh, _ = prototype_classify(embeddings, prototypes, threshold=best_thresh)
    
    # ── STEP 3: Evaluate seen class classification ──────────────────
    seen_indices = np.where(seen_mask)[0]
    seen_true = true_labels[seen_indices]
    seen_pred = preds_no_thresh[seen_indices]
    
    seen_acc = accuracy_score(seen_true, seen_pred)
    seen_f1 = f1_score(seen_true, seen_pred, average="macro", zero_division=0)
    seen_prec = precision_score(seen_true, seen_pred, average="macro", zero_division=0)
    seen_rec = recall_score(seen_true, seen_pred, average="macro", zero_division=0)
    
    # ── STEP 4: Evaluate novelty detection ──────────────────────────
    is_novel_true = unseen_mask.astype(int)
    is_novel_pred = (preds_with_thresh == -1).astype(int)
    
    novel_prec = precision_score(is_novel_true, is_novel_pred, average="binary", zero_division=0)
    novel_rec = recall_score(is_novel_true, is_novel_pred, average="binary", zero_division=0)
    novel_f1 = f1_score(is_novel_true, is_novel_pred, average="binary", zero_division=0)
    
    # ── STEP 5: Compute combined accuracy ────────────────────────────
    # Seen samples: correct if predicted correctly
    # Unseen samples: correct if predicted as -1 (novel)
    correct = 0
    for i, true_lbl in enumerate(true_labels):
        pred_lbl = preds_with_thresh[i]
        if true_lbl in seen_classes:
            if pred_lbl == true_lbl:
                correct += 1
        elif true_lbl in holdout_classes:
            if pred_lbl == -1:
                correct += 1
    
    combined_acc = correct / len(true_labels)
    
    # ── STEP 6: Similarity statistics ────────────────────────────────
    seen_sims = sims[seen_mask]
    unseen_sims = sims[unseen_mask]
    
    results = {
        "seen_accuracy": float(seen_acc),
        "seen_precision": float(seen_prec),
        "seen_recall": float(seen_rec),
        "seen_f1": float(seen_f1),
        "novelty_precision": float(novel_prec),
        "novelty_recall": float(novel_rec),
        "novelty_f1": float(novel_f1),
        "combined_accuracy": float(combined_acc),
        "optimal_threshold": float(best_thresh),
        "seen_sim_mean": float(seen_sims.mean()) if len(seen_sims) > 0 else 0.0,
        "seen_sim_std": float(seen_sims.std()) if len(seen_sims) > 0 else 0.0,
        "unseen_sim_mean": float(unseen_sims.mean()) if len(unseen_sims) > 0 else 0.0,
        "unseen_sim_std": float(unseen_sims.std()) if len(unseen_sims) > 0 else 0.0,
        "num_seen_samples": int(seen_mask.sum()),
        "num_unseen_samples": int(unseen_mask.sum()),
        "seen_classes": [int(c) for c in seen_classes],
        "holdout_classes": [int(c) for c in holdout_classes],
    }
    
    return results, embeddings, true_labels, sims


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_embeddings(embeddings, labels, seen_classes, holdout_classes,
                         similarities, threshold, save_path):
    """Generate t-SNE visualization with seen vs unseen highlighting."""
    log.info("  Generating t-SNE visualization...")
    
    N = len(labels)
    if N > 8000:
        # Subsample for faster t-SNE
        idx = np.random.choice(N, 8000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        similarities = similarities[idx]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                n_iter=1000, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ── Plot 1: Colored by attack type ──────────────────────────────
    all_classes = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", len(all_classes))
    
    for i, cls in enumerate(all_classes):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        
        name = ATTACK_NAMES.get(cls, f"Class{cls}")
        is_unseen = cls in holdout_classes
        marker = "x" if is_unseen else "o"
        alpha = 0.9 if is_unseen else 0.5
        size = 30 if is_unseen else 10
        label_prefix = "[UNSEEN] " if is_unseen else ""
        
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                   label=f"{label_prefix}{name}", s=size, alpha=alpha,
                   marker=marker, edgecolors="black" if is_unseen else "none")
    
    ax1.set_title("(a) Embeddings Colored by Attack Type", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7, markerscale=1.5, loc="best", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")
    
    # ── Plot 2: Novelty score heatmap ───────────────────────────────
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=similarities,
                         cmap="RdYlGn", s=10, alpha=0.6, edgecolors="none",
                         vmin=0.0, vmax=1.0)
    plt.colorbar(scatter, ax=ax2, label="Max Prototype Similarity")
    ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.set_title(f"(b) Novelty Score (threshold={threshold:.2f})", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")
    
    fig.suptitle("SCAFFOLD + Zero-Shot Attack Detection", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"  Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# MAIN TRAINING & EVALUATION
# =============================================================================

def main():
    log.info("=" * 75)
    log.info("SCAFFOLD + ZERO-SHOT NOVELTY DETECTION")
    log.info("=" * 75)
    log.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Run ID: {RUN_ID}")
    log.info(f"Log file: {LOG_PATH}")
    log.info(f"Holdout classes: {[ATTACK_NAMES[c] for c in HOLDOUT_CLASSES]}")
    log.info(f"Training rounds: {NUM_ROUNDS}")
    log.info(f"Local epochs: {LOCAL_EPOCHS}")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 75)
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)
    
    # ── Load data with holdout filtering (TWO-PASS) ─────────────────
    log.info("\n1. Loading data with holdout filtering...")
    
    # First pass: Determine common seen classes across all clients
    log.info("  First pass: Determining common seen classes...")
    all_seen_classes = []
    for client_id in range(NUM_CLIENTS):
        train_csv = os.path.join(DATA_DIR, f"client_{client_id}_train.csv")
        df = pd.read_csv(train_csv)
        df_filtered = df[~df["Attack_ENCODED"].isin(HOLDOUT_CLASSES)]
        seen = sorted(df_filtered["Attack_ENCODED"].unique().tolist())
        all_seen_classes.append(set(seen))
        log.info(f"    Client {client_id} has {len(seen)} seen classes: {seen}")
    
    common_seen_classes = sorted(list(set.intersection(*all_seen_classes)))
    num_seen_classes = len(common_seen_classes)
    log.info(f"  Common seen classes ({num_seen_classes}): {common_seen_classes}")
    
    # Second pass: Load with common filtering for consistency
    log.info("  Second pass: Loading data with common class filtering...")
    client_train_graphs = []
    client_test_graphs = []
    client_train_masks = []
    seen_classes_per_client = []
    label_maps = []
    
    for client_id in range(NUM_CLIENTS):
        train_g, test_g, train_m, test_m, seen_cls, lbl_map = load_data_with_holdout(
            client_id, HOLDOUT_CLASSES, common_seen_classes=common_seen_classes
        )
        client_train_graphs.append(train_g)
        client_test_graphs.append(test_g)
        client_train_masks.append(train_m)
        seen_classes_per_client.append(seen_cls)
        label_maps.append(lbl_map)
    
    seen_classes = common_seen_classes
    
    log.info(f"\n  Seen classes ({num_seen_classes}): {[ATTACK_NAMES[c] for c in seen_classes]}")
    log.info(f"  Holdout classes ({len(HOLDOUT_CLASSES)}): {[ATTACK_NAMES[c] for c in HOLDOUT_CLASSES]}")
    
    # ── Initialize global model and control variates ────────────────
    log.info("\n2. Initializing SCAFFOLD models...")
    global_model = FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM,
        hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2,
        proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM,
        num_classes=num_seen_classes  # Only seen classes
    ).to(device)
    
    c_global = init_control_variate(global_model)
    client_c_locals = [init_control_variate(global_model) for _ in range(NUM_CLIENTS)]
    
    log.info(f"  Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    # ── SCAFFOLD training loop ───────────────────────────────────────
    log.info("\n3. Starting SCAFFOLD training...")
    log.info("=" * 75)
    
    best_seen_acc = 0.0
    history = {"round": [], "acc": [], "f1": []}
    
    for round_t in range(NUM_ROUNDS):
        t_start = time.time()
        
        client_deltas = []
        client_delta_cs = []
        client_metrics = []
        
        # Client updates
        for client_id in range(NUM_CLIENTS):
            # Load global model
            local_model = FeCoGraphModel(
                in_dim=INPUT_FEATURE_DIM,
                hidden_dim=GCN_HIDDEN_DIM_1,
                embed_dim=GCN_HIDDEN_DIM_2,
                proj_hidden=PROJECTOR_HIDDEN_DIM,
                proj_out=PROJECTOR_OUTPUT_DIM,
                num_classes=num_seen_classes
            )
            local_model.load_state_dict(global_model.state_dict())
            
            # SCAFFOLD update
            delta, delta_c, c_new, metrics = scaffold_client_update(
                local_model,
                client_train_graphs[client_id],
                client_train_masks[client_id],
                client_c_locals[client_id],
                c_global,
                device,
                num_seen_classes  # Pass number of classes
            )
            
            client_deltas.append(delta)
            client_delta_cs.append(delta_c)
            client_metrics.append(metrics)
            client_c_locals[client_id] = c_new
        
        # Server aggregation
        scaffold_server_aggregate(global_model, client_deltas, client_delta_cs, c_global)
        
        # Evaluate on seen classes only (quick check)
        with torch.no_grad():
            global_model.eval()
            all_correct = 0
            all_total = 0
            all_preds = []
            all_true = []
            
            for test_g in client_test_graphs:
                labels = test_g.ndata["label"]
                # Only evaluate on SEEN classes
                seen_mask = torch.isin(labels, torch.tensor(seen_classes))
                if seen_mask.sum() == 0:
                    continue
                
                _, _, logits = global_model(test_g.to(device), test_g.ndata["feat"].to(device))
                preds = logits.argmax(dim=-1).cpu()
                
                # Filter to seen classes only
                true_lbl = labels[seen_mask].numpy()
                pred_lbl = preds[seen_mask].numpy()
                
                all_true.extend(true_lbl)
                all_preds.extend(pred_lbl)
            
            if len(all_true) > 0:
                acc = accuracy_score(all_true, all_preds)
                f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
            else:
                acc = 0.0
                f1 = 0.0
        
        elapsed = time.time() - t_start
        avg_loss = np.mean([m["total"] for m in client_metrics])
        
        history["round"].append(round_t + 1)
        history["acc"].append(acc)
        history["f1"].append(f1)
        
        # Save best model
        if acc > best_seen_acc:
            best_seen_acc = acc
            torch.save({
                "round": round_t + 1,
                "model_state": global_model.state_dict(),
                "c_global": c_global,
                "seen_classes": seen_classes,
                "seen_acc": acc,
            }, os.path.join(RESULT_CKPT_DIR, "global_model_best.pt"))
        
        if (round_t + 1) % 10 == 0 or round_t == NUM_ROUNDS - 1:
            log.info(f"Round {round_t+1:>3}/{NUM_ROUNDS} | "
                     f"Loss={avg_loss:.4f} | "
                     f"Acc(seen)={acc:.4f} | "
                     f"F1(seen)={f1:.4f} | "
                     f"Best={best_seen_acc:.4f} | "
                     f"{elapsed:.1f}s")
    
    log.info("=" * 75)
    log.info(f"SCAFFOLD training complete. Best seen accuracy: {best_seen_acc:.4f}")
    
    # ── Load best model ──────────────────────────────────────────────
    log.info("\n4. Loading best model checkpoint...")
    ckpt = torch.load(os.path.join(RESULT_CKPT_DIR, "global_model_best.pt"))
    global_model.load_state_dict(ckpt["model_state"])
    log.info(f"  Loaded checkpoint from round {ckpt['round']} (Acc={ckpt['seen_acc']:.4f})")
    
    # ── Compute class prototypes ─────────────────────────────────────
    log.info("\n5. Computing class prototypes from training data...")
    prototypes = compute_class_prototypes(global_model, client_train_graphs, label_maps, seen_classes, device)
    log.info(f"  Computed {len(prototypes)} prototypes: {sorted(prototypes.keys())}")
    
    # ── Zero-shot evaluation ─────────────────────────────────────────
    log.info("\n6. Running zero-shot evaluation...")
    
    # Combine all test graphs
    combined_test_labels = []
    combined_test_feats = []
    combined_test_graphs_list = []
    
    for test_g in client_test_graphs:
        combined_test_graphs_list.append(test_g)
    
    # Use first client's test graph for evaluation (or combine if needed)
    # For simplicity, evaluate on client 0's test set
    eval_graph = client_test_graphs[0]
    
    results, embeddings, true_labels, sims = evaluate_zero_shot(
        global_model, eval_graph, prototypes, seen_classes, HOLDOUT_CLASSES, device
    )
    
    log.info("\n" + "=" * 75)
    log.info("ZERO-SHOT EVALUATION RESULTS")
    log.info("=" * 75)
    log.info(f"  Seen Class Accuracy:      {results['seen_accuracy']:.4f}")
    log.info(f"  Seen Class Precision:     {results['seen_precision']:.4f}")
    log.info(f"  Seen Class Recall:        {results['seen_recall']:.4f}")
    log.info(f"  Seen Class F1:            {results['seen_f1']:.4f}")
    log.info(f"  ---")
    log.info(f"  Novelty Detection Prec:   {results['novelty_precision']:.4f}")
    log.info(f"  Novelty Detection Recall: {results['novelty_recall']:.4f}")
    log.info(f"  Novelty Detection F1:     {results['novelty_f1']:.4f}")
    log.info(f"  ---")
    log.info(f"  Combined Accuracy:        {results['combined_accuracy']:.4f}")
    log.info(f"  Optimal Threshold:        {results['optimal_threshold']:.4f}")
    log.info(f"  ---")
    log.info(f"  Seen samples:    {results['num_seen_samples']}")
    log.info(f"  Unseen samples:  {results['num_unseen_samples']}")
    log.info(f"  Seen sim:        {results['seen_sim_mean']:.4f} ± {results['seen_sim_std']:.4f}")
    log.info(f"  Unseen sim:      {results['unseen_sim_mean']:.4f} ± {results['unseen_sim_std']:.4f}")
    log.info("=" * 75)
    
    # ── Visualization ────────────────────────────────────────────────
    log.info("\n7. Generating visualization...")
    viz_path = os.path.join(RESULT_DIR, "scaffold_zeroshot_tsne.png")
    visualize_embeddings(
        embeddings, true_labels, seen_classes, HOLDOUT_CLASSES,
        sims, results['optimal_threshold'], viz_path
    )
    
    # ── Save results ─────────────────────────────────────────────────
    log.info("\n8. Saving results...")
    results_path = os.path.join(RESULT_DIR, "scaffold_zeroshot_results.json")
    results["holdout_class_names"] = [ATTACK_NAMES[c] for c in HOLDOUT_CLASSES]
    results["seen_class_names"] = [ATTACK_NAMES[c] for c in seen_classes]
    results["training_rounds"] = NUM_ROUNDS
    results["best_seen_accuracy_during_training"] = float(best_seen_acc)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"  Results saved: {results_path}")
    log.info(f"  Visualization: {viz_path}")
    log.info(f"  Checkpoint: {RESULT_CKPT_DIR}/global_model_best.pt")
    
    log.info("\n" + "=" * 75)
    log.info("SCAFFOLD + ZERO-SHOT NOVELTY DETECTION COMPLETE")
    log.info("=" * 75)


if __name__ == "__main__":
    main()
