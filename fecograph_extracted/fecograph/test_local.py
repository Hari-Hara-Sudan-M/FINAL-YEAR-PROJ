#!/usr/bin/env python3
# =============================================================================
# LOCAL TEST (no FL, no sockets)
# Run this on a single machine to verify the full pipeline works.
#
# Usage:
#   python test_local.py --csv /path/to/CIC-IDS2018_processed.csv
# =============================================================================

import os
import sys
import time
import argparse
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    LABEL_COL, ATTACK_COL, DEVICE, INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1,
    GCN_HIDDEN_DIM_2, PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI,
    LOCAL_EPOCHS, LOCAL_LR, LAMBDA_CE, TEMPERATURE, SEED,
    CLASSIFICATION_MODE
)
from data.line_graph import build_line_graph
from data.augmentation import augment_graph
from models.model import FeCoGraphModel, PersonalizedModel
from models.losses import JointLoss, ClassificationLoss
from fl.federated import evaluate_model, compute_class_weights
from utils.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--csv",       required=True, help="Path to processed CSV")
parser.add_argument("--rows",      type=int, default=5000, help="Rows to sample for quick test")
parser.add_argument("--epochs",    type=int, default=10)
parser.add_argument("--fl_scheme", default="ditto", choices=["fedavg","ditto"])
args = parser.parse_args()

set_seed(SEED)
device = torch.device(DEVICE)

print("=" * 70)
print("FeCoGraph Local Pipeline Test")
print("=" * 70)

# ── 1. Load small sample of processed CSV ────────────────────────────────────
print(f"\n[1] Loading {args.rows} rows from: {args.csv}")
df = pd.read_csv(args.csv, nrows=args.rows)
print(f"    Shape: {df.shape}")
label_col = ATTACK_COL if CLASSIFICATION_MODE == "multiclass" else LABEL_COL
num_classes = NUM_CLASSES_MULTI if CLASSIFICATION_MODE == "multiclass" else NUM_CLASSES_BINARY
print(f"    Mode: {CLASSIFICATION_MODE}  label_col={label_col}  num_classes={num_classes}")
print(f"    Label dist: {df[label_col].value_counts().to_dict()}")

# ── 2. Build Line Graph ───────────────────────────────────────────────────────
print(f"\n[2] Building Line Graph L(G)...")
t0 = time.time()
graph = build_line_graph(df, label_col=label_col)
print(f"    Nodes: {graph.num_nodes():,}  Edges: {graph.num_edges():,}  ({time.time()-t0:.1f}s)")

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
print(f"\n[3] Creating train/test masks (30/70 split)")
N = graph.num_nodes()
indices = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
n_train = int(N * 0.30)
train_mask = torch.zeros(N, dtype=torch.bool)
test_mask  = torch.zeros(N, dtype=torch.bool)
train_mask[indices[:n_train]] = True
test_mask[indices[n_train:]]  = True
print(f"    Train: {train_mask.sum().item():,}  Test: {test_mask.sum().item():,}")

# ── 4. Graph Augmentation ─────────────────────────────────────────────────────
print(f"\n[4] Adaptive Graph Augmentation G → G1, G2  (Eq. 1-5)")
G1, G2 = augment_graph(graph)
print(f"    G  : {graph.num_nodes():,} nodes, {graph.num_edges():,} edges")
print(f"    G1 : {G1.num_nodes():,} nodes, {G1.num_edges():,} edges (attribute augmented)")
print(f"    G2 : {G2.num_nodes():,} nodes, {G2.num_edges():,} edges (topology augmented)")

# ── 5. Model Init ─────────────────────────────────────────────────────────────
print(f"\n[5] Initializing FeCoGraph Model  (GCN encoder Eq. 6-7)")
model = FeCoGraphModel(
    in_dim=INPUT_FEATURE_DIM,
    hidden_dim=GCN_HIDDEN_DIM_1,
    embed_dim=GCN_HIDDEN_DIM_2,
    proj_hidden=PROJECTOR_HIDDEN_DIM,
    proj_out=PROJECTOR_OUTPUT_DIM,
    num_classes=num_classes
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {total_params:,}")

personal_model = PersonalizedModel(
    in_dim=INPUT_FEATURE_DIM,
    hidden_dim=GCN_HIDDEN_DIM_1,
    embed_dim=GCN_HIDDEN_DIM_2,
    proj_hidden=PROJECTOR_HIDDEN_DIM,
    proj_out=PROJECTOR_OUTPUT_DIM,
    num_classes=num_classes
).to(device)

# ── 6. Training Loop ──────────────────────────────────────────────────────────
print(f"\n[6] Training for {args.epochs} epochs (FL scheme: {args.fl_scheme})")
graph  = graph.to(device)
G1     = G1.to(device)
G2     = G2.to(device)
feat   = graph.ndata["feat"]
labels = graph.ndata["label"]
train_labels = labels[train_mask]

num_classes    = labels.max().item() + 1
class_weights  = compute_class_weights(train_labels, num_classes).to(device)
joint_loss_fn  = JointLoss(lambda_ce=LAMBDA_CE, temperature=TEMPERATURE,
                            class_weights=class_weights)
ce_fn_p        = ClassificationLoss(class_weights=class_weights)

optimizer_w = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)
optimizer_p = torch.optim.Adam(personal_model.parameters(), lr=LOCAL_LR)

for epoch in range(args.epochs):
    model.train()
    optimizer_w.zero_grad()

    # Contrastive branch: G1, G2
    _, z1, _ = model(G1, G1.ndata["feat"])
    _, z2, _ = model(G2, G2.ndata["feat"])

    # Classification branch: G
    _, _, logits = model(graph, feat)

    total, l_s, l_c = joint_loss_fn(
        z1[train_mask], z2[train_mask],
        logits[train_mask], train_labels
    )
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer_w.step()

    # Ditto personalized update
    if args.fl_scheme == "ditto":
        optimizer_p.zero_grad()
        logits_p = personal_model(graph, feat)
        l_ce_p   = ce_fn_p(logits_p[train_mask], train_labels)
        from config.config import MU
        reg = 0.0
        for (pp, pw) in zip(personal_model.model.parameters(), model.parameters()):
            reg += ((pp - pw.detach())**2).sum()
        loss_p = l_ce_p + (MU/2)*reg
        loss_p.backward()
        optimizer_p.step()

    if (epoch+1) % max(1, args.epochs//5) == 0:
        print(f"    Epoch {epoch+1:>3}/{args.epochs} | "
              f"Total={total.item():.4f} | SupCon={l_s.item():.4f} | CE={l_c.item():.4f}")

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
print(f"\n[7] Evaluation on test set")
test_metrics_global = evaluate_model(model, graph, test_mask, device, use_personalized=False)
print(f"    Global Model  → Acc={test_metrics_global['accuracy']:.4f} | "
      f"Prec={test_metrics_global['precision']:.4f} | "
      f"Rec={test_metrics_global['recall']:.4f} | "
      f"F1={test_metrics_global['f1']:.4f}")

if args.fl_scheme == "ditto":
    test_metrics_pers = evaluate_model(personal_model, graph, test_mask, device, use_personalized=True)
    print(f"    Personal Model→ Acc={test_metrics_pers['accuracy']:.4f} | "
          f"Prec={test_metrics_pers['precision']:.4f} | "
          f"Rec={test_metrics_pers['recall']:.4f} | "
          f"F1={test_metrics_pers['f1']:.4f}")

print("\n" + "=" * 70)
print("Local pipeline test PASSED. You are ready for FL training.")
print("=" * 70)
