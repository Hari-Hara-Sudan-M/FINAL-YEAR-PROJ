# =============================================================================
# STEP 1: Split Processed Dataset into Client Partitions
# Run this ONCE on the machine that has the processed CSV.
# It creates per-client CSV files you copy to each laptop.
#
# Usage:
#   python scripts/split_dataset.py --csv /path/to/CIC-IDS2018_processed.csv
# =============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    NUM_CLIENTS, DATA_DIR, LABEL_COL, ATTACK_COL, SEED,
    TRAIN_RATIO, MAX_NODES_PER_CLIENT, CLASSIFICATION_MODE
)


def dirichlet_split(df, label_col, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset across clients using Latent Dirichlet Allocation (LDA)
    strategy as described in paper Section V-A5:
    'partitioned into subgraphs based on the LDA Strategy'
    Each client gets a non-IID subset mimicking real distributed scenarios.
    """
    np.random.seed(seed)
    classes = df[label_col].unique()
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = df[df[label_col] == cls].index.tolist()
        np.random.shuffle(cls_indices)

        # Draw proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Assign indices proportionally
        splits = (proportions * len(cls_indices)).astype(int)
        # Fix rounding so total == len(cls_indices)
        splits[-1] = len(cls_indices) - splits[:-1].sum()

        start = 0
        for i, n in enumerate(splits):
            client_indices[i].extend(cls_indices[start:start + n])
            start += n

    return client_indices


def main():
    parser = argparse.ArgumentParser(description="Split dataset for federated clients")
    parser.add_argument("--csv", required=True, help="Path to processed CSV file")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha (lower=more non-IID). Default 0.5")
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS,
                        help=f"Number of clients (default from config: {NUM_CLIENTS})")
    parser.add_argument("--mode", default=CLASSIFICATION_MODE,
                        choices=["binary", "multiclass"],
                        help="Classification mode (uses ATTACK_COL for multiclass)")
    parser.add_argument("--out_dir", default=DATA_DIR, help="Output directory")
    args = parser.parse_args()

    split_col = ATTACK_COL if args.mode == "multiclass" else LABEL_COL

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print(f"FeCoGraph Dataset Splitter")
    print(f"  CSV       : {args.csv}")
    print(f"  Mode      : {args.mode}")
    print(f"  Split col : {split_col}")
    print(f"  Clients   : {args.num_clients}")
    print(f"  Alpha (a) : {args.alpha}  (lower = more non-IID)")
    print(f"  Output    : {args.out_dir}")
    print("=" * 70)

    print("\nLoading processed CSV...")
    df = pd.read_csv(args.csv)
    print(f"  Shape: {df.shape}")
    print(f"  Label distribution:\n{df[split_col].value_counts()}\n")

    # ── Cap size for memory ──────────────────────────────────────────────────
    total_cap = args.num_clients * MAX_NODES_PER_CLIENT
    if len(df) > total_cap:
        print(f"  Capping to {total_cap:,} rows ({MAX_NODES_PER_CLIENT:,} per client)...")
        df = df.sample(n=total_cap, random_state=SEED).reset_index(drop=True)

    # ── Dirichlet Split ──────────────────────────────────────────────────────
    print("Splitting with Dirichlet LDA strategy...")
    client_indices = dirichlet_split(
        df, split_col, args.num_clients,
        alpha=args.alpha, seed=SEED
    )

    # ── Save each client partition ───────────────────────────────────────────
    for client_id, indices in enumerate(client_indices):
        client_df = df.loc[indices].reset_index(drop=True)

        try:
            train_df, test_df = train_test_split(
                client_df,
                test_size=(1 - TRAIN_RATIO),
                stratify=client_df[split_col],
                random_state=SEED
            )
        except ValueError:
            train_df, test_df = train_test_split(
                client_df,
                test_size=(1 - TRAIN_RATIO),
                random_state=SEED
            )

        train_path = os.path.join(args.out_dir, f"client_{client_id}_train.csv")
        test_path  = os.path.join(args.out_dir, f"client_{client_id}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        label_dist = client_df[split_col].value_counts().to_dict()
        print(f"  Client {client_id}: {len(client_df):>7,} rows | "
              f"train={len(train_df):,}  test={len(test_df):,} | "
              f"labels={label_dist}")

    print(f"\nDone! Files saved to: {args.out_dir}")
    print("\nNext steps:")
    print(f"  → Copy client_0_train.csv + client_0_test.csv  →  Client Laptop 0")
    print(f"  → Copy client_1_train.csv + client_1_test.csv  →  Client Laptop 1")
    print(f"  (Repeat for more clients if NUM_CLIENTS > 2)")


if __name__ == "__main__":
    main()
