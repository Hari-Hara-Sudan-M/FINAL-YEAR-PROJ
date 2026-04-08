# =============================================================================
# FeCoGraph Global Configuration
# =============================================================================

import os

# ─────────────────────────────────────────────
# FEDERATED LEARNING SETTINGS
# ─────────────────────────────────────────────
NUM_CLIENTS = 2                  # <── Change this before running (matches number of client laptops)
NUM_ROUNDS = 100                 # Communication rounds T
LOCAL_EPOCHS = 5                 # Local epochs r
PERSONALIZED_EPOCHS = 5          # Personalized epochs s
LOCAL_LR = 0.001                 # Local learning rate η_l
PERSONALIZED_LR = 0.001          # Personalized learning rate η_p
MU = 0.1                         # Interpolation factor µ (Ditto regularization)
FL_SCHEME = "ditto"              # "fedavg" or "ditto"
CLIENT_FRACTION = 1.0            # Fraction of clients selected per round

# ─────────────────────────────────────────────
# NETWORK SETTINGS (Server ↔ Clients)
# ─────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"          # Server listens on all interfaces
SERVER_PORT = 9999
# On client machines, set SERVER_HOST to the server's IP address
# e.g., SERVER_HOST = "192.168.1.100"

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE (Paper Section IV-B3)
# ─────────────────────────────────────────────
INPUT_FEATURE_DIM = 40           # Number of flow features after preprocessing
GCN_HIDDEN_DIM_1 = 64            # First GCN layer hidden units
GCN_HIDDEN_DIM_2 = 32            # Second GCN layer hidden units (final embedding h)
PROJECTOR_HIDDEN_DIM = 256       # Projector MLP hidden dim (tuned: 256 in paper)
PROJECTOR_OUTPUT_DIM = 128       # Projector output dimension
NUM_CLASSES_BINARY = 2           # Binary: Benign / Malicious
NUM_CLASSES_MULTI = 15           # Multiclass: CIC-IDS2018 has labels 0-14 (15 values)
CLASSIFICATION_MODE = "multiclass"   # "binary" or "multiclass"

# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────
MAX_NODES_PER_CLIENT = 50000     # Cap graph size per client for memory
TRAIN_RATIO = 0.30               # 30% train / 70% test split in split_dataset.py
TEST_RATIO = 0.70

# ─────────────────────────────────────────────
# LABEL PROPORTION (Paper Fig.7 experiment)
# ─────────────────────────────────────────────
# Controls what fraction of TRAINING nodes are treated as labeled.
# The GCN still sees the full graph for message passing — only the
# supervised loss (contrastive + CE) is computed on labeled nodes only.
#
# Paper Fig.7 tests: 0.1, 0.3, 0.5, 0.7
# Default: 0.3 (matches your BMTA=0.9817 result)
#
# To run label proportion experiments, change this value and re-run:
#   LABEL_PROPORTION = 0.1  ->  10% labeled
#   LABEL_PROPORTION = 0.3  ->  30% labeled (default)
#   LABEL_PROPORTION = 0.5  ->  50% labeled
#   LABEL_PROPORTION = 0.7  ->  70% labeled
LABEL_PROPORTION = 0.3

# ─────────────────────────────────────────────
# GRAPH AUGMENTATION (Equations 1-5 in paper)
# ─────────────────────────────────────────────
PE = 0.3          # Edge drop scaling hyper-parameter p_e
PF = 0.3          # Feature mask scaling hyper-parameter p_f
P_TAU = 0.7       # Cut-off probability p_τ

# ─────────────────────────────────────────────
# CONTRASTIVE LEARNING (Equations 8-12 in paper)
# ─────────────────────────────────────────────
CONTRASTIVE_MODE = "supcon"  # "supcon" = label-aware (Eq.8-9), "sslcon" = self-supervised (Table VI ablation)
TEMPERATURE = 0.3        # τ temperature parameter
LAMBDA_CE = 0.07         # λ_ce weighting cross-entropy vs contrastive loss
                         # (best on IDS2018 per Table IV: 0.07)
BATCH_SIZE = 512         # Mini-batch size for contrastive loss computation

# ─────────────────────────────────────────────
# DATASET & DATA PATHS
# ─────────────────────────────────────────────
DATASET_NAME = "CIC-IDS2018"
PROCESSED_CSV = "CIC-IDS2018_processed.csv"   # Output of your preprocessing
DATA_DIR = "./data"                             # Where split client data is stored
LABEL_COL = "Label_ENCODED"                    # Binary label column after preprocessing
ATTACK_COL = "Attack_ENCODED"                  # Multiclass label column

# ─────────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────────
SEED = 42
DEVICE = "cpu"           # "cuda" if GPU available, else "cpu"

# ─────────────────────────────────────────────
# LOGGING & CHECKPOINTS
# ─────────────────────────────────────────────
LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
LOG_EVERY = 10           # Log every N rounds
