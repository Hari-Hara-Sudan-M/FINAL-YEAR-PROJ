# FeCoGraph — Federated Graph Contrastive Learning for Network Intrusion Detection

**Paper:** *FeCoGraph: Label-Aware Federated Graph Contrastive Learning for Few-Shot Network Intrusion Detection* (IEEE TIFS 2025)

**Project:** Full Python implementation of the FeCoGraph framework on 3 laptops using federated learning (FL) over LAN.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [How to Run](#5-how-to-run)
6. [Experiments Implemented](#6-experiments-implemented)
7. [Configuration Reference](#7-configuration-reference)
8. [What Is Implemented (Paper Coverage)](#8-what-is-implemented-paper-coverage)
9. [What Is NOT Yet Implemented](#9-what-is-not-yet-implemented)
10. [Results Achieved](#10-results-achieved)
11. [Known Issues and Fixes](#11-known-issues-and-fixes)

---

## 1. Project Overview

FeCoGraph is a **Federated Graph Contrastive Learning** framework for **Network Intrusion Detection System (NIDS)**. It solves two key challenges:

- **Few-shot problem**: Real network environments have very few labeled attack samples
- **Privacy problem**: Raw network traffic data cannot be shared between organizations

### How it works (high level)

```
Each client has local network flow data
           ↓
Flows are converted to a Line Graph L(G)
  (flows become nodes, shared-IP flows become edges)
           ↓
Two augmented views G1 (attribute-masked) and G2 (topology-dropped) are generated
           ↓
A 2-layer GCN Encoder learns embeddings h for each flow
           ↓
CONTRASTIVE TASK:  G1, G2 → Encoder → Projector → z → SupCon Loss  ┐
CLASSIFICATION TASK: G  → Encoder → Classifier → y → CE Loss        ├─ Joint Loss F_k
           ↓
Only global model weights Δ_k = w_k - w_t are sent to server (no raw data)
           ↓
Server aggregates: FedAvg or Ditto (personalized FL)
           ↓
Each client keeps a private personalized model θ_k (never shared)
```

---

## 2. Project Structure

```
fecograph/
│
├── server.py                   # FL Server — aggregates model updates (Algorithm 1, server side)
├── client.py                   # FL Client — local training (Algorithm 1, client side)
│
├── config/
│   └── config.py               # All hyperparameters and settings (edit this to change experiments)
│
├── data/
│   ├── line_graph.py           # Line graph L(G) construction (Paper Section IV-B1)
│   └── augmentation.py         # Adaptive augmentation G→G1,G2 (Paper Equations 1-5)
│
├── models/
│   ├── model.py                # GCN Encoder + Projector + Classifier (Paper Equations 6-7)
│   └── losses.py               # SupCon loss, SSLCon loss, CE loss, Joint loss (Eq 8-12)
│
├── fl/
│   ├── federated.py            # Local training, FedAvg/Ditto, evaluation (Algorithm 1)
│   └── comm.py                 # Socket communication between server and clients
│
├── scripts/
│   └── split_dataset.py        # Splits processed CSV into per-client train/test using LDA
│
├── utils/
│   └── utils.py                # Helper functions
│
├── data/                       # (created after split_dataset.py runs)
│   ├── client_0_train.csv
│   ├── client_0_test.csv
│   ├── client_1_train.csv
│   └── client_1_test.csv
│
├── checkpoints/                # Saved model weights
│   ├── global_model_best_local.pt
│   ├── global_model_best_pers.pt
│   └── global_model_final.pt
│
├── logs/
│   ├── server.log
│   ├── client_0.log
│   ├── client_1.log
│   └── training_history.csv    # Per-round metrics for all experiments
│
└── requirements.txt
```

---

## 3. Environment Setup

### Requirements
- Python 3.10+
- PyTorch 2.6.0
- DGL 1.1.3
- 3 laptops connected on the same LAN (or same machine for testing)

### Install dependencies

```bash
# Create and activate environment
conda create -n feco_env python=3.10
conda activate feco_env

# Install PyTorch (CPU)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Install DGL (CPU)
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html

# Install remaining packages
pip install -r requirements.txt
```

### requirements.txt contents

```
numpy
pandas
scikit-learn
torch==2.6.0
dgl==1.1.3
tqdm
```

> Security note: `dgl` currently has a published advisory with no patched release available upstream. This project now hard-disables DGL RPC/distributed imports at runtime, requires authenticated FL socket payloads via `FECOGRAPH_COMM_SECRET`, and defaults to loopback-only network binding unless `FECOGRAPH_ALLOW_REMOTE_NETWORK=1` is explicitly set.

---

## 4. Dataset Preparation

### Supported Datasets (Paper Table III)

| Dataset | Type | Classes (Binary) | Classes (Multiclass) |
|---|---|---|---|
| CIC-IDS2018 | ✅ Implemented | Benign / Attack | 7 attack types |
| NF-BoT-IoT-v2 | ❌ Not yet implemented | Benign / Attack | 4 attack types |
| NF-ToN-IoT-v2 | ❌ Not yet implemented | Benign / Attack | 5 attack types |

### Step 1 — Preprocess raw CSV

Your raw dataset CSV must have these columns (or equivalents):
- `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` — source/destination IP
- `L4_SRC_PORT`, `L4_DST_PORT` — ports
- Flow feature columns (packet counts, byte counts, durations, etc.)
- `Label` — binary label string ("Benign" / "Attack")
- `Attack` — attack type string (for multiclass)

After preprocessing, your CSV must have:
- `Label_ENCODED` — 0=Benign, 1=Attack (binary label)
- `Attack_ENCODED` — integer encoded attack type (multiclass label)

The dataset used: `CIC-IDS2018_processed.csv` (377,873 rows, 49 features)

### Step 2 — Split dataset into per-client files

```bash
# Split into 2 clients using LDA Dirichlet distribution (non-IID)
python scripts/split_dataset.py \
    --csv CIC-IDS2018_processed.csv \
    --num_clients 2 \
    --alpha 0.5 \
    --output_dir ./data
```

This creates:
- `data/client_0_train.csv` — 25,353 training flows for Client 0
- `data/client_0_test.csv`  — 59,159 test flows for Client 0
- `data/client_1_train.csv` — 4,646 training flows for Client 1
- `data/client_1_test.csv`  — 10,842 test flows for Client 1

The LDA split ensures **non-IID** distribution (different attack distributions per client), which is realistic for federated settings.

---

## 5. How to Run

### Option A — All on one machine (testing)

Open 3 terminals in the project directory:

```bash
# Required for authenticated model/metric transport (same value in all terminals)
export FECOGRAPH_COMM_SECRET="replace-with-strong-shared-secret"

# Terminal 1: Start server
python server.py

# Terminal 2: Start Client 0
python client.py --client_id 0 --server_ip 127.0.0.1

# Terminal 3: Start Client 1
python client.py --client_id 1 --server_ip 127.0.0.1
```

### Option B — Across 3 laptops (real federated)

```bash
# Laptop 1 (Server machine):
# Required on all 3 laptops (same value everywhere):
export FECOGRAPH_COMM_SECRET="replace-with-strong-shared-secret"

# Explicit opt-in to remote network use (default runtime mode is loopback-only):
export FECOGRAPH_ALLOW_REMOTE_NETWORK=1
python server.py

# Laptop 2 (Client 0) — replace IP with server laptop's IP:
export FECOGRAPH_COMM_SECRET="replace-with-strong-shared-secret"
export FECOGRAPH_ALLOW_REMOTE_NETWORK=1
python client.py --client_id 0 --server_ip 192.168.x.x

# Laptop 3 (Client 1) — replace IP with server laptop's IP:
export FECOGRAPH_COMM_SECRET="replace-with-strong-shared-secret"
export FECOGRAPH_ALLOW_REMOTE_NETWORK=1
python client.py --client_id 1 --server_ip 192.168.x.x
```

**Important:** Each client machine must have its own copy of `data/client_<id>_train.csv` and `data/client_<id>_test.csv`. Raw data is never transferred over the network.

### What happens during training

Each FL round:
1. Server sends global model weights `w_t` to all clients
2. Each client runs local training:
   - **Contrastive Task**: G1, G2 → Encoder → Projector → SupCon Loss (Eq.17)
   - **Classification Task**: G → Encoder → Classifier → CE + Ditto regularization (Eq.18)
3. Each client sends only `Δ_k = w_k - w_t` back to server (not raw data, not full model)
4. Server aggregates: `w^{t+1} = w^t + (1/|S|) Σ Δ_k` (FedAvg, Eq.14)
5. Personalized model `θ_k` stays private on each client (never sent to server)

### Training output (per round)

```
[CLIENT-0] Round 50 done | Train loss=6.41 | SupCon=6.89 | CE=0.049 | TestAcc(local)=0.9986 | TestAcc(pers)=0.9989 | F1=0.9456 | Time=113.8s
[SERVER]   Round 50/100 | SupCon=6.59 | CE=0.083 | Acc_local=0.9809 | F1_local=0.9537 | Acc_pers=0.9810 | F1_pers=0.9493 | BMTA_pers=0.9810
```

---

## 6. Experiments Implemented

### Experiment 1 — Binary Classification, CIC-IDS2018 (DONE ✅)

**Config settings:**
```python
CLASSIFICATION_MODE = "binary"
FL_SCHEME = "ditto"
NUM_ROUNDS = 100
LABEL_PROPORTION = 0.3
CONTRASTIVE_MODE = "supcon"
```

**Result achieved:**
- BMTA (Best Mean Test Accuracy) = **0.9817**
- Final F1 (personalized model) = **0.9630**

---

### Experiment 2 — Label Proportion Experiment (Paper Fig.7)

Tests FeCoGraph with 10%, 30%, 50%, 70% labeled training data.

**How to run:** Change one line in `config.py` and re-run FL each time:

```python
# In config/config.py — change LABEL_PROPORTION:
LABEL_PROPORTION = 0.1   # Run 1: 10% labeled
LABEL_PROPORTION = 0.3   # Run 2: 30% labeled (already done, BMTA=0.9817)
LABEL_PROPORTION = 0.5   # Run 3: 50% labeled
LABEL_PROPORTION = 0.7   # Run 4: 70% labeled
```

Run FL for each value and record BMTA. Paper shows FeCoGraph maintains high accuracy even with only 10% labeled data.

**Status:** Config and client code implemented ✅. Need to run the 3 remaining proportions.

---

### Experiment 3 — FedAvg vs Ditto Comparison (Paper Fig.8/9)

Compares two FL aggregation strategies:
- **FedAvg**: Standard federated averaging, one shared global model
- **Ditto**: Personalized FL, each client has a private model θ_k

**How to run:** Change `FL_SCHEME` in config and re-run FL:

```python
# In config/config.py:
FL_SCHEME = "fedavg"    # Run 1: standard FedAvg
FL_SCHEME = "ditto"     # Run 2: personalized Ditto (already done, BMTA=0.9817)
```

Both are fully implemented in `fl/federated.py`. Just re-run with FedAvg to get comparison data.

**Status:** Both FedAvg and Ditto implemented ✅. Need to run FedAvg to get comparison numbers.

---

### Experiment 4 — Ablation: SupCon vs SSLCon (Paper Table VI)

Proves that using class labels in contrastive learning (SupCon) is better than self-supervised contrastive learning without labels (SSLCon / SimCLR-style).

**How to run:** Change `CONTRASTIVE_MODE` in config and re-run FL:

```python
# In config/config.py:
CONTRASTIVE_MODE = "supcon"   # Run 1: label-aware supervised (already done, BMTA=0.9817)
CONTRASTIVE_MODE = "sslcon"   # Run 2: self-supervised (SimCLR-style, no labels used)
```

- **SupCon**: Positive pairs = flows of the same attack class. Uses labels. (Eq.8-9)
- **SSLCon**: Positive pairs = only the two augmented views of the same flow. No labels used.

Both are implemented in `models/losses.py`. `JointLoss` automatically picks the right one from config.

**Status:** Both implemented ✅. Need to run SSLCon to get ablation numbers.

---

## 7. Configuration Reference

All settings are in `config/config.py`. Key settings to change for different experiments:

```python
# ── Federated Learning ──────────────────────────────────────
NUM_CLIENTS = 2           # Number of clients (match your setup)
NUM_ROUNDS = 100          # Training rounds T
LOCAL_EPOCHS = 5          # Local epochs r (Algorithm 1)
PERSONALIZED_EPOCHS = 5   # Personalized epochs s (Ditto)
LOCAL_LR = 0.001          # Learning rate
MU = 0.1                  # Ditto regularization strength
FL_SCHEME = "ditto"       # "fedavg" or "ditto"

# ── Model ───────────────────────────────────────────────────
CLASSIFICATION_MODE = "binary"   # "binary" or "multiclass"
NUM_CLASSES_BINARY = 2           # 2 for binary
NUM_CLASSES_MULTI = 7            # Adjust per dataset

# ── Experiments ─────────────────────────────────────────────
LABEL_PROPORTION = 0.3    # Fig.7: try 0.1, 0.3, 0.5, 0.7
CONTRASTIVE_MODE = "supcon"  # Table VI: "supcon" or "sslcon"

# ── Loss hyperparameters (Paper Table IV) ───────────────────
LAMBDA_CE = 0.07          # Best on IDS2018 (Table IV)
TEMPERATURE = 0.3         # τ temperature

# ── Dataset ─────────────────────────────────────────────────
LABEL_COL = "Label_ENCODED"    # Binary label column
ATTACK_COL = "Attack_ENCODED"  # Multiclass label column
```

---

## 8. What Is Implemented (Paper Coverage)

### Core Algorithm

| Paper Component | Equation / Section | File | Status |
|---|---|---|---|
| Line Graph L(G) construction | Section IV-B1 | `data/line_graph.py` | ✅ Done |
| Topology augmentation (edge drop) | Eq.1-2 | `data/augmentation.py` | ✅ Done |
| Attribute augmentation (feature mask) | Eq.3-5 | `data/augmentation.py` | ✅ Done |
| GCN Encoder (2-layer) | Eq.6-7 | `models/model.py` | ✅ Done |
| Projector MLP (h → z) | Section IV-B3 | `models/model.py` | ✅ Done |
| Classifier head | Eq.10 | `models/model.py` | ✅ Done |
| Supervised Contrastive Loss (SupCon) | Eq.8-9 | `models/losses.py` | ✅ Done |
| Self-Supervised Contrastive Loss (SSLCon) | Table VI ablation | `models/losses.py` | ✅ Done |
| Cross-Entropy Loss | Eq.10-11 | `models/losses.py` | ✅ Done |
| Joint Loss | Eq.12 | `models/losses.py` | ✅ Done |
| Contrastive Task (global model update) | Eq.17 | `fl/federated.py` | ✅ Done |
| Classification Task (personalized model) | Eq.18 | `fl/federated.py` | ✅ Done |
| FedAvg aggregation | Eq.13-14 | `fl/federated.py` | ✅ Done |
| Ditto personalized FL | Eq.15-16 | `fl/federated.py` | ✅ Done |
| Full Algorithm 1 | Algorithm 1 | `server.py` + `client.py` | ✅ Done |
| Batched contrastive loss (OOM fix) | Section V-A5 | `models/losses.py` | ✅ Done |
| LDA Dirichlet non-IID data split | Section V-A | `scripts/split_dataset.py` | ✅ Done |
| Weight delta transfer (privacy) | Algorithm 1 line 11 | `client.py` | ✅ Done |
| Personalized model stays private | Algorithm 1 | `client.py` | ✅ Done |

### Experiments

| Paper Experiment | Status |
|---|---|
| Binary classification — CIC-IDS2018 | ✅ Done (BMTA=0.9817, F1=0.9630) |
| Label proportion (10/30/50/70%) — Fig.7 | ✅ Code ready, 30% run done, others pending |
| FedAvg vs Ditto comparison — Fig.8/9 | ✅ Code ready, Ditto done, FedAvg run pending |
| Ablation SupCon vs SSLCon — Table VI | ✅ Code ready, SupCon done, SSLCon run pending |
| Dual metrics (local + personalized model) | ✅ Done |
| Checkpoint saving (best local + best pers) | ✅ Done |
| Training history CSV export | ✅ Done |

---

## 9. What Is NOT Yet Implemented

These parts of the paper are not yet implemented and are needed to fully reproduce the paper results.

### 9.1 Multiclass Classification

**Paper:** Table III reports results for both binary AND multiclass classification on all datasets.

**What needs to change:**
- `config.py`: Set `CLASSIFICATION_MODE = "multiclass"` and `NUM_CLASSES_MULTI = 7` (for CIC-IDS2018)
- `data/line_graph.py`: Use `ATTACK_COL = "Attack_ENCODED"` instead of `LABEL_COL` when building the graph
- `models/model.py`: The classifier output dimension must match `NUM_CLASSES_MULTI`
- `models/losses.py`: Class weights need to handle 7 classes instead of 2
- `scripts/split_dataset.py`: LDA split must use attack labels for stratification

Currently the code has `NUM_CLASSES_MULTI = 7` defined but nothing actually uses it — `CLASSIFICATION_MODE = "binary"` is hardcoded in the graph loader.

**How to implement:** In `data/line_graph.py`, change the label assignment to read from `ATTACK_COL` when `CLASSIFICATION_MODE == "multiclass"`. Everything else (model, FL, loss) works automatically since num_classes flows from config.

---

### 9.2 NF-BoT-IoT-v2 Dataset

**Paper:** Table III reports FeCoGraph on 3 datasets. We only have CIC-IDS2018.

**What needs to be done:**
- Download NF-BoT-IoT-v2 from the NF-V2 repository
- Preprocess to match the expected CSV format (same columns as CIC-IDS2018)
- Encode labels: `Label_ENCODED` (0=Benign, 1=Attack) and `Attack_ENCODED` (0-3 for 4 attack types)
- Run `split_dataset.py` with the new CSV
- Run full FL training and record BMTA

---

### 9.3 NF-ToN-IoT-v2 Dataset

**Paper:** Same as above — third dataset in Table III.

**What needs to be done:**
- Download NF-ToN-IoT-v2
- Preprocess to match expected CSV format
- Encode labels (0=Benign, 1=Attack; 0-4 for 5 attack types)
- Run split and FL training

---

### 9.4 t-SNE Embedding Visualization (Paper Fig.6)

**Paper:** Fig.6 shows t-SNE plots of node embeddings before and after FeCoGraph training to visualize how the contrastive learning separates attack classes in embedding space.

**What needs to be done:**
- After training, extract embeddings from the GCN encoder for all test nodes
- Run t-SNE dimensionality reduction (sklearn.manifold.TSNE)
- Plot with color per class (binary or multiclass)
- Show before training (random init) vs after training

**Approximate code needed:**
```python
# Load trained model
model = FeCoGraphModel(...)
model.load_state_dict(torch.load("checkpoints/global_model_final.pt"))
model.eval()

# Extract embeddings
with torch.no_grad():
    h = model.encoder(graph, graph.ndata["feat"])  # [N, 32]

# t-SNE
from sklearn.manifold import TSNE
emb_2d = TSNE(n_components=2).fit_transform(h.numpy())

# Plot
import matplotlib.pyplot as plt
plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap="tab10")
plt.savefig("tsne.png")
```

---

### 9.5 FL Convergence Plot (Paper Fig.8/9)

**Paper:** Shows accuracy vs. FL round curves comparing FedAvg vs Ditto over 100 rounds.

**What needs to be done:**
- The data already exists in `logs/training_history.csv` for the Ditto run
- Need to run `FL_SCHEME = "fedavg"` to get FedAvg curve data
- Plot both curves on the same graph (round vs BMTA)

---

### 9.6 Hyperparameter Sensitivity (Paper Table IV, Table V)

**Paper Table IV:** Tests different λ_ce values: {0.03, 0.05, 0.07, 0.3, 0.5, 0.7}
**Paper Table V:** Tests different τ (temperature) values: {0.1, 0.3, 0.5, 0.7}

**How to run:** Change `LAMBDA_CE` or `TEMPERATURE` in config and re-run FL for each value. Already implemented, just needs runs.

---

### 9.7 E-ResGAT Baseline (Paper Table III)

**Paper:** Compares against E-ResGAT (a more advanced graph baseline not yet implemented).

**Note:** E-GraphSAGE and Anomal-E baselines are implemented in `scripts/run_baselines.py` but E-ResGAT is missing.

---

## 10. Results Achieved

### Binary Classification — CIC-IDS2018 (100 FL Rounds, Ditto)

| Client | Test Accuracy (Local) | Test Accuracy (Pers) | F1 (Pers) |
|---|---|---|---|
| Client 0 | 0.9970 | 0.9994 | 0.9718 |
| Client 1 | 0.9639 | 0.9637 | 0.9542 |
| **Mean (BMTA)** | **0.9817** | **0.9817** | **0.9630** |

**Training details:**
- Dataset: CIC-IDS2018 (377,873 flows total)
- Client 0 training nodes: 25,353 (mostly benign — 25,220 benign, 133 attack)
- Client 1 training nodes: 4,646 (mostly attack — 1,183 benign, 3,463 attack)
- Round time: ~113 seconds per round (CPU only)
- Total training time: ~3.1 hours for 100 rounds

---

## 11. Known Issues and Fixes

### Issue 1 — Augmentation crash: `RuntimeError: Expected p_in >= 0`
**Cause:** CSV contains empty cells in some flow features → NaN/inf after `log()` → `keep_prob` outside [0,1]
**Fix (in `data/augmentation.py`):** Added `.clamp(0.0, 1.0)` on all probabilities and `torch.nan_to_num()` on features before Bernoulli sampling.

### Issue 2 — Client 0 OOM crash (10 GB allocation)
**Cause:** Contrastive loss built a full [50,706 × 50,706] similarity matrix in one shot
**Fix (in `models/losses.py`):** Implemented `batched_joint_loss()` — processes embeddings in chunks of 512. Matrix is now [1024 × 1024] per batch.

### Issue 3 — Server crash when client disconnects
**Cause:** `ConnectionAbortedError` propagated and killed the server process
**Fix (in `server.py`):** All socket operations wrapped in try/except — failed clients removed gracefully and training continues with remaining clients.

### Issue 4 — Windows console encoding error
**Cause:** Windows terminal (cp1252) cannot display Unicode `→` character in log messages
**Fix (in `server.py`):** Changed `→` to `->` in all log messages.

---

## Continuing This Project

When starting a new chat, provide this README and the current project zip file. The immediate next steps are:

1. **Run SSLCon ablation** — change `CONTRASTIVE_MODE = "sslcon"` and run FL → get Table VI numbers
2. **Run FedAvg** — change `FL_SCHEME = "fedavg"` and run FL → get Fig.8/9 comparison
3. **Implement multiclass** — modify `data/line_graph.py` to use `ATTACK_COL` when `CLASSIFICATION_MODE = "multiclass"` → re-run FL
4. **Preprocess and run BoT-IoT and ToN-IoT** — reproduce full Table III
5. **t-SNE visualization** — extract embeddings and plot Fig.6
