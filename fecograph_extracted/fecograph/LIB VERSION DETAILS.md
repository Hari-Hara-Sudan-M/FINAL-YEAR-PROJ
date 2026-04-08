# FeCoGraph: Federated Graph Contrastive Learning for NIDS
## Full Implementation — 3-Laptop Setup (1 Server + 2 Clients)

---

## Python Version
**Use Python 3.8.x** (paper uses 3.8.8 — most compatible with DGL 1.x + PyTorch 1.13)

---

## Project Structure
```
fecograph/
├── config/
│   └── config.py              ← Global settings — EDIT THIS BEFORE RUNNING
├── data/
│   ├── line_graph.py          ← Line graph converter (Paper Section IV-B1)
│   └── augmentation.py        ← Adaptive augmentation Eq.1-5
├── models/
│   ├── model.py               ← GCN Encoder+Projector+Classifier Eq.6-7
│   └── losses.py              ← SupCon + CE + Joint Loss Eq.8-12
├── fl/
│   ├── federated.py           ← FedAvg + Ditto (Algorithm 1)
│   └── comm.py                ← Socket communication
├── scripts/
│   └── split_dataset.py       ← Split CSV into per-client partitions
├── utils/
│   └── utils.py
├── server.py                  ← Run on Server Laptop
├── client.py                  ← Run on Client Laptops
├── test_local.py              ← Single-machine test before FL
└── requirements.txt
```

---

## Step-by-Step Setup

### Step 1 — Install Python 3.8 on ALL 3 laptops
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# Create virtual environment on EACH laptop
python3.8 -m venv feco_env
source feco_env/bin/activate       # Linux/Mac
# feco_env\Scripts\activate        # Windows
```

### Step 2 — Install dependencies on ALL 3 laptops
```bash
pip install --upgrade pip

# Install PyTorch 1.13 (CPU-only — works on all laptops)
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu

# Install DGL for PyTorch 1.13 + CPU
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html

# Install PyTorch Geometric
pip install torch-geometric==2.3.0

# Other dependencies
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 tqdm matplotlib
```

### Step 3 — Copy this project folder to ALL 3 laptops
```bash
# On all laptops the folder structure must be identical
# just the data files differ per client laptop
```

---

## Running the Project

### PHASE A: On the machine with the preprocessed CSV (your Kaggle output)

**Step A1 — Set number of clients in config:**
```python
# config/config.py — change this line:
NUM_CLIENTS = 2
```

**Step A2 — Split dataset:**
```bash
python scripts/split_dataset.py \
    --csv /path/to/CIC-IDS2018_processed.csv \
    --num_clients 2 \
    --alpha 0.5

# This creates:
#   data/client_0_train.csv  data/client_0_test.csv
#   data/client_1_train.csv  data/client_1_test.csv
```

**Step A3 — Copy data to client laptops:**
```bash
# Send to Client Laptop 0:
scp data/client_0_train.csv  user@CLIENT0_IP:~/fecograph/data/
scp data/client_0_test.csv   user@CLIENT0_IP:~/fecograph/data/

# Send to Client Laptop 1:
scp data/client_1_train.csv  user@CLIENT1_IP:~/fecograph/data/
scp data/client_1_test.csv   user@CLIENT1_IP:~/fecograph/data/
```

---

### PHASE B: Quick local test (optional, single machine)
```bash
# Test the entire pipeline on one laptop before running FL
python test_local.py --csv /path/to/CIC-IDS2018_processed.csv --rows 5000 --epochs 10

# Expected output:
# [1] Loading data...
# [2] Building Line Graph L(G)...
# [3] Creating train/test masks...
# [4] Adaptive Graph Augmentation G → G1, G2
# [5] Initializing FeCoGraph Model...
# [6] Training...
# [7] Evaluation → Acc=0.xxxx | F1=0.xxxx
# Local pipeline test PASSED.
```

---

### PHASE C: Federated Learning (3 laptops in parallel)

**Step C1 — Find Server Laptop's IP:**
```bash
# On Server Laptop:
ip addr show        # Linux
ipconfig            # Windows
# Note the LAN IP, e.g. 192.168.1.100
```

**Step C2 — Update config on ALL laptops:**
```python
# config/config.py:
NUM_CLIENTS  = 2
NUM_ROUNDS   = 100
LOCAL_EPOCHS = 5
FL_SCHEME    = "ditto"    # or "fedavg"
SERVER_PORT  = 9999
```

**Step C3 — Start Server FIRST:**
```bash
# On Server Laptop:
source feco_env/bin/activate
python server.py
# → Waiting for 2 clients...
```

**Step C4 — Start each Client (AFTER server is running):**
```bash
# On Client Laptop 0:
source feco_env/bin/activate
python client.py --client_id 0 --server_ip 192.168.1.100

# On Client Laptop 1:
source feco_env/bin/activate
python client.py --client_id 1 --server_ip 192.168.1.100
```

**Training starts automatically once all clients connect.**

---

## Configuration Reference (config/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLIENTS` | 2 | **Change this first** |
| `NUM_ROUNDS` | 100 | FL communication rounds T |
| `LOCAL_EPOCHS` | 5 | Local training epochs r |
| `PERSONALIZED_EPOCHS` | 5 | Ditto personalized epochs s |
| `LOCAL_LR` | 0.001 | Local learning rate η_l |
| `PERSONALIZED_LR` | 0.001 | Personalized learning rate η_p |
| `MU` | 0.1 | Ditto regularization µ |
| `FL_SCHEME` | "ditto" | "fedavg" or "ditto" |
| `LAMBDA_CE` | 0.07 | λ_ce joint loss weight (best IDS2018, Table IV) |
| `TEMPERATURE` | 0.3 | τ contrastive temperature |
| `CLASSIFICATION_MODE` | "binary" | "binary" or "multiclass" |
| `INPUT_FEATURE_DIM` | 40 | Flow features (from your preprocessing output) |
| `SERVER_PORT` | 9999 | TCP port for FL communication |

---

## Outputs

- **Logs:** `logs/server.log`, `logs/client_0.log`, `logs/client_1.log`
- **Checkpoints:** `checkpoints/global_model_best.pt`, `checkpoints/global_model_final.pt`
- **Metrics reported:** Accuracy, Precision, Recall, F1-score (macro) per round
- **Primary metric:** BMTA (Best Mean Testing Accuracy across clients) — paper standard

---

## Paper → Code Mapping

| Paper Element | Code Location |
|--------------|--------------|
| Line Graph L(G) construction | `data/line_graph.py` |
| Eq.1-2 topology augmentation | `data/augmentation.py:topology_augmentation()` |
| Eq.3-5 attribute augmentation | `data/augmentation.py:attribute_augmentation()` |
| Eq.6-7 GCN Encoder | `models/model.py:GCNLayer, GCNEncoder` |
| Eq.8-9 Supervised Contrastive Loss | `models/losses.py:SupervisedContrastiveLoss` |
| Eq.10-11 Cross-Entropy Loss | `models/losses.py:ClassificationLoss` |
| Eq.12 Joint Loss | `models/losses.py:JointLoss` |
| Eq.13-14 FedAvg | `fl/federated.py:server_aggregate()` |
| Eq.15-16 Ditto personalization | `fl/federated.py:local_train_client()` |
| Eq.17 Global objective F_k | `fl/federated.py` (contrastive + CE) |
| Eq.18 Local objective f_k | `fl/federated.py` (CE only for θ_k) |
| Algorithm 1 (full workflow) | `server.py` + `client.py` combined |
| LDA dataset split | `scripts/split_dataset.py:dirichlet_split()` |

---

## Troubleshooting

**"No module named dgl"**
→ Make sure you activated the virtual environment: `source feco_env/bin/activate`

**"Connection refused"**
→ Start server.py BEFORE client.py. Check firewall: `sudo ufw allow 9999`

**"FileNotFoundError: client_0_train.csv"**
→ Run `split_dataset.py` first and copy files to each client laptop

**Memory issues with large graphs**
→ Reduce `MAX_NODES_PER_CLIENT` in config.py

**DGL version conflict**
→ `pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html`
