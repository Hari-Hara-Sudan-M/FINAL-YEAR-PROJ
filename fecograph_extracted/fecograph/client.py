#!/usr/bin/env python3
# =============================================================================
# FeCoGraph CLIENT  (runs on each Client Laptop)
#
# Algorithm 1 — Client side (lines 3-12):
#   Line 4:  Graph augmentation G_k → G1_k, G2_k
#   Line 5:  Compute F_k (contrastive loss) on G1, G2
#   Line 6-7: Update w_k for r local epochs
#   Line 8:  Compute f_k (CE loss) on G
#   Line 9-10: Update θ_k for s personalized epochs (Ditto)
#   Line 11: Send Δ_k = w_k - w_t back to server
#
# Usage:
#   python client.py --client_id 0 --server_ip 192.168.x.x
#   python client.py --client_id 1 --server_ip 192.168.x.x
# =============================================================================

import os
import sys
import copy
import time
import socket
import logging
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    SERVER_HOST, SERVER_PORT, DATA_DIR, DEVICE, LOG_DIR,
    FL_SCHEME, INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI, CLASSIFICATION_MODE,
    TRAIN_RATIO, LABEL_PROPORTION, LABEL_COL, ATTACK_COL, LOCAL_LR, SEED
)
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import local_train_client, evaluate_model
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph
from utils.security import isolated_connect_host


parser = argparse.ArgumentParser(description="FeCoGraph Federated Client")
parser.add_argument("--client_id",  type=int, required=True, help="Client ID (0, 1, ...)")
parser.add_argument("--server_ip",  type=str, default=SERVER_HOST, help="Server IP address")
parser.add_argument("--server_port",type=int, default=SERVER_PORT)
parser.add_argument("--data_dir",   type=str, default=DATA_DIR)
parser.add_argument("--fl_scheme",  type=str, default=FL_SCHEME, choices=["fedavg", "ditto"])
args = parser.parse_args()

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [CLIENT-{args.client_id}] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"client_{args.client_id}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(f"FeCoGraph.client{args.client_id}")



def load_client_data(client_id, data_dir):
    train_csv = os.path.join(data_dir, f"client_{client_id}_train.csv")
    test_csv  = os.path.join(data_dir, f"client_{client_id}_test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Train CSV not found: {train_csv}\n"
            f"Run: python scripts/split_dataset.py --csv <path_to_processed.csv>"
        )

    label_col = ATTACK_COL if CLASSIFICATION_MODE == "multiclass" else LABEL_COL

    log.info(f"Loading train graph from {train_csv} (label_col={label_col})")
    train_graph = csv_to_line_graph(train_csv, label_col=label_col)

    log.info(f"Loading test graph from {test_csv}")
    test_graph  = csv_to_line_graph(test_csv,  label_col=label_col)

    labels     = train_graph.ndata["label"]
    N          = train_graph.num_nodes()
    n_labeled  = max(1, int(N * LABEL_PROPORTION))

    # Stratified sampling across all classes to keep class balance
    unique_classes = labels.unique()
    selected_parts = []
    for cls in unique_classes:
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        n_cls = max(1, min(int(n_labeled * len(cls_idx) / N), len(cls_idx)))
        perm = torch.randperm(len(cls_idx))[:n_cls]
        selected_parts.append(cls_idx[perm])
    selected = torch.cat(selected_parts)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[selected] = True

    test_mask  = torch.ones(test_graph.num_nodes(), dtype=torch.bool)

    log.info(f"Train graph: {train_graph.num_nodes():,} nodes | "
             f"Test graph: {test_graph.num_nodes():,} nodes")
    log.info(f"Label proportion: {LABEL_PROPORTION*100:.0f}%  "
             f"({train_mask.sum().item()} labeled nodes)")
    log.info(f"Label dist (train): {torch.bincount(train_graph.ndata['label'])}")
    log.info(f"Label dist (labeled subset): {torch.bincount(labels[train_mask])}")

    return train_graph, test_graph, train_mask, test_mask



def make_model():
    num_classes = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
    return FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM,
        hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2,
        proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM,
        num_classes=num_classes
    )

def make_personal_model():
    num_classes = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
    return PersonalizedModel(
        in_dim=INPUT_FEATURE_DIM,
        hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2,
        proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM,
        num_classes=num_classes
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main client loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)

    log.info("=" * 60)
    log.info(f"FeCoGraph Client {args.client_id}")
    log.info(f"FL Scheme: {args.fl_scheme.upper()}")
    log.info(f"Device: {device}")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_graph, test_graph, train_mask, test_mask = load_client_data(
        args.client_id, args.data_dir
    )

    # ── Initialize models ─────────────────────────────────────────────────────
    local_model        = make_model().to(device)
    personalized_model = make_personal_model().to(device) if args.fl_scheme == "ditto" else None

    # ── Connect to server ─────────────────────────────────────────────────────
    connect_ip = isolated_connect_host(args.server_ip)
    log.info(f"Connecting to server {connect_ip}:{args.server_port}...")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((connect_ip, args.server_port))
            break
        except ConnectionRefusedError:
            log.info("  Server not ready, retrying in 3s...")
            time.sleep(3)

    # Register with server
    send_object(sock, make_message(MsgType.REGISTER,
                                   payload={"client_id": args.client_id}))
    log.info("Connected and registered with server.")

    round_num = 0
    running   = True

    # ── FL Training Loop ──────────────────────────────────────────────────────
    while running:
        msg = recv_object(sock)
        if msg is None:
            log.warning("Lost connection to server.")
            break

        if msg["type"] == MsgType.SHUTDOWN:
            log.info("Received SHUTDOWN from server. Training complete.")
            running = False
            break

        if msg["type"] == MsgType.ROUND_START:
            round_num = msg["round"]
            global_sd = msg["payload"]

            # Load global model weights w_t  (Algorithm 1, line 6: set w_k^t = w_t)
            local_model.load_state_dict(global_sd, strict=True)
            local_model.to(device)

            t_round = time.time()
            log.info(f"Round {round_num+1}: Starting local training...")

            # ── Algorithm 1, lines 4-11: Local training ───────────────────────
            delta, personalized_model, train_metrics = local_train_client(
                global_model=local_model,
                personalized_model=personalized_model,
                graph=train_graph,
                train_mask=train_mask,
                device=device,
                round_num=round_num,
                fl_scheme=args.fl_scheme,
            )

            test_metrics_local = evaluate_model(
                local_model, test_graph, test_mask, device, use_personalized=False
            )

            if args.fl_scheme == "ditto" and personalized_model is not None:
                test_metrics_pers = evaluate_model(
                    personalized_model, test_graph, test_mask, device, use_personalized=True
                )
            else:
                test_metrics_pers = test_metrics_local

            elapsed = time.time() - t_round
            log.info(
                f"  Round {round_num+1} done | "
                f"Train loss={train_metrics['total']:.4f} | "
                f"SupCon={train_metrics['supcon']:.4f} | "
                f"CE={train_metrics['ce']:.4f} | "
                f"TestAcc(local)={test_metrics_local['accuracy']:.4f} | "
                f"TestAcc(pers)={test_metrics_pers['accuracy']:.4f} | "
                f"F1={test_metrics_pers['f1']:.4f} | "
                f"Time={elapsed:.1f}s"
            )

            response_payload = {
                "delta": delta,
                "metrics": {
                    **train_metrics,
                    # Local model (w_k) metrics
                    "test_acc_local": test_metrics_local["accuracy"],
                    "test_f1_local":  test_metrics_local["f1"],
                    # Personalized model (theta_k) metrics
                    "test_acc_pers":  test_metrics_pers["accuracy"],
                    "test_f1_pers":   test_metrics_pers["f1"],
                    "test_precision": test_metrics_pers["precision"],
                    "test_recall":    test_metrics_pers["recall"],
                }
            }
            send_object(sock, make_message(MsgType.CLIENT_DELTA,
                                           payload=response_payload,
                                           round_num=round_num))

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(local_model.state_dict(),
               f"checkpoints/client_{args.client_id}_local_final.pt")
    if personalized_model:
        torch.save(personalized_model.state_dict(),
                   f"checkpoints/client_{args.client_id}_personal_final.pt")

    sock.close()
    log.info("Client done.")


if __name__ == "__main__":
    main()
