# =============================================================================
# SSLCon Experiment Runner (Table VI ablation)
#
# Single script: launches server + 2 clients, saves results to
# results/sslcon_binary/. Does NOT modify config.py.
#
# Usage:
#   python run_sslcon.py
# =============================================================================

import os
import sys
import csv
import copy
import time
import socket
import threading
import logging
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    SERVER_HOST, SERVER_PORT, NUM_CLIENTS, NUM_ROUNDS,
    CLIENT_FRACTION, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, CLASSIFICATION_MODE,
    LABEL_PROPORTION, LABEL_COL, ATTACK_COL,
    LOCAL_LR, SEED, DATA_DIR, LOCAL_EPOCHS
)
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import local_train_client, evaluate_model, server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph
from utils.security import isolated_bind_host

# ── Override: force SSLCon mode ──────────────────────────────────────────────
import config.config as cfg
cfg.CONTRASTIVE_MODE = "sslcon"

RESULT_DIR = os.path.join("results", "sslcon_binary")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)

FL_SCHEME = "ditto"
PORT = 9998  # Use different port to avoid conflict with any running server


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(f"%(asctime)s [{name}] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_client_data(client_id, log):
    label_col = ATTACK_COL if CLASSIFICATION_MODE == "multiclass" else LABEL_COL
    train_csv = os.path.join(DATA_DIR, f"client_{client_id}_train.csv")
    test_csv = os.path.join(DATA_DIR, f"client_{client_id}_test.csv")

    log.info(f"Loading train graph from {train_csv} (label_col={label_col})")
    train_graph = csv_to_line_graph(train_csv, label_col=label_col)
    log.info(f"Loading test graph from {test_csv}")
    test_graph = csv_to_line_graph(test_csv, label_col=label_col)

    labels = train_graph.ndata["label"]
    N = train_graph.num_nodes()
    n_labeled = max(1, int(N * LABEL_PROPORTION))

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
    test_mask = torch.ones(test_graph.num_nodes(), dtype=torch.bool)

    log.info(f"Train: {N:,} nodes | Test: {test_graph.num_nodes():,} nodes | "
             f"Labeled: {train_mask.sum().item()}")
    return train_graph, test_graph, train_mask, test_mask


def make_model():
    nc = NUM_CLASSES_BINARY
    return FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc,
    )


def client_worker(client_id, port):
    log = setup_logger(f"CLIENT-{client_id}",
                       os.path.join(RESULT_LOG_DIR, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id)
    np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)

    log.info("=" * 60)
    log.info(f"FeCoGraph Client {client_id} (SSLCon mode)")
    log.info(f"FL Scheme: {FL_SCHEME.upper()} | Device: {device}")
    log.info("=" * 60)

    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id, log)
    local_model = make_model().to(device)
    personal_model = PersonalizedModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NUM_CLASSES_BINARY,
    ).to(device)

    time.sleep(2)
    connect_ip = "127.0.0.1"
    log.info(f"Connecting to server {connect_ip}:{port}...")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((connect_ip, port))
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)

    send_object(sock, make_message(MsgType.REGISTER,
                                   payload={"client_id": client_id}))
    log.info("Connected and registered.")

    running = True
    while running:
        msg = recv_object(sock)
        if msg is None:
            break
        if msg["type"] == MsgType.SHUTDOWN:
            log.info("Received SHUTDOWN. Done.")
            break
        if msg["type"] == MsgType.ROUND_START:
            round_num = msg["round"]
            local_model.load_state_dict(msg["payload"], strict=True)
            local_model.to(device)

            t0 = time.time()
            delta, personal_model, train_metrics = local_train_client(
                global_model=local_model,
                personalized_model=personal_model,
                graph=train_graph, train_mask=train_mask,
                device=device, round_num=round_num, fl_scheme=FL_SCHEME,
            )
            test_local = evaluate_model(local_model, test_graph, test_mask,
                                        device, use_personalized=False)
            test_pers = evaluate_model(personal_model, test_graph, test_mask,
                                       device, use_personalized=True)
            elapsed = time.time() - t0
            log.info(f"  Round {round_num+1} | Loss={train_metrics['total']:.4f} | "
                     f"Acc(L)={test_local['accuracy']:.4f} | "
                     f"Acc(P)={test_pers['accuracy']:.4f} | "
                     f"F1(P)={test_pers['f1']:.4f} | {elapsed:.1f}s")

            send_object(sock, make_message(MsgType.CLIENT_DELTA, payload={
                "delta": delta,
                "metrics": {
                    **train_metrics,
                    "test_acc_local": test_local["accuracy"],
                    "test_f1_local": test_local["f1"],
                    "test_acc_pers": test_pers["accuracy"],
                    "test_f1_pers": test_pers["f1"],
                    "test_precision": test_pers["precision"],
                    "test_recall": test_pers["recall"],
                }
            }, round_num=round_num))

    torch.save(local_model.state_dict(),
               os.path.join(RESULT_CKPT_DIR, f"client_{client_id}_local_final.pt"))
    torch.save(personal_model.state_dict(),
               os.path.join(RESULT_CKPT_DIR, f"client_{client_id}_personal_final.pt"))
    sock.close()
    log.info("Client done.")


def server_worker(port):
    log = setup_logger("SERVER",
                       os.path.join(RESULT_LOG_DIR, "server.log"))

    global_model = make_model().to(DEVICE)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [],
               "mean_f1_local": [], "mean_f1_pers": []}
    bmta_local = 0.0
    bmta_pers = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((isolated_bind_host("0.0.0.0"), port))
    server_sock.listen(NUM_CLIENTS + 2)

    log.info(f"SSLCon Server on port {port} | Waiting for {NUM_CLIENTS} clients...")

    while len(client_conns) < NUM_CLIENTS:
        conn, addr = server_sock.accept()
        msg = recv_object(conn)
        if msg and msg["type"] == MsgType.REGISTER:
            cid = msg["payload"]["client_id"]
            client_conns[cid] = conn
            log.info(f"  Client {cid} registered from {addr}")

    log.info(f"All {NUM_CLIENTS} clients connected. Starting SSLCon training...")
    log.info("=" * 65)

    for round_t in range(NUM_ROUNDS):
        if not client_conns:
            break
        t_start = time.time()
        selected = list(client_conns.keys())
        global_sd = {k: v.cpu() for k, v in global_model.state_dict().items()}

        failed = []
        for cid in selected:
            try:
                send_object(client_conns[cid],
                            make_message(MsgType.ROUND_START,
                                         payload=global_sd, round_num=round_t))
            except Exception:
                failed.append(cid)
        selected = [c for c in selected if c not in failed]

        client_deltas = []
        client_metrics = []
        for cid in selected:
            try:
                resp = recv_object(client_conns[cid])
                if resp and resp["type"] == MsgType.CLIENT_DELTA:
                    client_deltas.append(resp["payload"]["delta"])
                    client_metrics.append(resp["payload"]["metrics"])
            except Exception:
                pass

        if client_deltas:
            global_model = server_aggregate(global_model, client_deltas, selected)

        elapsed = time.time() - t_start
        if client_metrics:
            mean_acc_l = float(np.mean([m["test_acc_local"] for m in client_metrics]))
            mean_acc_p = float(np.mean([m["test_acc_pers"] for m in client_metrics]))
            mean_f1_l = float(np.mean([m["test_f1_local"] for m in client_metrics]))
            mean_f1_p = float(np.mean([m["test_f1_pers"] for m in client_metrics]))
            avg_supcon = float(np.mean([m.get("supcon", 0) for m in client_metrics]))
            avg_ce = float(np.mean([m.get("ce", 0) for m in client_metrics]))

            if mean_acc_l > bmta_local:
                bmta_local = mean_acc_l
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(),
                            "bmta_local": bmta_local, "bmta_pers": bmta_pers},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_local.pt"))
            if mean_acc_p > bmta_pers:
                bmta_pers = mean_acc_p
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(),
                            "bmta_local": bmta_local, "bmta_pers": bmta_pers},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_pers.pt"))

            history["round"].append(round_t + 1)
            history["loss_supcon"].append(avg_supcon)
            history["loss_ce"].append(avg_ce)
            history["mean_acc_local"].append(mean_acc_l)
            history["mean_acc_pers"].append(mean_acc_p)
            history["mean_f1_local"].append(mean_f1_l)
            history["mean_f1_pers"].append(mean_f1_p)

            if (round_t + 1) % LOG_EVERY == 0 or round_t == NUM_ROUNDS - 1:
                log.info(f"Round {round_t+1:>3}/{NUM_ROUNDS} | "
                         f"SSLCon={avg_supcon:.4f} | CE={avg_ce:.4f} | "
                         f"Acc_L={mean_acc_l:.4f} | F1_L={mean_f1_l:.4f} | "
                         f"Acc_P={mean_acc_p:.4f} | F1_P={mean_f1_p:.4f} | "
                         f"BMTA_P={bmta_pers:.4f} | {elapsed:.1f}s")

    log.info("=" * 65)
    log.info("SSLCon TRAINING COMPLETE")
    log.info(f"  BMTA (Local):       {bmta_local:.4f}")
    log.info(f"  BMTA (Personalized): {bmta_pers:.4f}")

    torch.save({"round": NUM_ROUNDS, "model_state": global_model.state_dict(),
                "bmta_local": bmta_local, "bmta_pers": bmta_pers},
               os.path.join(RESULT_CKPT_DIR, "global_model_final.pt"))

    # Save history CSV
    hist_path = os.path.join(RESULT_LOG_DIR, "training_history.csv")
    if history["round"]:
        keys = list(history.keys())
        with open(hist_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(len(history["round"])):
                writer.writerow({k: history[k][i] for k in keys})
    log.info(f"  History saved: {hist_path}")

    for cid, conn in client_conns.items():
        try:
            send_object(conn, make_message(MsgType.SHUTDOWN))
            conn.close()
        except Exception:
            pass
    server_sock.close()
    log.info("Server shut down.")


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph SSLCon Experiment (Table VI ablation)")
    print(f"Results will be saved to: {RESULT_DIR}")
    print(f"Using port: {PORT}")
    print("=" * 70)

    server_thread = threading.Thread(target=server_worker, args=(PORT,), daemon=True)
    server_thread.start()
    time.sleep(3)

    client_threads = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, PORT), daemon=False)
        t.start()
        client_threads.append(t)
        time.sleep(1)

    server_thread.join()
    for t in client_threads:
        t.join(timeout=30)

    print("\n" + "=" * 70)
    print("SSLCon experiment DONE!")
    print(f"Results in: {RESULT_DIR}")
    print("=" * 70)
