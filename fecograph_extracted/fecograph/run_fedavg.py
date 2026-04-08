# =============================================================================
# FedAvg Experiment (Fig. 8 — FedAvg vs Ditto comparison)
#
# Runs FeCoGraph with FedAvg (no personalized model). Results saved to
# results/fedavg_binary/. Does NOT modify config.py.
#
# Usage:  python run_fedavg.py
# =============================================================================

import os, sys, csv, time, socket, threading, logging
import numpy as np, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    NUM_CLIENTS, NUM_ROUNDS, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, CLASSIFICATION_MODE,
    LABEL_PROPORTION, LABEL_COL, ATTACK_COL,
    SEED, DATA_DIR
)
from models.model import FeCoGraphModel
from fl.federated import local_train_client, evaluate_model, server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph
from utils.security import isolated_bind_host

RESULT_DIR = os.path.join("results", "fedavg_binary")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)

FL_SCHEME = "fedavg"
PORT = 9997


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(f"%(asctime)s [{name}] %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def load_client_data(client_id, log):
    label_col = ATTACK_COL if CLASSIFICATION_MODE == "multiclass" else LABEL_COL
    train_csv = os.path.join(DATA_DIR, f"client_{client_id}_train.csv")
    test_csv = os.path.join(DATA_DIR, f"client_{client_id}_test.csv")
    log.info(f"Loading graphs (label_col={label_col})")
    train_graph = csv_to_line_graph(train_csv, label_col=label_col)
    test_graph = csv_to_line_graph(test_csv, label_col=label_col)
    labels = train_graph.ndata["label"]
    N = train_graph.num_nodes()
    n_labeled = max(1, int(N * LABEL_PROPORTION))
    selected_parts = []
    for cls in labels.unique():
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        n_cls = max(1, min(int(n_labeled * len(cls_idx) / N), len(cls_idx)))
        selected_parts.append(cls_idx[torch.randperm(len(cls_idx))[:n_cls]])
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[torch.cat(selected_parts)] = True
    test_mask = torch.ones(test_graph.num_nodes(), dtype=torch.bool)
    log.info(f"Train: {N:,} | Test: {test_graph.num_nodes():,} | Labeled: {train_mask.sum().item()}")
    return train_graph, test_graph, train_mask, test_mask


def make_model():
    return FeCoGraphModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NUM_CLASSES_BINARY)


def client_worker(client_id, port):
    log = setup_logger(f"CLIENT-{client_id}",
                       os.path.join(RESULT_LOG_DIR, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id); np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)
    log.info("=" * 60)
    log.info(f"FeCoGraph Client {client_id} (FedAvg mode)")
    log.info("=" * 60)

    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id, log)
    local_model = make_model().to(device)

    time.sleep(2)
    log.info(f"Connecting to 127.0.0.1:{port}...")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port)); break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)

    send_object(sock, make_message(MsgType.REGISTER, payload={"client_id": client_id}))
    log.info("Connected.")

    running = True
    while running:
        msg = recv_object(sock)
        if msg is None: break
        if msg["type"] == MsgType.SHUTDOWN:
            log.info("SHUTDOWN received."); break
        if msg["type"] == MsgType.ROUND_START:
            round_num = msg["round"]
            local_model.load_state_dict(msg["payload"], strict=True)
            local_model.to(device)
            t0 = time.time()
            delta, _, train_metrics = local_train_client(
                global_model=local_model, personalized_model=None,
                graph=train_graph, train_mask=train_mask,
                device=device, round_num=round_num, fl_scheme="fedavg")
            test_m = evaluate_model(local_model, test_graph, test_mask, device, use_personalized=False)
            log.info(f"  Round {round_num+1} | Acc={test_m['accuracy']:.4f} | "
                     f"F1={test_m['f1']:.4f} | {time.time()-t0:.1f}s")
            send_object(sock, make_message(MsgType.CLIENT_DELTA, payload={
                "delta": delta, "metrics": {
                    **train_metrics,
                    "test_acc_local": test_m["accuracy"], "test_f1_local": test_m["f1"],
                    "test_acc_pers": test_m["accuracy"], "test_f1_pers": test_m["f1"],
                    "test_precision": test_m["precision"], "test_recall": test_m["recall"],
                }}, round_num=round_num))

    torch.save(local_model.state_dict(),
               os.path.join(RESULT_CKPT_DIR, f"client_{client_id}_local_final.pt"))
    sock.close(); log.info("Done.")


def server_worker(port):
    log = setup_logger("SERVER", os.path.join(RESULT_LOG_DIR, "server.log"))
    global_model = make_model().to(DEVICE)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [],
               "mean_f1_local": [], "mean_f1_pers": []}
    bmta = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((isolated_bind_host("0.0.0.0"), port)); server_sock.listen(NUM_CLIENTS + 2)
    log.info(f"FedAvg Server on port {port} | Waiting for {NUM_CLIENTS} clients...")

    while len(client_conns) < NUM_CLIENTS:
        conn, addr = server_sock.accept()
        msg = recv_object(conn)
        if msg and msg["type"] == MsgType.REGISTER:
            cid = msg["payload"]["client_id"]
            client_conns[cid] = conn
            log.info(f"  Client {cid} from {addr}")

    log.info(f"All clients connected. Starting FedAvg training...")
    log.info("=" * 65)

    for round_t in range(NUM_ROUNDS):
        if not client_conns: break
        t_start = time.time()
        selected = list(client_conns.keys())
        global_sd = {k: v.cpu() for k, v in global_model.state_dict().items()}
        failed = []
        for cid in selected:
            try: send_object(client_conns[cid], make_message(MsgType.ROUND_START, payload=global_sd, round_num=round_t))
            except: failed.append(cid)
        selected = [c for c in selected if c not in failed]
        client_deltas, client_metrics = [], []
        for cid in selected:
            try:
                resp = recv_object(client_conns[cid])
                if resp and resp["type"] == MsgType.CLIENT_DELTA:
                    client_deltas.append(resp["payload"]["delta"])
                    client_metrics.append(resp["payload"]["metrics"])
            except: pass
        if client_deltas:
            global_model = server_aggregate(global_model, client_deltas, selected)
        if client_metrics:
            ma = float(np.mean([m["test_acc_local"] for m in client_metrics]))
            mf = float(np.mean([m["test_f1_local"] for m in client_metrics]))
            sc = float(np.mean([m.get("supcon", 0) for m in client_metrics]))
            ce = float(np.mean([m.get("ce", 0) for m in client_metrics]))
            if ma > bmta:
                bmta = ma
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(), "bmta": bmta},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_local.pt"))
            history["round"].append(round_t+1); history["loss_supcon"].append(sc)
            history["loss_ce"].append(ce); history["mean_acc_local"].append(ma)
            history["mean_acc_pers"].append(ma); history["mean_f1_local"].append(mf)
            history["mean_f1_pers"].append(mf)
            if (round_t+1) % LOG_EVERY == 0 or round_t == NUM_ROUNDS-1:
                log.info(f"Round {round_t+1:>3}/{NUM_ROUNDS} | SupCon={sc:.4f} | CE={ce:.4f} | "
                         f"Acc={ma:.4f} | F1={mf:.4f} | BMTA={bmta:.4f} | {time.time()-t_start:.1f}s")

    log.info("=" * 65)
    log.info(f"FedAvg TRAINING COMPLETE | BMTA: {bmta:.4f}")
    torch.save({"round": NUM_ROUNDS, "model_state": global_model.state_dict(), "bmta": bmta},
               os.path.join(RESULT_CKPT_DIR, "global_model_final.pt"))
    hist_path = os.path.join(RESULT_LOG_DIR, "training_history.csv")
    if history["round"]:
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history.keys())); w.writeheader()
            for i in range(len(history["round"])): w.writerow({k: history[k][i] for k in history})
    log.info(f"  History saved: {hist_path}")
    for cid, conn in client_conns.items():
        try: send_object(conn, make_message(MsgType.SHUTDOWN)); conn.close()
        except: pass
    server_sock.close(); log.info("Server shut down.")


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph FedAvg Experiment (Fig. 8)")
    print(f"Results -> {RESULT_DIR}"); print(f"Port: {PORT}")
    print("=" * 70)
    server_thread = threading.Thread(target=server_worker, args=(PORT,), daemon=True)
    server_thread.start(); time.sleep(3)
    threads = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, PORT)); t.start()
        threads.append(t); time.sleep(1)
    server_thread.join()
    for t in threads: t.join(timeout=30)
    print(f"\nFedAvg experiment DONE! Results in: {RESULT_DIR}")
