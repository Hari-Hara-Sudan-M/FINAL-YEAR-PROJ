# =============================================================================
# Lambda Sensitivity Experiment (Table IV)
#
# Runs FeCoGraph with different LAMBDA_CE values: 0.01, 0.03, 0.05, 0.07, 0.1, 0.5
# Each run saves to results/lambda_X/. Does NOT modify config.py.
#
# Usage:  python run_lambda.py
# =============================================================================

import os, sys, csv, time, socket, threading, logging
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    NUM_CLIENTS, NUM_ROUNDS, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI,
    LABEL_PROPORTION, LABEL_COL, ATTACK_COL,
    SEED, DATA_DIR
)
import config.config as cfg
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import local_train_client, evaluate_model, server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph

MODE = "multiclass"  # Paper Table IV is multiclass
NC = NUM_CLASSES_MULTI

LAMBDA_VALUES = [0.03, 0.05, 0.07, 0.3, 0.5, 0.7]  # Paper Table IV exact values
BASE_PORT = 9990


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


def load_client_data(client_id):
    label_col = ATTACK_COL if MODE == "multiclass" else LABEL_COL
    train_graph = csv_to_line_graph(os.path.join(DATA_DIR, f"client_{client_id}_train.csv"), label_col=label_col)
    test_graph = csv_to_line_graph(os.path.join(DATA_DIR, f"client_{client_id}_test.csv"), label_col=label_col)
    labels = train_graph.ndata["label"]
    N = train_graph.num_nodes()
    n_labeled = max(1, int(N * LABEL_PROPORTION))
    parts = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        parts.append(idx[torch.randperm(len(idx))[:max(1, min(int(n_labeled * len(idx) / N), len(idx)))]])
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[torch.cat(parts)] = True
    return train_graph, test_graph, train_mask, torch.ones(test_graph.num_nodes(), dtype=torch.bool)


def make_model():
    return FeCoGraphModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NC)


def client_worker(client_id, port, result_log_dir, result_ckpt_dir):
    log = setup_logger(f"C{client_id}-L{cfg.LAMBDA_CE}",
                       os.path.join(result_log_dir, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id); np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)
    log.info(f"Client {client_id} | LAMBDA_CE={cfg.LAMBDA_CE}")
    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id)
    local_model = make_model().to(device)
    personal_model = PersonalizedModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NC).to(device)

    time.sleep(2)
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port)); break
        except (ConnectionRefusedError, OSError): time.sleep(2)
    send_object(sock, make_message(MsgType.REGISTER, payload={"client_id": client_id}))

    while True:
        msg = recv_object(sock)
        if msg is None or msg["type"] == MsgType.SHUTDOWN: break
        if msg["type"] == MsgType.ROUND_START:
            local_model.load_state_dict(msg["payload"], strict=True); local_model.to(device)
            delta, personal_model, tm = local_train_client(
                local_model, personal_model, train_graph, train_mask, device, msg["round"], fl_scheme="ditto")
            tl = evaluate_model(local_model, test_graph, test_mask, device, False)
            tp = evaluate_model(personal_model, test_graph, test_mask, device, True)
            log.info(f"  R{msg['round']+1} Acc_P={tp['accuracy']:.4f} F1_P={tp['f1']:.4f}")
            send_object(sock, make_message(MsgType.CLIENT_DELTA, payload={
                "delta": delta, "metrics": {**tm,
                    "test_acc_local": tl["accuracy"], "test_f1_local": tl["f1"],
                    "test_acc_pers": tp["accuracy"], "test_f1_pers": tp["f1"],
                    "test_precision": tp["precision"], "test_recall": tp["recall"]}
            }, round_num=msg["round"]))
    torch.save(local_model.state_dict(), os.path.join(result_ckpt_dir, f"client_{client_id}_final.pt"))
    sock.close()


def server_worker(port, result_log_dir, result_ckpt_dir, lam_val):
    log = setup_logger(f"SRV-L{lam_val}", os.path.join(result_log_dir, "server.log"))
    global_model = make_model().to(DEVICE)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [], "mean_f1_local": [], "mean_f1_pers": []}
    bmta_p = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port)); server_sock.listen(5)
    log.info(f"Lambda={lam_val} Server on port {port}")

    while len(client_conns) < NUM_CLIENTS:
        conn, addr = server_sock.accept()
        msg = recv_object(conn)
        if msg and msg["type"] == MsgType.REGISTER:
            client_conns[msg["payload"]["client_id"]] = conn
            log.info(f"  Client {msg['payload']['client_id']} connected")
    log.info("All clients connected. Training...")

    for rt in range(NUM_ROUNDS):
        if not client_conns: break
        t0 = time.time()
        selected = list(client_conns.keys())
        gsd = {k: v.cpu() for k, v in global_model.state_dict().items()}
        for cid in selected:
            try: send_object(client_conns[cid], make_message(MsgType.ROUND_START, payload=gsd, round_num=rt))
            except: pass
        deltas, metrics = [], []
        for cid in selected:
            try:
                r = recv_object(client_conns[cid])
                if r and r["type"] == MsgType.CLIENT_DELTA:
                    deltas.append(r["payload"]["delta"]); metrics.append(r["payload"]["metrics"])
            except: pass
        if deltas: global_model = server_aggregate(global_model, deltas, selected)
        if metrics:
            ma = float(np.mean([m["test_acc_pers"] for m in metrics]))
            mf = float(np.mean([m["test_f1_pers"] for m in metrics]))
            sc = float(np.mean([m.get("supcon", 0) for m in metrics]))
            ce = float(np.mean([m.get("ce", 0) for m in metrics]))
            mal = float(np.mean([m["test_acc_local"] for m in metrics]))
            mfl = float(np.mean([m["test_f1_local"] for m in metrics]))
            if ma > bmta_p:
                bmta_p = ma
                torch.save({"round": rt+1, "model_state": global_model.state_dict(), "bmta": bmta_p},
                           os.path.join(result_ckpt_dir, "global_model_best_pers.pt"))
            history["round"].append(rt+1); history["loss_supcon"].append(sc); history["loss_ce"].append(ce)
            history["mean_acc_local"].append(mal); history["mean_acc_pers"].append(ma)
            history["mean_f1_local"].append(mfl); history["mean_f1_pers"].append(mf)
            if (rt+1) % LOG_EVERY == 0 or rt == NUM_ROUNDS-1:
                log.info(f"R{rt+1:>3}/{NUM_ROUNDS} | Acc_P={ma:.4f} F1_P={mf:.4f} BMTA={bmta_p:.4f} | {time.time()-t0:.1f}s")

    log.info(f"DONE | Lambda={lam_val} | BMTA={bmta_p:.4f}")
    torch.save({"round": NUM_ROUNDS, "model_state": global_model.state_dict(), "bmta": bmta_p},
               os.path.join(result_ckpt_dir, "global_model_final.pt"))
    hp = os.path.join(result_log_dir, "training_history.csv")
    if history["round"]:
        with open(hp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history.keys())); w.writeheader()
            for i in range(len(history["round"])): w.writerow({k: history[k][i] for k in history})
    for c in client_conns.values():
        try: send_object(c, make_message(MsgType.SHUTDOWN)); c.close()
        except: pass
    server_sock.close()


def run_one(lam_val, port):
    cfg.LAMBDA_CE = lam_val
    tag = str(lam_val).replace(".", "p")
    rdir = os.path.join("results", f"lambda_{tag}")
    rlog = os.path.join(rdir, "logs"); rckpt = os.path.join(rdir, "checkpoints")
    os.makedirs(rlog, exist_ok=True); os.makedirs(rckpt, exist_ok=True)

    print(f"\n{'='*70}\nRunning LAMBDA_CE = {lam_val} | Results -> {rdir}\n{'='*70}")
    st = threading.Thread(target=server_worker, args=(port, rlog, rckpt, lam_val), daemon=True)
    st.start(); time.sleep(3)
    cts = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, port, rlog, rckpt))
        t.start(); cts.append(t); time.sleep(1)
    st.join()
    for t in cts: t.join(timeout=30)
    print(f"Lambda={lam_val} DONE!\n")


def generate_summary():
    import pandas as pd
    print("\n" + "=" * 70)
    print("LAMBDA SENSITIVITY SUMMARY (Paper Table IV)")
    print("=" * 70)
    rows = []
    for lv in LAMBDA_VALUES:
        tag = str(lv).replace(".", "p")
        hp = os.path.join("results", f"lambda_{tag}", "logs", "training_history.csv")
        if os.path.exists(hp):
            df = pd.read_csv(hp)
            bmta = df["mean_acc_pers"].max()
            best_f1 = df["mean_f1_pers"].max()
            final = df.iloc[-1]
            rows.append({"lambda": lv, "BMTA": round(bmta*100, 2),
                         "Best_F1": round(best_f1*100, 2),
                         "Final_Acc": round(final["mean_acc_pers"]*100, 2),
                         "Final_F1": round(final["mean_f1_pers"]*100, 2)})
            print(f"  λ={lv:<5} | BMTA={bmta*100:.2f}% | F1={best_f1*100:.2f}%")
    if rows:
        summary_path = os.path.join("results", "lambda_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nSummary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph Lambda Sensitivity (Paper Table IV - Multiclass)")
    print(f"Values: {LAMBDA_VALUES}")
    print("=" * 70)
    for i, lv in enumerate(LAMBDA_VALUES):
        run_one(lv, BASE_PORT + i)
    generate_summary()
    print("\nALL LAMBDA EXPERIMENTS DONE! Results in: results/lambda_*/")
