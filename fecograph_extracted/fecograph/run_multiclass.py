# =============================================================================
# Multiclass Label Proportion Experiment (Paper Fig. 7)
#
# Runs FeCoGraph in MULTICLASS mode with label proportions: 0.1, 0.3, 0.5, 0.7
# Saves per-attack F1 scores for Fig. 7 bar chart generation.
#
# Results saved to: results/labelprop_Xp_multiclass/
#
# Usage:  python run_multiclass.py
# =============================================================================

import os, sys, csv, json, time, socket, threading, logging
import numpy as np, torch
from sklearn.metrics import f1_score as sklearn_f1, accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    NUM_CLIENTS, NUM_ROUNDS, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_MULTI, ATTACK_COL, SEED, DATA_DIR
)
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import local_train_client, evaluate_model, server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph

PROP_VALUES = [0.1, 0.3, 0.5, 0.7]
BASE_PORT = 9960
NC = NUM_CLASSES_MULTI

ATTACK_NAMES = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

# Paper groups for Fig. 7a (CIC-IDS2018)
ATTACK_GROUPS = {
    "BruteForce": [2, 3, 11, 14],
    "Bot": [1],
    "DoS": [7, 8, 9, 10],
    "DDoS": [4, 5, 6],
    "Infiltration": [12],
    "Web Attacks": [],
}


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


def load_client_data(client_id, label_proportion):
    train_graph = csv_to_line_graph(
        os.path.join(DATA_DIR, f"client_{client_id}_train.csv"), label_col=ATTACK_COL)
    test_graph = csv_to_line_graph(
        os.path.join(DATA_DIR, f"client_{client_id}_test.csv"), label_col=ATTACK_COL)
    labels = train_graph.ndata["label"]
    N = train_graph.num_nodes()
    n_labeled = max(1, int(N * label_proportion))
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


@torch.no_grad()
def detailed_evaluate(model, graph, mask, device, use_personalized=False):
    model.eval()
    feat = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].cpu().numpy()
    if use_personalized:
        logits = model(graph.to(device), feat)
    else:
        _, _, logits = model(graph.to(device), feat)
    preds = logits.argmax(dim=-1).cpu().numpy()
    mask_np = mask.cpu().numpy()
    y_true = labels[mask_np]
    y_pred = preds[mask_np]

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(sklearn_f1(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class F1
    per_class_f1 = {}
    unique_labels = sorted(set(y_true) | set(y_pred))
    for cls in unique_labels:
        cls_true = (y_true == cls)
        cls_pred = (y_pred == cls)
        tp = int((cls_true & cls_pred).sum())
        fp = int((~cls_true & cls_pred).sum())
        fn = int((cls_true & ~cls_pred).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[int(cls)] = round(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, 4)

    # Group into paper categories
    grouped_f1 = {}
    for group_name, class_ids in ATTACK_GROUPS.items():
        vals = [per_class_f1.get(c, 0.0) for c in class_ids if c in per_class_f1]
        grouped_f1[group_name] = round(np.mean(vals) * 100, 2) if vals else 0.0

    overall["per_class_f1"] = per_class_f1
    overall["grouped_f1"] = grouped_f1
    return overall


def client_worker(client_id, port, result_log_dir, result_ckpt_dir, prop_val):
    log = setup_logger(f"C{client_id}-MC-P{prop_val}",
                       os.path.join(result_log_dir, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id); np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)
    log.info(f"Client {client_id} | MULTICLASS | LABEL_PROPORTION={prop_val}")

    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id, prop_val)
    log.info(f"Labeled: {train_mask.sum().item()} / {train_graph.num_nodes()}")
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

    # Final detailed evaluation with per-class F1
    final = detailed_evaluate(personal_model, test_graph, test_mask, device, True)
    with open(os.path.join(result_ckpt_dir, f"client_{client_id}_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    log.info(f"Final: Acc={final['accuracy']:.4f} F1={final['f1_macro']:.4f}")
    log.info(f"Grouped F1: {final['grouped_f1']}")

    torch.save(local_model.state_dict(), os.path.join(result_ckpt_dir, f"client_{client_id}_final.pt"))
    sock.close()


def server_worker(port, result_log_dir, result_ckpt_dir, prop_val):
    log = setup_logger(f"SRV-MC-P{prop_val}", os.path.join(result_log_dir, "server.log"))
    global_model = make_model().to(DEVICE)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [], "mean_f1_local": [], "mean_f1_pers": []}
    bmta_p = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port)); server_sock.listen(5)
    log.info(f"Multiclass LabelProp={prop_val} on port {port}")

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

    log.info(f"DONE | Multiclass LabelProp={prop_val} | BMTA={bmta_p:.4f}")
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


def run_one(prop_val, port):
    tag = str(prop_val).replace(".", "p")
    rdir = os.path.join("results", f"labelprop_{tag}_multiclass")
    rlog = os.path.join(rdir, "logs"); rckpt = os.path.join(rdir, "checkpoints")
    os.makedirs(rlog, exist_ok=True); os.makedirs(rckpt, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MULTICLASS | LABEL_PROPORTION={prop_val} | Results -> {rdir}")
    print(f"{'='*70}")
    st = threading.Thread(target=server_worker, args=(port, rlog, rckpt, prop_val), daemon=True)
    st.start(); time.sleep(3)
    cts = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, port, rlog, rckpt, prop_val))
        t.start(); cts.append(t); time.sleep(1)
    st.join()
    for t in cts: t.join(timeout=30)
    print(f"Multiclass LabelProp={prop_val} DONE!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph Multiclass Label Proportion ")
    print(f"Proportions: {PROP_VALUES}")
    print("=" * 70)

    for i, pv in enumerate(PROP_VALUES):
        tag = str(pv).replace(".", "p")
        check = os.path.join("results", f"labelprop_{tag}_multiclass", "logs", "server.log")
        if os.path.exists(check):
            with open(check) as f:
                if any("shut down" in l or "DONE" in l for l in f):
                    print(f"  Skipping prop={pv} (already done)")
                    continue
        run_one(pv, BASE_PORT + i)

    print("\n" + "=" * 70)
    print("ALL MULTICLASS EXPERIMENTS DONE!")
    print("Results in: results/labelprop_*_multiclass/")
    print("=" * 70)
