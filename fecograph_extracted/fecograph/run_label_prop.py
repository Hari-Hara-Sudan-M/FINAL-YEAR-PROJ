# =============================================================================
# Label Proportion Experiment (Paper Fig. 7)
#
# Runs FeCoGraph with different LABEL_PROPORTION values: 0.1, 0.3, 0.5, 0.7
# For BOTH binary and multiclass classification.
#
# Saves:
#   - results/labelprop_Xp/logs/          (server, client logs, history CSV)
#   - results/labelprop_Xp/checkpoints/   (model checkpoints)
#   - results/labelprop_summary.csv        (all proportions comparison)
#   - results/labelprop_fig7.png           (Paper Fig. 7 bar chart)
#
# Usage:  python run_label_prop.py
# =============================================================================

import os, sys, csv, time, socket, threading, logging, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score as sklearn_f1, accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    NUM_CLIENTS, NUM_ROUNDS, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI,
    LABEL_COL, ATTACK_COL, SEED, DATA_DIR
)
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import local_train_client, evaluate_model, server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph

PROP_VALUES = [0.1, 0.3, 0.5, 0.7]
BASE_PORT = 9970

ATTACK_NAMES_IDS2018 = {
    0: "Benign", 1: "Bot", 2: "BruteForce-Web", 3: "BruteForce-XSS",
    4: "DDoS-HOIC", 5: "DDoS-LOIC-UDP", 6: "DDoS-LOIC-HTTP",
    7: "DoS-GoldenEye", 8: "DoS-Hulk", 9: "DoS-SlowHTTPTest",
    10: "DoS-Slowloris", 11: "FTP-BruteForce", 12: "Infilteration",
    13: "Unknown-13", 14: "SSH-Bruteforce",
}

# Paper Fig. 7 groups sub-attacks into these 6 categories + Benign
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


def load_client_data(client_id, label_proportion, mode="binary"):
    label_col = ATTACK_COL if mode == "multiclass" else LABEL_COL
    train_graph = csv_to_line_graph(os.path.join(DATA_DIR, f"client_{client_id}_train.csv"), label_col=label_col)
    test_graph = csv_to_line_graph(os.path.join(DATA_DIR, f"client_{client_id}_test.csv"), label_col=label_col)
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


def make_model(mode="binary"):
    nc = NUM_CLASSES_MULTI if mode == "multiclass" else NUM_CLASSES_BINARY
    return FeCoGraphModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc)


@torch.no_grad()
def detailed_evaluate(model, graph, mask, device, use_personalized=False):
    """Evaluate and return overall + per-class metrics."""
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
        "f1": float(sklearn_f1(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class F1
    unique_labels = sorted(set(y_true) | set(y_pred))
    per_class_f1 = {}
    for cls in unique_labels:
        cls_mask = y_true == cls
        if cls_mask.sum() == 0:
            per_class_f1[int(cls)] = 0.0
        else:
            cls_pred = (y_pred == cls)
            tp = int((cls_mask & cls_pred).sum())
            fp = int((~cls_mask & cls_pred).sum())
            fn = int((cls_mask & ~cls_pred).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_f1[int(cls)] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    overall["per_class_f1"] = per_class_f1
    return overall


def client_worker(client_id, port, result_log_dir, result_ckpt_dir, prop_val, mode):
    log = setup_logger(f"C{client_id}-P{prop_val}-{mode}",
                       os.path.join(result_log_dir, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id); np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)
    log.info(f"Client {client_id} | LABEL_PROPORTION={prop_val} | mode={mode}")
    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id, prop_val, mode)
    log.info(f"Labeled: {train_mask.sum().item()} / {train_graph.num_nodes()}")
    local_model = make_model(mode).to(device)
    nc = NUM_CLASSES_MULTI if mode == "multiclass" else NUM_CLASSES_BINARY
    personal_model = PersonalizedModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc).to(device)

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

    # Final detailed eval with per-class F1
    final_metrics = detailed_evaluate(personal_model, test_graph, test_mask, device, use_personalized=True)
    metrics_path = os.path.join(result_ckpt_dir, f"client_{client_id}_detailed_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    log.info(f"Per-class F1: {final_metrics['per_class_f1']}")

    torch.save(local_model.state_dict(), os.path.join(result_ckpt_dir, f"client_{client_id}_final.pt"))
    sock.close()


def server_worker(port, result_log_dir, result_ckpt_dir, prop_val, mode):
    log = setup_logger(f"SRV-P{prop_val}-{mode}", os.path.join(result_log_dir, "server.log"))
    global_model = make_model(mode).to(DEVICE)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [], "mean_f1_local": [], "mean_f1_pers": []}
    bmta_p = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port)); server_sock.listen(5)
    log.info(f"LabelProp={prop_val} mode={mode} Server on port {port}")

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

    log.info(f"DONE | LabelProp={prop_val} mode={mode} | BMTA={bmta_p:.4f}")
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


def run_one(prop_val, port, mode="binary"):
    tag = str(prop_val).replace(".", "p")
    rdir = os.path.join("results", f"labelprop_{tag}_{mode}")
    rlog = os.path.join(rdir, "logs"); rckpt = os.path.join(rdir, "checkpoints")
    os.makedirs(rlog, exist_ok=True); os.makedirs(rckpt, exist_ok=True)

    print(f"\n{'='*70}\nLABEL_PROPORTION={prop_val} | mode={mode} | Results -> {rdir}\n{'='*70}")
    st = threading.Thread(target=server_worker, args=(port, rlog, rckpt, prop_val, mode), daemon=True)
    st.start(); time.sleep(3)
    cts = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, port, rlog, rckpt, prop_val, mode))
        t.start(); cts.append(t); time.sleep(1)
    st.join()
    for t in cts: t.join(timeout=30)
    print(f"LabelProp={prop_val} mode={mode} DONE!\n")
    return rdir


def generate_summary_and_plots(all_results):
    """Generate summary CSV and Fig. 7 bar chart from per-class F1 scores."""
    results_base = "results"
    summary_rows = []

    for (prop_val, mode), rdir in all_results.items():
        ckpt_dir = os.path.join(rdir, "checkpoints")
        log_dir = os.path.join(rdir, "logs")

        # Read per-client detailed metrics and average
        all_per_class = {}
        overall_acc, overall_f1 = [], []
        for cid in range(NUM_CLIENTS):
            mpath = os.path.join(ckpt_dir, f"client_{cid}_detailed_metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    m = json.load(f)
                overall_acc.append(m["accuracy"])
                overall_f1.append(m["f1"])
                for cls_str, f1_val in m.get("per_class_f1", {}).items():
                    cls = int(cls_str)
                    if cls not in all_per_class:
                        all_per_class[cls] = []
                    all_per_class[cls].append(f1_val)

        avg_per_class = {cls: np.mean(vals) for cls, vals in all_per_class.items()}
        avg_acc = np.mean(overall_acc) if overall_acc else 0.0
        avg_f1 = np.mean(overall_f1) if overall_f1 else 0.0

        # Read BMTA from server log history
        hist_path = os.path.join(log_dir, "training_history.csv")
        bmta = 0.0
        if os.path.exists(hist_path):
            import pandas as pd
            df = pd.read_csv(hist_path)
            bmta = df["mean_acc_pers"].max() if "mean_acc_pers" in df.columns else 0.0

        row = {
            "label_proportion": prop_val,
            "mode": mode,
            "accuracy": round(avg_acc * 100, 2),
            "f1_macro": round(avg_f1 * 100, 2),
            "bmta": round(bmta * 100, 2),
        }

        if mode == "multiclass":
            for group_name, class_ids in ATTACK_GROUPS.items():
                group_f1_vals = [avg_per_class.get(c, 0.0) for c in class_ids if c in avg_per_class]
                row[f"F1_{group_name}"] = round(np.mean(group_f1_vals) * 100, 2) if group_f1_vals else 0.0
        else:
            row["F1_Benign"] = round(avg_per_class.get(0, 0.0) * 100, 2)
            row["F1_Malicious"] = round(avg_per_class.get(1, 0.0) * 100, 2)

        summary_rows.append(row)

    # Save summary CSV
    summary_path = os.path.join(results_base, "labelprop_summary.csv")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in summary_rows: w.writerow(r)
        print(f"\nSummary saved: {summary_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("LABEL PROPORTION RESULTS SUMMARY")
    print("=" * 90)
    for r in summary_rows:
        print(f"  Prop={r['label_proportion']} mode={r['mode']} | "
              f"Acc={r['accuracy']}% | F1={r['f1_macro']}% | BMTA={r['bmta']}%")
        for k, v in r.items():
            if k.startswith("F1_"):
                print(f"    {k}: {v}%")
    print("=" * 90)

    # Generate Fig. 7 bar chart (multiclass only)
    multi_rows = [r for r in summary_rows if r["mode"] == "multiclass"]
    if multi_rows:
        categories = list(ATTACK_GROUPS.keys())
        props = [r["label_proportion"] for r in multi_rows]
        n_props = len(props)
        n_cats = len(categories)
        x = np.arange(n_cats)
        width = 0.8 / n_props
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, r in enumerate(multi_rows):
            vals = [r.get(f"F1_{cat}", 0.0) for cat in categories]
            bars = ax.bar(x + i * width - (n_props - 1) * width / 2, vals,
                          width, label=f"{int(r['label_proportion']*100)}% labeled",
                          color=colors[i % len(colors)])
        ax.set_xlabel("Attack Category", fontsize=12)
        ax.set_ylabel("F1-score (%)", fontsize=12)
        ax.set_title("Performance with Different Label Proportions (Paper Fig. 7a)", fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.legend(fontsize=10); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)

        # Add value table below
        table_data = []
        for r in multi_rows:
            row_data = [f"{r.get(f'F1_{cat}', 0.0):.1f}" for cat in categories]
            table_data.append(row_data)
        row_labels = [f"{int(r['label_proportion']*100)}% labeled" for r in multi_rows]
        table = ax.table(cellText=table_data, rowLabels=row_labels,
                         colLabels=categories, loc="bottom", bbox=[0, -0.45, 1, 0.3])
        table.auto_set_font_size(False); table.set_fontsize(8)
        fig.subplots_adjust(bottom=0.35)

        fig_path = os.path.join(results_base, "labelprop_fig7.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Fig. 7 chart saved: {fig_path}")
        plt.close(fig)


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph Label Proportion Experiment (Paper Fig. 7)")
    print(f"Proportions: {PROP_VALUES}")
    print(f"Modes: binary + multiclass")
    print("=" * 70)

    all_results = {}
    port_idx = 0

    # Run binary for all proportions
    for pv in PROP_VALUES:
        rdir = run_one(pv, BASE_PORT + port_idx, mode="binary")
        all_results[(pv, "binary")] = rdir
        port_idx += 1

    # Run multiclass for all proportions (for Fig. 7 per-attack F1)
    for pv in PROP_VALUES:
        rdir = run_one(pv, BASE_PORT + port_idx, mode="multiclass")
        all_results[(pv, "multiclass")] = rdir
        port_idx += 1

    generate_summary_and_plots(all_results)

    print("\n" + "=" * 70)
    print("ALL LABEL PROPORTION EXPERIMENTS DONE!")
    print("Results in: results/labelprop_*/")
    print("=" * 70)
