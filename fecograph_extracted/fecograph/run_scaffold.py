# =============================================================================
# NOVELTY 1: SCAFFOLD Federated Learning
#
# SCAFFOLD (Stochastic Controlled Averaging for FL) corrects client drift
# using control variates. Provably faster convergence than FedAvg on
# non-IID data (Karimireddy et al., ICML 2020).
#
# Key difference from paper's FedAvg (Eq 13-14) and Ditto (Eq 15-16):
#   FedAvg:    w_k ← w_k - η·∇F_k(w_k)                    [drifts on non-IID]
#   Ditto:     θ_k ← θ_k - η·(∇f_k(θ_k) + μ(θ_k - w*))   [post-hoc fix]
#   SCAFFOLD:  w_k ← w_k - η·(∇F_k(w_k) - c_k + c)        [corrected gradient]
#
# Each client maintains a control variate c_k that tracks its gradient
# drift from the global average. The server maintains c (global control).
# The corrected gradient (∇F - c_k + c) removes client-specific bias,
# making updates more aligned with the true global objective.
#
# Results saved to: results/scaffold_binary/
#
# Usage:  python run_scaffold.py
# =============================================================================

import os, sys, csv, copy, time, socket, threading, logging
import numpy as np, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    NUM_CLIENTS, NUM_ROUNDS, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, CLASSIFICATION_MODE,
    LABEL_PROPORTION, LABEL_COL, ATTACK_COL,
    LOCAL_LR, LOCAL_EPOCHS, LAMBDA_CE, TEMPERATURE, BATCH_SIZE,
    SEED, DATA_DIR
)
from models.model import FeCoGraphModel, PersonalizedModel
from models.losses import SupervisedContrastiveLoss, ClassificationLoss
from fl.federated import evaluate_model, server_aggregate, compute_class_weights
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph
from data.augmentation import augment_graph

RESULT_DIR = os.path.join("results", "scaffold_binary")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)

PORT = 9955


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
    label_col = ATTACK_COL if CLASSIFICATION_MODE == "multiclass" else LABEL_COL
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
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NUM_CLASSES_BINARY)


def init_control_variate(model):
    """Initialize control variate c as zeros (same shape as model params)."""
    return {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}


def scaffold_local_train(global_model, graph, train_mask, device,
                         c_local, c_global):
    """
    SCAFFOLD local training with variance-reduced gradients.

    Instead of: w ← w - η·∇F(w)                     (FedAvg, Eq 13)
    We use:     w ← w - η·(∇F(w) - c_local + c_global)  (SCAFFOLD)

    This corrects the gradient direction by subtracting client-specific
    bias (c_local) and adding global correction (c_global).
    """
    global_model.to(device)
    global_model.train()

    w_t = {n: p.clone().detach() for n, p in global_model.named_parameters()}

    labels = graph.ndata["label"].to(device)
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).to(device)
    num_classes = int(labels.max().item()) + 1
    class_weights = compute_class_weights(labels[train_indices], num_classes).to(device)

    con_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    ce_fn = ClassificationLoss(class_weights=class_weights)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=LOCAL_LR)

    G1, G2 = augment_graph(graph)

    metrics = {"total": [], "supcon": [], "ce": []}

    local_steps = 0
    for epoch in range(LOCAL_EPOCHS):
        optimizer.zero_grad()

        _, z1, _ = global_model(G1.to(device), G1.ndata["feat"].to(device))
        _, z2, _ = global_model(G2.to(device), G2.ndata["feat"].to(device))
        _, _, y = global_model(graph.to(device), graph.ndata["feat"].to(device))

        perm = torch.randperm(len(train_indices), device=device)
        b_idx = train_indices[perm[:min(BATCH_SIZE, len(train_indices))]]
        b_lbl = labels[b_idx]

        if len(b_lbl) < 2:
            continue

        l_con = con_fn(z1[b_idx], z2[b_idx], b_lbl)
        l_ce = ce_fn(y[b_idx], b_lbl)
        loss = (1.0 - LAMBDA_CE) * l_con + LAMBDA_CE * l_ce
        loss.backward()

        for name, param in global_model.named_parameters():
            if param.grad is not None:
                correction = (-c_local[name].to(device) + c_global[name].to(device))
                param.grad.data.add_(correction)

        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=5.0)
        optimizer.step()
        local_steps += 1

        metrics["total"].append(loss.item())
        metrics["supcon"].append(l_con.item())
        metrics["ce"].append(l_ce.item())

    delta = {n: p.data.cpu() - w_t[n].cpu() for n, p in global_model.named_parameters()}

    c_local_new = {}
    K = max(1, local_steps)
    for name in c_local:
        c_local_new[name] = (
            c_local[name].cpu()
            - c_global[name].cpu()
            + (w_t[name].cpu() - global_model.state_dict()[name].cpu()) / (LOCAL_LR * K)
        )

    delta_c = {n: c_local_new[n] - c_local[n] for n in c_local}

    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return delta, delta_c, c_local_new, avg_metrics


def client_worker(client_id, port):
    log = setup_logger(f"C{client_id}-SCAFFOLD",
                       os.path.join(RESULT_LOG_DIR, f"client_{client_id}.log"))
    torch.manual_seed(SEED + client_id); np.random.seed(SEED + client_id)
    device = torch.device(DEVICE)

    log.info("=" * 60)
    log.info(f"FeCoGraph Client {client_id} (SCAFFOLD FL)")
    log.info("=" * 60)

    train_graph, test_graph, train_mask, test_mask = load_client_data(client_id)
    log.info(f"Labeled: {train_mask.sum().item()} / {train_graph.num_nodes()}")

    local_model = make_model().to(device)
    c_local = init_control_variate(local_model)

    time.sleep(2)
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port)); break
        except (ConnectionRefusedError, OSError): time.sleep(2)
    send_object(sock, make_message(MsgType.REGISTER, payload={"client_id": client_id}))
    log.info("Connected.")

    while True:
        msg = recv_object(sock)
        if msg is None or msg["type"] == MsgType.SHUTDOWN: break
        if msg["type"] == MsgType.ROUND_START:
            round_num = msg["round"]
            payload = msg["payload"]
            global_sd = payload["model_state"]
            c_global = payload["c_global"]

            local_model.load_state_dict(global_sd, strict=True)
            local_model.to(device)

            t0 = time.time()
            delta, delta_c, c_local, train_metrics = scaffold_local_train(
                local_model, train_graph, train_mask, device,
                c_local, c_global
            )

            test_m = evaluate_model(local_model, test_graph, test_mask, device, False)
            elapsed = time.time() - t0
            log.info(f"  R{round_num+1} Acc={test_m['accuracy']:.4f} F1={test_m['f1']:.4f} | {elapsed:.1f}s")

            send_object(sock, make_message(MsgType.CLIENT_DELTA, payload={
                "delta": delta,
                "delta_c": delta_c,
                "metrics": {
                    **train_metrics,
                    "test_acc_local": test_m["accuracy"],
                    "test_f1_local": test_m["f1"],
                    "test_acc_pers": test_m["accuracy"],
                    "test_f1_pers": test_m["f1"],
                    "test_precision": test_m["precision"],
                    "test_recall": test_m["recall"],
                }
            }, round_num=round_num))

    torch.save(local_model.state_dict(),
               os.path.join(RESULT_CKPT_DIR, f"client_{client_id}_final.pt"))
    sock.close(); log.info("Done.")


def server_worker(port):
    log = setup_logger("SRV-SCAFFOLD", os.path.join(RESULT_LOG_DIR, "server.log"))

    global_model = make_model().to(DEVICE)
    c_global = init_control_variate(global_model)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [],
               "mean_f1_local": [], "mean_f1_pers": []}
    bmta = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port)); server_sock.listen(5)
    log.info(f"SCAFFOLD Server on port {port}")

    while len(client_conns) < NUM_CLIENTS:
        conn, addr = server_sock.accept()
        msg = recv_object(conn)
        if msg and msg["type"] == MsgType.REGISTER:
            client_conns[msg["payload"]["client_id"]] = conn
            log.info(f"  Client {msg['payload']['client_id']} connected")
    log.info("All clients connected. Starting SCAFFOLD training...")
    log.info("=" * 65)

    for round_t in range(NUM_ROUNDS):
        if not client_conns: break
        t_start = time.time()
        selected = list(client_conns.keys())

        # Send global model + global control variate c to clients
        global_sd = {k: v.cpu() for k, v in global_model.state_dict().items()}
        payload = {"model_state": global_sd, "c_global": c_global}

        for cid in selected:
            try:
                send_object(client_conns[cid],
                            make_message(MsgType.ROUND_START, payload=payload, round_num=round_t))
            except: pass

        client_deltas, client_delta_cs, client_metrics = [], [], []
        for cid in selected:
            try:
                resp = recv_object(client_conns[cid])
                if resp and resp["type"] == MsgType.CLIENT_DELTA:
                    client_deltas.append(resp["payload"]["delta"])
                    client_delta_cs.append(resp["payload"]["delta_c"])
                    client_metrics.append(resp["payload"]["metrics"])
            except: pass

        if client_deltas:
            # Standard aggregation: w_{t+1} = w_t + (1/|S|) * Σ Δ_k
            global_model = server_aggregate(global_model, client_deltas, selected)

            # SCAFFOLD server update: c = c + (1/N) * Σ Δc_k
            for name in c_global:
                delta_c_sum = torch.stack([dc[name] for dc in client_delta_cs]).mean(dim=0)
                c_global[name] = c_global[name] + delta_c_sum

        elapsed = time.time() - t_start
        if client_metrics:
            ma = float(np.mean([m["test_acc_local"] for m in client_metrics]))
            mf = float(np.mean([m["test_f1_local"] for m in client_metrics]))
            sc = float(np.mean([m.get("supcon", 0) for m in client_metrics]))
            ce = float(np.mean([m.get("ce", 0) for m in client_metrics]))

            if ma > bmta:
                bmta = ma
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(),
                            "bmta": bmta, "c_global": c_global},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_local.pt"))

            history["round"].append(round_t+1); history["loss_supcon"].append(sc)
            history["loss_ce"].append(ce); history["mean_acc_local"].append(ma)
            history["mean_acc_pers"].append(ma); history["mean_f1_local"].append(mf)
            history["mean_f1_pers"].append(mf)

            if (round_t+1) % LOG_EVERY == 0 or round_t == NUM_ROUNDS-1:
                log.info(f"R{round_t+1:>3}/{NUM_ROUNDS} | SupCon={sc:.4f} CE={ce:.4f} | "
                         f"Acc={ma:.4f} F1={mf:.4f} BMTA={bmta:.4f} | {elapsed:.1f}s")

    log.info("=" * 65)
    log.info(f"SCAFFOLD TRAINING COMPLETE | BMTA={bmta:.4f}")
    torch.save({"round": NUM_ROUNDS, "model_state": global_model.state_dict(),
                "bmta": bmta, "c_global": c_global},
               os.path.join(RESULT_CKPT_DIR, "global_model_final.pt"))
    hp = os.path.join(RESULT_LOG_DIR, "training_history.csv")
    if history["round"]:
        with open(hp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history.keys())); w.writeheader()
            for i in range(len(history["round"])): w.writerow({k: history[k][i] for k in history})
    log.info(f"  History saved: {hp}")
    for c in client_conns.values():
        try: send_object(c, make_message(MsgType.SHUTDOWN)); c.close()
        except: pass
    server_sock.close(); log.info("Server shut down.")


if __name__ == "__main__":
    print("=" * 70)
    print("NOVELTY: FeCoGraph + SCAFFOLD FL")
    print("SCAFFOLD corrects client drift via control variates")
    print(f"Results -> {RESULT_DIR}")
    print("=" * 70)

    st = threading.Thread(target=server_worker, args=(PORT,), daemon=True)
    st.start(); time.sleep(3)
    threads = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(cid, PORT))
        t.start(); threads.append(t); time.sleep(1)
    st.join()
    for t in threads: t.join(timeout=30)
    print(f"\nSCAFFOLD experiment DONE! Results in: {RESULT_DIR}")
