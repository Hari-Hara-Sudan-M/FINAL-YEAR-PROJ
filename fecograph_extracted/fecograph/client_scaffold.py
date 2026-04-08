# =============================================================================
# SCAFFOLD FL Client (Novelty - separate terminal version)
#
# Terminal 1:  python server_scaffold.py
# Terminal 2:  python client_scaffold.py --client_id 0
# Terminal 3:  python client_scaffold.py --client_id 1
#
# Key difference from original client.py (FedAvg/Ditto):
#   FedAvg:    w_k ← w_k - η·∇F_k(w_k)                    [drifts on non-IID]
#   SCAFFOLD:  w_k ← w_k - η·(∇F_k(w_k) - c_k + c)        [corrected gradient]
#
# Each client maintains a control variate c_k that tracks gradient drift.
# Server sends global control c. The correction (-c_k + c) removes
# client-specific bias from the gradient before each update.
# =============================================================================

import os, sys, time, socket, logging, argparse
from datetime import datetime
import numpy as np, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    SERVER_HOST, SERVER_PORT, DATA_DIR, DEVICE,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI, CLASSIFICATION_MODE,
    LABEL_PROPORTION, LABEL_COL, ATTACK_COL,
    LOCAL_LR, LOCAL_EPOCHS, LAMBDA_CE, TEMPERATURE, BATCH_SIZE, SEED,
    CONTRASTIVE_MODE,
)
from models.model import FeCoGraphModel, PersonalizedModel
from fl.federated import (
    evaluate_model,
    compute_class_weights,
    classification_task_step,
    contrastive_task_step,
)
from fl.comm import send_object, recv_object, make_message, MsgType
from data.line_graph import csv_to_line_graph
from data.augmentation import augment_graph

RESULT_DIR = os.path.join("results", f"scaffold_{CLASSIFICATION_MODE}_{CONTRASTIVE_MODE}")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)
RUN_ID = os.environ.get("SCAFFOLD_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

parser = argparse.ArgumentParser(description="SCAFFOLD FL Client")
parser.add_argument("--client_id", type=int, required=True)
parser.add_argument("--server_ip", type=str, default=SERVER_HOST)
parser.add_argument("--server_port", type=int, default=SERVER_PORT)
args = parser.parse_args()
CLIENT_LOG_PATH = os.path.join(RESULT_LOG_DIR, f"client_{args.client_id}_{RUN_ID}.log")

logging.basicConfig(
    level=logging.INFO,
    format=f"[CLIENT-{args.client_id}-SCAFFOLD] %(message)s",
    handlers=[
        logging.FileHandler(CLIENT_LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(f"scaffold.client{args.client_id}")


def make_model():
    nc = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
    return FeCoGraphModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc)


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


def scaffold_local_train(model, graph, train_mask, device, c_local, c_global, contrastive_mode):
    """
    SCAFFOLD local training with variance-reduced gradients.
    Uses canonical correction: g_corrected = ∇F(w) - c_local + c_global.
    """
    model.to(device); model.train()
    w_t = {n: p.clone().detach() for n, p in model.named_parameters()}

    labels = graph.ndata["label"].to(device)
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).to(device)
    num_classes = int(labels.max().item()) + 1
    class_weights = compute_class_weights(labels[train_indices], num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)

    G1, G2 = augment_graph(graph)
    metrics = {"total": [], "supcon": [], "ce": []}

    local_steps = 0
    for epoch in range(LOCAL_EPOCHS):
        optimizer.zero_grad()

        loss, l_con, l_ce = contrastive_task_step(
            model=model,
            graph=graph,
            G1=G1,
            G2=G2,
            train_indices=train_indices,
            labels=labels,
            class_weights=class_weights,
            device=device,
            lambda_ce=LAMBDA_CE,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            contrastive_mode=contrastive_mode,
        )

        if loss <= 0.0:
            continue

        # Canonical SCAFFOLD correction: grad <- grad - c_k + c
        for name, param in model.named_parameters():
            if param.grad is not None:
                correction = (-c_local[name].to(device) + c_global[name].to(device))
                param.grad.data.add_(correction)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        local_steps += 1

        metrics["total"].append(loss)
        metrics["supcon"].append(l_con)
        metrics["ce"].append(l_ce)

    # Delta = w_k - w_t
    delta = {n: p.data.cpu() - w_t[n].cpu() for n, p in model.named_parameters()}

    # Canonical SCAFFOLD Option-II control variate update:
    # c_k_new = c_k - c + (w_t - w_k) / (eta * K)
    c_local_new = {}
    K = max(1, local_steps)
    for name in c_local:
        c_local_new[name] = (
            c_local[name].cpu()
            - c_global[name].cpu()
            + (w_t[name].cpu() - model.state_dict()[name].cpu()) / (LOCAL_LR * K)
        )
    delta_c = {n: c_local_new[n] - c_local[n] for n in c_local}

    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return delta, delta_c, c_local_new, avg_metrics


def main():
    log.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Run ID: {RUN_ID}")
    log.info(f"Log file: {CLIENT_LOG_PATH}")
    torch.manual_seed(SEED + args.client_id)
    np.random.seed(SEED + args.client_id)
    device = torch.device(DEVICE)

    log.info("=" * 60)
    log.info(f"FeCoGraph SCAFFOLD Client {args.client_id}")
    log.info(f"Contrastive mode: {CONTRASTIVE_MODE}")
    log.info(f"Device: {device}")
    log.info("=" * 60)

    train_graph, test_graph, train_mask, test_mask = load_client_data(args.client_id)
    log.info(f"Train: {train_graph.num_nodes():,} | Test: {test_graph.num_nodes():,} | "
             f"Labeled: {train_mask.sum().item()}")

    local_model = make_model().to(device)
    nc = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
    personal_model = PersonalizedModel(
        in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc
    ).to(device)
    c_local = {n: torch.zeros_like(p.data) for n, p in local_model.named_parameters()}

    connect_ip = "127.0.0.1" if args.server_ip == "0.0.0.0" else args.server_ip
    log.info(f"Connecting to server {connect_ip}:{args.server_port}...")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((connect_ip, args.server_port))
            break
        except (ConnectionRefusedError, OSError):
            log.info("  Server not ready, retrying in 3s...")
            time.sleep(3)

    send_object(sock, make_message(MsgType.REGISTER, payload={"client_id": args.client_id}))
    log.info("Connected and registered.")

    running = True
    while running:
        msg = recv_object(sock)
        if msg is None:
            log.warning("Lost connection to server."); break
        if msg["type"] == MsgType.SHUTDOWN:
            log.info("Received SHUTDOWN. Training complete."); break
        if msg["type"] == MsgType.ROUND_START:
            round_num = msg["round"]
            payload = msg["payload"]
            global_sd = payload["model_state"]
            c_global = payload["c_global"]

            local_model.load_state_dict(global_sd, strict=True)
            local_model.to(device)

            t0 = time.time()
            delta, delta_c, c_local, train_metrics = scaffold_local_train(
                local_model, train_graph, train_mask, device, c_local, c_global, CONTRASTIVE_MODE
            )

            # Personalized model update (same as Ditto Eq 15-16)
            labels = train_graph.ndata["label"].to(device)
            train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).to(device)
            num_classes = int(labels.max().item()) + 1
            cw = compute_class_weights(labels[train_indices], num_classes).to(device)
            personal_model.to(device); personal_model.train()
            classification_task_step(
                personalized_model=personal_model,
                global_model=local_model,
                graph=train_graph,
                train_indices=train_indices,
                labels=labels,
                class_weights=cw,
                device=device,
                batch_size=BATCH_SIZE,
            )

            test_local = evaluate_model(local_model, test_graph, test_mask, device, False)
            test_pers = evaluate_model(personal_model, test_graph, test_mask, device, True)
            elapsed = time.time() - t0

            log.info(f"  Round {round_num+1} | "
                     f"Loss={train_metrics['total']:.4f} | "
                     f"Acc(G)={test_local['accuracy']:.4f} | "
                     f"Acc(P)={test_pers['accuracy']:.4f} | "
                     f"F1(P)={test_pers['f1']:.4f} | "
                     f"Time={elapsed:.1f}s")

            send_object(sock, make_message(MsgType.CLIENT_DELTA, payload={
                "delta": delta,
                "delta_c": delta_c,
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
               os.path.join(RESULT_CKPT_DIR, f"client_{args.client_id}_local_final.pt"))
    torch.save(personal_model.state_dict(),
               os.path.join(RESULT_CKPT_DIR, f"client_{args.client_id}_personal_final.pt"))
    sock.close()
    log.info("Client done.")


if __name__ == "__main__":
    main()
