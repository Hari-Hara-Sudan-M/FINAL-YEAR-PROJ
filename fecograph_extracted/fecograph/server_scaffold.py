# =============================================================================
# SCAFFOLD FL Server (Novelty - separate terminal version)
#
# Terminal 1:  python server_scaffold.py
# Terminal 2:  python client_scaffold.py --client_id 0
# Terminal 3:  python client_scaffold.py --client_id 1
#
# Results saved to: results/scaffold_binary/
# =============================================================================

import os, sys, csv, time, socket, logging
from datetime import datetime
import numpy as np, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    SERVER_HOST, SERVER_PORT, NUM_CLIENTS, NUM_ROUNDS,
    CLIENT_FRACTION, DEVICE, LOG_EVERY,
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, CLASSIFICATION_MODE, NUM_CLASSES_MULTI,
    CONTRASTIVE_MODE,
)
from models.model import FeCoGraphModel
from fl.federated import server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType

RESULT_DIR = os.path.join("results", f"scaffold_{CLASSIFICATION_MODE}_{CONTRASTIVE_MODE}")
RESULT_LOG_DIR = os.path.join(RESULT_DIR, "logs")
RESULT_CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(RESULT_LOG_DIR, exist_ok=True)
os.makedirs(RESULT_CKPT_DIR, exist_ok=True)
RUN_ID = os.environ.get("SCAFFOLD_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
SERVER_LOG_PATH = os.path.join(RESULT_LOG_DIR, f"server_{RUN_ID}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[SERVER-SCAFFOLD] %(message)s",
    handlers=[
        logging.FileHandler(SERVER_LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("scaffold.server")


def make_model():
    nc = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
    return FeCoGraphModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=nc)


def init_control_variate(model):
    return {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}


def main():
    log.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Run ID: {RUN_ID}")
    log.info(f"Log file: {SERVER_LOG_PATH}")
    log.info(f"Contrastive mode: {CONTRASTIVE_MODE}")
    global_model = make_model().to(DEVICE)
    c_global = init_control_variate(global_model)
    client_conns = {}
    history = {"round": [], "loss_supcon": [], "loss_ce": [],
               "mean_acc_local": [], "mean_acc_pers": [],
               "mean_f1_local": [], "mean_f1_pers": []}
    bmta = 0.0
    bmta_pers = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((SERVER_HOST, SERVER_PORT))
    server_sock.listen(NUM_CLIENTS + 2)

    log.info(f"SCAFFOLD Server listening on {SERVER_HOST}:{SERVER_PORT}")
    log.info(f"Waiting for {NUM_CLIENTS} clients...")

    while len(client_conns) < NUM_CLIENTS:
        conn, addr = server_sock.accept()
        msg = recv_object(conn)
        if msg and msg["type"] == MsgType.REGISTER:
            cid = msg["payload"]["client_id"]
            client_conns[cid] = conn
            log.info(f"  Client {cid} registered from {addr}")

    log.info(f"All {NUM_CLIENTS} clients connected. Starting SCAFFOLD training...")
    log.info("=" * 65)

    for round_t in range(NUM_ROUNDS):
        if not client_conns:
            log.error("No clients remaining."); break

        t_start = time.time()
        selected = list(client_conns.keys())

        # Send global model + global control variate c to each client
        global_sd = {k: v.cpu() for k, v in global_model.state_dict().items()}
        payload = {"model_state": global_sd, "c_global": c_global}

        failed = []
        for cid in selected:
            try:
                send_object(client_conns[cid],
                            make_message(MsgType.ROUND_START, payload=payload, round_num=round_t))
            except Exception as e:
                log.warning(f"  Send failed to client {cid}: {e}")
                failed.append(cid)
        selected = [c for c in selected if c not in failed]

        # Receive deltas + control variate deltas from clients
        client_deltas, client_delta_cs, client_metrics = [], [], []
        for cid in selected:
            try:
                resp = recv_object(client_conns[cid])
                if resp and resp["type"] == MsgType.CLIENT_DELTA:
                    client_deltas.append(resp["payload"]["delta"])
                    client_delta_cs.append(resp["payload"]["delta_c"])
                    client_metrics.append(resp["payload"]["metrics"])
            except Exception as e:
                log.warning(f"  Recv failed from client {cid}: {e}")

        if client_deltas:
            # Standard model aggregation: w_{t+1} = w_t + (1/|S|) * Σ Δ_k
            global_model = server_aggregate(global_model, client_deltas, selected)

            # SCAFFOLD control variate update: c = c + (1/N) * Σ Δc_k
            for name in c_global:
                delta_c_avg = torch.stack([dc[name] for dc in client_delta_cs]).mean(dim=0)
                c_global[name] = c_global[name] + delta_c_avg

        elapsed = time.time() - t_start
        if client_metrics:
            ma_l = float(np.mean([m["test_acc_local"] for m in client_metrics]))
            mf_l = float(np.mean([m["test_f1_local"] for m in client_metrics]))
            ma_p = float(np.mean([m["test_acc_pers"] for m in client_metrics]))
            mf_p = float(np.mean([m["test_f1_pers"] for m in client_metrics]))
            sc = float(np.mean([m.get("supcon", 0) for m in client_metrics]))
            ce = float(np.mean([m.get("ce", 0) for m in client_metrics]))

            if ma_l > bmta:
                bmta = ma_l
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(),
                            "bmta": bmta, "c_global": c_global},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_local.pt"))
            if ma_p > bmta_pers:
                bmta_pers = ma_p
                torch.save({"round": round_t+1, "model_state": global_model.state_dict(),
                            "bmta_pers": bmta_pers, "c_global": c_global},
                           os.path.join(RESULT_CKPT_DIR, "global_model_best_pers.pt"))

            history["round"].append(round_t+1)
            history["loss_supcon"].append(sc); history["loss_ce"].append(ce)
            history["mean_acc_local"].append(ma_l); history["mean_acc_pers"].append(ma_p)
            history["mean_f1_local"].append(mf_l); history["mean_f1_pers"].append(mf_p)

            if (round_t+1) % LOG_EVERY == 0 or round_t == NUM_ROUNDS-1:
                log.info(f"Round {round_t+1:>3}/{NUM_ROUNDS} | "
                         f"SupCon={sc:.4f} | CE={ce:.4f} | "
                         f"Acc_L={ma_l:.4f} | Acc_P={ma_p:.4f} | "
                         f"F1_P={mf_p:.4f} | BMTA_P={bmta_pers:.4f} | {elapsed:.1f}s")

    log.info("=" * 65)
    log.info("SCAFFOLD + PERSONALIZATION TRAINING COMPLETE")
    log.info(f"  BMTA (Local):        {bmta:.4f}")
    log.info(f"  BMTA (Personalized): {bmta_pers:.4f}")
    torch.save({"round": NUM_ROUNDS, "model_state": global_model.state_dict(),
                "bmta": bmta, "c_global": c_global},
               os.path.join(RESULT_CKPT_DIR, "global_model_final.pt"))
    log.info(f"  Final checkpoint saved.")

    hp = os.path.join(RESULT_LOG_DIR, f"training_history_{RUN_ID}.csv")
    if history["round"]:
        with open(hp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history.keys())); w.writeheader()
            for i in range(len(history["round"])): w.writerow({k: history[k][i] for k in history})
    log.info(f"  History saved: {hp}")

    for cid, conn in client_conns.items():
        try: send_object(conn, make_message(MsgType.SHUTDOWN)); conn.close()
        except: pass
    server_sock.close()
    log.info("Server shut down cleanly.")


if __name__ == "__main__":
    main()
