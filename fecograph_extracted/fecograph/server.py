

import os
import sys
import time
import socket
import logging
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    SERVER_HOST, SERVER_PORT, NUM_CLIENTS, NUM_ROUNDS,
    CLIENT_FRACTION, DEVICE, LOG_DIR, CHECKPOINT_DIR,
    LOG_EVERY, FL_SCHEME, INPUT_FEATURE_DIM,
    GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI, CLASSIFICATION_MODE
)
from models.model import FeCoGraphModel
from fl.federated import server_aggregate
from fl.comm import send_object, recv_object, make_message, MsgType

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("FeCoGraph.server")



class FLServer:
    def __init__(self):
        num_classes = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
        self.global_model = FeCoGraphModel(
            in_dim=INPUT_FEATURE_DIM,
            hidden_dim=GCN_HIDDEN_DIM_1,
            embed_dim=GCN_HIDDEN_DIM_2,
            proj_hidden=PROJECTOR_HIDDEN_DIM,
            proj_out=PROJECTOR_OUTPUT_DIM,
            num_classes=num_classes
        ).to(DEVICE)

        self.client_conns = {}   
        self.history = {
            "round": [],
            "loss_supcon": [], "loss_ce": [],
            "mean_acc_local": [], "mean_acc_pers": [],
            "mean_f1_local":  [], "mean_f1_pers":  [],
        }

        self.bmta_local = 0.0
        self.bmta_pers  = 0.0

    def start(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((SERVER_HOST, SERVER_PORT))
        server_sock.listen(NUM_CLIENTS + 2)

        log.info(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")
        log.info(f"Waiting for {NUM_CLIENTS} clients to connect...")

        while len(self.client_conns) < NUM_CLIENTS:
            conn, addr = server_sock.accept()
            msg = recv_object(conn)
            if msg and msg["type"] == MsgType.REGISTER:
                cid = msg["payload"]["client_id"]
                self.client_conns[cid] = conn
                log.info(f"  Client {cid} registered from {addr}")

        log.info(f"All {NUM_CLIENTS} clients connected.")
        log.info(f"FL Scheme : {FL_SCHEME.upper()}")
        log.info(f"Rounds    : {NUM_ROUNDS}")
        log.info(f"Weight transfer: Server sends w_t → Clients send Delta_k back")
        log.info("=" * 65)

        # ── Main FL loop — Algorithm 1 ────────────────────────────────────────
        for round_t in range(NUM_ROUNDS):
            if not self.client_conns:
                log.error("No clients remaining. Stopping.")
                break

            t_start = time.time()

            # Line 2: Select subset S_t of clients
            n_select = max(1, int(len(self.client_conns) * CLIENT_FRACTION))
            selected = list(self.client_conns.keys())[:n_select]

            # Serialize global model w_t to send to clients
            # Only the global model (w_k) weights are transferred — not theta_k
            global_sd = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

            # Send w_t to each selected client
            failed_send = []
            for cid in selected:
                try:
                    msg = make_message(MsgType.ROUND_START,
                                       payload=global_sd,
                                       round_num=round_t)
                    send_object(self.client_conns[cid], msg)
                except Exception as e:
                    log.warning(f"  Send failed to client {cid}: {e}")
                    failed_send.append(cid)

            selected = [c for c in selected if c not in failed_send]
            self._remove_clients(failed_send)
            if not selected:
                log.warning("  No clients available this round, skipping.")
                continue

            # Receive Delta_k from each client (Algorithm 1, line 11)
            client_deltas  = []
            client_metrics = []
            failed_recv    = []

            for cid in selected:
                try:
                    response = recv_object(self.client_conns[cid])
                    if response and response["type"] == MsgType.CLIENT_DELTA:
                        client_deltas.append(response["payload"]["delta"])
                        client_metrics.append(response["payload"]["metrics"])
                    else:
                        log.warning(f"  Client {cid} sent unexpected msg, skipping.")
                        failed_recv.append(cid)
                except Exception as e:
                    log.warning(f"  Client {cid} recv error round {round_t+1}: {e}")
                    failed_recv.append(cid)

            self._remove_clients(failed_recv)

            # Lines 13-14: FedAvg aggregation (Eq.14)
            # w^{t+1} = w^t + (1/|S_t|) * sum(Delta_k)
            if client_deltas:
                self.global_model = server_aggregate(
                    self.global_model, client_deltas, selected
                )

            elapsed = time.time() - t_start

            if client_metrics:
                acc_local = [m.get("test_acc_local", 0.0) for m in client_metrics]
                f1_local  = [m.get("test_f1_local",  0.0) for m in client_metrics]

                acc_pers  = [m.get("test_acc_pers",  0.0) for m in client_metrics]
                f1_pers   = [m.get("test_f1_pers",   0.0) for m in client_metrics]

                mean_acc_local = float(np.mean(acc_local))
                mean_acc_pers  = float(np.mean(acc_pers))
                mean_f1_local  = float(np.mean(f1_local))
                mean_f1_pers   = float(np.mean(f1_pers))

                avg_supcon = float(np.mean([m.get("supcon", 0) for m in client_metrics]))
                avg_ce     = float(np.mean([m.get("ce",     0) for m in client_metrics]))

                # Update BMTA (paper primary metric)
                if mean_acc_local > self.bmta_local:
                    self.bmta_local = mean_acc_local
                    self._save_checkpoint(round_t, "best_local")
                if mean_acc_pers > self.bmta_pers:
                    self.bmta_pers = mean_acc_pers
                    self._save_checkpoint(round_t, "best_pers")

                # Store history
                self.history["round"].append(round_t + 1)
                self.history["loss_supcon"].append(avg_supcon)
                self.history["loss_ce"].append(avg_ce)
                self.history["mean_acc_local"].append(mean_acc_local)
                self.history["mean_acc_pers"].append(mean_acc_pers)
                self.history["mean_f1_local"].append(mean_f1_local)
                self.history["mean_f1_pers"].append(mean_f1_pers)

                if (round_t + 1) % LOG_EVERY == 0 or round_t == NUM_ROUNDS - 1:
                    log.info(
                        f"Round {round_t+1:>3}/{NUM_ROUNDS} | "
                        f"Clients={len(client_deltas)} | "
                        f"SupCon={avg_supcon:.4f} | CE={avg_ce:.4f} | "
                        f"Acc_local={mean_acc_local:.4f} | F1_local={mean_f1_local:.4f} | "
                        f"Acc_pers={mean_acc_pers:.4f} | F1_pers={mean_f1_pers:.4f} | "
                        f"BMTA_pers={self.bmta_pers:.4f} | "
                        f"Time={elapsed:.1f}s"
                    )
            else:
                if (round_t + 1) % LOG_EVERY == 0:
                    log.info(f"Round {round_t+1:>3}/{NUM_ROUNDS} | No metrics | Time={elapsed:.1f}s")

        log.info("=" * 65)
        log.info("TRAINING COMPLETE")
        log.info(f"  BMTA (Local model)       : {self.bmta_local:.4f}")
        log.info(f"  BMTA (Personalized model): {self.bmta_pers:.4f}")
        self._save_checkpoint(NUM_ROUNDS - 1, "final")
        self._save_history()

        # Send shutdown to all remaining clients
        for cid, conn in list(self.client_conns.items()):
            try:
                send_object(conn, make_message(MsgType.SHUTDOWN))
                conn.close()
            except:
                pass

        server_sock.close()
        log.info("Server shut down cleanly.")

    # ─────────────────────────────────────────────────────────────────────────
    def _remove_clients(self, client_ids: list):
        for cid in client_ids:
            if cid in self.client_conns:
                try:
                    self.client_conns[cid].close()
                except:
                    pass
                del self.client_conns[cid]

    def _save_checkpoint(self, round_t: int, tag: str):
        path = os.path.join(CHECKPOINT_DIR, f"global_model_{tag}.pt")
        torch.save({
            "round":       round_t + 1,
            "model_state": self.global_model.state_dict(),
            "bmta_local":  self.bmta_local,
            "bmta_pers":   self.bmta_pers,
        }, path)
        log.info(f"  Checkpoint saved: {path}")

    def _save_history(self):
        import csv
        path = os.path.join(LOG_DIR, "training_history.csv")
        if not self.history["round"]:
            return
        keys = list(self.history.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(len(self.history["round"])):
                writer.writerow({k: self.history[k][i] for k in keys})
        log.info(f"  Training history saved: {path}")



if __name__ == "__main__":
    server = FLServer()
    server.start()
