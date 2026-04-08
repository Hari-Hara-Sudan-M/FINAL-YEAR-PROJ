# =============================================================================
# Network Communication Utilities (Server ↔ Clients)
# Simple socket-based model weight exchange over LAN
# =============================================================================

import os
import sys
import socket
import pickle
import struct
import logging

logger = logging.getLogger("FeCoGraph.comm")


def send_object(sock: socket.socket, obj):
    """Serialize and send a Python object over socket with length prefix."""
    data = pickle.dumps(obj)
    length = struct.pack(">I", len(data))  # 4-byte big-endian length
    sock.sendall(length + data)


def recv_object(sock: socket.socket):
    """Receive a length-prefixed serialized object from socket."""
    # First 4 bytes = length
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack(">I", raw_len)[0]
    data = _recv_exact(sock, length)
    if not data:
        return None
    return pickle.loads(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

class MsgType:
    GLOBAL_MODEL = "global_model"    # Server → Client: send w_t
    CLIENT_DELTA = "client_delta"    # Client → Server: send Δ_k
    ROUND_START  = "round_start"     # Server → Client: start round t
    ROUND_END    = "round_end"       # Client → Server: done with round
    EVAL_RESULT  = "eval_result"     # Client → Server: evaluation metrics
    SHUTDOWN     = "shutdown"        # Server → Client: stop training
    REGISTER     = "register"        # Client → Server: initial handshake


def make_message(msg_type: str, payload=None, round_num: int = 0) -> dict:
    return {
        "type": msg_type,
        "round": round_num,
        "payload": payload,
    }
