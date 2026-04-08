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
import hashlib
import hmac

from utils.security import is_trusted_peer_host

logger = logging.getLogger("FeCoGraph.comm")

_SIG_LEN = hashlib.sha256().digest_size
_MAX_MESSAGE_BYTES = int(os.environ.get("FECOGRAPH_MAX_MESSAGE_BYTES", str(256 * 1024 * 1024)))


def _comm_secret() -> bytes:
    secret = os.environ.get("FECOGRAPH_COMM_SECRET", "").encode("utf-8")
    if not secret:
        raise RuntimeError(
            "FECOGRAPH_COMM_SECRET is required for authenticated socket messaging. "
            "Set a strong shared secret for server and all clients."
        )
    return secret


def _assert_trusted_peer(sock: socket.socket):
    try:
        peer_host, _ = sock.getpeername()
    except OSError:
        return
    if not is_trusted_peer_host(peer_host):
        raise ConnectionError(f"Rejected untrusted peer host: {peer_host}")


def send_object(sock: socket.socket, obj):
    """Serialize and send a Python object over socket with length prefix."""
    _assert_trusted_peer(sock)
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    signature = hmac.new(_comm_secret(), data, hashlib.sha256).digest()
    payload = signature + data
    length = struct.pack(">I", len(payload))  # 4-byte big-endian length
    sock.sendall(length + payload)


def recv_object(sock: socket.socket):
    """Receive a length-prefixed serialized object from socket."""
    _assert_trusted_peer(sock)
    # First 4 bytes = length
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack(">I", raw_len)[0]
    if length <= _SIG_LEN or length > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid message length: {length}")
    payload = _recv_exact(sock, length)
    if not payload:
        return None
    signature = payload[:_SIG_LEN]
    data = payload[_SIG_LEN:]
    expected = hmac.new(_comm_secret(), data, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Rejected message with invalid authentication signature")
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
