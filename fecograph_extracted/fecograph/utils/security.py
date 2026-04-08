import importlib.abc
import ipaddress
import logging
import os
import sys


logger = logging.getLogger("FeCoGraph.security")

_DGL_RPC_BLOCK_ENABLED = False
_BLOCKED_DGL_PREFIXES = ("dgl.distributed", "dgl.rpc")


class _BlockedDGLFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _BLOCKED_DGL_PREFIXES or fullname.startswith(tuple(p + "." for p in _BLOCKED_DGL_PREFIXES)):
            raise ImportError(
                "Blocked import: DGL RPC/distributed modules are disabled due to unresolved upstream security advisory."
            )
        return None


def disable_dgl_rpc_imports():
    """Block DGL RPC/distributed imports at runtime."""
    global _DGL_RPC_BLOCK_ENABLED
    if _DGL_RPC_BLOCK_ENABLED:
        return

    loaded_blocked = [
        name for name in sys.modules
        if name in _BLOCKED_DGL_PREFIXES or name.startswith(tuple(p + "." for p in _BLOCKED_DGL_PREFIXES))
    ]
    if loaded_blocked:
        raise RuntimeError(
            "DGL RPC/distributed modules are already loaded; cannot enforce security guard. "
            f"Loaded modules: {loaded_blocked}"
        )

    sys.meta_path.insert(0, _BlockedDGLFinder())
    _DGL_RPC_BLOCK_ENABLED = True


def allow_remote_network() -> bool:
    return os.environ.get("FECOGRAPH_ALLOW_REMOTE_NETWORK", "").strip().lower() in {"1", "true", "yes"}


def isolated_bind_host(configured_host: str) -> str:
    """Force loopback binding unless remote network is explicitly allowed."""
    if allow_remote_network():
        return configured_host
    if configured_host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Remote binding disabled by default; overriding host '%s' -> '127.0.0.1'. "
            "Set FECOGRAPH_ALLOW_REMOTE_NETWORK=1 to allow remote interfaces.",
            configured_host,
        )
    return "127.0.0.1"


def isolated_connect_host(configured_host: str) -> str:
    """Force loopback connect target unless remote network is explicitly allowed."""
    if allow_remote_network():
        return configured_host
    return "127.0.0.1"


def is_trusted_peer_host(host: str) -> bool:
    """Allow loopback/private/link-local peers only."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local
