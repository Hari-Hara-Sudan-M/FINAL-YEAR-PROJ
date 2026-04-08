# =============================================================================
# Utility functions: metrics printing, seed, device
# =============================================================================

import os
import random
import numpy as np
import torch
import logging


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cpu") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_metrics(metrics: dict, prefix: str = ""):
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    print(f"{prefix}  " + " | ".join(parts))


def log_round_summary(log, round_t, num_rounds, train_m, test_m, elapsed):
    log.info(
        f"Round {round_t+1:>3}/{num_rounds} | "
        f"Loss={train_m.get('total', 0):.4f} | "
        f"SupCon={train_m.get('supcon', 0):.4f} | "
        f"CE={train_m.get('ce', 0):.4f} | "
        f"Acc={test_m.get('accuracy', 0):.4f} | "
        f"F1={test_m.get('f1', 0):.4f} | "
        f"{elapsed:.1f}s"
    )
