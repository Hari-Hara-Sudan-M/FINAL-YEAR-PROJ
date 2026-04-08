
import os
import sys
import numpy as np
import pandas as pd
import torch
import dgl
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LABEL_COL, ATTACK_COL, INPUT_FEATURE_DIM,
    NUM_CLASSES_BINARY, CLASSIFICATION_MODE
)


EXCLUDE_COLS = [
    "Label", "Attack",
    "Label_ENCODED", "Attack_ENCODED",
    "IPV4_SRC_ADDR", "L4_SRC_PORT",
    "IPV4_DST_ADDR", "L4_DST_PORT",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if df[c].dtype in [np.float32, np.float64, np.int32, np.int64]:
            cols.append(c)
    return cols


def build_traffic_graph(df: pd.DataFrame):

    
    if "SRC_ADDR_PORT" in df.columns and "DST_ADDR_PORT" in df.columns:
        src_col, dst_col = "SRC_ADDR_PORT", "DST_ADDR_PORT"
    elif "IPV4_SRC_ADDR" in df.columns and "IPV4_DST_ADDR" in df.columns:
    
        src_col, dst_col = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
    else:
        raise ValueError("Cannot find src/dst address columns in dataframe.")

    
    all_hosts = pd.concat([df[src_col].astype(str), df[dst_col].astype(str)], ignore_index=True)
    unique_hosts = all_hosts.unique()
    node_map = {h: i for i, h in enumerate(unique_hosts)}

    src_list = df[src_col].astype(str).map(node_map).tolist()
    dst_list = df[dst_col].astype(str).map(node_map).tolist()

    return src_list, dst_list, node_map


def build_line_graph(df: pd.DataFrame, label_col: str = None) -> dgl.DGLGraph:

    print(f"  Building line graph from {len(df):,} flows...")

    src_list, dst_list, node_map = build_traffic_graph(df)
    num_flows = len(df)

 
    host_to_flows = defaultdict(list)
    for flow_idx, (s, d) in enumerate(zip(src_list, dst_list)):
        host_to_flows[s].append(flow_idx)
        host_to_flows[d].append(flow_idx)

    lg_src, lg_dst = [], []

    for host_id, flows in host_to_flows.items():
        if len(flows) < 2:
            continue

        if len(flows) > 500:
            flows = flows[:500]
        for i in range(len(flows)):
            for j in range(i + 1, len(flows)):
                u, v = flows[i], flows[j]
                lg_src.append(u)
                lg_dst.append(v)
                lg_src.append(v)  # undirected
                lg_dst.append(u)

    # Remove duplicate edges
    edges = list(set(zip(lg_src, lg_dst)))
    if len(edges) == 0:
        # Fallback: isolated nodes still form a valid graph
        print("  Warning: No shared-host edges found. Using isolated node graph.")
        lg_src_t = torch.zeros(0, dtype=torch.long)
        lg_dst_t = torch.zeros(0, dtype=torch.long)
    else:
        lg_src_t = torch.tensor([e[0] for e in edges], dtype=torch.long)
        lg_dst_t = torch.tensor([e[1] for e in edges], dtype=torch.long)

    g = dgl.graph((lg_src_t, lg_dst_t), num_nodes=num_flows)


    feat_cols = get_feature_cols(df)

    feat_cols = feat_cols[:INPUT_FEATURE_DIM]

    feat_df = df[feat_cols].copy()

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_array = feat_df.values.astype(np.float32)

    feat_array = np.clip(feat_array, -1e6, 1e6)
    if feat_array.shape[1] < INPUT_FEATURE_DIM:
        pad = np.zeros((feat_array.shape[0], INPUT_FEATURE_DIM - feat_array.shape[1]), dtype=np.float32)
        feat_array = np.concatenate([feat_array, pad], axis=1)

    g.ndata["feat"] = torch.tensor(feat_array, dtype=torch.float32)

    if label_col and label_col in df.columns:
        labels = df[label_col].values.astype(np.int64)
        g.ndata["label"] = torch.tensor(labels, dtype=torch.long)


    g = dgl.add_self_loop(g)

    num_edges = g.num_edges()
    print(f"  Line graph L(G): {g.num_nodes():,} nodes | {num_edges:,} edges")
    return g


def get_label_col_for_mode() -> str:
    """Return the correct label column based on CLASSIFICATION_MODE."""
    if CLASSIFICATION_MODE == "multiclass":
        return ATTACK_COL
    return LABEL_COL


def csv_to_line_graph(csv_path: str, label_col: str = None) -> dgl.DGLGraph:
    df = pd.read_csv(csv_path)
    if label_col is None:
        label_col = get_label_col_for_mode()
    return build_line_graph(df, label_col=label_col)


if __name__ == "__main__":
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python line_graph.py <path_to_client_csv>")
        sys.exit(1)
    g = csv_to_line_graph(sys.argv[1])
    print(f"Graph: {g}")
    print(f"Node feat shape : {g.ndata['feat'].shape}")
    if "label" in g.ndata:
        print(f"Label distribution: {torch.bincount(g.ndata['label'])}")
