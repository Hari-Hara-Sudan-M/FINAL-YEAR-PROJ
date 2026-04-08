import os
import sys
import torch
import dgl
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PE, PF, P_TAU


def compute_degree_centrality(g: dgl.DGLGraph) -> torch.Tensor:

    degrees = g.in_degrees().float() + g.out_degrees().float()
    # Normalize to [0,1]
    d_min = degrees.min()
    d_max = degrees.max()
    if d_max - d_min < 1e-8:
        return torch.ones(g.num_nodes())
    return (degrees - d_min) / (d_max - d_min + 1e-8)


def topology_augmentation(g: dgl.DGLGraph, centrality: torch.Tensor,
                           pe: float = PE, p_tau: float = P_TAU) -> dgl.DGLGraph:

    src, dst = g.edges()
    src, dst = src.long(), dst.long()

    w_uv = (centrality[src] + centrality[dst]) / 2.0
    w_uv = w_uv.clamp(min=1e-8)


    s_uv = torch.log(w_uv)                   
    s_max = s_uv.max()
    mu_s  = s_uv.mean()

    denom = (s_max - mu_s).clamp(min=1e-8)
    p_drop = ((s_max - s_uv) / denom * pe).clamp(0.0, p_tau)  
    p_drop = torch.nan_to_num(p_drop, nan=0.0)                

    
    keep_mask = torch.bernoulli((1.0 - p_drop).clamp(0.0, 1.0)).bool()

    kept_src = src[keep_mask]
    kept_dst = dst[keep_mask]

    g_aug = dgl.graph((kept_src, kept_dst), num_nodes=g.num_nodes())
    g_aug.ndata["feat"] = g.ndata["feat"].clone()
    if "label" in g.ndata:
        g_aug.ndata["label"] = g.ndata["label"].clone()
    g_aug = dgl.add_self_loop(g_aug)
    return g_aug



def attribute_augmentation(g: dgl.DGLGraph, centrality: torch.Tensor,
                            pf: float = PF, p_tau: float = P_TAU) -> dgl.DGLGraph:
    
    X = g.ndata["feat"].clone()              
    N, F = X.shape

    phi = centrality.unsqueeze(1)          
    w_f = (X.abs() * phi).sum(dim=0)        
    w_f = w_f.clamp(min=1e-8)

    s_f   = torch.log(w_f)

    s_max = s_f.max()
    mu_s  = s_f.mean()
    denom = (s_max - mu_s).clamp(min=1e-8)
    p_mask = ((s_max - s_f) / denom * pf).clamp(0.0, p_tau) 


    keep_prob = (1.0 - p_mask).clamp(0.0, 1.0)          
    keep_prob = torch.nan_to_num(keep_prob, nan=1.0)    
    mask = torch.bernoulli(keep_prob.unsqueeze(0).expand(N, -1))  
    X_aug = X * mask

    g_aug = dgl.graph((g.edges()[0], g.edges()[1]), num_nodes=N)
    g_aug.ndata["feat"] = X_aug
    if "label" in g.ndata:
        g_aug.ndata["label"] = g.ndata["label"].clone()
    return g_aug



def augment_graph(g: dgl.DGLGraph) -> tuple:

    centrality = compute_degree_centrality(g)

    G1 = attribute_augmentation(g, centrality, pf=PF, p_tau=P_TAU)

    G2 = topology_augmentation(g, centrality, pe=PE, p_tau=P_TAU)

    return G1, G2


if __name__ == "__main__":
   
    import dgl
   
    src = torch.tensor([0, 1, 2, 3, 0, 2])
    dst = torch.tensor([1, 2, 3, 0, 3, 1])
    g = dgl.graph((src, dst), num_nodes=4)
    g.ndata["feat"] = torch.randn(4, 40)
    g.ndata["label"] = torch.tensor([0, 1, 0, 1])

    G1, G2 = augment_graph(g)
    print(f"Original G : {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Augmented G1: {G1.num_nodes()} nodes, {G1.num_edges()} edges")
    print(f"Augmented G2: {G2.num_nodes()} nodes, {G2.num_edges()} edges")
    print("Augmentation OK")
