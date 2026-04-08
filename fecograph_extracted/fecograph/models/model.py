
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI, CLASSIFICATION_MODE
)



class GCNLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # W^k maps concatenated [h_v^{k-1} || h_N(v)^k] → h_v^k
        self.agg = GraphConv(in_dim, in_dim, norm="both", bias=False)  # aggregation
        self.W   = nn.Linear(in_dim * 2, out_dim, bias=True)            # Eq.7 weight
        self.bn  = nn.BatchNorm1d(out_dim)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        h_prev = h
        h_agg  = self.agg(g, h)                          # h_N(v)^k  (Eq.6)
        h_cat  = torch.cat([h_prev, h_agg], dim=-1)      # CONCAT     (Eq.7)
        h_new  = self.W(h_cat)                           # W^k · CONCAT
        h_new  = self.bn(h_new)
        h_new  = F.relu(h_new)
        return h_new


class GCNEncoder(nn.Module):

    def __init__(self,
                 in_dim:     int = INPUT_FEATURE_DIM,
                 hidden_dim: int = GCN_HIDDEN_DIM_1,
                 out_dim:    int = GCN_HIDDEN_DIM_2):
        super().__init__()
        self.layer1 = GCNLayer(in_dim,     hidden_dim)   # 40 → 64
        self.layer2 = GCNLayer(hidden_dim, out_dim)      # 64 → 32
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:

        h = self.layer1(g, feat)
        h = self.dropout(h)
        h = self.layer2(g, h)
        return h   

class Projector(nn.Module):

    def __init__(self,
                 in_dim:     int = GCN_HIDDEN_DIM_2,
                 hidden_dim: int = PROJECTOR_HIDDEN_DIM,
                 out_dim:    int = PROJECTOR_OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        z = F.normalize(z, p=2, dim=-1)   # Unit hypersphere normalization
        return z


# ─────────────────────────────────────────────────────────────────────────────
# Classifier (FC head)
# ─────────────────────────────────────────────────────────────────────────────

class Classifier(nn.Module):

    def __init__(self,
                 in_dim:     int = GCN_HIDDEN_DIM_2,
                 num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES_BINARY if CLASSIFICATION_MODE == "binary" else NUM_CLASSES_MULTI
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)   # logits [N, num_classes]


class FeCoGraphModel(nn.Module):
    """
    Full model with shared encoder + two branches:
      Branch A (contrastive): Encoder → Projector  →  supervised contrastive loss
      Branch B (classify)   : Encoder → Classifier →  cross-entropy loss

    Paper: 'backbone model is divided into two branches for supervised
            classification and contrastive learning, respectively'
    """
    def __init__(self,
                 in_dim:      int = INPUT_FEATURE_DIM,
                 hidden_dim:  int = GCN_HIDDEN_DIM_1,
                 embed_dim:   int = GCN_HIDDEN_DIM_2,
                 proj_hidden: int = PROJECTOR_HIDDEN_DIM,
                 proj_out:    int = PROJECTOR_OUTPUT_DIM,
                 num_classes: int = None):
        super().__init__()
        self.encoder    = GCNEncoder(in_dim, hidden_dim, embed_dim)
        self.projector  = Projector(embed_dim, proj_hidden, proj_out)
        self.classifier = Classifier(embed_dim, num_classes)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        """
        Returns:
            h : node embeddings [N, embed_dim]        — shared representation
            z : projected embeddings [N, proj_out]    — for contrastive loss
            y : class logits [N, num_classes]          — for cross-entropy
        """
        h = self.encoder(g, feat)      # Eq. 6-7
        z = self.projector(h)          # contrastive branch
        y = self.classifier(h)         # classification branch
        return h, z, y

    def encode(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """Inference: return encoder output only (projector discarded)."""
        return self.encoder(g, feat)

    def classify(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """Inference: full forward → class logits."""
        h = self.encoder(g, feat)
        return self.classifier(h)

    def get_flat_params(self) -> torch.Tensor:
        """Flatten all parameters to 1D tensor (for FL weight sharing)."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor):
        """Load 1D flat parameter vector back into model."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel



class PersonalizedModel(nn.Module):
    """
    Personalized model θ_k for client k  (Ditto, Eq.15-16).
    Same architecture as global model but kept local — not uploaded to server.
    Uses only cross-entropy loss (Eq.18 in paper).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.model = FeCoGraphModel(**kwargs)

    def forward(self, g, feat):
        return self.model.classify(g, feat)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.model.load_state_dict(sd, strict)


if __name__ == "__main__":
    import dgl
    # Quick smoke test
    g = dgl.rand_graph(100, 400)
    g = dgl.add_self_loop(g)
    feat = torch.randn(100, INPUT_FEATURE_DIM)
    g.ndata["feat"] = feat
    model = FeCoGraphModel()
    h, z, y = model(g, feat)
    print(f"h: {h.shape}  z: {z.shape}  y: {y.shape}")
    print("Model OK")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
