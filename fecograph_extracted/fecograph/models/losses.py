# =============================================================================
# Loss Functions  (Paper Section IV-B4, Equations 8-12)
#
# Equation 8-9:  Label-Aware Supervised Contrastive Loss  L_supcon
# Equation 10-11: Cross-Entropy Loss  L_ce
# Equation 12:   Joint Loss  L = (1-λ_ce)·L_supcon + λ_ce·L_ce
# =============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TEMPERATURE, LAMBDA_CE, BATCH_SIZE, CONTRASTIVE_MODE


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss  (Equations 8-9)
# ─────────────────────────────────────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Label-aware supervised contrastive loss (Eq. 8-9).

    Given anchor embedding z_i and its 2N augmented sample set:
      - Positive set P(i): samples from the SAME class (in both views)
      - Negative set: samples from DIFFERENT classes

    L_supcon = (1/2N) Σ_i L_supcon_i
    L_supcon_i = (1/|P(i)|) Σ_{p∈P(i)} -log[ exp(z_i·z_p/τ) / Σ_{k≠i} exp(z_i·z_k/τ) ]

    Paper: 'Motivated by visual supervised contrastive learning [33]'
    Reference: Khosla et al. NeurIPS 2020 (Supervised Contrastive Learning)
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        z1     : projected embeddings from view 1  [N, D]  (L2 normalized)
        z2     : projected embeddings from view 2  [N, D]  (L2 normalized)
        labels : class labels                       [N]

        Returns scalar loss.
        """
        N = z1.size(0)
        if N == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Concatenate both views → [2N, D]
        z    = torch.cat([z1, z2], dim=0)         # [2N, D]
        lbl  = torch.cat([labels, labels], dim=0) # [2N]

        # Cosine similarity matrix  [2N, 2N]  — z is already L2 normalized
        sim = torch.mm(z, z.T) / self.tau          # [2N, 2N]

        # Exclude self-similarity (diagonal)
        mask_self = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)

        # Positive mask: same label, not self
        lbl_eq = (lbl.unsqueeze(0) == lbl.unsqueeze(1))  # [2N, 2N]
        mask_pos = lbl_eq & mask_self

        # For numerical stability: subtract row max
        sim_masked = sim.masked_fill(~mask_self, float('-inf'))
        sim_max, _ = sim_masked.max(dim=1, keepdim=True)
        sim_exp    = torch.exp(sim - sim_max.detach())

        # Denominator: sum over all k ≠ i
        denom = (sim_exp * mask_self.float()).sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Log probability
        log_prob = sim - sim_max.detach() - torch.log(denom)

        # Only average over positive pairs per anchor
        num_pos = mask_pos.float().sum(dim=1).clamp(min=1)   # |P(i)|

        # Eq. 9
        loss_per_anchor = -(log_prob * mask_pos.float()).sum(dim=1) / num_pos

        # Skip anchors with no positives (edge case for minority classes)
        valid = mask_pos.any(dim=1)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        loss = loss_per_anchor[valid].mean()   # Eq. 8  (1/2N factor via mean)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Self-Supervised Contrastive Loss  (Table VI ablation — SimCLR-style)
# ─────────────────────────────────────────────────────────────────────────────

class SelfSupervisedContrastiveLoss(nn.Module):
    """
    Self-supervised contrastive loss (SSLCon) for ablation study (Table VI).

    Unlike SupCon, no label information is used.
    For each node i, the only positive is the same node i in the other view.
    All other 2N-2 nodes are treated as negatives (SimCLR/NT-Xent style).

    Paper Table VI: 'SSLCon denotes self-supervised graph contrastive learning module'
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        """
        z1     : projected embeddings from view 1  [N, D]  (L2 normalized)
        z2     : projected embeddings from view 2  [N, D]  (L2 normalized)
        labels : IGNORED (accepted for API compatibility with SupCon)

        Returns scalar loss.
        """
        N = z1.size(0)
        if N == 0:
            return torch.tensor(0.0, requires_grad=True)

        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        sim = torch.mm(z, z.T) / self.tau  # [2N, 2N]

        mask_self = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)

        # Positive mask: node i in view1 pairs with node i in view2 (and vice versa)
        # z[i] <-> z[i+N]  and  z[i+N] <-> z[i]
        mask_pos = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=z.device)
        for i in range(N):
            mask_pos[i, i + N] = True
            mask_pos[i + N, i] = True

        sim_masked = sim.masked_fill(~mask_self, float('-inf'))
        sim_max, _ = sim_masked.max(dim=1, keepdim=True)
        sim_exp = torch.exp(sim - sim_max.detach())

        denom = (sim_exp * mask_self.float()).sum(dim=1, keepdim=True).clamp(min=1e-8)

        log_prob = sim - sim_max.detach() - torch.log(denom)

        loss_per_anchor = -(log_prob * mask_pos.float()).sum(dim=1)

        loss = loss_per_anchor.mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss  (Equations 10-11)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationLoss(nn.Module):
    """
    Standard cross-entropy loss  L_ce  (Eq. 10-11).

    ŷ = softmax(W · Enc(x) + b)   (Eq.10)
    L_ce = -Σ_i Σ_j y_ij log(ŷ_ij)  (Eq.11)
    """
    def __init__(self, num_classes: int = None, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Joint Loss  (Equation 12)
# ─────────────────────────────────────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    L = (1 - λ_ce) · L_contrastive  +  λ_ce · L_ce      (Eq. 12)

    λ_ce controls trade-off: smaller → more weight on contrastive
    Paper best values: 0.07 on IDS2018 (Table IV)

    contrastive_mode:
      "supcon" → label-aware supervised contrastive (Eq.8-9)
      "sslcon" → self-supervised SimCLR-style (Table VI ablation)
    """
    def __init__(self, lambda_ce: float = LAMBDA_CE,
                 temperature: float = TEMPERATURE,
                 class_weights: torch.Tensor = None,
                 contrastive_mode: str = CONTRASTIVE_MODE):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.contrastive_mode = contrastive_mode
        if contrastive_mode == "sslcon":
            self.contrastive = SelfSupervisedContrastiveLoss(temperature)
        else:
            self.contrastive = SupervisedContrastiveLoss(temperature)
        self.ce_loss = ClassificationLoss(class_weights=class_weights)

    def forward(self,
                z1: torch.Tensor,       # view-1 projections  [N, D]
                z2: torch.Tensor,       # view-2 projections  [N, D]
                logits: torch.Tensor,   # classifier output   [N, C]
                labels: torch.Tensor    # ground truth        [N]
                ) -> tuple:
        """
        Returns:
            total_loss   : scalar
            loss_supcon  : scalar  (for logging)
            loss_ce      : scalar  (for logging)
        """
        l_con = self.contrastive(z1, z2, labels)
        l_ce  = self.ce_loss(logits, labels)
        total = (1 - self.lambda_ce) * l_con + self.lambda_ce * l_ce
        return total, l_con, l_ce


# ─────────────────────────────────────────────────────────────────────────────
# Batched Joint Loss (for large graphs — avoids CUDA OOM)
# ─────────────────────────────────────────────────────────────────────────────

def batched_joint_loss(z1_all, z2_all, logits_all, labels_all,
                       lambda_ce=LAMBDA_CE, temperature=TEMPERATURE,
                       batch_size=BATCH_SIZE, class_weights=None):
    """
    Paper: 'we calculate supervised contrastive loss in batches to avoid
            the issue of CUDA out of memory'

    Splits the 2N embeddings into mini-batches and averages the loss.
    """
    joint = JointLoss(lambda_ce=lambda_ce, temperature=temperature,
                      class_weights=class_weights)
    N = z1_all.size(0)
    total, sup, ce_val = 0.0, 0.0, 0.0
    n_batches = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        loss, l_s, l_c = joint(
            z1_all[start:end],
            z2_all[start:end],
            logits_all[start:end],
            labels_all[start:end]
        )
        total  += loss.item()
        sup    += l_s.item()
        ce_val += l_c.item()
        n_batches += 1
        loss.backward(retain_graph=(end < N))

    return total / n_batches, sup / n_batches, ce_val / n_batches


if __name__ == "__main__":
    z1 = F.normalize(torch.randn(16, 128), dim=-1)
    z2 = F.normalize(torch.randn(16, 128), dim=-1)
    logits = torch.randn(16, 2)
    labels = torch.randint(0, 2, (16,))

    loss_fn_sup = JointLoss(contrastive_mode="supcon")
    total, l_s, l_c = loss_fn_sup(z1, z2, logits, labels)
    print(f"[SupCon] Total={total:.4f}  Con={l_s:.4f}  CE={l_c:.4f}")

    loss_fn_ssl = JointLoss(contrastive_mode="sslcon")
    total, l_s, l_c = loss_fn_ssl(z1, z2, logits, labels)
    print(f"[SSLCon] Total={total:.4f}  Con={l_s:.4f}  CE={l_c:.4f}")

    print("Both loss modes OK")
