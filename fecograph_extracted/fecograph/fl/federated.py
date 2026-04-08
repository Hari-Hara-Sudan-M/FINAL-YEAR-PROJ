

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LOCAL_EPOCHS, PERSONALIZED_EPOCHS, LOCAL_LR, PERSONALIZED_LR,
    MU, LAMBDA_CE, TEMPERATURE, BATCH_SIZE, FL_SCHEME, DEVICE,
    CONTRASTIVE_MODE
)
from models.losses import SupervisedContrastiveLoss, SelfSupervisedContrastiveLoss, ClassificationLoss
from data.augmentation import augment_graph

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts  = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights


def contrastive_task_step(model, graph, G1, G2,
                           train_indices, labels,
                           class_weights, device,
                           lambda_ce=LAMBDA_CE,
                           temperature=TEMPERATURE,
                           batch_size=BATCH_SIZE,
                           contrastive_mode=CONTRASTIVE_MODE):
    if contrastive_mode == "sslcon":
        con_fn = SelfSupervisedContrastiveLoss(temperature=temperature)
    else:
        con_fn = SupervisedContrastiveLoss(temperature=temperature)
    ce_fn = ClassificationLoss(class_weights=class_weights)

    _, z1_all, _    = model(G1.to(device), G1.ndata["feat"].to(device))
    _, z2_all, _    = model(G2.to(device), G2.ndata["feat"].to(device))
    _, _,  y_all    = model(graph.to(device), graph.ndata["feat"].to(device))

    total_val  = 0.0
    con_val    = 0.0
    ce_val     = 0.0
    n_batches  = 0

    perm         = torch.randperm(len(train_indices), device=device)
    shuffled_idx = train_indices[perm]
    N_labeled    = len(shuffled_idx)

    for start in range(0, N_labeled, batch_size):
        end   = min(start + batch_size, N_labeled)
        b_idx = shuffled_idx[start:end]
        b_lbl = labels[b_idx]

        if len(b_lbl) < 2:
            continue

        z1_b    = z1_all[b_idx]
        z2_b    = z2_all[b_idx]
        logit_b = y_all[b_idx]

        l_con = con_fn(z1_b, z2_b, b_lbl)
        l_ce  = ce_fn(logit_b, b_lbl)
        loss  = (1.0 - lambda_ce) * l_con + lambda_ce * l_ce

        is_last = (end >= N_labeled)
        loss.backward(retain_graph=not is_last)

        total_val += loss.item()
        con_val   += l_con.item()
        ce_val    += l_ce.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_val / n_batches, con_val / n_batches, ce_val / n_batches


def classification_task_step(personalized_model, global_model,
                              graph, train_indices, labels,
                              class_weights, device,
                              batch_size=BATCH_SIZE):

    ce_fn_p     = ClassificationLoss(class_weights=class_weights)
    optimizer_p = torch.optim.Adam(personalized_model.parameters(),
                                    lr=PERSONALIZED_LR)
    feat_G = graph.ndata["feat"].to(device)

    total_ce = 0.0

    for epoch in range(PERSONALIZED_EPOCHS):
        optimizer_p.zero_grad()

        logits_p = personalized_model(graph.to(device), feat_G)

        perm  = torch.randperm(len(train_indices), device=device)
        b_idx = train_indices[perm[:min(batch_size, len(train_indices))]]
        l_ce  = ce_fn_p(logits_p[b_idx], labels[b_idx])

        reg = torch.tensor(0.0, device=device)
        for pp, pw in zip(personalized_model.model.parameters(),
                          global_model.parameters()):
            reg = reg + ((pp - pw.detach()) ** 2).sum()
        reg = (MU / 2.0) * reg

        loss_p = l_ce + reg
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(personalized_model.parameters(), max_norm=5.0)
        optimizer_p.step()

        total_ce += l_ce.item()

    return total_ce / PERSONALIZED_EPOCHS if PERSONALIZED_EPOCHS > 0 else 0.0



def local_train_client(global_model, personalized_model,
                       graph, train_mask,
                       device, round_num,
                       fl_scheme=FL_SCHEME):

    global_model.to(device)
    global_model.train()

    # Save w_t for Delta computation and Ditto regularization
    w_t_params = {
        n: p.clone().detach().cpu()
        for n, p in global_model.named_parameters()
    }

    labels        = graph.ndata["label"].to(device)
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1).to(device)
    labeled_labels = labels[train_indices]

    num_classes   = int(labels.max().item()) + 1
    class_weights = compute_class_weights(labeled_labels, num_classes).to(device)

    optimizer_w = torch.optim.Adam(global_model.parameters(), lr=LOCAL_LR)


    G1, G2 = augment_graph(graph)

    metrics = {"total": [], "supcon": [], "ce": []}
    for epoch in range(LOCAL_EPOCHS):
        global_model.train()
        optimizer_w.zero_grad()

        t_loss, s_loss, c_loss = contrastive_task_step(
            model=global_model,
            graph=graph,
            G1=G1, G2=G2,
            train_indices=train_indices,
            labels=labels,
            class_weights=class_weights,
            device=device,
            lambda_ce=LAMBDA_CE,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
        )

        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=5.0)
        optimizer_w.step()

        metrics["total"].append(t_loss)
        metrics["supcon"].append(s_loss)
        metrics["ce"].append(c_loss)

    if fl_scheme == "ditto" and personalized_model is not None:
        personalized_model.to(device)
        personalized_model.train()
        classification_task_step(
            personalized_model=personalized_model,
            global_model=global_model,
            graph=graph,
            train_indices=train_indices,
            labels=labels,
            class_weights=class_weights,
            device=device,
            batch_size=BATCH_SIZE,
        )

    delta = {
        n: p.data.cpu() - w_t_params[n]
        for n, p in global_model.named_parameters()
    }

    avg_metrics = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in metrics.items()
    }
    return delta, personalized_model, avg_metrics



def server_aggregate(global_model, client_deltas: list, selected_clients: list):

    if not client_deltas:
        return global_model

    # Average all client deltas
    avg_delta = {}
    for name in client_deltas[0].keys():
        stacked = torch.stack([cd[name].float() for cd in client_deltas])
        avg_delta[name] = stacked.mean(dim=0)

    # Apply averaged delta to global model
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_delta:
                param.data += avg_delta[name].to(param.device)

    return global_model



@torch.no_grad()
def evaluate_model(model, graph, mask, device, use_personalized=False):

    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)
    model.eval()
    feat   = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].cpu().numpy()

    if use_personalized:
        logits = model(graph.to(device), feat)
    else:
        _, _, logits = model(graph.to(device), feat)

    preds   = logits.argmax(dim=-1).cpu().numpy()
    mask_np = mask.cpu().numpy()
    y_true  = labels[mask_np]
    y_pred  = preds[mask_np]

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
