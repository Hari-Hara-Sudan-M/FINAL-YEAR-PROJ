# =============================================================================
# Generate All Paper Visualizations & Summary Tables
#
# Reads all experiment results and generates:
#   - Table III:  Binary & Multiclass performance comparison
#   - Table IV:   Lambda sensitivity (multiclass)
#   - Table V:    Temperature sensitivity (multiclass)
#   - Table VI:   SupCon vs SSLCon ablation
#   - Fig. 6:     t-SNE embeddings
#   - Fig. 7:     Label proportion bar chart
#   - Fig. 8:     FL convergence (FedAvg vs Ditto)
#   - Fig. 9/10:  Convergence under different settings
#
# Usage:  python generate_all_plots.py
# =============================================================================

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    INPUT_FEATURE_DIM, GCN_HIDDEN_DIM_1, GCN_HIDDEN_DIM_2,
    PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM,
    NUM_CLASSES_BINARY, NUM_CLASSES_MULTI, SEED
)
from models.model import FeCoGraphModel
from data.line_graph import csv_to_line_graph

RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_history(folder_name):
    path = os.path.join(RESULTS_DIR, folder_name, "logs", "training_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_bmta(df):
    if df is not None and "mean_acc_pers" in df.columns:
        return df["mean_acc_pers"].max()
    if df is not None and "mean_acc_local" in df.columns:
        return df["mean_acc_local"].max()
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# TABLE VI: SupCon vs SSLCon Ablation
# ═══════════════════════════════════════════════════════════════════════
def generate_table_vi():
    print("\n" + "=" * 70)
    print(" SupCon vs SSLCon Ablation Study")
    print("=" * 70)

    rows = []
    for name, folder in [("SupCon (Ditto)", "supcon_binary"), ("SSLCon (Ditto)", "sslcon_binary")]:
        df = load_history(folder)
        if df is not None:
            bmta = get_bmta(df)
            best_f1 = df["mean_f1_pers"].max() if "mean_f1_pers" in df.columns else 0
            final = df.iloc[-1]
            rows.append({
                "Method": name,
                "BMTA (%)": round(bmta * 100, 2),
                "Best F1 (%)": round(best_f1 * 100, 2),
                "Final Acc (%)": round(final.get("mean_acc_pers", 0) * 100, 2),
                "Final F1 (%)": round(final.get("mean_f1_pers", 0) * 100, 2),
            })
            print(f"  {name:<20} BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "table_vi_ablation.csv"), index=False)
        print(f"  Saved: {OUTPUT_DIR}/table_vi_ablation.csv")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: FedAvg vs Ditto Convergence
# ═══════════════════════════════════════════════════════════════════════
def generate_fig8():
    print("\n" + "=" * 70)
    print(" FedAvg vs Ditto Convergence")
    print("=" * 70)

    ditto_df = load_history("supcon_binary")
    fedavg_df = load_history("fedavg_binary")

    if ditto_df is None and fedavg_df is None:
        print("  No data available"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax = axes[0]
    if ditto_df is not None:
        ax.plot(ditto_df["round"], ditto_df["mean_acc_pers"] * 100, label="Ditto+SupCon (Personal)", linewidth=2, color="green")
        ax.plot(ditto_df["round"], ditto_df["mean_acc_local"] * 100, label="Ditto+SupCon (Global)", linewidth=1.5, linestyle="--", color="blue")
    if fedavg_df is not None:
        ax.plot(fedavg_df["round"], fedavg_df["mean_acc_local"] * 100, label="FedAvg+SupCon", linewidth=1.5, color="red")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy Convergence", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # F1
    ax = axes[1]
    if ditto_df is not None:
        ax.plot(ditto_df["round"], ditto_df["mean_f1_pers"] * 100, label="Ditto+SupCon (Personal)", linewidth=2, color="green")
        ax.plot(ditto_df["round"], ditto_df["mean_f1_local"] * 100, label="Ditto+SupCon (Global)", linewidth=1.5, linestyle="--", color="blue")
    if fedavg_df is not None:
        ax.plot(fedavg_df["round"], fedavg_df["mean_f1_local"] * 100, label="FedAvg+SupCon", linewidth=1.5, color="red")
    ax.set_xlabel("Communication Round", fontsize=12); ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.set_title("F1 Score Convergence", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(" FedAvg vs Ditto Convergence (Binary, CIC-IDS2018)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig8_fedavg_vs_ditto.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TABLE IV: Lambda Sensitivity
# ═══════════════════════════════════════════════════════════════════════
def generate_table_iv():
    print("\n" + "=" * 70)
    print("Lambda Sensitivity (Multiclass)")
    print("=" * 70)

    lambda_folders = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("lambda_")])
    rows = []
    for folder in lambda_folders:
        df = load_history(folder)
        if df is not None:
            lam = folder.replace("lambda_", "").replace("p", ".")
            bmta = get_bmta(df)
            best_f1 = df["mean_f1_pers"].max() if "mean_f1_pers" in df.columns else df["mean_f1_local"].max()
            rows.append({"Lambda": lam, "BMTA (%)": round(bmta*100, 2), "Best F1 (%)": round(best_f1*100, 2)})
            print(f"  lambda={lam:<6} BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "table_iv_lambda.csv"), index=False)
        print(f"  Saved: {OUTPUT_DIR}/table_iv_lambda.csv")


# ═══════════════════════════════════════════════════════════════════════
# TABLE V: Temperature Sensitivity
# ═══════════════════════════════════════════════════════════════════════
def generate_table_v():
    print("\n" + "=" * 70)
    print(" Temperature Sensitivity (Multiclass)")
    print("=" * 70)

    temp_folders = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("temp_")])
    rows = []
    for folder in temp_folders:
        df = load_history(folder)
        if df is not None:
            tau = folder.replace("temp_", "").replace("p", ".")
            bmta = get_bmta(df)
            best_f1 = df["mean_f1_pers"].max() if "mean_f1_pers" in df.columns else df["mean_f1_local"].max()
            rows.append({"Tau": tau, "BMTA (%)": round(bmta*100, 2), "Best F1 (%)": round(best_f1*100, 2)})
            print(f"  tau={tau:<6} BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "table_v_temperature.csv"), index=False)
        print(f"  Saved: {OUTPUT_DIR}/table_v_temperature.csv")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Label Proportion (Binary + Multiclass)
# ═══════════════════════════════════════════════════════════════════════
def generate_fig7():
    print("\n" + "=" * 70)
    print(" Label Proportion Impact")
    print("=" * 70)

    props = [0.1, 0.3, 0.5, 0.7]

    # Binary results
    print("  Binary:")
    binary_rows = []
    for pv in props:
        tag = str(pv).replace(".", "p")
        df = load_history(f"labelprop_{tag}_binary")
        if df is not None:
            bmta = get_bmta(df)
            best_f1 = df["mean_f1_pers"].max() if "mean_f1_pers" in df.columns else 0
            binary_rows.append({"Proportion": pv, "BMTA (%)": round(bmta*100, 2), "F1 (%)": round(best_f1*100, 2)})
            print(f"    {int(pv*100)}% -> BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")

    # Multiclass per-attack results
    print("  Multiclass per-attack F1:")
    attack_groups = {"BruteForce": [], "Bot": [], "DoS": [], "DDoS": [], "Infiltration": [], "Web Attacks": []}
    multi_rows = []
    for pv in props:
        tag = str(pv).replace(".", "p")
        ckpt_dir = os.path.join(RESULTS_DIR, f"labelprop_{tag}_multiclass", "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue

        grouped = {}
        count = 0
        for cid in range(10):
            mpath = os.path.join(ckpt_dir, f"client_{cid}_metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    m = json.load(f)
                for gname, gf1 in m.get("grouped_f1", {}).items():
                    if gname not in grouped: grouped[gname] = []
                    grouped[gname].append(gf1)
                count += 1

        if grouped:
            row = {"Proportion": f"{int(pv*100)}%"}
            for gname in attack_groups:
                val = round(np.mean(grouped.get(gname, [0])), 2) if gname in grouped else 0.0
                row[gname] = val
            multi_rows.append(row)
            print(f"    {int(pv*100)}% -> {row}")

    # Generate bar chart if multiclass data exists
    if multi_rows:
        categories = [k for k in attack_groups.keys()]
        n_props = len(multi_rows)
        n_cats = len(categories)
        x = np.arange(n_cats)
        width = 0.8 / max(n_props, 1)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, row in enumerate(multi_rows):
            vals = [row.get(cat, 0.0) for cat in categories]
            ax.bar(x + i * width - (n_props - 1) * width / 2, vals,
                   width, label=row["Proportion"] + " labeled",
                   color=colors[i % len(colors)])
        ax.set_xlabel("Attack Category", fontsize=12)
        ax.set_ylabel("F1-score (%)", fontsize=12)
        ax.set_title(" Performance with Different Label Proportions (CIC-IDS2018)", fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.legend(fontsize=10); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)

        # Value table
        table_data = [[str(row.get(cat, 0.0)) for cat in categories] for row in multi_rows]
        row_labels = [row["Proportion"] + " labeled" for row in multi_rows]
        table = ax.table(cellText=table_data, rowLabels=row_labels,
                         colLabels=categories, loc="bottom", bbox=[0, -0.45, 1, 0.3])
        table.auto_set_font_size(False); table.set_fontsize(8)
        fig.subplots_adjust(bottom=0.35)

        path = os.path.join(OUTPUT_DIR, "fig7_label_proportion.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    # Save CSV
    if binary_rows:
        pd.DataFrame(binary_rows).to_csv(os.path.join(OUTPUT_DIR, "fig7_binary_results.csv"), index=False)
    if multi_rows:
        pd.DataFrame(multi_rows).to_csv(os.path.join(OUTPUT_DIR, "fig7_multiclass_results.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: t-SNE Visualization
# ═══════════════════════════════════════════════════════════════════════
def generate_fig6():
    print("\n" + "=" * 70)
    print(" t-SNE Embedding Visualization")
    print("=" * 70)

    ckpt_path = os.path.join(RESULTS_DIR, "supcon_binary", "checkpoints", "global_model_best_pers.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(RESULTS_DIR, "supcon_binary", "checkpoints", "global_model_best_local.pt")
    if not os.path.exists(ckpt_path):
        print("  No checkpoint found for t-SNE, skipping."); return

    from sklearn.manifold import TSNE

    model = FeCoGraphModel(in_dim=INPUT_FEATURE_DIM, hidden_dim=GCN_HIDDEN_DIM_1,
        embed_dim=GCN_HIDDEN_DIM_2, proj_hidden=PROJECTOR_HIDDEN_DIM,
        proj_out=PROJECTOR_OUTPUT_DIM, num_classes=NUM_CLASSES_BINARY)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_csv = os.path.join("data", "client_0_test.csv")
    graph = csv_to_line_graph(test_csv, label_col="Label_ENCODED")
    labels = graph.ndata["label"].numpy()

    with torch.no_grad():
        raw_feat = graph.ndata["feat"].numpy()
        embeddings = model.encode(graph, graph.ndata["feat"]).numpy()

    rng = np.random.RandomState(SEED)
    N = len(labels)
    if N > 8000:
        idx = rng.choice(N, 8000, replace=False)
        raw_feat = raw_feat[idx]; embeddings = embeddings[idx]; labels = labels[idx]

    print("  Running t-SNE on raw features...")
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000, init="pca", learning_rate="auto")
    coords_raw = tsne_raw.fit_transform(raw_feat)

    print("  Running t-SNE on learned embeddings...")
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000, init="pca", learning_rate="auto")
    coords_emb = tsne_emb.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    name_map = {0: "Benign", 1: "Malicious"}
    colors = {0: "#3498db", 1: "#e74c3c"}

    for cls in [0, 1]:
        mask = labels == cls
        ax1.scatter(coords_raw[mask, 0], coords_raw[mask, 1], c=colors[cls],
                    label=name_map[cls], s=6, alpha=0.5, edgecolors="none")
        ax2.scatter(coords_emb[mask, 0], coords_emb[mask, 1], c=colors[cls],
                    label=name_map[cls], s=6, alpha=0.5, edgecolors="none")

    ax1.set_title("(a) Raw Edge Features", fontsize=13)
    ax1.legend(fontsize=10, markerscale=4); ax1.grid(True, alpha=0.2)
    ax2.set_title("(b) FeCoGraph Encoder Embeddings", fontsize=13)
    ax2.legend(fontsize=10, markerscale=4); ax2.grid(True, alpha=0.2)
    fig.suptitle("Fig. 6: t-SNE Visualization of Flow Representations", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig6_tsne.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# CONVERGENCE OVERVIEW (all experiments)
# ═══════════════════════════════════════════════════════════════════════
def generate_convergence_overview():
    print("\n" + "=" * 70)
    print("Convergence Overview (all experiments)")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Lambda convergence
    ax = axes[0, 0]
    for folder in sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("lambda_")]):
        df = load_history(folder)
        if df is not None:
            lam = folder.replace("lambda_", "").replace("p", ".")
            col = "mean_acc_pers" if "mean_acc_pers" in df.columns else "mean_acc_local"
            ax.plot(df["round"], df[col]*100, label=f"λ={lam}", linewidth=1.2)
    ax.set_title("Lambda Sensitivity (Table IV)", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Temperature convergence
    ax = axes[0, 1]
    for folder in sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("temp_")]):
        df = load_history(folder)
        if df is not None:
            tau = folder.replace("temp_", "").replace("p", ".")
            col = "mean_acc_pers" if "mean_acc_pers" in df.columns else "mean_acc_local"
            ax.plot(df["round"], df[col]*100, label=f"τ={tau}", linewidth=1.2)
    ax.set_title("Temperature Sensitivity (Table V)", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Label proportion convergence (binary)
    ax = axes[1, 0]
    for pv in [0.1, 0.3, 0.5, 0.7]:
        tag = str(pv).replace(".", "p")
        df = load_history(f"labelprop_{tag}_binary")
        if df is not None:
            col = "mean_acc_pers" if "mean_acc_pers" in df.columns else "mean_acc_local"
            ax.plot(df["round"], df[col]*100, label=f"{int(pv*100)}% labeled", linewidth=1.5)
    ax.set_title("Label Proportion (Fig. 7 - Binary)", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # SupCon vs SSLCon vs FedAvg vs SCAFFOLD
    ax = axes[1, 1]
    for name, folder, style, col_key in [
        ("Ditto+SupCon", "supcon_binary", "-", "mean_acc_pers"),
        ("Ditto+SSLCon", "sslcon_binary", "--", "mean_acc_pers"),
        ("FedAvg+SupCon", "fedavg_binary", "-.", "mean_acc_local"),
        ("SCAFFOLD+SupCon", "scaffold_binary_supcon", ":", "mean_acc_local"),
        ("SCAFFOLD", "scaffold_binary", (0, (3,1,1,1)), "mean_acc_local"),
    ]:
        df = load_history(folder)
        if df is not None:
            col = col_key if col_key in df.columns else "mean_acc_local"
            ax.plot(df["round"], df[col]*100, label=name, linewidth=1.5, linestyle=style)
    ax.set_title("FL Algorithm Comparison", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("FeCoGraph Experiment Results Overview", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "convergence_overview.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# SCAFFOLD COMPARISON PLOT (NOVELTY)
# ═══════════════════════════════════════════════════════════════════════
def generate_scaffold_comparison():
    print("\n" + "=" * 70)
    print(" SCAFFOLD vs FedAvg vs Ditto Comparison (NOVELTY)")
    print("=" * 70)

    experiments = [
        ("FedAvg+SupCon",       "fedavg_binary",        "mean_acc_local", "mean_f1_local",  "red",    "-."),
        ("Ditto+SupCon",        "supcon_binary",         "mean_acc_pers",  "mean_f1_pers",   "green",  "-"),
        ("SCAFFOLD (CE only)",  "scaffold_binary",       "mean_acc_local", "mean_f1_local",  "purple", "--"),
        ("SCAFFOLD+SupCon",     "scaffold_binary_supcon","mean_acc_local", "mean_f1_local",  "orange", ":"),
    ]

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, folder, acc_col, f1_col, color, style in experiments:
        df = load_history(folder)
        if df is None:
            print(f"  SKIP: {folder} not found")
            continue
        a_col = acc_col if acc_col in df.columns else "mean_acc_local"
        f_col = f1_col  if f1_col  in df.columns else "mean_f1_local"
        bmta    = df[a_col].max()
        best_f1 = df[f_col].max()
        print(f"  {name:<25} BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")
        rows.append({"Method": name, "BMTA (%)": round(bmta*100,2), "Best F1 (%)": round(best_f1*100,2)})
        axes[0].plot(df["round"], df[a_col]*100, label=name, linewidth=2, color=color, linestyle=style)
        axes[1].plot(df["round"], df[f_col]*100, label=name, linewidth=2, color=color, linestyle=style)

    for ax, ylabel, title in [
        (axes[0], "Test Accuracy (%)", "Test Accuracy Convergence"),
        (axes[1], "Macro F1 (%)",      "F1 Score Convergence"),
    ]:
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("SCAFFOLD vs FedAvg vs Ditto (Binary, CIC-IDS2018)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scaffold_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    if rows:
        csv_path = os.path.join(OUTPUT_DIR, "scaffold_comparison.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# MASTER SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════
def generate_master_summary():
    print("\n" + "=" * 70)
    print("MASTER SUMMARY — ALL EXPERIMENTS")
    print("=" * 70)

    rows = []
    for folder in sorted(os.listdir(RESULTS_DIR)):
        if folder == "plots": continue
        df = load_history(folder)
        if df is not None:
            bmta = get_bmta(df)
            col_f1 = "mean_f1_pers" if "mean_f1_pers" in df.columns else "mean_f1_local"
            best_f1 = df[col_f1].max()
            rows.append({"Experiment": folder, "BMTA (%)": round(bmta*100, 2), "Best F1 (%)": round(best_f1*100, 2)})
            print(f"  {folder:<35} BMTA={bmta*100:.2f}%  F1={best_f1*100:.2f}%")

    if rows:
        path = os.path.join(OUTPUT_DIR, "master_summary.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("FeCoGraph — Generate All Paper Plots & Tables")
    print("=" * 70)

    generate_master_summary()
    generate_table_vi()
    generate_table_iv()
    generate_table_v()
    generate_fig8()
    generate_fig7()
    generate_fig6()
    generate_convergence_overview()
    generate_scaffold_comparison()

    print("\n" + "=" * 70)
    print("ALL PLOTS & TABLES GENERATED!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
