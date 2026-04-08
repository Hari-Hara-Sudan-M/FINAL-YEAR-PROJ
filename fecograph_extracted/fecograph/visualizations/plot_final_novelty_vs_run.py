"""
Generate final comparison graph:
- "Run Result" metrics from binary training detailed metrics (client averages)
- "Novelty Result" metrics from SCAFFOLD zero-shot results
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RUN_CLIENT_METRICS = [
    RESULTS_DIR / "labelprop_0p3_binary" / "checkpoints" / "client_0_detailed_metrics.json",
    RESULTS_DIR / "labelprop_0p3_binary" / "checkpoints" / "client_1_detailed_metrics.json",
]
NOVELTY_METRICS_FILE = RESULTS_DIR / "scaffold_zeroshot" / "scaffold_zeroshot_results.json"

OUTPUT_PNG = PLOTS_DIR / "final_novelty_vs_run_metrics.png"
OUTPUT_CSV = PLOTS_DIR / "final_novelty_vs_run_metrics.csv"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_metrics():
    loaded = [_load_json(path) for path in RUN_CLIENT_METRICS]
    return {
        "accuracy": float(np.mean([m["accuracy"] for m in loaded])),
        "precision": float(np.mean([m["precision"] for m in loaded])),
        "recall": float(np.mean([m["recall"] for m in loaded])),
        "f1": float(np.mean([m["f1"] for m in loaded])),
    }


def load_novelty_metrics():
    data = _load_json(NOVELTY_METRICS_FILE)
    return {
        "accuracy": float(data["combined_accuracy"]),
        "precision": float(data["novelty_precision"]),
        "recall": float(data["novelty_recall"]),
        "f1": float(data["novelty_f1"]),
    }


def save_csv(run_metrics, novelty_metrics):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Run Result (Avg Client)", run_metrics),
        ("Novelty Result (Zero-Shot)", novelty_metrics),
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8") as f:
        f.write("Method,Accuracy,Precision,Recall,F1\n")
        for name, metrics in rows:
            f.write(
                f"{name},{metrics['accuracy']:.6f},{metrics['precision']:.6f},"
                f"{metrics['recall']:.6f},{metrics['f1']:.6f}\n"
            )


def plot_final_graph(run_metrics, novelty_metrics):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    run_vals = [run_metrics["accuracy"], run_metrics["precision"], run_metrics["recall"], run_metrics["f1"]]
    novelty_vals = [
        novelty_metrics["accuracy"],
        novelty_metrics["precision"],
        novelty_metrics["recall"],
        novelty_metrics["f1"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars_run = ax.bar(
        x - width / 2,
        np.array(run_vals) * 100,
        width,
        label="Run Result (Avg Client)",
        color="#2E86AB",
        edgecolor="black",
        linewidth=1.2,
    )
    bars_novel = ax.bar(
        x + width / 2,
        np.array(novelty_vals) * 100,
        width,
        label="Novelty Result (Zero-Shot)",
        color="#F18F01",
        edgecolor="black",
        linewidth=1.2,
    )

    for bars in (bars_run, bars_novel):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title("Final Metrics Comparison: Run Result vs Novelty Result", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right")
    ax.text(
        0.01,
        -0.15,
        "Note: Novelty accuracy uses combined_accuracy from scaffold_zeroshot_results.json",
        transform=ax.transAxes,
        fontsize=9,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for path in RUN_CLIENT_METRICS + [NOVELTY_METRICS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    run_metrics = load_run_metrics()
    novelty_metrics = load_novelty_metrics()
    save_csv(run_metrics, novelty_metrics)
    plot_final_graph(run_metrics, novelty_metrics)

    print(f"Saved graph: {OUTPUT_PNG}")
    print(f"Saved table: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
