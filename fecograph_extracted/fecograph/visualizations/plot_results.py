"""
Visualization script for SCAFFOLD Zero-Shot Results
Creates comprehensive graphs to display and compare accuracy metrics
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6A4C93'
}

def load_results(results_path):
    """Load results JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_overall_metrics(results, save_path):
    """
    Bar chart comparing overall accuracy metrics
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = {
        'Seen\nAccuracy': results['seen_accuracy'],
        'Seen\nPrecision': results['seen_precision'],
        'Seen\nRecall': results['seen_recall'],
        'Seen\nF1': results['seen_f1'],
        'Novelty\nPrecision': results['novelty_precision'],
        'Novelty\nRecall': results['novelty_recall'],
        'Novelty\nF1': results['novelty_f1'],
        'Combined\nAccuracy': results['combined_accuracy']
    }
    
    labels = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]  # Convert to percentage
    
    # Color bars by category
    colors = [
        COLORS['primary'], COLORS['primary'], COLORS['primary'], COLORS['primary'],
        COLORS['secondary'], COLORS['secondary'], COLORS['secondary'],
        COLORS['success']
    ]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add 98% target line
    ax.axhline(y=98, color=COLORS['danger'], linestyle='--', linewidth=2, label='Target (98%)')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('SCAFFOLD Zero-Shot: Overall Performance Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_seen_vs_novel_comparison(results, save_path):
    """
    Grouped bar chart comparing Seen vs Novel detection metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Precision', 'Recall', 'F1 Score']
    seen_values = [
        results['seen_precision'] * 100,
        results['seen_recall'] * 100,
        results['seen_f1'] * 100
    ]
    novel_values = [
        results['novelty_precision'] * 100,
        results['novelty_recall'] * 100,
        results['novelty_f1'] * 100
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, seen_values, width, label='Seen Classes',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, novel_values, width, label='Novel Classes',
                   color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Seen vs Novel Attack Detection Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_confusion_summary(results, save_path):
    """
    Summary visualization showing classification breakdown
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Sample distribution
    seen_count = results['num_seen_samples']
    unseen_count = results['num_unseen_samples']
    
    sizes = [seen_count, unseen_count]
    labels = [f'Seen Classes\n({seen_count:,} samples)', 
              f'Novel Classes\n({unseen_count:,} samples)']
    colors = [COLORS['primary'], COLORS['warning']]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'},
                                         wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Test Set Distribution', fontsize=13, fontweight='bold', pad=15)
    
    # Right: Accuracy comparison
    accuracies = [
        results['seen_accuracy'] * 100,
        results['combined_accuracy'] * 100
    ]
    labels_acc = ['Seen Class\nAccuracy', 'Combined\nAccuracy']
    colors_acc = [COLORS['primary'], COLORS['success']]
    
    bars = ax2.barh(labels_acc, accuracies, color=colors_acc, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add 98% target line
    ax2.axvline(x=98, color=COLORS['danger'], linestyle='--', linewidth=2, label='Target (98%)')
    
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Metrics', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(0, 105)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_similarity_distributions(results, save_path):
    """
    Violin-style plot showing similarity score distributions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract similarity statistics
    seen_mean = results['seen_sim_mean']
    seen_std = results['seen_sim_std']
    unseen_mean = results['unseen_sim_mean']
    unseen_std = results['unseen_sim_std']
    threshold = results['optimal_threshold']
    
    # Create box plot representation
    data = [
        [seen_mean - seen_std, seen_mean, seen_mean + seen_std],
        [unseen_mean - unseen_std, unseen_mean, unseen_mean + unseen_std]
    ]
    
    positions = [1, 2]
    labels = ['Seen Classes', 'Novel Classes']
    colors_box = [COLORS['primary'], COLORS['warning']]
    
    # Plot boxes
    for i, (pos, color) in enumerate(zip(positions, colors_box)):
        mean_val = data[i][1]
        std_val = results['seen_sim_std'] if i == 0 else results['unseen_sim_std']
        
        # Box for ±1 std
        ax.bar(pos, height=2*std_val, width=0.5, bottom=mean_val-std_val,
               color=color, alpha=0.3, edgecolor=color, linewidth=2)
        
        # Mean line
        ax.plot([pos-0.25, pos+0.25], [mean_val, mean_val],
                color=color, linewidth=3, label=f'{labels[i]} (μ={mean_val:.3f})')
    
    # Threshold line
    ax.axhline(y=threshold, color=COLORS['danger'], linestyle='--', linewidth=2,
               label=f'Threshold={threshold:.3f}')
    
    # Add annotations
    ax.text(1, seen_mean, f'μ={seen_mean:.3f}\nσ={seen_std:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, unseen_mean, f'μ={unseen_mean:.3f}\nσ={unseen_std:.3f}',
            ha='center', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Cosine Similarity to Nearest Prototype', fontsize=12, fontweight='bold')
    ax.set_title('Similarity Score Distributions: Seen vs Novel Classes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_class_names_info(results, save_path):
    """
    Display class information in a table format
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    seen_names = results.get('seen_class_names', [])
    holdout_names = results.get('holdout_class_names', [])
    
    # Split into two columns if needed
    max_rows = max(len(seen_names), len(holdout_names))
    
    table_data = []
    table_data.append(['Seen Classes (Training)', 'Novel/Holdout Classes (Zero-Shot Test)'])
    
    for i in range(max_rows):
        seen = seen_names[i] if i < len(seen_names) else ''
        holdout = holdout_names[i] if i < len(holdout_names) else ''
        table_data.append([seen, holdout])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['info'])
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if table_data[i][j]:
                cell.set_facecolor('lightgray' if i % 2 == 0 else 'white')
            cell.set_edgecolor('black')
    
    plt.title('Attack Classes: Training vs Zero-Shot Detection',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_summary_report(results, save_path):
    """
    Create a single comprehensive summary figure with all key metrics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('SCAFFOLD Zero-Shot Novelty Detection - Complete Results Summary',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Overall Metrics (top row, span 3 columns)
    ax1 = fig.add_subplot(gs[0, :])
    metrics = {
        'Seen\nAcc': results['seen_accuracy'] * 100,
        'Seen\nF1': results['seen_f1'] * 100,
        'Novel\nPrec': results['novelty_precision'] * 100,
        'Novel\nRec': results['novelty_recall'] * 100,
        'Novel\nF1': results['novelty_f1'] * 100,
        'Combined\nAcc': results['combined_accuracy'] * 100
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors_bar = [COLORS['primary'], COLORS['primary'], COLORS['warning'], 
                  COLORS['warning'], COLORS['warning'], COLORS['success']]
    
    bars = ax1.bar(labels, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.axhline(y=98, color=COLORS['danger'], linestyle='--', linewidth=2, label='Target (98%)')
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Sample Distribution (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    seen_count = results['num_seen_samples']
    unseen_count = results['num_unseen_samples']
    wedges, texts, autotexts = ax2.pie([seen_count, unseen_count],
                                         labels=[f'Seen\n{seen_count:,}', f'Novel\n{unseen_count:,}'],
                                         colors=[COLORS['primary'], COLORS['warning']],
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 9, 'fontweight': 'bold'},
                                         wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    for autotext in autotexts:
        autotext.set_color('white')
    ax2.set_title('Test Samples', fontsize=11, fontweight='bold')
    
    # 3. Similarity Distributions (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    seen_mean = results['seen_sim_mean']
    unseen_mean = results['unseen_sim_mean']
    threshold = results['optimal_threshold']
    x = [1, 2]
    means = [seen_mean, unseen_mean]
    colors_sim = [COLORS['primary'], COLORS['warning']]
    bars = ax3.bar(x, means, color=colors_sim, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=threshold, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Threshold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Seen', 'Novel'], fontsize=10)
    ax3.set_ylabel('Similarity', fontsize=10, fontweight='bold')
    ax3.set_title('Mean Similarity Scores', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Training Info (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    info_text = (
        f"Training Configuration\n"
        f"{'='*30}\n"
        f"Training Rounds: {results.get('training_rounds', 'N/A')}\n"
        f"Best Seen Acc (Training): {results.get('best_seen_accuracy_during_training', 0)*100:.2f}%\n"
        f"Optimal Threshold: {results['optimal_threshold']:.4f}\n"
        f"\n"
        f"Seen Classes: {len(results.get('seen_class_names', []))}\n"
        f"Holdout Classes: {len(results.get('holdout_class_names', []))}\n"
        f"\n"
        f"Similarity Statistics\n"
        f"{'='*30}\n"
        f"Seen:  μ={results['seen_sim_mean']:.3f}, σ={results['seen_sim_std']:.3f}\n"
        f"Novel: μ={results['unseen_sim_mean']:.3f}, σ={results['unseen_sim_std']:.3f}\n"
    )
    ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Class Names (bottom row, span 3 columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    seen_names = results.get('seen_class_names', [])
    holdout_names = results.get('holdout_class_names', [])
    
    seen_text = "Seen Classes: " + ", ".join(seen_names)
    holdout_text = "Holdout Classes: " + ", ".join(holdout_names)
    
    ax5.text(0.05, 0.7, seen_text, fontsize=10, wrap=True,
             bbox=dict(boxstyle='round', facecolor=COLORS['primary'], alpha=0.3))
    ax5.text(0.05, 0.3, holdout_text, fontsize=10, wrap=True,
             bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.3))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    print("\n" + "="*70)
    print("SCAFFOLD ZERO-SHOT RESULTS VISUALIZATION")
    print("="*70 + "\n")
    
    # Paths
    results_dir = Path("../results/scaffold_zeroshot")
    results_file = results_dir / "scaffold_zeroshot_results.json"
    viz_output_dir = Path("./output")
    viz_output_dir.mkdir(exist_ok=True)
    
    # Check if results exist
    if not results_file.exists():
        print(f"❌ Error: Results file not found at {results_file}")
        print("   Please run the training script first: python run_scaffold_zeroshot.py")
        return
    
    # Load results
    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    print(f"✓ Results loaded successfully\n")
    
    # Generate all plots
    print("Generating visualizations...\n")
    
    plot_overall_metrics(results, viz_output_dir / "1_overall_metrics.png")
    plot_seen_vs_novel_comparison(results, viz_output_dir / "2_seen_vs_novel.png")
    plot_confusion_summary(results, viz_output_dir / "3_confusion_summary.png")
    plot_similarity_distributions(results, viz_output_dir / "4_similarity_distributions.png")
    plot_class_names_info(results, viz_output_dir / "5_class_names.png")
    create_summary_report(results, viz_output_dir / "0_complete_summary.png")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {viz_output_dir.absolute()}")
    print("\nGenerated files:")
    print("  0_complete_summary.png         - All metrics in one view")
    print("  1_overall_metrics.png          - Bar chart of all metrics")
    print("  2_seen_vs_novel.png            - Comparison of seen vs novel detection")
    print("  3_confusion_summary.png        - Sample distribution and accuracy")
    print("  4_similarity_distributions.png - Similarity score analysis")
    print("  5_class_names.png              - Attack class information")
    print()


if __name__ == "__main__":
    main()
