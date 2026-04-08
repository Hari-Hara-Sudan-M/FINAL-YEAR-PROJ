"""
Real FL Comparison: SCAFFOLD vs FedAvg using actual training logs
Parses training_history.csv files to create accurate comparison graphs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'scaffold': '#2E86AB',     # Blue - SCAFFOLD
    'fedavg': '#F18F01',       # Orange - FedAvg
    'target': '#C73E1D',       # Red - Target
    'success': '#06A77D',      # Green
}

def load_training_history(csv_path):
    """Load training history CSV"""
    df = pd.read_csv(csv_path)
    return df

def plot_accuracy_convergence(fedavg_df, scaffold_df, save_path):
    """
    Main convergence plot comparing FedAvg vs SCAFFOLD accuracy over rounds
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Extract data
    fedavg_rounds = fedavg_df['round'].values
    fedavg_acc = fedavg_df['mean_acc_pers'].values * 100  # Personalized accuracy
    
    scaffold_rounds = scaffold_df['round'].values
    scaffold_acc = scaffold_df['mean_acc_pers'].values * 100
    
    # Plot lines
    ax.plot(fedavg_rounds, fedavg_acc, color=COLORS['fedavg'], linewidth=3,
            label='FedAvg (Normal FL)', marker='o', markersize=5, markevery=10)
    ax.plot(scaffold_rounds, scaffold_acc, color=COLORS['scaffold'], linewidth=3,
            label='SCAFFOLD FL', marker='s', markersize=5, markevery=10)
    
    # Add target line
    ax.axhline(y=98, color=COLORS['target'], linestyle='--', linewidth=2.5,
               label='Target (98%)', alpha=0.8)
    
    # Mark final values
    final_fedavg = fedavg_acc[-1]
    final_scaffold = scaffold_acc[-1]
    
    ax.plot(fedavg_rounds[-1], final_fedavg, 'o', color=COLORS['fedavg'],
            markersize=15, markeredgecolor='black', markeredgewidth=2)
    ax.plot(scaffold_rounds[-1], final_scaffold, 's', color=COLORS['scaffold'],
            markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    # Add annotations for final values
    ax.text(fedavg_rounds[-1] + 2, final_fedavg,
            f'FedAvg: {final_fedavg:.2f}%',
            fontsize=11, fontweight='bold', color=COLORS['fedavg'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['fedavg'], linewidth=2))
    ax.text(scaffold_rounds[-1] + 2, final_scaffold,
            f'SCAFFOLD: {final_scaffold:.2f}%',
            fontsize=11, fontweight='bold', color=COLORS['scaffold'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['scaffold'], linewidth=2))
    
    # Styling
    ax.set_xlabel('Training Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('FL Methods Comparison: SCAFFOLD vs FedAvg Training Convergence',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, max(fedavg_rounds[-1], scaffold_rounds[-1]) + 5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_final_metrics_comparison(fedavg_df, scaffold_df, save_path):
    """
    Bar chart comparing final accuracy and F1 score
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract final metrics
    fedavg_final_acc = fedavg_df['mean_acc_pers'].iloc[-1] * 100
    fedavg_final_f1 = fedavg_df['mean_f1_pers'].iloc[-1] * 100
    
    scaffold_final_acc = scaffold_df['mean_acc_pers'].iloc[-1] * 100
    scaffold_final_f1 = scaffold_df['mean_f1_pers'].iloc[-1] * 100
    
    metrics = ['Accuracy', 'F1 Score']
    fedavg_values = [fedavg_final_acc, fedavg_final_f1]
    scaffold_values = [scaffold_final_acc, scaffold_final_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fedavg_values, width, label='FedAvg (Normal FL)',
                   color=COLORS['fedavg'], alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, scaffold_values, width, label='SCAFFOLD FL',
                   color=COLORS['scaffold'], alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add target line
    ax.axhline(y=98, color=COLORS['target'], linestyle='--', linewidth=2.5,
               label='Target (98%)', alpha=0.8)
    
    # Add improvement annotations
    for i, (fed, scaf) in enumerate(zip(fedavg_values, scaffold_values)):
        if scaf != fed:
            improvement = scaf - fed
            sign = '+' if improvement > 0 else ''
            color = COLORS['success'] if improvement > 0 else COLORS['target']
            ax.annotate(f'{sign}{improvement:.2f}%',
                       xy=(x[i] + width/2, scaf + 1),
                       xytext=(x[i] + width/2, scaf + 4),
                       ha='center', fontsize=11, fontweight='bold',
                       color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Styling
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Comparison: SCAFFOLD vs FedAvg',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comprehensive_comparison(fedavg_df, scaffold_df, save_path):
    """
    Comprehensive 2x2 grid comparison
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    fig.suptitle('Comprehensive FL Comparison: SCAFFOLD vs FedAvg (From Real Training Logs)',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Accuracy convergence (top left)
    fedavg_acc = fedavg_df['mean_acc_pers'].values * 100
    scaffold_acc = scaffold_df['mean_acc_pers'].values * 100
    
    ax1.plot(fedavg_df['round'], fedavg_acc, color=COLORS['fedavg'],
            linewidth=2.5, label='FedAvg', marker='o', markersize=4, markevery=10)
    ax1.plot(scaffold_df['round'], scaffold_acc, color=COLORS['scaffold'],
            linewidth=2.5, label='SCAFFOLD', marker='s', markersize=4, markevery=10)
    ax1.axhline(y=98, color=COLORS['target'], linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Test Accuracy Convergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. F1 Score convergence (top right)
    fedavg_f1 = fedavg_df['mean_f1_pers'].values * 100
    scaffold_f1 = scaffold_df['mean_f1_pers'].values * 100
    
    ax2.plot(fedavg_df['round'], fedavg_f1, color=COLORS['fedavg'],
            linewidth=2.5, label='FedAvg', marker='o', markersize=4, markevery=10)
    ax2.plot(scaffold_df['round'], scaffold_f1, color=COLORS['scaffold'],
            linewidth=2.5, label='SCAFFOLD', marker='s', markersize=4, markevery=10)
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) F1 Score Convergence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 3. Final metrics bar chart (bottom left)
    metrics = ['Accuracy', 'F1 Score']
    fedavg_finals = [fedavg_acc[-1], fedavg_f1[-1]]
    scaffold_finals = [scaffold_acc[-1], scaffold_f1[-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax3.bar(x - width/2, fedavg_finals, width, label='FedAvg',
                   color=COLORS['fedavg'], alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, scaffold_finals, width, label='SCAFFOLD',
                   color=COLORS['scaffold'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.axhline(y=98, color=COLORS['target'], linestyle='--', linewidth=2)
    ax3.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Final Performance Metrics', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Training info and stats (bottom right)
    ax4.axis('off')
    
    info_text = (
        f"Training Statistics\n"
        f"{'='*50}\n\n"
        f"FedAvg (Normal FL):\n"
        f"  Total Rounds:     {len(fedavg_df)}\n"
        f"  Final Accuracy:   {fedavg_acc[-1]:.2f}%\n"
        f"  Final F1 Score:   {fedavg_f1[-1]:.2f}%\n"
        f"  Best Accuracy:    {fedavg_acc.max():.2f}%\n"
        f"\n"
        f"SCAFFOLD FL:\n"
        f"  Total Rounds:     {len(scaffold_df)}\n"
        f"  Final Accuracy:   {scaffold_acc[-1]:.2f}%\n"
        f"  Final F1 Score:   {scaffold_f1[-1]:.2f}%\n"
        f"  Best Accuracy:    {scaffold_acc.max():.2f}%\n"
        f"\n"
        f"{'='*50}\n"
        f"Improvement (SCAFFOLD vs FedAvg):\n"
        f"  Accuracy:  {scaffold_acc[-1] - fedavg_acc[-1]:+.2f}%\n"
        f"  F1 Score:  {scaffold_f1[-1] - fedavg_f1[-1]:+.2f}%\n"
        f"\n"
        f"Target Achievement:\n"
        f"  FedAvg:    {'✓ YES' if fedavg_acc[-1] >= 98 else '✗ NO'} ({fedavg_acc[-1]:.2f}% {'≥' if fedavg_acc[-1] >= 98 else '<'} 98%)\n"
        f"  SCAFFOLD:  {'✓ YES' if scaffold_acc[-1] >= 98 else '✗ NO'} ({scaffold_acc[-1]:.2f}% {'≥' if scaffold_acc[-1] >= 98 else '<'} 98%)\n"
    )
    
    ax4.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=1))
    ax4.set_title('(D) Summary Statistics', fontsize=12, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main comparison pipeline using real training logs"""
    print("\n" + "="*70)
    print("FL COMPARISON: SCAFFOLD vs FedAvg (REAL TRAINING DATA)")
    print("="*70 + "\n")
    
    # Paths to training history CSV files
    fedavg_csv = Path("../results/fedavg_binary/logs/training_history.csv")
    scaffold_csv = Path("../results/scaffold_multiclass_supcon/logs/training_history_20260325_104854.csv")
    
    # Check if files exist
    if not fedavg_csv.exists():
        print(f"❌ Error: FedAvg history not found at {fedavg_csv}")
        return
    if not scaffold_csv.exists():
        print(f"❌ Error: SCAFFOLD history not found at {scaffold_csv}")
        return
    
    # Load data
    print("Loading training histories...")
    fedavg_df = load_training_history(fedavg_csv)
    scaffold_df = load_training_history(scaffold_csv)
    
    print(f"✓ FedAvg:    {len(fedavg_df)} rounds, Final Acc={fedavg_df['mean_acc_pers'].iloc[-1]*100:.2f}%")
    print(f"✓ SCAFFOLD:  {len(scaffold_df)} rounds, Final Acc={scaffold_df['mean_acc_pers'].iloc[-1]*100:.2f}%")
    print()
    
    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating comparison visualizations...\n")
    
    plot_accuracy_convergence(
        fedavg_df, scaffold_df,
        output_dir / "real_fl_comparison_convergence.png"
    )
    
    plot_final_metrics_comparison(
        fedavg_df, scaffold_df,
        output_dir / "real_fl_comparison_final.png"
    )
    
    plot_comprehensive_comparison(
        fedavg_df, scaffold_df,
        output_dir / "real_fl_comparison_comprehensive.png"
    )
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  real_fl_comparison_convergence.png     - Training curves ⭐")
    print("  real_fl_comparison_final.png           - Final metrics bars")
    print("  real_fl_comparison_comprehensive.png   - All-in-one view")
    print()


if __name__ == "__main__":
    main()
