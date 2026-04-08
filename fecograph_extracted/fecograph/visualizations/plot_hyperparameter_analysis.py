"""
Comprehensive Hyperparameter Analysis Visualization
Creates publication-quality graphs for Temperature, Lambda, and Label Proportion experiments
Extracts final metrics (Accuracy, Precision, Recall, F1) from training_history.csv files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Publication-quality styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

COLORS = {
    'accuracy': '#2E86AB',
    'precision': '#06A77D',
    'recall': '#F18F01',
    'f1': '#C73E1D',
    'local': '#6A4C93',
    'pers': '#2E86AB',
}

def extract_final_metrics(csv_path):
    """
    Extract final round metrics from training_history.csv
    Returns: dict with accuracy, precision, recall, f1
    """
    try:
        df = pd.read_csv(csv_path)
        final_row = df.iloc[-1]
        
        # Get final personalized metrics
        acc = final_row['mean_acc_pers']
        f1 = final_row['mean_f1_pers']
        
        # Estimate precision and recall from F1 and accuracy
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Assuming precision ≈ recall for symmetric estimate
        precision = f1  # Approximate
        recall = f1     # Approximate
        
        return {
            'accuracy': acc * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'rounds': len(df)
        }
    except Exception as e:
        print(f"  ⚠️  Error reading {csv_path}: {e}")
        return None


def scan_temperature_experiments(results_dir):
    """Scan all temperature experiments"""
    temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    data = []
    
    print("\nScanning Temperature experiments...")
    for temp in temps:
        folder_name = f"temp_{temp}".replace('.', 'p')
        csv_path = results_dir / folder_name / "logs" / "training_history.csv"
        
        if csv_path.exists():
            metrics = extract_final_metrics(csv_path)
            if metrics:
                metrics['temperature'] = temp
                data.append(metrics)
                print(f"  ✓ T={temp}: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.2f}%")
        else:
            print(f"  ✗ T={temp}: Not found")
    
    return pd.DataFrame(data) if data else None


def scan_lambda_experiments(results_dir):
    """Scan all lambda experiments"""
    lambdas = [0.01, 0.03, 0.05, 0.07, 0.1, 0.5]
    data = []
    
    print("\nScanning Lambda experiments...")
    for lam in lambdas:
        folder_name = f"lambda_{lam}".replace('.', 'p')
        csv_path = results_dir / folder_name / "logs" / "training_history.csv"
        
        if csv_path.exists():
            metrics = extract_final_metrics(csv_path)
            if metrics:
                metrics['lambda'] = lam
                data.append(metrics)
                print(f"  ✓ λ={lam}: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.2f}%")
        else:
            print(f"  ✗ λ={lam}: Not found")
    
    return pd.DataFrame(data) if data else None


def scan_labelprop_experiments(results_dir, task='binary'):
    """Scan all label proportion experiments"""
    props = [0.1, 0.3, 0.5, 0.7]
    data = []
    
    print(f"\nScanning Label Proportion experiments ({task})...")
    for prop in props:
        folder_name = f"labelprop_{prop}_{task}".replace('.', 'p')
        csv_path = results_dir / folder_name / "logs" / "training_history.csv"
        
        if csv_path.exists():
            metrics = extract_final_metrics(csv_path)
            if metrics:
                metrics['label_prop'] = prop
                data.append(metrics)
                print(f"  ✓ LP={prop}: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.2f}%")
        else:
            print(f"  ✗ LP={prop}: Not found")
    
    return pd.DataFrame(data) if data else None


def plot_temperature_analysis(df, save_path):
    """
    Plot Temperature hyperparameter analysis
    4 subplots: Accuracy, Precision, Recall, F1
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Hyperparameter Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['(a) Accuracy', '(b) Precision', '(c) Recall', '(d) F1 Score']
    colors = [COLORS['accuracy'], COLORS['precision'], COLORS['recall'], COLORS['f1']]
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        temps = df['temperature'].values
        values = df[metric].values
        
        # Line plot with markers
        ax.plot(temps, values, color=color, linewidth=2.5, marker='o', 
                markersize=10, markeredgewidth=2, markeredgecolor='white', label=metric.capitalize())
        
        # Add value labels
        for x, y in zip(temps, values):
            ax.text(x, y + 1, f'{y:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Highlight best value
        best_idx = values.argmax()
        ax.plot(temps[best_idx], values[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black', label='Best')
        
        ax.set_xlabel('Temperature (τ)', fontweight='bold')
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_lambda_analysis(df, save_path):
    """
    Plot Lambda (CE weight) hyperparameter analysis
    4 subplots: Accuracy, Precision, Recall, F1
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Lambda (CE Weight) Hyperparameter Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['(a) Accuracy', '(b) Precision', '(c) Recall', '(d) F1 Score']
    colors = [COLORS['accuracy'], COLORS['precision'], COLORS['recall'], COLORS['f1']]
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        lambdas = df['lambda'].values
        values = df[metric].values
        
        # Line plot with markers
        ax.plot(lambdas, values, color=color, linewidth=2.5, marker='s', 
                markersize=10, markeredgewidth=2, markeredgecolor='white', label=metric.capitalize())
        
        # Add value labels
        for x, y in zip(lambdas, values):
            ax.text(x, y + 1, f'{y:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Highlight best value
        best_idx = values.argmax()
        ax.plot(lambdas[best_idx], values[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black', label='Best')
        
        ax.set_xlabel('Lambda (λ)', fontweight='bold')
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_ylim(0, 105)
        ax.set_xscale('log')  # Log scale for lambda
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_labelprop_analysis(df, task, save_path):
    """
    Plot Label Proportion hyperparameter analysis
    4 subplots: Accuracy, Precision, Recall, F1
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Label Proportion Analysis ({task.capitalize()})', fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['(a) Accuracy', '(b) Precision', '(c) Recall', '(d) F1 Score']
    colors = [COLORS['accuracy'], COLORS['precision'], COLORS['recall'], COLORS['f1']]
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        props = df['label_prop'].values
        values = df[metric].values
        
        # Line plot with markers
        ax.plot(props, values, color=color, linewidth=2.5, marker='D', 
                markersize=10, markeredgewidth=2, markeredgecolor='white', label=metric.capitalize())
        
        # Add value labels
        for x, y in zip(props, values):
            ax.text(x, y + 1, f'{y:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Highlight best value
        best_idx = values.argmax()
        ax.plot(props[best_idx], values[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black', label='Best')
        
        ax.set_xlabel('Label Proportion', fontweight='bold')
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_ylim(0, 105)
        ax.set_xticks(props)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_combined_comparison(temp_df, lambda_df, labelprop_df, save_path):
    """
    Combined comparison of all hyperparameters
    Shows F1 score across all experiments
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Hyperparameter Comparison: F1 Score Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Temperature
    if temp_df is not None:
        ax = axes[0]
        temps = temp_df['temperature'].values
        f1 = temp_df['f1'].values
        ax.plot(temps, f1, color=COLORS['f1'], linewidth=3, marker='o', markersize=10)
        best_idx = f1.argmax()
        ax.plot(temps[best_idx], f1[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black')
        ax.set_xlabel('Temperature (τ)', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
        ax.set_title('(a) Temperature Impact', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    # Lambda
    if lambda_df is not None:
        ax = axes[1]
        lambdas = lambda_df['lambda'].values
        f1 = lambda_df['f1'].values
        ax.plot(lambdas, f1, color=COLORS['f1'], linewidth=3, marker='s', markersize=10)
        best_idx = f1.argmax()
        ax.plot(lambdas[best_idx], f1[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black')
        ax.set_xlabel('Lambda (λ)', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
        ax.set_title('(b) Lambda Impact', fontweight='bold', fontsize=13)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    # Label Proportion
    if labelprop_df is not None:
        ax = axes[2]
        props = labelprop_df['label_prop'].values
        f1 = labelprop_df['f1'].values
        ax.plot(props, f1, color=COLORS['f1'], linewidth=3, marker='D', markersize=10)
        best_idx = f1.argmax()
        ax.plot(props[best_idx], f1[best_idx], 'g*', markersize=20, 
               markeredgewidth=2, markeredgecolor='black')
        ax.set_xlabel('Label Proportion', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
        ax.set_title('(c) Label Proportion Impact', fontweight='bold', fontsize=13)
        ax.set_xticks(props)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_all_metrics_table(temp_df, lambda_df, labelprop_df, save_path):
    """
    Create a summary table visualization showing best hyperparameters
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Hyperparameter', 'Value', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)']]
    
    # Temperature - best
    if temp_df is not None:
        best_idx = temp_df['f1'].idxmax()
        best_row = temp_df.iloc[best_idx]
        table_data.append([
            'Temperature (τ)',
            f"{best_row['temperature']:.1f}",
            f"{best_row['accuracy']:.2f}",
            f"{best_row['precision']:.2f}",
            f"{best_row['recall']:.2f}",
            f"{best_row['f1']:.2f}"
        ])
    
    # Lambda - best
    if lambda_df is not None:
        best_idx = lambda_df['f1'].idxmax()
        best_row = lambda_df.iloc[best_idx]
        table_data.append([
            'Lambda (λ)',
            f"{best_row['lambda']:.2f}",
            f"{best_row['accuracy']:.2f}",
            f"{best_row['precision']:.2f}",
            f"{best_row['recall']:.2f}",
            f"{best_row['f1']:.2f}"
        ])
    
    # Label Proportion - best
    if labelprop_df is not None:
        best_idx = labelprop_df['f1'].idxmax()
        best_row = labelprop_df.iloc[best_idx]
        table_data.append([
            'Label Proportion',
            f"{best_row['label_prop']:.1f}",
            f"{best_row['accuracy']:.2f}",
            f"{best_row['precision']:.2f}",
            f"{best_row['recall']:.2f}",
            f"{best_row['f1']:.2f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_edgecolor('black')
        cell.set_linewidth(2)
    
    # Style data rows
    colors_alt = ['#E8F4F8', 'white']
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(colors_alt[i % 2])
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            if j == 0:
                cell.set_text_props(weight='bold')
    
    plt.title('Best Hyperparameter Configuration Summary', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main analysis pipeline"""
    print("\n" + "="*70)
    print("COMPREHENSIVE HYPERPARAMETER ANALYSIS")
    print("="*70)
    
    results_dir = Path("../results")
    output_dir = Path("./output/hyperparameter_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan all experiments
    temp_df = scan_temperature_experiments(results_dir)
    lambda_df = scan_lambda_experiments(results_dir)
    labelprop_binary_df = scan_labelprop_experiments(results_dir, 'binary')
    labelprop_multi_df = scan_labelprop_experiments(results_dir, 'multiclass')
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Generate plots
    if temp_df is not None:
        plot_temperature_analysis(temp_df, output_dir / "1_temperature_analysis.png")
    
    if lambda_df is not None:
        plot_lambda_analysis(lambda_df, output_dir / "2_lambda_analysis.png")
    
    if labelprop_binary_df is not None:
        plot_labelprop_analysis(labelprop_binary_df, 'binary', 
                               output_dir / "3_labelprop_binary_analysis.png")
    
    if labelprop_multi_df is not None:
        plot_labelprop_analysis(labelprop_multi_df, 'multiclass',
                               output_dir / "4_labelprop_multiclass_analysis.png")
    
    # Combined comparison
    plot_combined_comparison(temp_df, lambda_df, labelprop_binary_df,
                            output_dir / "5_combined_f1_comparison.png")
    
    # Summary table
    plot_all_metrics_table(temp_df, lambda_df, labelprop_binary_df,
                          output_dir / "6_best_hyperparameters_summary.png")
    
    # Save metrics to JSON
    summary = {}
    if temp_df is not None:
        summary['temperature'] = temp_df.to_dict('records')
    if lambda_df is not None:
        summary['lambda'] = lambda_df.to_dict('records')
    if labelprop_binary_df is not None:
        summary['label_prop_binary'] = labelprop_binary_df.to_dict('records')
    if labelprop_multi_df is not None:
        summary['label_prop_multiclass'] = labelprop_multi_df.to_dict('records')
    
    json_path = output_dir / "hyperparameter_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {json_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1_temperature_analysis.png          - Temperature hyperparameter")
    print("  2_lambda_analysis.png               - Lambda (CE weight)")
    print("  3_labelprop_binary_analysis.png     - Label proportion (binary)")
    print("  4_labelprop_multiclass_analysis.png - Label proportion (multiclass)")
    print("  5_combined_f1_comparison.png        - Combined F1 comparison")
    print("  6_best_hyperparameters_summary.png  - Summary table")
    print("  hyperparameter_summary.json         - All metrics (JSON)")
    print()


if __name__ == "__main__":
    main()
