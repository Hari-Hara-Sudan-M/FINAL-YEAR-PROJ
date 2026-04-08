# SCAFFOLD Zero-Shot Results Visualization

This folder contains scripts to visualize and compare the results from SCAFFOLD zero-shot novelty detection experiments.

## 📁 Files

- **`plot_results.py`** - Main visualization script that generates comprehensive graphs
- **`output/`** - Output directory for generated plots (created automatically)

## 🚀 Quick Start

### Step 1: Run Training (if not done yet)

```bash
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
python run_scaffold_zeroshot.py
```

This will create: `results/scaffold_zeroshot/scaffold_zeroshot_results.json`

### Step 2: Generate Visualizations

```bash
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations"
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
python plot_results.py
```

## 📊 Generated Plots

The script generates 6 comprehensive visualizations:

### 1. **Complete Summary** (`0_complete_summary.png`)
- All-in-one dashboard with all key metrics
- Sample distribution, similarity scores, training config
- **Best for presentations and reports**

### 2. **Overall Metrics** (`1_overall_metrics.png`)
- Bar chart comparing all performance metrics
- Shows: Seen Acc, Seen F1, Novel Prec/Rec/F1, Combined Acc
- Includes 98% target line
- **Best for comparing against target accuracy**

### 3. **Seen vs Novel Comparison** (`2_seen_vs_novel.png`)
- Grouped bar chart comparing Seen vs Novel detection
- Side-by-side comparison of Precision, Recall, F1
- **Best for understanding detection quality**

### 4. **Confusion Summary** (`3_confusion_summary.png`)
- Pie chart showing test sample distribution
- Horizontal bar chart with accuracy metrics
- **Best for understanding dataset balance**

### 5. **Similarity Distributions** (`4_similarity_distributions.png`)
- Shows mean similarity scores with standard deviations
- Displays optimal threshold line
- **Best for understanding decision boundaries**

### 6. **Class Names Info** (`5_class_names.png`)
- Table showing which attacks are in seen vs holdout classes
- **Best for documentation and explaining experiment setup**

---

## ✅ Final Result Graph (Novelty vs Run Result)

To generate a single final comparison graph for **Accuracy, Precision, Recall, F1**:

```bash
cd /home/runner/work/FINAL-YEAR-PROJ/FINAL-YEAR-PROJ/fecograph_extracted/fecograph/visualizations
python plot_final_novelty_vs_run.py
```

Outputs:
- `/home/runner/work/FINAL-YEAR-PROJ/FINAL-YEAR-PROJ/fecograph_extracted/fecograph/results/plots/final_novelty_vs_run_metrics.png`
- `/home/runner/work/FINAL-YEAR-PROJ/FINAL-YEAR-PROJ/fecograph_extracted/fecograph/results/plots/final_novelty_vs_run_metrics.csv`

Data used:
- Run result: average of
  - `results/labelprop_0p3_binary/checkpoints/client_0_detailed_metrics.json`
  - `results/labelprop_0p3_binary/checkpoints/client_1_detailed_metrics.json`
- Novelty result:
  - `results/scaffold_zeroshot/scaffold_zeroshot_results.json`

## 📋 Output Format

All plots are saved as high-resolution PNG files (300 DPI) suitable for:
- Research papers
- Presentations
- Technical reports
- Thesis documentation

## 🎨 Customization

To customize plots, edit `plot_results.py`:

```python
# Change color scheme
COLORS = {
    'primary': '#2E86AB',    # Seen class metrics
    'secondary': '#A23B72',   # (unused)
    'success': '#06A77D',     # Combined accuracy
    'warning': '#F18F01',     # Novel class metrics
    'danger': '#C73E1D',      # Target line
    'info': '#6A4C93'         # Headers
}

# Change figure sizes
fig, ax = plt.subplots(figsize=(12, 6))  # (width, height) in inches

# Change DPI (resolution)
plt.savefig(save_path, dpi=300)  # Higher = better quality, larger file
```

## 📈 Understanding the Metrics

### Seen Class Metrics
- **Accuracy**: How well the model classifies known attack types
- **Precision/Recall/F1**: Detailed performance on known attacks

### Novelty Detection Metrics
- **Precision**: Of samples flagged as "novel", how many are truly novel?
- **Recall**: Of all truly novel samples, how many did we detect?
- **F1**: Harmonic mean of precision and recall

### Combined Metrics
- **Combined Accuracy**: Overall correctness (seen correct + novel detected)
- **Optimal Threshold**: Best similarity cutoff for novel detection

### Target
- **98% Accuracy**: Project goal for zero-shot detection

## 🐛 Troubleshooting

### Error: Results file not found
```
❌ Error: Results file not found at results/scaffold_zeroshot/scaffold_zeroshot_results.json
```
**Solution**: Run the training script first:
```bash
python run_scaffold_zeroshot.py
```

### Error: Module not found
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution**: Activate virtual environment:
```bash
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
```

### Plot looks wrong or missing data
**Solution**: Check that training completed successfully and results.json is valid JSON

## 💡 Tips

1. **Run training first** - Visualization requires completed training results
2. **Check output folder** - All plots saved to `visualizations/output/`
3. **Start with summary** - Look at `0_complete_summary.png` first for overview
4. **Compare metrics** - Use `1_overall_metrics.png` to see if target (98%) is met
5. **Analyze failures** - If accuracy is low, check `4_similarity_distributions.png` for overlap

## 🔄 Re-running

You can re-run visualization anytime after training:
```bash
cd visualizations
python plot_results.py
```

It will regenerate all plots from the saved results JSON.
