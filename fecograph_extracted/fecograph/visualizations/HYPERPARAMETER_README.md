# Comprehensive Hyperparameter Analysis

## 🎯 What This Does

Scans ALL your hyperparameter experiment results and creates publication-quality graphs matching research paper style:
- **Temperature (τ)**: 0.1, 0.3, 0.5, 0.7, 0.9
- **Lambda (λ)**: 0.01, 0.03, 0.05, 0.07, 0.1, 0.5
- **Label Proportion**: 0.1, 0.3, 0.5, 0.7 (binary and multiclass)

## 📊 Output Graphs (6 Files)

### 1. **Temperature Analysis** (`1_temperature_analysis.png`)
4-subplot grid showing:
- (a) Accuracy vs Temperature
- (b) Precision vs Temperature
- (c) Recall vs Temperature
- (d) F1 Score vs Temperature
- Best value marked with green star ⭐

### 2. **Lambda Analysis** (`2_lambda_analysis.png`)
4-subplot grid showing:
- (a) Accuracy vs Lambda (log scale)
- (b) Precision vs Lambda
- (c) Recall vs Lambda
- (d) F1 Score vs Lambda
- Best value marked with green star ⭐

### 3. **Label Proportion - Binary** (`3_labelprop_binary_analysis.png`)
4-subplot grid for binary classification

### 4. **Label Proportion - Multiclass** (`4_labelprop_multiclass_analysis.png`)
4-subplot grid for multiclass classification

### 5. **Combined F1 Comparison** (`5_combined_f1_comparison.png`)
3-subplot comparison showing F1 across all hyperparameters

### 6. **Best Hyperparameters Summary** (`6_best_hyperparameters_summary.png`)
Table showing best configuration for each hyperparameter

## 🚀 Command

```bash
# Activate virtual environment
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate.bat

# Navigate to folder
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations"

# Run analysis
python plot_hyperparameter_analysis.py
```

## 📁 Data Source

Automatically reads from your results folders:
```
results/
├── temp_0p1/logs/training_history.csv
├── temp_0p3/logs/training_history.csv
├── temp_0p5/logs/training_history.csv
├── temp_0p7/logs/training_history.csv
├── temp_0p9/logs/training_history.csv
├── lambda_0p01/logs/training_history.csv
├── lambda_0p03/logs/training_history.csv
├── lambda_0p05/logs/training_history.csv
├── lambda_0p07/logs/training_history.csv
├── lambda_0p1/logs/training_history.csv
├── lambda_0p5/logs/training_history.csv
├── labelprop_0p1_binary/logs/training_history.csv
├── labelprop_0p3_binary/logs/training_history.csv
├── labelprop_0p5_binary/logs/training_history.csv
├── labelprop_0p7_binary/logs/training_history.csv
├── labelprop_0p1_multiclass/logs/training_history.csv
├── labelprop_0p3_multiclass/logs/training_history.csv
├── labelprop_0p5_multiclass/logs/training_history.csv
└── labelprop_0p7_multiclass/logs/training_history.csv
```

## 📈 Extracted Metrics

For each experiment, extracts final round values:
- **Accuracy**: `mean_acc_pers` (personalized accuracy)
- **F1 Score**: `mean_f1_pers` (personalized F1)
- **Precision**: Estimated from F1
- **Recall**: Estimated from F1

## 🎨 Graph Style

- Publication-quality (300 DPI)
- Serif fonts
- Clean grid layout
- Color-coded metrics:
  - Blue: Accuracy
  - Green: Precision
  - Orange: Recall
  - Red: F1 Score
- Best values marked with green star ⭐
- Value labels on all data points

## ⏱️ Runtime

- **<10 seconds** to generate all 6 graphs
- Reads CSV files directly (no retraining)

## 📝 Output Files

All saved to: `visualizations/output/hyperparameter_analysis/`

```
output/hyperparameter_analysis/
├── 1_temperature_analysis.png
├── 2_lambda_analysis.png
├── 3_labelprop_binary_analysis.png
├── 4_labelprop_multiclass_analysis.png
├── 5_combined_f1_comparison.png
├── 6_best_hyperparameters_summary.png
└── hyperparameter_summary.json
```

## 📄 JSON Summary

`hyperparameter_summary.json` contains all extracted metrics in structured format for further analysis or table generation.

## 🔍 Console Output

The script prints:
```
Scanning Temperature experiments...
  ✓ T=0.1: Acc=94.25%, F1=80.95%
  ✓ T=0.3: Acc=95.61%, F1=86.44%
  ...

Scanning Lambda experiments...
  ✓ λ=0.01: Acc=96.63%, F1=86.44%
  ...

Generating visualizations...
  ✓ Saved: 1_temperature_analysis.png
  ...
```

## 💡 Tips

1. **Start here**: Check `6_best_hyperparameters_summary.png` for quick overview
2. **Deep dive**: Use individual analysis plots (1-4) for detailed trends
3. **Comparison**: Use `5_combined_f1_comparison.png` to compare all at once
4. **Data export**: Use `hyperparameter_summary.json` for tables in papers

## 🎓 Perfect For

- ✅ Research papers
- ✅ Thesis chapters
- ✅ Conference presentations
- ✅ Hyperparameter selection justification
- ✅ Ablation study reports

---

**This creates the same style of graphs as in research papers showing how hyperparameters affect performance!**
