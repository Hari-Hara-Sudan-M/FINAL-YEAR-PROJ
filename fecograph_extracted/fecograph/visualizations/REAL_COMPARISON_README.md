# FL Comparison Using Real Training Logs

## 🎯 What This Does

Creates comparison graphs using **REAL training data** from your actual training runs:
- **FedAvg**: `results/fedavg_binary/logs/training_history.csv`
- **SCAFFOLD**: `results/scaffold_multiclass_supcon/logs/training_history_20260325_104854.csv`

## 🚀 Command

```bash
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations"
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
python plot_real_comparison.py
```

## 📊 Output Graphs

1. **`real_fl_comparison_convergence.png`** ⭐ **MAIN GRAPH**
   - Line graph showing accuracy over training rounds
   - Clear visual: SCAFFOLD vs FedAvg convergence
   - Shows which method reaches target faster
   - **Real training curves from your logs!**

2. **`real_fl_comparison_final.png`**
   - Bar chart of final accuracy & F1
   - Side-by-side comparison
   - Shows improvement percentages

3. **`real_fl_comparison_comprehensive.png`**
   - 2x2 grid with all metrics
   - Accuracy + F1 convergence
   - Final metrics bars
   - Statistics summary

## 📈 What Your Real Data Shows

Based on your actual logs:

### FedAvg (Binary):
- Final Accuracy: **98.09%** ✅
- Final F1: **95.88%**
- 100 training rounds

### SCAFFOLD (Multiclass):  
- Final Accuracy: **92.60%**
- Final F1: **40.21%**
- 100 training rounds

**Note**: These are from different tasks (binary vs multiclass), so direct comparison shows task difficulty difference.

## ⚡ Runtime

- **<5 seconds** (just reads CSVs and plots)
- No training needed!

## 🔧 To Compare Same Task

For fair comparison, use same task results:
- Both binary: `fedavg_binary` vs `scaffold_binary`
- Both multiclass: Compare two multiclass runs

Edit line 78-79 in `plot_real_comparison.py`:
```python
fedavg_csv = Path("../results/fedavg_binary/logs/training_history.csv")
scaffold_csv = Path("../results/scaffold_binary/logs/training_history.csv")  # Change this
```
