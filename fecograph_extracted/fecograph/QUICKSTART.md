# SCAFFOLD Zero-Shot Visualization - Quick Start Guide

## 📂 What Was Created

```
D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\
│
├── run_scaffold_zeroshot.py       # Main training script (ALREADY EXISTS - FIXED)
├── run_commands.bat               # Windows batch script (NEW)
├── run_commands.sh                # Linux/Mac bash script (NEW)
│
└── visualizations/                # NEW FOLDER
    ├── plot_results.py            # Visualization script
    ├── README.md                  # Documentation
    └── output/                    # Generated plots (created when run)
```

## 🚀 EASIEST WAY TO RUN

### Option 1: Use Interactive Menu (Windows)

Double-click: `run_commands.bat`

OR in terminal:
```cmd
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
run_commands.bat
```

### Option 2: Manual Commands (Windows)

```cmd
# Activate virtual environment
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate

# Step 1: Run training
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
python run_scaffold_zeroshot.py

# Step 2: Generate visualizations
cd visualizations
python plot_results.py
```

## 📊 What You'll Get

### After Training (Step 1):
- `results/scaffold_zeroshot/scaffold_zeroshot_results.json` - All metrics
- `results/scaffold_zeroshot/scaffold_zeroshot_tsne.png` - t-SNE embedding plot
- `results/scaffold_zeroshot/checkpoints/global_model_best.pt` - Best model

### After Visualization (Step 2):
- `visualizations/output/0_complete_summary.png` - **ALL METRICS IN ONE VIEW** ⭐
- `visualizations/output/1_overall_metrics.png` - Bar chart of all metrics
- `visualizations/output/2_seen_vs_novel.png` - Seen vs Novel comparison
- `visualizations/output/3_confusion_summary.png` - Sample distribution
- `visualizations/output/4_similarity_distributions.png` - Similarity analysis
- `visualizations/output/5_class_names.png` - Class information table

## 📈 Understanding the Results

### Key Metrics to Check:

1. **Combined Accuracy** - Must be **> 98%** (target)
2. **Novelty Detection F1** - How well we detect new attacks
3. **Seen Class Accuracy** - How well we classify known attacks

### Visual Indicators:

✅ **Good Result**: Bars exceed red dashed line (98% target)  
⚠️ **Needs Tuning**: Bars below target line

## 🎯 What Was Fixed

### Problem:
Different clients had inconsistent class labels after filtering holdout classes:
- Client 0: 8 classes
- Client 1: 11 classes
- **Result**: Dimension mismatch error

### Solution:
Implemented **two-pass data loading**:
1. **First pass**: Find common seen classes across ALL clients
2. **Second pass**: Load data with consistent label mapping

### Expected Console Output:
```
First pass: Determining common seen classes...
  Client 0 has 8 seen classes: [0, 1, 4, 6, 7, 8, 12, 14]
  Client 1 has 11 seen classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 14]
Common seen classes (8): [0, 1, 4, 6, 7, 8, 12, 14]
Second pass: Loading data with common class filtering...
```

## ⏱️ Expected Runtime

- **Training**: ~30-60 minutes (50 rounds, depends on GPU/CPU)
- **Visualization**: ~10 seconds

## 🔍 What to Do After Running

1. **Check console output** for final metrics
2. **Open** `visualizations/output/0_complete_summary.png` 
3. **Compare** combined accuracy against 98% target
4. **If < 98%**: Tune hyperparameters (see below)

## 🛠️ If Accuracy is Low (< 98%)

Edit `run_scaffold_zeroshot.py` and modify:

```python
# Increase training rounds
NUM_ROUNDS = 100  # Default: 50

# Try different holdout classes (use more common attacks)
HOLDOUT_CLASSES = [9, 10, 11]  # Current
# Alternative: [12, 13, 14] or [1, 2, 3]

# Adjust learning rates
LOCAL_LR = 0.001  # Default: 0.0005
GLOBAL_LR = 1.0   # Default: 1.0

# Increase local epochs
LOCAL_EPOCHS = 10  # Default: 5
```

Then re-run training and visualization.

## 📝 Notes

- Training runs on GPU if available (`cuda`), otherwise CPU
- Results are saved automatically after each run
- You can re-run visualization anytime without re-training
- All plots are high-resolution (300 DPI) suitable for papers/reports

## ❓ Troubleshooting

### "Module not found" error
**Solution**: Make sure virtual environment is activated:
```cmd
D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
```

### "Results file not found" when visualizing
**Solution**: Run training first:
```cmd
python run_scaffold_zeroshot.py
```

### Training takes too long
**Solution**: Reduce NUM_ROUNDS to 10-20 for quick testing:
```python
NUM_ROUNDS = 10  # In run_scaffold_zeroshot.py
```

## 📧 File Structure After Running

```
results/scaffold_zeroshot/
├── scaffold_zeroshot_results.json    # All metrics (JSON)
├── scaffold_zeroshot_tsne.png        # t-SNE visualization
├── checkpoints/
│   └── global_model_best.pt          # Best model weights
└── logs/
    └── training_YYYYMMDD_HHMMSS.log  # Training log

visualizations/output/
├── 0_complete_summary.png            # ⭐ START HERE
├── 1_overall_metrics.png
├── 2_seen_vs_novel.png
├── 3_confusion_summary.png
├── 4_similarity_distributions.png
└── 5_class_names.png
```

---

**Good luck! 🚀**
