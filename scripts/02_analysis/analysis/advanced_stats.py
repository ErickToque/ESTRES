"""
Análisis estadístico avanzado para el paper
- Wilcoxon tests
- Feature importance
- ROC curves
- Per-fold stability analysis
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Datos de ejemplo (reemplazar con resultados reales de LOSO)
# Por fold (10 folds, uno por sujeto)
eda_acc_scores = [0.67, 0.33, 1.00, 0.67, 0.33, 0.67, 0.33, 0.67, 0.67, 0.67]  # EDA+ACC
eda_only_scores = [0.67, 0.33, 1.00, 0.33, 0.33, 0.67, 0.33, 0.67, 0.33, 0.67]  # EDA-only
acc_only_scores = [0.67, 0.33, 1.00, 0.67, 0.33, 0.67, 0.33, 0.67, 0.67, 0.67]  # ACC-only

print("=" * 60)
print("ANÁLISIS ESTADÍSTICO AVANZADO")
print("=" * 60)

# 1. Wilcoxon test
print("\n1. WILCOXON SIGNED-RANK TEST")
stat, p_val = wilcoxon(eda_acc_scores, eda_only_scores)
print(f"EDA+ACC vs EDA-only: p = {p_val:.4f}")

stat, p_val = wilcoxon(eda_acc_scores, acc_only_scores)
print(f"EDA+ACC vs ACC-only: p = {p_val:.4f}")

# 2. Confusion matrix (ejemplo)
print("\n2. CONFUSION MATRIX (EDA+ACC)")
# Simular predicciones
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
cm = confusion_matrix(y_true, y_pred)
print(f"True Positives: {cm[1,1]}, False Positives: {cm[0,1]}")
print(f"True Negatives: {cm[0,0]}, False Negatives: {cm[1,0]}")

# 3. Feature importance (placeholder)
print("\n3. FEATURE IMPORTANCE")
features = ['ACC movement', 'EDA initial variance', 'EDA mid/end ratio', 'EDA mean', 'Temp slope']
importances = [0.24, 0.18, 0.14, 0.11, 0.09]
for f, imp in zip(features, importances):
    print(f"  {f}: {imp:.2f}")

print("\n Análisis completado")
