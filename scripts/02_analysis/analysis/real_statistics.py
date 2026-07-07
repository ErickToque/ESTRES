"""
Cálculo de estadísticas REALES basadas en los resultados del ablation study
"""
import numpy as np
from scipy.stats import wilcoxon

# ============================================================================
# RESULTADOS REALES DEL ABLATION STUDY (debes completar con tus datos)
# ============================================================================
# Estos son los accuracy por fold (LOSO) para cada modelo
# Cada fold corresponde a un sujeto (S1 a S10)

# EJEMPLO - Reemplazar con tus resultados reales
eda_only_folds = [0.67, 0.33, 1.00, 0.33, 0.33, 0.67, 0.33, 0.67, 0.33, 0.67]  # 10 valores
eda_acc_folds = [0.67, 0.33, 1.00, 0.67, 0.33, 0.67, 0.33, 0.67, 0.67, 0.67]  # 10 valores
acc_only_folds = [0.67, 0.33, 1.00, 0.67, 0.33, 0.67, 0.33, 0.67, 0.67, 0.67]  # 10 valores

print("=" * 70)
print("ESTADÍSTICAS REALES DEL ABLATION STUDY")
print("=" * 70)

# 1. Wilcoxon tests
print("\n1. WILCOXON SIGNED-RANK TEST")
stat, p_eda_vs_edaacc = wilcoxon(eda_only_folds, eda_acc_folds)
print(f"EDA-only vs EDA+ACC: p = {p_eda_vs_edaacc:.4f}")

stat, p_acc_vs_edaacc = wilcoxon(acc_only_folds, eda_acc_folds)
print(f"ACC-only vs EDA+ACC: p = {p_acc_vs_edaacc:.4f}")

# 2. Interpretación
print("\n2. INTERPRETACIÓN")
if p_eda_vs_edaacc < 0.05:
    print(" La mejora de EDA+ACC sobre EDA-only es ESTADÍSTICAMENTE SIGNIFICATIVA")
else:
    print(" La mejora de EDA+ACC sobre EDA-only NO es estadísticamente significativa")

# 3. Matriz de confusión (si tienes las predicciones reales)
print("\n3. PARA ACTUALIZAR: Necesitas las predicciones reales por muestra")
print("   Ejecuta ablation_study_v3.py y guarda las predicciones")

print("\n Para obtener resultados reales, ejecuta primero ablation_study_v3.py")
