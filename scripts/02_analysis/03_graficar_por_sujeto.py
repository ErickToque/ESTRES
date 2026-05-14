"""
Genera gráficas por sujeto mostrando evolución de features y grades
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# Cargar datos
df = pd.read_csv('results/features_individual_grades.csv')

# Orden de exámenes
exam_order = {'midterm_1': 0, 'midterm_2': 1, 'Final': 2}
exam_labels = ['Midterm 1', 'Midterm 2', 'Final']

participants = df['participant'].unique()
feature_cols = [c for c in df.columns if c.startswith(('eda_', 'hrv_', 'temp_'))]

# Seleccionar features más relevantes
key_features = ['eda_mean', 'eda_num_scr', 'hrv_rmssd', 'hrv_mean_hr', 'temp_mean']

print("Generando gráficas por sujeto...")

for p in participants:
    subject_data = df[df['participant'] == p].copy()
    subject_data['exam_num'] = subject_data['exam'].map(exam_order)
    subject_data = subject_data.sort_values('exam_num')
    
    grades = subject_data['grade'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Gráfico de calificaciones
    ax = axes[0]
    ax.plot(exam_labels, grades, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(exam_labels, grades, alpha=0.2)
    ax.set_ylabel('Grade')
    ax.set_title(f'{p} - Grades')
    ax.grid(True, alpha=0.3)
    
    # Gráficos de features
    idx = 1
    for feat in key_features:
        if feat in subject_data.columns:
            ax = axes[idx]
            values = subject_data[feat].values
            ax.plot(exam_labels, values, 'ro-', linewidth=2, markersize=8)
            ax.fill_between(exam_labels, values, alpha=0.2)
            ax.set_ylabel(feat.replace('_', ' '))
            ax.set_title(feat)
            ax.grid(True, alpha=0.3)
            idx += 1
    
    # Esconder ejes no usados
    for i in range(idx, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'analysis/figuras_intra/{p}_tendencias.png', dpi=150)
    plt.close()
    print(f"   ✅ {p}")

print("\n✅ Figuras guardadas en analysis/figuras_intra/")
