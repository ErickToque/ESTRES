"""
Genera figuras para el paper basadas en análisis intra-sujeto
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

# Sujetos a mostrar (los mejores)
show_subjects = ['S2', 'S3', 'S10']

print("Generando figuras...")

# Figura 1: Evolución grades vs features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, p in enumerate(show_subjects):
    subject_data = df[df['participant'] == p].copy()
    subject_data['exam_num'] = subject_data['exam'].map(exam_order)
    subject_data = subject_data.sort_values('exam_num')
    
    # Grades
    ax = axes[i, 0]
    grades = subject_data['grade'].values
    ax.plot(exam_labels, grades, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(exam_labels, grades, alpha=0.2)
    ax.set_ylabel('Grade')
    ax.set_title(f'{p}: Grades')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Mejor feature para este sujeto (según análisis previo)
    if p == 'S2':
        best_feat = 'hrv_mean_hr'
    elif p == 'S3':
        best_feat = 'hrv_mean_ibi'
    else:
        best_feat = 'temp_slope'
    
    ax = axes[i, 1]
    if best_feat in subject_data.columns:
        values = subject_data[best_feat].values
        # Normalizar para visualización
        values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)
        ax.plot(exam_labels, values_norm, 'ro-', linewidth=2, markersize=8)
        ax.fill_between(exam_labels, values_norm, alpha=0.2)
        ax.set_ylabel(f'{best_feat} (norm)')
        ax.set_title(f'{p}: {best_feat}')
        ax.grid(True, alpha=0.3)
    
    # Correlación
    ax = axes[i, 2]
    if best_feat in subject_data.columns:
        values = subject_data[best_feat].values
        corr = np.corrcoef(grades, values)[0, 1]
        ax.scatter(grades, values, s=100, c='blue', alpha=0.7)
        # Línea de tendencia
        z = np.polyfit(grades, values, 1)
        p_line = np.poly1d(z)
        ax.plot(grades, p_line(grades), 'r--', linewidth=2)
        ax.set_xlabel('Grade')
        ax.set_ylabel(best_feat)
        ax.set_title(f'{p}: r={corr:.3f}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figuras_paper/figura_intra_sujeto.png', dpi=300)
print("✅ Figura intra-sujeto guardada")

# Figura 2: Comparación modelos por sujeto
model_data = pd.read_csv('analysis/intra_sujeto/comparacion_modelos.csv')

fig, ax = plt.subplots(figsize=(12, 6))
subjects = model_data['participant'].unique()
x = np.arange(len(subjects))
width = 0.2

colors = {'EDA': 'blue', 'HRV': 'green', 'TEMP': 'orange', 'MULTIMODAL': 'red'}

for i, model in enumerate(['EDA', 'HRV', 'TEMP', 'MULTIMODAL']):
    model_subset = model_data[model_data['model'] == model]
    values = []
    for s in subjects:
        val = model_subset[model_subset['participant'] == s]['avg_correlation'].values
        values.append(val[0] if len(val) > 0 else 0)
    ax.bar(x + i*width, values, width, label=model, color=colors.get(model, 'gray'))

ax.set_xlabel('Subject')
ax.set_ylabel('Average Correlation |r|')
ax.set_title('Model Performance by Subject')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(subjects)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('analysis/figuras_paper/figura_modelos_por_sujeto.png', dpi=300)
print("✅ Figura modelos por sujeto guardada")

# Figura 3: Heatmap de correlaciones por sujeto y feature
print("\nGenerando heatmap...")

# Seleccionar features más relevantes
feature_cols = ['eda_mean', 'eda_num_scr', 'hrv_mean_hr', 'hrv_rmssd', 'temp_mean', 'temp_slope']
corr_matrix = []

for p in df['participant'].unique():
    subject_data = df[df['participant'] == p].copy()
    subject_data['exam_num'] = subject_data['exam'].map(exam_order)
    subject_data = subject_data.sort_values('exam_num')
    grades = subject_data['grade'].values
    
    row = []
    for col in feature_cols:
        if col in subject_data.columns:
            values = subject_data[col].values
            if len(np.unique(values)) > 1:
                corr = np.corrcoef(grades, values)[0, 1]
            else:
                corr = 0
        else:
            corr = 0
        row.append(corr)
    corr_matrix.append(row)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(np.arange(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=45, ha='right')
ax.set_yticks(np.arange(len(df['participant'].unique())))
ax.set_yticklabels(df['participant'].unique())
ax.set_xlabel('Features')
ax.set_ylabel('Subject')
ax.set_title('Correlation between Physiological Features and Grades by Subject')
plt.colorbar(im, label='Correlation (r)')
plt.tight_layout()
plt.savefig('analysis/figuras_paper/figura_heatmap_correlaciones.png', dpi=300)
print("✅ Heatmap guardado")

print("\n✅ Todas las figuras guardadas en analysis/figuras_paper/")
