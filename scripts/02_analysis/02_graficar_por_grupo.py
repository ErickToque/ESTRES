"""
Gráficas comparativas entre grupos (clasificación por promedio)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

df = pd.read_csv('analysis/clasificacion_correcta/datos_con_grupo.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (senal, color, ax) in enumerate(zip(['eda', 'hr', 'temp'], 
                                            ['blue', 'red', 'green'], axes)):
    
    alto = df[df['grupo_estudiante'] == 'Alto'][senal].dropna()
    bajo = df[df['grupo_estudiante'] == 'Bajo'][senal].dropna()
    
    # Boxplot
    bp = ax.boxplot([alto, bajo], labels=['Alto\nrendimiento', 'Bajo\nrendimiento'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # Test estadístico
    stat, p_valor = mannwhitneyu(alto, bajo, alternative='two-sided')
    
    ax.set_ylabel(senal.upper())
    ax.set_title(f'{senal.upper()}\np = {p_valor:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Añadir significancia
    if p_valor < 0.05:
        ax.text(0.5, 0.95, '* p<0.05', transform=ax.transAxes, 
                ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/clasificacion_correcta/comparacion_grupos.png', dpi=150)
print("✅ Figura guardada: analysis/clasificacion_correcta/comparacion_grupos.png")

# Evolución temporal por grupo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = ['Midterm 1', 'Midterm 2', 'Final']

for i, senal in enumerate(['eda', 'hr', 'temp']):
    ax = axes[i]
    
    for grupo, color, label in [('Alto', 'green', 'Alto rendimiento'),
                                  ('Bajo', 'red', 'Bajo rendimiento')]:
        valores = []
        for exam in exams:
            subset = df[(df['grupo_estudiante'] == grupo) & (df['exam'] == exam)]
            vals = subset[senal].dropna()
            valores.append(np.mean(vals) if len(vals) > 0 else 0)
        
        ax.plot(exam_labels, valores, 'o-', color=color, linewidth=2, 
                markersize=8, label=label)
    
    ax.set_xlabel('Examen')
    ax.set_ylabel(senal.upper())
    ax.set_title(f'Evolución de {senal.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/clasificacion_correcta/evolucion_por_grupo.png', dpi=150)
print("✅ Figura guardada: analysis/clasificacion_correcta/evolucion_por_grupo.png")
