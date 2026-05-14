"""
Gráficas de evolución temporal de bioseñales durante exámenes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import sys
sys.path.append('.')

from scripts.utils import load_signal

participants = ['S1', 'S2', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1 (90 min)', 
               'midterm_2': 'Midterm 2 (90 min)', 
               'Final': 'Final (180 min)'}

print("Generando gráficas de evolución temporal...")

# Crear figura comparativa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, p in enumerate(participants[:4]):
    ax = axes[idx]
    
    for exam in exams:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None:
            continue
        
        eda = eda_df['eda'].values * 10
        fs = 4.0
        tiempo = np.arange(len(eda)) / fs / 60  # minutos
        
        # Suavizado para ver tendencia
        eda_smooth = uniform_filter1d(eda, size=min(300, len(eda)//10))
        
        # Color según examen
        color = {'midterm_1': 'blue', 'midterm_2': 'green', 'Final': 'red'}[exam]
        label = exam_labels[exam]
        
        ax.plot(tiempo, eda_smooth, color=color, linewidth=1.5, label=label)
    
    ax.set_xlabel('Tiempo (minutos)')
    ax.set_ylabel('EDA (μS)')
    ax.set_title(f'{p} - Evolución de EDA durante exámenes')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/evolucion/evolucion_eda_por_sujeto.png', dpi=150)
print("✅ Evolución EDA por sujeto guardada")

# Figura: Promedio de todos los sujetos
fig, ax = plt.subplots(figsize=(10, 6))

for exam in exams:
    all_eda = []
    max_len = 0
    
    for p in participants:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None:
            continue
        eda = eda_df['eda'].values * 10
        all_eda.append(eda)
        max_len = max(max_len, len(eda))
    
    # Interpolar a longitud común
    from scipy import interpolate
    eda_interp = []
    for eda in all_eda:
        x_old = np.linspace(0, 1, len(eda))
        x_new = np.linspace(0, 1, max_len)
        f = interpolate.interp1d(x_old, eda, kind='linear')
        eda_interp.append(f(x_new))
    
    eda_mean = np.mean(eda_interp, axis=0)
    eda_std = np.std(eda_interp, axis=0)
    
    tiempo = np.arange(max_len) / 4 / 60  # minutos
    color = {'midterm_1': 'blue', 'midterm_2': 'green', 'Final': 'red'}[exam]
    
    ax.plot(tiempo, eda_mean, color=color, linewidth=2, label=exam_labels[exam])
    ax.fill_between(tiempo, eda_mean - eda_std, eda_mean + eda_std, 
                    color=color, alpha=0.2)

ax.set_xlabel('Tiempo (minutos)')
ax.set_ylabel('EDA (μS)')
ax.set_title('Evolución promedio de EDA durante exámenes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/evolucion/evolucion_eda_promedio.png', dpi=150)
print("✅ Evolución EDA promedio guardada")

print("\n✅ Gráficas guardadas en analysis/evolucion/")
