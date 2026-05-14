"""
Análisis de evolución de temperatura durante exámenes
La temperatura suele DISMINUIR por vasoconstricción por estrés
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import sys
sys.path.append('.')

from scripts.utils import load_signal

participants = ['S1', 'S2', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, p in enumerate(participants):
    ax = axes[idx]
    
    for exam in exams:
        temp_df = load_signal(p, exam, 'TEMP')
        if temp_df is None:
            continue
        
        temp = temp_df['temp'].values
        fs = 4.0
        tiempo = np.arange(len(temp)) / fs / 60
        
        temp_smooth = uniform_filter1d(temp, size=min(300, len(temp)//10))
        
        color = {'midterm_1': 'blue', 'midterm_2': 'green', 'Final': 'red'}[exam]
        ax.plot(tiempo, temp_smooth, color=color, linewidth=1.5, 
                label=exam_labels[exam])
    
    ax.set_xlabel('Tiempo (minutos)')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title(f'{p} - Evolución de temperatura')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/evolucion/evolucion_temperatura.png', dpi=150)
print("✅ Evolución temperatura guardada")
