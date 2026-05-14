"""
Compara la evolución temporal entre midterms y final
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
exam_colors = {'midterm_1': 'blue', 'midterm_2': 'green', 'Final': 'red'}
exam_labels = {'midterm_1': 'Midterm 1 (90 min)', 
               'midterm_2': 'Midterm 2 (90 min)', 
               'Final': 'Final (180 min)'}

# Crear figura con 3 paneles
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, exam in enumerate(exams):
    ax = axes[i]
    
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
    if len(all_eda) > 0:
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
        
        ax.plot(tiempo, eda_mean, color=exam_colors[exam], linewidth=2)
        ax.fill_between(tiempo, eda_mean - eda_std, eda_mean + eda_std, 
                        color=exam_colors[exam], alpha=0.2)
        
        # Añadir línea de tendencia
        z = np.polyfit(tiempo, eda_mean, 1)
        p_line = np.poly1d(z)
        ax.plot(tiempo, p_line(tiempo), 'k--', linewidth=1, alpha=0.5)
        
        # Calcular pendiente
        pendiente = z[0]  # μS por minuto
        if pendiente > 0:
            tendencia = f"↑ +{pendiente:.3f} μS/min"
        else:
            tendencia = f"↓ {pendiente:.3f} μS/min"
        
        ax.set_xlabel('Tiempo (minutos)')
        ax.set_ylabel('EDA (μS)')
        ax.set_title(f'{exam_labels[exam]}\nTendencia: {tendencia}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/evolucion/comparacion_examenes.png', dpi=150)
print("✅ Comparación entre exámenes guardada")

# Figura: EDA inicial vs final
print("\n📊 EDA: Inicio vs Final del examen")
data = []

for exam in exams:
    for p in participants:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None:
            continue
        eda = eda_df['eda'].values * 10
        n = len(eda)
        inicio = np.mean(eda[:min(100, n//10)])  # primeros 25 segundos
        final = np.mean(eda[-min(100, n//10):])   # últimos 25 segundos
        data.append({
            'participant': p,
            'exam': exam,
            'inicio': inicio,
            'final': final,
            'cambio': final - inicio
        })

df = pd.DataFrame(data)
print(df.groupby('exam').agg({'cambio': ['mean', 'std']}))

df.to_csv('analysis/evolucion/eda_inicio_final.csv', index=False)
print("\n✅ Guardado: analysis/evolucion/eda_inicio_final.csv")
