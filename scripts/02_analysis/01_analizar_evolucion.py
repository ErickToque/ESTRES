"""
Análisis de la evolución de bioseñales durante los exámenes
No necesita grades - solo muestra tendencias temporales
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import sys
sys.path.append('.')

from scripts.utils import load_signal

# Configuración
participants = ['S1', 'S2', 'S9', 'S10']  # Solo buena calidad EDA
exams = ['midterm_1', 'midterm_2', 'Final']
exam_durations = {'midterm_1': 90, 'midterm_2': 90, 'Final': 180}  # minutos

print("=" * 70)
print("ANÁLISIS DE EVOLUCIÓN DE BIOSEÑALES DURANTE EXÁMENES")
print("=" * 70)

# Almacenar resultados
resultados = []

for exam in exams:
    print(f"\n📚 {exam.upper()} (duración: {exam_durations[exam]} min)")
    print("-" * 50)
    
    # Para este examen, calcular promedios por cuartil
    all_eda = []
    all_hrv = []
    all_temp = []
    
    for p in participants:
        eda_df = load_signal(p, exam, 'EDA')
        ibi_df = load_ibi(p, exam)
        temp_df = load_signal(p, exam, 'TEMP')
        
        if eda_df is None:
            continue
        
        eda = eda_df['eda'].values * 10  # corrección x10
        fs = 4.0
        n_samples = len(eda)
        
        # Dividir en 4 cuartiles temporales
        cuartiles = np.array_split(eda, 4)
        tiempos = [f"Q{i+1}" for i in range(4)]
        
        eda_medias = [np.mean(q) for q in cuartiles]
        all_eda.append(eda_medias)
        
        # HRV si está disponible
        if ibi_df is not None:
            if 'ibi' in ibi_df.columns:
                ibi = ibi_df['ibi'].values
            else:
                ibi = ibi_df.iloc[:, 1].values
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
            if len(ibi) > 10:
                hr = 60.0 / np.mean(ibi)
                all_hrv.append(hr)
        
        # Temperatura
        if temp_df is not None:
            temp = temp_df['temp'].values
            temp_cuartiles = np.array_split(temp, 4)
            temp_medias = [np.mean(t) for t in temp_cuartiles]
            all_temp.append(temp_medias)
    
    # Promedios entre participantes
    if all_eda:
        eda_promedio = np.mean(all_eda, axis=0)
        eda_std = np.std(all_eda, axis=0)
        
        print(f"  EDA (μS) por cuartil:")
        for i, (m, s) in enumerate(zip(eda_promedio, eda_std)):
            print(f"    Q{i+1}: {m:.2f} ± {s:.2f}")
        
        # Tendencia
        if eda_promedio[-1] > eda_promedio[0]:
            tendencia = "↑ AUMENTA durante el examen"
        else:
            tendencia = "↓ DISMINUYE durante el examen"
        print(f"  Tendencia EDA: {tendencia}")
        
        resultados.append({
            'exam': exam,
            'signal': 'EDA',
            'q1': eda_promedio[0],
            'q2': eda_promedio[1],
            'q3': eda_promedio[2],
            'q4': eda_promedio[3],
            'trend': tendencia
        })
    
    if all_hrv:
        hr_promedio = np.mean(all_hrv)
        print(f"  HRV: HR promedio = {hr_promedio:.1f} bpm")
    
    if all_temp:
        temp_promedio = np.mean(all_temp, axis=0)
        print(f"  Temperatura (°C) por cuartil:")
        for i, t in enumerate(temp_promedio):
            print(f"    Q{i+1}: {t:.1f}")

# Guardar resultados
df_res = pd.DataFrame(resultados)
df_res.to_csv('analysis/evolucion/evolucion_por_examen.csv', index=False)
print("\n✅ Guardado: analysis/evolucion/evolucion_por_examen.csv")
