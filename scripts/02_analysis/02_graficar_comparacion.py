"""
Gráficas comparativas entre grupos de rendimiento
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Cargar calificaciones
def load_individual_grades():
    grades_raw = {}
    with open('data/wearable-exam-stress/StudentGrades.txt', 'r', encoding='latin1') as f:
        lines = f.readlines()
    current_exam = None
    for line in lines:
        line = line.strip()
        if 'MIDTERM 1' in line.upper():
            current_exam = 'midterm_1'
        elif 'MIDTERM 2' in line.upper():
            current_exam = 'midterm_2'
        elif 'FINAL' in line.upper():
            current_exam = 'Final'
        elif line.startswith('S') and current_exam:
            parts = line.split()
            if len(parts) >= 2:
                student_num = int(parts[0].strip()[1:])
                student_key = f'S{student_num}'
                if student_key not in grades_raw:
                    grades_raw[student_key] = {}
                grades_raw[student_key][current_exam] = float(parts[-1])
    return grades_raw

grades_raw = load_individual_grades()

participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

# Calcular promedio
avg_grades = {}
for p in participants:
    if p in grades_raw:
        grades = list(grades_raw[p].values())
        final_idx = list(grades_raw[p].keys()).index('Final') if 'Final' in grades_raw[p] else -1
        if final_idx >= 0:
            grades[final_idx] = grades[final_idx] / 2
        avg_grades[p] = np.mean(grades)

alto = [p for p in avg_grades if avg_grades[p] >= 80]
medio = [p for p in avg_grades if 70 <= avg_grades[p] < 80]
bajo = [p for p in avg_grades if avg_grades[p] < 70]

# Evolución temporal por grupo
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. EDA por grupo
ax = axes[0, 0]
for grupo, nombre, color in [(alto, 'Alto (≥80)', 'green'), 
                              (medio, 'Medio (70-80)', 'orange'),
                              (bajo, 'Bajo (<70)', 'red')]:
    if len(grupo) == 0:
        continue
    valores = []
    for exam in exams:
        eda_vals = []
        for p in grupo:
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is not None:
                eda = eda_df['eda'].values * 10
                eda_no_ceros = eda[eda > 0.01]
                if len(eda_no_ceros) > 0:
                    eda_vals.append(np.mean(eda_no_ceros))
        valores.append(np.mean(eda_vals) if eda_vals else 0)
    ax.plot(exam_labels.values(), valores, 'o-', color=color, linewidth=2, markersize=8, label=nombre)
ax.set_ylabel('EDA (μS)')
ax.set_title('Electrodermal Activity por grupo')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. HR por grupo
ax = axes[0, 1]
for grupo, nombre, color in [(alto, 'Alto (≥80)', 'green'), 
                              (medio, 'Medio (70-80)', 'orange'),
                              (bajo, 'Bajo (<70)', 'red')]:
    if len(grupo) == 0:
        continue
    valores = []
    for exam in exams:
        hr_vals = []
        for p in grupo:
            ibi_df = load_ibi(p, exam)
            if ibi_df is not None:
                if 'ibi' in ibi_df.columns:
                    ibi = ibi_df['ibi'].values
                else:
                    ibi = ibi_df.iloc[:, 1].values
                ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                if len(ibi) > 5:
                    hr_vals.append(60.0 / np.mean(ibi))
        valores.append(np.mean(hr_vals) if hr_vals else 0)
    ax.plot(exam_labels.values(), valores, 'o-', color=color, linewidth=2, markersize=8, label=nombre)
ax.set_ylabel('Heart Rate (bpm)')
ax.set_title('Frecuencia Cardíaca por grupo')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Temperatura por grupo
ax = axes[1, 0]
for grupo, nombre, color in [(alto, 'Alto (≥80)', 'green'), 
                              (medio, 'Medio (70-80)', 'orange'),
                              (bajo, 'Bajo (<70)', 'red')]:
    if len(grupo) == 0:
        continue
    valores = []
    for exam in exams:
        temp_vals = []
        for p in grupo:
            temp_df = load_signal(p, exam, 'TEMP')
            if temp_df is not None:
                temp_vals.append(np.mean(temp_df['temp'].values))
        valores.append(np.mean(temp_vals) if temp_vals else 0)
    ax.plot(exam_labels.values(), valores, 'o-', color=color, linewidth=2, markersize=8, label=nombre)
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperatura por grupo')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Relación estrés vs rendimiento (cambio de HR durante examen)
ax = axes[1, 1]
datos_relacion = []
for p in participants:
    ibi_m1 = load_ibi(p, 'midterm_1')
    ibi_final = load_ibi(p, 'Final')
    if ibi_m1 is not None and ibi_final is not None:
        if 'ibi' in ibi_m1.columns:
            hr_m1 = 60 / np.mean(ibi_m1['ibi'].values)
            hr_final = 60 / np.mean(ibi_final['ibi'].values)
        else:
            hr_m1 = 60 / np.mean(ibi_m1.iloc[:, 1].values)
            hr_final = 60 / np.mean(ibi_final.iloc[:, 1].values)
        cambio_hr = hr_final - hr_m1
        if p in avg_grades:
            datos_relacion.append((avg_grades[p], cambio_hr))

datos_relacion.sort(key=lambda x: x[0])
grades_plot = [d[0] for d in datos_relacion]
cambio_plot = [d[1] for d in datos_relacion]

ax.bar(range(len(grades_plot)), cambio_plot, color=['green' if g>80 else 'orange' if g>70 else 'red' for g in grades_plot])
ax.set_xticks(range(len(grades_plot)))
ax.set_xticklabels([f'S{i+1}' for i in range(len(grades_plot))], rotation=45)
ax.set_ylabel('Cambio HR (Final - Midterm 1) bpm')
ax.set_title('Cambio en frecuencia cardíaca vs Rendimiento')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/grupos_rendimiento/comparacion_grupos_completa.png', dpi=150)
print("✅ Figura comparativa guardada: analysis/grupos_rendimiento/comparacion_grupos_completa.png")
