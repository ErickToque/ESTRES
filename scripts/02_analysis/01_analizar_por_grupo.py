"""
Comparación de evolución fisiológica por grupo de rendimiento
- Alto rendimiento (grade >80)
- Bajo rendimiento (grade <70)
- Rendimiento medio (70-80)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import signal
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Cargar calificaciones individuales
def load_individual_grades():
    """Cargar calificaciones por sujeto y examen"""
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
                student_id = parts[0].strip()
                student_num = int(student_id[1:])
                student_key = f'S{student_num}'
                try:
                    grade = float(parts[-1])
                    if student_key not in grades_raw:
                        grades_raw[student_key] = {}
                    grades_raw[student_key][current_exam] = grade
                except:
                    pass
    return grades_raw

grades_raw = load_individual_grades()

# Definir grupos
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

# Clasificar por rendimiento PROMEDIO
avg_grades = {}
for p in participants:
    if p in grades_raw:
        grades = list(grades_raw[p].values())
        # Convertir final (sobre 200) a escala 100
        final_idx = list(grades_raw[p].keys()).index('Final') if 'Final' in grades_raw[p] else -1
        if final_idx >= 0:
            grades[final_idx] = grades[final_idx] / 2
        avg_grades[p] = np.mean(grades)

alto_rendimiento = [p for p in avg_grades if avg_grades[p] >= 80]
medio_rendimiento = [p for p in avg_grades if 70 <= avg_grades[p] < 80]
bajo_rendimiento = [p for p in avg_grades if avg_grades[p] < 70]

print("=" * 70)
print("ANÁLISIS POR GRUPO DE RENDIMIENTO")
print("=" * 70)
print(f"\nAlto rendimiento (≥80): {alto_rendimiento}")
print(f"  Grades: {[avg_grades[p] for p in alto_rendimiento]}")
print(f"\nMedio rendimiento (70-80): {medio_rendimiento}")
print(f"  Grades: {[avg_grades[p] for p in medio_rendimiento]}")
print(f"\nBajo rendimiento (<70): {bajo_rendimiento}")
print(f"  Grades: {[avg_grades[p] for p in bajo_rendimiento]}")

# Análisis por examen
exams = ['midterm_1', 'midterm_2', 'Final']

resultados = []

for exam in exams:
    print(f"\n{'='*50}")
    print(f"EXAMEN: {exam.upper()}")
    print(f"{'='*50}")
    
    for grupo, nombre in [(alto_rendimiento, 'Alto'), (medio_rendimiento, 'Medio'), (bajo_rendimiento, 'Bajo')]:
        if len(grupo) == 0:
            continue
            
        eda_vals = []
        hr_vals = []
        temp_vals = []
        
        for p in grupo:
            # EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is not None:
                eda = eda_df['eda'].values * 10
                # Evitar ceros para la media
                eda_no_ceros = eda[eda > 0.01]
                if len(eda_no_ceros) > 0:
                    eda_vals.append(np.mean(eda_no_ceros))
            
            # HRV (frecuencia cardíaca)
            ibi_df = load_ibi(p, exam)
            if ibi_df is not None:
                if 'ibi' in ibi_df.columns:
                    ibi = ibi_df['ibi'].values
                else:
                    ibi = ibi_df.iloc[:, 1].values
                ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                if len(ibi) > 5:
                    hr = 60.0 / np.mean(ibi)
                    hr_vals.append(hr)
            
            # Temperatura
            temp_df = load_signal(p, exam, 'TEMP')
            if temp_df is not None:
                temp = temp_df['temp'].values
                temp_vals.append(np.mean(temp))
        
        if eda_vals:
            resultados.append({
                'grupo': nombre,
                'examen': exam,
                'senal': 'EDA',
                'media': np.mean(eda_vals),
                'std': np.std(eda_vals),
                'n': len(eda_vals)
            })
            print(f"\n{nombre} rendimiento - EDA: {np.mean(eda_vals):.2f} ± {np.std(eda_vals):.2f} μS (n={len(eda_vals)})")
        
        if hr_vals:
            resultados.append({
                'grupo': nombre,
                'examen': exam,
                'senal': 'HR',
                'media': np.mean(hr_vals),
                'std': np.std(hr_vals),
                'n': len(hr_vals)
            })
            print(f"{nombre} rendimiento - HR: {np.mean(hr_vals):.1f} ± {np.std(hr_vals):.1f} bpm (n={len(hr_vals)})")
        
        if temp_vals:
            resultados.append({
                'grupo': nombre,
                'examen': exam,
                'senal': 'TEMP',
                'media': np.mean(temp_vals),
                'std': np.std(temp_vals),
                'n': len(temp_vals)
            })
            print(f"{nombre} rendimiento - TEMP: {np.mean(temp_vals):.1f} ± {np.std(temp_vals):.1f} °C (n={len(temp_vals)})")

# Guardar resultados
df_res = pd.DataFrame(resultados)
df_res.to_csv('analysis/grupos_rendimiento/comparacion_grupos.csv', index=False)
print("\n✅ Guardado: analysis/grupos_rendimiento/comparacion_grupos.csv")
