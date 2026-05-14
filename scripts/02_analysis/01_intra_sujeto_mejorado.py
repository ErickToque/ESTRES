"""
ANÁLISIS INTRA-SUJETO MEJORADO
Hallazgo: Cada individuo tiene su propia firma fisiológica
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Cargar calificaciones
def load_grades_individual():
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

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

print("=" * 70)
print("HALLAZGO 1: ANÁLISIS INTRA-SUJETO")
print("Cada individuo tiene su propia firma fisiológica")
print("=" * 70)

# Almacenar resultados
intra_results = []

for p in participants:
    print(f"\n📌 {p}:")
    
    grades = []
    eda_vals = []
    hr_vals = []
    temp_vals = []
    
    for exam in exams:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        
        grade = grades_raw[p][exam]
        if exam == 'Final':
            grade = grade / 2
        
        grades.append(grade)
        
        # EDA
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is not None:
            eda = eda_df['eda'].values * 10
            eda_no_ceros = eda[eda > 0.01]
            if len(eda_no_ceros) > 0:
                eda_vals.append(np.mean(eda_no_ceros))
            else:
                eda_vals.append(np.nan)
        else:
            eda_vals.append(np.nan)
        
        # HR
        ibi_df = load_ibi(p, exam)
        if ibi_df is not None:
            if 'ibi' in ibi_df.columns:
                ibi = ibi_df['ibi'].values
            else:
                ibi = ibi_df.iloc[:, 1].values
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
            if len(ibi) > 5:
                hr_vals.append(60.0 / np.mean(ibi))
            else:
                hr_vals.append(np.nan)
        else:
            hr_vals.append(np.nan)
        
        # TEMP
        temp_df = load_signal(p, exam, 'TEMP')
        if temp_df is not None:
            temp_vals.append(np.mean(temp_df['temp'].values))
        else:
            temp_vals.append(np.nan)
    
    # Calcular correlaciones intra-sujeto
    grades_arr = np.array(grades)
    
    # EDA
    eda_arr = np.array(eda_vals)
    mask = ~np.isnan(eda_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        eda_corr, eda_p = pearsonr(grades_arr[mask], eda_arr[mask])
        eda_sig = "✅" if eda_p < 0.05 else " "
        print(f"   EDA: r={eda_corr:.3f} (p={eda_p:.3f}) {eda_sig}")
        intra_results.append({'participant': p, 'signal': 'EDA', 'correlation': eda_corr, 'p_value': eda_p})
    
    # HR
    hr_arr = np.array(hr_vals)
    mask = ~np.isnan(hr_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        hr_corr, hr_p = pearsonr(grades_arr[mask], hr_arr[mask])
        hr_sig = "✅" if hr_p < 0.05 else " "
        print(f"   HR:  r={hr_corr:.3f} (p={hr_p:.3f}) {hr_sig}")
        intra_results.append({'participant': p, 'signal': 'HR', 'correlation': hr_corr, 'p_value': hr_p})
    
    # TEMP
    temp_arr = np.array(temp_vals)
    mask = ~np.isnan(temp_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        temp_corr, temp_p = pearsonr(grades_arr[mask], temp_arr[mask])
        temp_sig = "✅" if temp_p < 0.05 else " "
        print(f"   TEMP: r={temp_corr:.3f} (p={temp_p:.3f}) {temp_sig}")
        intra_results.append({'participant': p, 'signal': 'TEMP', 'correlation': temp_corr, 'p_value': temp_p})

# Resumen
print("\n" + "=" * 70)
print("RESUMEN INTRA-SUJETO")
print("=" * 70)

df_intra = pd.DataFrame(intra_results)
print(f"\nCorrelaciones significativas (p<0.05): {len(df_intra[df_intra['p_value'] < 0.05])} de {len(df_intra)}")

for signal in ['EDA', 'HR', 'TEMP']:
    sig = df_intra[(df_intra['signal'] == signal) & (df_intra['p_value'] < 0.05)]
    print(f"{signal}: {len(sig)} sujetos con correlación significativa")

# Guardar
df_intra.to_csv('analysis/enfoque_mejorado/intra_sujeto_correlaciones.csv', index=False)
print("\n✅ Guardado: analysis/enfoque_mejorado/intra_sujeto_correlaciones.csv")
