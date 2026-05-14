"""
ANÁLISIS INTRA-SUJETO FINAL
Correlaciones entre señales fisiológicas y calificaciones por sujeto
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

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
print("RESULTADO 1: ANÁLISIS INTRA-SUJETO")
print("¿Cada individuo tiene su propia firma fisiológica?")
print("=" * 70)

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
    
    # Correlaciones
    grades_arr = np.array(grades)
    
    # EDA
    eda_arr = np.array(eda_vals)
    mask = ~np.isnan(eda_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        corr, p_val = pearsonr(grades_arr[mask], eda_arr[mask])
        sig = "✅" if p_val < 0.05 else " "
        print(f"   EDA: r={corr:.3f} (p={p_val:.3f}) {sig}")
        intra_results.append({'participant': p, 'signal': 'EDA', 'correlation': corr, 'p_value': p_val})
    else:
        print(f"   EDA: datos insuficientes")
    
    # HR
    hr_arr = np.array(hr_vals)
    mask = ~np.isnan(hr_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        corr, p_val = pearsonr(grades_arr[mask], hr_arr[mask])
        sig = "✅" if p_val < 0.05 else " "
        print(f"   HR:  r={corr:.3f} (p={p_val:.3f}) {sig}")
        intra_results.append({'participant': p, 'signal': 'HR', 'correlation': corr, 'p_value': p_val})
    else:
        print(f"   HR: datos insuficientes")
    
    # TEMP
    temp_arr = np.array(temp_vals)
    mask = ~np.isnan(temp_arr) & ~np.isnan(grades_arr)
    if mask.sum() >= 3:
        corr, p_val = pearsonr(grades_arr[mask], temp_arr[mask])
        sig = "✅" if p_val < 0.05 else " "
        print(f"   TEMP: r={corr:.3f} (p={p_val:.3f}) {sig}")
        intra_results.append({'participant': p, 'signal': 'TEMP', 'correlation': corr, 'p_value': p_val})
    else:
        print(f"   TEMP: datos insuficientes")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN INTRA-SUJETO")
print("=" * 70)

df_intra = pd.DataFrame(intra_results)
significativas = df_intra[df_intra['p_value'] < 0.05]
print(f"\nCorrelaciones significativas (p<0.05): {len(significativas)} de {len(df_intra)}")

for _, row in significativas.iterrows():
    print(f"   {row['participant']} - {row['signal']}: r={row['correlation']:.3f} (p={row['p_value']:.3f})")

df_intra.to_csv('paper_resultados/intra_sujeto_resultados.csv', index=False)
print("\n✅ Guardado: paper_resultados/intra_sujeto_resultados.csv")
