"""
Análisis estadístico robusto de relación señales-grades
- Regresión lineal (relación continua)
- ANOVA con grupos basados en percentiles (más justo)
- Correlación de Spearman (no paramétrica)
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Cargar datos completos
def load_all_data():
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    
    # Cargar calificaciones
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
    
    data = []
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade = grades_raw[p][exam]
            if exam == 'Final':
                grade = grade / 2
            
            # EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is not None:
                eda = eda_df['eda'].values * 10
                eda_no_ceros = eda[eda > 0.01]
                eda_mean = np.mean(eda_no_ceros) if len(eda_no_ceros) > 0 else np.nan
            else:
                eda_mean = np.nan
            
            # HR
            ibi_df = load_ibi(p, exam)
            if ibi_df is not None:
                if 'ibi' in ibi_df.columns:
                    ibi = ibi_df['ibi'].values
                else:
                    ibi = ibi_df.iloc[:, 1].values
                ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                hr = 60.0 / np.mean(ibi) if len(ibi) > 5 else np.nan
            else:
                hr = np.nan
            
            # TEMP
            temp_df = load_signal(p, exam, 'TEMP')
            if temp_df is not None:
                temp = temp_df['temp'].values
                temp_mean = np.mean(temp)
            else:
                temp_mean = np.nan
            
            data.append({
                'participant': p,
                'exam': exam,
                'grade': grade,
                'eda': eda_mean,
                'hr': hr,
                'temp': temp_mean
            })
    
    return pd.DataFrame(data)

df = load_all_data()
df = df.dropna()

print("=" * 60)
print("ANÁLISIS ESTADÍSTICO ROBUSTO")
print("=" * 60)

print(f"\nMuestras totales: {len(df)}")
print(f"Rango de grades: {df['grade'].min():.1f} - {df['grade'].max():.1f}")
print(f"Media de grades: {df['grade'].mean():.1f} ± {df['grade'].std():.1f}")

# 1. Correlaciones (Pearson y Spearman)
print("\n" + "=" * 60)
print("1. CORRELACIONES CON CALIFICACIONES")
print("=" * 60)

for senal in ['eda', 'hr', 'temp']:
    datos = df[senal].dropna()
    grades = df.loc[datos.index, 'grade']
    
    pearson_r, pearson_p = pearsonr(datos, grades)
    spearman_r, spearman_p = spearmanr(datos, grades)
    
    print(f"\n{senal.upper()}:")
    print(f"  Pearson: r={pearson_r:.3f}, p={pearson_p:.4f}")
    print(f"  Spearman: ρ={spearman_r:.3f}, p={spearman_p:.4f}")
    
    if pearson_p < 0.05:
        print(f"  → Significativo (p<0.05)")

# 2. ANOVA con grupos basados en percentiles (más justo que umbrales arbitrarios)
print("\n" + "=" * 60)
print("2. ANOVA POR GRUPOS (percentiles 33% y 67%)")
print("=" * 60)

# Crear grupos por percentil (más equilibrados)
grades_sorted = df['grade'].sort_values()
n = len(grades_sorted)
tercil1 = grades_sorted.iloc[n//3]
tercil2 = grades_sorted.iloc[2*n//3]

def asignar_grupo(grade):
    if grade <= tercil1:
        return 'Bajo'
    elif grade <= tercil2:
        return 'Medio'
    else:
        return 'Alto'

df['grupo'] = df['grade'].apply(asignar_grupo)

print(f"Puntos de corte: Bajo ≤ {tercil1:.1f}, Medio ≤ {tercil2:.1f}")
print(f"Distribución: Bajo={sum(df['grupo']=='Bajo')}, "
      f"Medio={sum(df['grupo']=='Medio')}, "
      f"Alto={sum(df['grupo']=='Alto')}")

for senal in ['eda', 'hr', 'temp']:
    grupos = []
    for g in ['Bajo', 'Medio', 'Alto']:
        valores = df[df['grupo'] == g][senal].dropna().values
        grupos.append(valores)
        print(f"\n{senal.upper()} - {g}: n={len(valores)}, media={np.mean(valores):.2f} ± {np.std(valores):.2f}")
    
    # ANOVA
    if all(len(g) > 0 for g in grupos):
        f_stat, p_valor = f_oneway(*grupos)
        print(f"  ANOVA: F={f_stat:.3f}, p={p_valor:.4f}")
        if p_valor < 0.05:
            print(f"  → Diferencia significativa entre grupos")

# 3. Regresión lineal con control por sujeto (efectos mixtos)
print("\n" + "=" * 60)
print("3. CORRELACIÓN PARCIAL (controlando por sujeto)")
print("=" * 60)

# Correlación intra-sujeto (promedio de correlaciones individuales)
from scipy.stats import pearsonr

intra_corrs = {'eda': [], 'hr': [], 'temp': []}

for p in df['participant'].unique():
    subj_data = df[df['participant'] == p]
    if len(subj_data) >= 2:
        for senal in ['eda', 'hr', 'temp']:
            senal_vals = subj_data[senal].dropna()
            grades_vals = subj_data.loc[senal_vals.index, 'grade']
            if len(senal_vals) >= 3:
                corr, _ = pearsonr(senal_vals, grades_vals)
                if not np.isnan(corr):
                    intra_corrs[senal].append(corr)

for senal in ['eda', 'hr', 'temp']:
    if intra_corrs[senal]:
        media_corr = np.mean(intra_corrs[senal])
        print(f"{senal.upper()}: correlación intra-sujeto media = {media_corr:.3f}")

# 4. Guardar resultados
df.to_csv('analysis/estadistica_robusta/datos_completos.csv', index=False)
print("\n✅ Datos guardados: analysis/estadistica_robusta/datos_completos.csv")
