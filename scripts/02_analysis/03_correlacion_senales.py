"""
Correlación entre señales fisiológicas y rendimiento académico
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

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
exams = ['midterm_1', 'midterm_2', 'Final']

# Recolectar datos
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
        
        # Temperatura
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

df = pd.DataFrame(data)

# Correlaciones
print("=" * 50)
print("CORRELACIONES CON CALIFICACIONES")
print("=" * 50)

for señal in ['eda', 'hr', 'temp']:
    corr = df[señal].corr(df['grade'])
    print(f"{señal.upper()}: r = {corr:.3f}")

# Por subgrupo (alto vs bajo rendimiento)
print("\n" + "=" * 50)
print("CORRELACIONES POR SUBGRUPO")
print("=" * 50)

alto = df[df['grade'] >= 80]
bajo = df[df['grade'] < 70]

for señal in ['eda', 'hr', 'temp']:
    corr_alto = alto[señal].corr(alto['grade']) if len(alto) > 1 else np.nan
    corr_bajo = bajo[señal].corr(bajo['grade']) if len(bajo) > 1 else np.nan
    print(f"{señal.upper()}: Alto r={corr_alto:.3f} | Bajo r={corr_bajo:.3f}")

# Matriz de correlación
corr_matrix = df[['grade', 'eda', 'hr', 'temp']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Matriz de correlación: Señales vs Calificaciones')
plt.tight_layout()
plt.savefig('analysis/grupos_rendimiento/matriz_correlacion.png', dpi=150)
print("\n✅ Matriz guardada: analysis/grupos_rendimiento/matriz_correlacion.png")

df.to_csv('analysis/grupos_rendimiento/datos_completos.csv', index=False)
print("✅ Datos guardados: analysis/grupos_rendimiento/datos_completos.csv")
