"""
¿Cómo cambia la correlación entre EDA y grade a lo largo del semestre?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scripts.utils import load_signal, load_grades_individual

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

# Para cada estudiante, calcular correlación EDA-grade usando:
# - Solo midterm_1 y midterm_2 (primer semestre)
# - Solo midterm_2 y Final (segundo semestre)
# - Las tres (completo)

results = []

for p in participants:
    grades = []
    eda_vals = []
    
    for exam in exams:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        grade = grades_raw[p][exam]
        if exam == 'Final':
            grade = grade / 2
        grades.append(grade)
        
        eda_sig = load_signal(p, exam, 'EDA')
        if eda_sig is not None:
            eda = eda_sig['data'] * 10
            eda_vals.append(np.mean(eda[eda > 0.01]) if np.any(eda > 0.01) else np.nan)
        else:
            eda_vals.append(np.nan)
    
    grades_arr = np.array(grades)
    eda_arr = np.array(eda_vals)
    mask = ~np.isnan(eda_arr) & ~np.isnan(grades_arr)
    
    if mask.sum() >= 2:
        # Correlación primeros 2 exámenes
        if mask[:2].sum() == 2:
            corr_early, _ = pearsonr(grades_arr[:2], eda_arr[:2])
        else:
            corr_early = np.nan
        
        # Correlación últimos 2 exámenes
        if mask[-2:].sum() == 2:
            corr_late, _ = pearsonr(grades_arr[-2:], eda_arr[-2:])
        else:
            corr_late = np.nan
        
        # Correlación completa
        if mask.sum() >= 3:
            corr_full, _ = pearsonr(grades_arr[mask], eda_arr[mask])
        else:
            corr_full = np.nan
        
        results.append({
            'participant': p,
            'corr_early': corr_early,
            'corr_late': corr_late,
            'corr_full': corr_full
        })

df = pd.DataFrame(results)

print("=" * 60)
print("EVOLUCIÓN DE CORRELACIÓN EDA-GRADE DURANTE EL SEMESTRE")
print("=" * 60)

print("\nCorrelación promedio EDA-grade:")
print(f"  Primeras 2 evaluaciones: {df['corr_early'].mean():.3f}")
print(f"  Últimas 2 evaluaciones:  {df['corr_late'].mean():.3f}")
print(f"  Semestre completo:       {df['corr_full'].mean():.3f}")

print("\n Tabla para el paper:")
print("\\begin{table}[htbp]")
print("\\caption{Evolution of EDA-grade correlation across the semester}")
print("\\centering")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("\\textbf{Participant} & \\textbf{Early (M1-M2)} & \\textbf{Late (M2-Final)} & \\textbf{Full} \\\\")
print("\\midrule")
for _, row in df.iterrows():
    early = f"{row['corr_early']:.3f}" if not np.isnan(row['corr_early']) else "---"
    late = f"{row['corr_late']:.3f}" if not np.isnan(row['corr_late']) else "---"
    full = f"{row['corr_full']:.3f}" if not np.isnan(row['corr_full']) else "---"
    print(f"{row['participant']} & {early} & {late} & {full} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:evolution}")
print("\\end{table}")
