"""
Comparación de correlaciones intra-sujeto entre grupos de rendimiento
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu
from scripts.utils import load_signal, load_ibi, load_grades_individual

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

# Calcular promedio por estudiante
averages = {}
for p in participants:
    grades = []
    for exam in exams:
        if p in grades_raw and exam in grades_raw[p]:
            grade = grades_raw[p][exam]
            if exam == 'Final':
                grade = grade / 2
            grades.append(grade)
    if grades:
        averages[p] = np.mean(grades)

high_performers = [p for p in averages if averages[p] >= 80]
low_performers = [p for p in averages if averages[p] < 80]

print("=" * 60)
print("CORRELACIONES POR GRUPO DE RENDIMIENTO")
print("=" * 60)

high_corrs = {'EDA': [], 'HR': [], 'TEMP': []}
low_corrs = {'EDA': [], 'HR': [], 'TEMP': []}

for p in participants:
    if p not in averages:
        continue
    
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
        
        eda_sig = load_signal(p, exam, 'EDA')
        if eda_sig is not None:
            eda = eda_sig['data'] * 10
            eda_vals.append(np.mean(eda[eda > 0.01]) if np.any(eda > 0.01) else np.nan)
        else:
            eda_vals.append(np.nan)
        
        ibi_df = load_ibi(p, exam)
        if ibi_df is not None:
            ibi = ibi_df['ibi'].values if 'ibi' in ibi_df.columns else ibi_df.iloc[:, 1].values
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
            hr_vals.append(60.0 / np.mean(ibi) if len(ibi) > 5 else np.nan)
        else:
            hr_vals.append(np.nan)
        
        temp_sig = load_signal(p, exam, 'TEMP')
        if temp_sig is not None:
            temp_vals.append(np.mean(temp_sig['data']))
        else:
            temp_vals.append(np.nan)
    
    grades_arr = np.array(grades)
    
    for name, vals, target in [('EDA', eda_vals, high_corrs if p in high_performers else low_corrs),
                                 ('HR', hr_vals, high_corrs if p in high_performers else low_corrs),
                                 ('TEMP', temp_vals, high_corrs if p in high_performers else low_corrs)]:
        vals_arr = np.array(vals)
        mask = ~np.isnan(vals_arr) & ~np.isnan(grades_arr)
        if mask.sum() >= 3:
            corr, _ = pearsonr(grades_arr[mask], vals_arr[mask])
            if not np.isnan(corr):
                target[name].append(corr)

print(f"\nAlto rendimiento (n={len(high_performers)}): {high_performers}")
print(f"Bajo rendimiento (n={len(low_performers)}): {low_performers}")

print("\n|r| promedio por grupo:")
for signal in ['EDA', 'HR', 'TEMP']:
    high_avg = np.mean(np.abs(high_corrs[signal])) if high_corrs[signal] else 0
    low_avg = np.mean(np.abs(low_corrs[signal])) if low_corrs[signal] else 0
    print(f"  {signal}: Alto = {high_avg:.3f}, Bajo = {low_avg:.3f}")
    
    if high_corrs[signal] and low_corrs[signal]:
        stat, p = mannwhitneyu(high_corrs[signal], low_corrs[signal])
        print(f"           Mann-Whitney p = {p:.4f}")

print("\n Tabla para el paper:")
print("\\begin{table}[htbp]")
print("\\caption{Average absolute correlations by performance group}")
print("\\centering")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Signal} & \\textbf{High performers} & \\textbf{Low performers} \\\\")
print("\\midrule")
for signal in ['EDA', 'HR', 'TEMP']:
    high_avg = np.mean(np.abs(high_corrs[signal])) if high_corrs[signal] else 0
    low_avg = np.mean(np.abs(low_corrs[signal])) if low_corrs[signal] else 0
    print(f"{signal} & {high_avg:.3f} & {low_avg:.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:group_correlations}")
print("\\end{table}")
