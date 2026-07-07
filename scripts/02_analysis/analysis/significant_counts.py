"""
Conteo de correlaciones significativas por señal
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scripts.utils import load_signal, load_ibi, load_grades_individual

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

results = []

for p in participants:
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
    
    for name, vals in [('EDA', eda_vals), ('HR', hr_vals), ('TEMP', temp_vals)]:
        vals_arr = np.array(vals)
        mask = ~np.isnan(vals_arr) & ~np.isnan(grades_arr)
        if mask.sum() >= 3:
            corr, p_val = pearsonr(grades_arr[mask], vals_arr[mask])
            results.append({
                'participant': p,
                'signal': name,
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'n_points': mask.sum()
            })

df = pd.DataFrame(results)

print("=" * 50)
print("CORRELACIONES SIGNIFICATIVAS POR SEÑAL")
print("=" * 50)

for signal in ['EDA', 'HR', 'TEMP']:
    sig = df[(df['signal'] == signal) & (df['significant'] == True)]
    total = df[df['signal'] == signal]
    print(f"{signal}: {len(sig)}/{len(total)} significativas (p<0.05)")

print("\n Tabla para el paper:")
print("\\begin{table}[htbp]")
print("\\caption{Number of significant intra-subject correlations by signal type}")
print("\\centering")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Signal} & \\textbf{Significant (p<0.05)} & \\textbf{Total} \\\\")
print("\\midrule")
for signal in ['EDA', 'HR', 'TEMP']:
    sig = len(df[(df['signal'] == signal) & (df['significant'] == True)])
    total = len(df[df['signal'] == signal])
    pct = sig/total*100 if total > 0 else 0
    print(f"{signal} & {sig}/{total} & {pct:.1f}\\% \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:significant_counts}")
print("\\end{table}")
