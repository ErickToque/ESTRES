"""
Figura: Distribución de correlaciones intra-sujeto por señal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scripts.utils import load_signal, load_ibi, load_grades_individual

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

correlations = {'EDA': [], 'HR': [], 'TEMP': []}

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
    
    for name, vals, target in [('EDA', eda_vals, correlations['EDA']),
                                 ('HR', hr_vals, correlations['HR']),
                                 ('TEMP', temp_vals, correlations['TEMP'])]:
        vals_arr = np.array(vals)
        mask = ~np.isnan(vals_arr) & ~np.isnan(grades_arr)
        if mask.sum() >= 3:
            corr, _ = pearsonr(grades_arr[mask], vals_arr[mask])
            target.append(corr)

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
fig, ax = plt.subplots(figsize=(4, 3))

data = [correlations['EDA'], correlations['HR'], correlations['TEMP']]
bp = ax.boxplot(data, labels=['EDA', 'HR', 'TEMP'], patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.7))
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
ax.set_ylabel('Correlation (r)')
ax.set_title('Distribution of intra-subject correlations')
ax.grid(True, alpha=0.3, axis='y')

for i, d in enumerate(data):
    y = np.mean(d) if d else 0
    ax.text(i+1, y + 0.05, f'n={len(d)}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('intra_correlation_distribution.png', dpi=300, bbox_inches='tight')
print(" Figura guardada: intra_correlation_distribution.png")
