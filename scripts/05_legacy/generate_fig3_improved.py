"""
Figura 3: Evolución temporal de señales por grupo
Texto mejorado para IEEE conference
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi, load_grades_individual

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.0
})

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = ['Midterm 1', 'Midterm 2', 'Final']

# Calcular promedios por estudiante
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

def get_group_data(group, signal_name):
    data = []
    for p in group:
        values = []
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                values.append(np.nan)
                continue
            
            if signal_name == 'EDA':
                sig = load_signal(p, exam, 'EDA')
                if sig is not None:
                    eda = sig['data'] * 10
                    eda_clean = eda[eda > 0.01]
                    val = np.mean(eda_clean) if len(eda_clean) > 0 else np.nan
                else:
                    val = np.nan
            elif signal_name == 'HR':
                ibi_df = load_ibi(p, exam)
                if ibi_df is not None:
                    ibi = ibi_df['ibi'].values if 'ibi' in ibi_df.columns else ibi_df.iloc[:, 1].values
                    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                    val = 60.0 / np.mean(ibi) if len(ibi) > 5 else np.nan
                else:
                    val = np.nan
            else:
                sig = load_signal(p, exam, 'TEMP')
                if sig is not None:
                    val = np.mean(sig['data'])
                else:
                    val = np.nan
            values.append(val)
        data.append(values)
    return np.array(data)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
signals = ['EDA', 'HR', 'TEMP']
colors = {'High': '#2E86AB', 'Low': '#E74C3C'}

for idx, signal_name in enumerate(signals):
    ax = axes[idx]
    x_pos = np.arange(len(exam_labels))
    
    for grupo, nombre, color in [(high_performers, 'High', colors['High']),
                                   (low_performers, 'Low', colors['Low'])]:
        if len(grupo) == 0:
            continue
        
        data = get_group_data(grupo, signal_name)
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        
        # Puntos individuales
        for sujeto_data in data:
            offset = -0.1 if nombre == 'High' else 0.1
            ax.scatter(x_pos + offset, sujeto_data, alpha=0.2, s=12, color=color)
        
        # Línea de promedio
        ax.plot(x_pos, mean_vals, 'o-', color=color, linewidth=1.2,
               markersize=4, markerfacecolor='white', markeredgewidth=1,
               label=f'{nombre} (n={len(grupo)})')
        
        # Banda de dispersión
        ax.fill_between(x_pos, mean_vals - std_vals, mean_vals + std_vals,
                        color=color, alpha=0.1)
    
    ax.set_xlabel('Examination')
    ax.set_ylabel(signal_name)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exam_labels, rotation=15, ha='right')
    ax.set_title(signal_name, fontsize=9, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('conference/figures/fig03_group_evolution.png', dpi=300, bbox_inches='tight')
print("✅ Figura 3 guardada: conference/figures/fig03_group_evolution.png")
