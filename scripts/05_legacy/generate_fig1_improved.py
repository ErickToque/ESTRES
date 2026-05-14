"""
Figura 1: Violin plots de EDA, HR, TEMP por grupo de rendimiento
Texto mejorado para IEEE conference
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi, load_grades_individual

# Configuración estilo IEEE
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.0
})

# Cargar datos
grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

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

# Recolectar datos
high_data = {'EDA': [], 'HR': [], 'TEMP': []}
low_data = {'EDA': [], 'HR': [], 'TEMP': []}

for p in participants:
    eda_vals = []
    hr_vals = []
    temp_vals = []
    
    for exam in exams:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        
        # EDA
        eda_sig = load_signal(p, exam, 'EDA')
        if eda_sig is not None:
            eda = eda_sig['data'] * 10
            eda_clean = eda[eda > 0.01]
            if len(eda_clean) > 0:
                eda_vals.append(np.mean(eda_clean))
        
        # HR
        ibi_df = load_ibi(p, exam)
        if ibi_df is not None:
            ibi = ibi_df['ibi'].values if 'ibi' in ibi_df.columns else ibi_df.iloc[:, 1].values
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
            if len(ibi) > 5:
                hr_vals.append(60.0 / np.mean(ibi))
        
        # TEMP
        temp_sig = load_signal(p, exam, 'TEMP')
        if temp_sig is not None:
            temp_vals.append(np.mean(temp_sig['data']))
    
    # Promedio por estudiante
    if len(eda_vals) >= 2:
        if p in high_performers:
            high_data['EDA'].append(np.mean(eda_vals))
        else:
            low_data['EDA'].append(np.mean(eda_vals))
    
    if len(hr_vals) >= 2:
        if p in high_performers:
            high_data['HR'].append(np.mean(hr_vals))
        else:
            low_data['HR'].append(np.mean(hr_vals))
    
    if len(temp_vals) >= 2:
        if p in high_performers:
            high_data['TEMP'].append(np.mean(temp_vals))
        else:
            low_data['TEMP'].append(np.mean(temp_vals))

# Crear figura
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))  # Ancho para dos columnas
signals = ['EDA', 'HR', 'TEMP']
ylabel = {'EDA': 'EDA (μS)', 'HR': 'Heart Rate (bpm)', 'TEMP': 'Temperature (°C)'}
colors = {'High': '#2E86AB', 'Low': '#E74C3C'}

for idx, signal in enumerate(signals):
    ax = axes[idx]
    
    high_vals = high_data[signal]
    low_vals = low_data[signal]
    
    if len(high_vals) == 0 or len(low_vals) == 0:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        continue
    
    # Violin plots
    parts = ax.violinplot([high_vals, low_vals], positions=[1, 2], 
                          widths=0.6, showmeans=True, showmedians=True)
    
    parts['bodies'][0].set_facecolor(colors['High'])
    parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor(colors['Low'])
    parts['bodies'][1].set_alpha(0.6)
    
    # Medians y means
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cmeans'].set_color('white')
    parts['cmeans'].set_marker('o')
    parts['cmeans'].set_markersize(4)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'High\n(n={len(high_vals)})', 
                        f'Low\n(n={len(low_vals)})'])
    ax.set_ylabel(ylabel[signal])
    ax.set_title(signal, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('conference/figures/fig01_signal_distribution_improved.png', dpi=300, bbox_inches='tight')
print("✅ Figura 1 guardada: conference/figures/fig01_signal_distribution_improved.png")
