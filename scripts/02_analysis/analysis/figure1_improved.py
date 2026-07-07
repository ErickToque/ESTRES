"""
FIGURA 1 MEJORADA: Violin plots de EDA, HR y TEMP por grupo de rendimiento
- Sin título principal
- Etiquetas en inglés
- Formato consistente con figura 7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi, load_grades_individual

# Configuración estilo paper
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# Cargar calificaciones
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

print("High performers:", high_performers)
print("Low performers:", low_performers)

# Recolectar datos para violin plots
data = []
for p in participants:
    for exam in exams:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        
        grade = grades_raw[p][exam]
        if exam == 'Final':
            grade = grade / 2
        grade_class = 'High' if grade >= 80 else 'Low'
        
        # EDA
        eda_sig = load_signal(p, exam, 'EDA')
        if eda_sig is not None:
            eda = eda_sig['data'] * 10
            eda_clean = eda[eda > 0.01]
            eda_val = np.mean(eda_clean) if len(eda_clean) > 0 else np.nan
        else:
            eda_val = np.nan
        
        # HR
        ibi_df = load_ibi(p, exam)
        if ibi_df is not None:
            ibi = ibi_df['ibi'].values if 'ibi' in ibi_df.columns else ibi_df.iloc[:, 1].values
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
            hr_val = 60.0 / np.mean(ibi) if len(ibi) > 5 else np.nan
        else:
            hr_val = np.nan
        
        # TEMP
        temp_sig = load_signal(p, exam, 'TEMP')
        if temp_sig is not None:
            temp_val = np.mean(temp_sig['data'])
        else:
            temp_val = np.nan
        
        data.append({
            'group': grade_class,
            'EDA': eda_val,
            'HR': hr_val,
            'TEMP': temp_val
        })

df = pd.DataFrame(data)

# Crear figura
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
signals = ['EDA', 'HR', 'TEMP']
ylabel = {'EDA': 'EDA (μS)', 'HR': 'Heart Rate (bpm)', 'TEMP': 'Temperature (°C)'}
colors = {'High': '#2E86AB', 'Low': '#E74C3C'}

for idx, signal in enumerate(signals):
    ax = axes[idx]
    
    # Separar datos por grupo
    high_data = df[df['group'] == 'High'][signal].dropna().values
    low_data = df[df['group'] == 'Low'][signal].dropna().values
    
    # Violin plots
    parts = ax.violinplot([high_data, low_data], positions=[1, 2], 
                          widths=0.6, showmeans=True, showmedians=True)
    
    # Colorear
    for i, color in enumerate([colors['High'], colors['Low']]):
        parts['bodies'][i].set_facecolor(color)
        parts['bodies'][i].set_alpha(0.6)
        parts['bodies'][i].set_edgecolor('black')
    
    # Personalizar
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['High\n(n={})'.format(len(high_data)), 
                        'Low\n(n={})'.format(len(low_data))])
    ax.set_ylabel(ylabel[signal])
    ax.set_title(signal)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir línea de referencia (opcional)
    if signal == 'EDA':
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

plt.tight_layout()
plt.savefig('conference/figures/fig01_signal_distribution_improved.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Figura 1 mejorada guardada: conference/figures/fig01_signal_distribution_improved.png")
print("   - Sin título principal")
print("   - Etiquetas en inglés")
print("   - Violin plots de EDA, HR, TEMP por grupo (High/Low)")
