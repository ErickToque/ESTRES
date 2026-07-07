"""
FIGURA 7 FINAL: Evolución de señales con promedio + banda de dispersión
- Leyenda única para toda la figura
- Sin título principal
- Estilo consistente con gráfico de ventanas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi, load_grades_individual

# Configuración
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
exam_labels = ['Midterm 1', 'Midterm 2', 'Final']

# Calcular promedio por estudiante para clasificación
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

print("Alto rendimiento:", high_performers)
print("Bajo rendimiento:", low_performers)

def get_group_data(group, signal_name):
    """Retorna matriz de valores (n_sujetos x 3 exámenes)"""
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
            else:  # TEMP
                sig = load_signal(p, exam, 'TEMP')
                if sig is not None:
                    val = np.mean(sig['data'])
                else:
                    val = np.nan
            values.append(val)
        data.append(values)
    return np.array(data)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
signals = ['EDA', 'HR', 'TEMP']
colors = {'Alto': '#2E86AB', 'Bajo': '#E74C3C'}
labels = {'Alto': 'High performers (n=4)', 'Bajo': 'Low performers (n=6)'}

for idx, signal_name in enumerate(signals):
    ax = axes[idx]
    
    for grupo, nombre, color in [(high_performers, 'Alto', colors['Alto']),
                                   (low_performers, 'Bajo', colors['Bajo'])]:
        if len(grupo) == 0:
            continue
        
        data = get_group_data(grupo, signal_name)
        
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        
        x_positions = np.arange(len(exam_labels))
        
        # Puntos individuales
        for i, sujeto_data in enumerate(data):
            offset = -0.1 if nombre == 'Alto' else 0.1
            ax.scatter(x_positions + offset, sujeto_data, alpha=0.25, s=15, 
                      color=color, edgecolors='none')
        
        # Línea de promedio
        ax.plot(x_positions, mean_vals, 'o-', color=color, linewidth=2,
               markersize=5, markerfacecolor='white', markeredgewidth=1.5,
               label=labels[nombre])
        
        # Banda de dispersión
        ax.fill_between(x_positions, mean_vals - std_vals, mean_vals + std_vals,
                        color=color, alpha=0.12)
        
        # Valores numéricos
        for i, (x, y) in enumerate(zip(x_positions, mean_vals)):
            if not np.isnan(y):
                ax.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 6),
                           textcoords='offset points', ha='center', fontsize=7)
    
    ax.set_xlabel('Examination')
    ax.set_ylabel(signal_name)
    ax.set_xticks(np.arange(len(exam_labels)))
    ax.set_xticklabels(exam_labels)
    ax.set_title(f'{signal_name}')
    ax.grid(True, alpha=0.3)

# Leyenda única para toda la figura
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
           ncol=2, fontsize=9, frameon=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('conference/figures/fig07_group_evolution_final.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Figura 7 final guardada: conference/figures/fig07_group_evolution_final.png")
print("   - Leyenda única para toda la figura")
print("   - Sin título principal")
print("   - Muestra evolución de EDA, HR y TEMP")
