"""
FIGURA 7: Evolución de señales con promedio + banda de dispersión
Estilo similar al gráfico de optimización de ventanas
- Línea: promedio del grupo
- Banda: desviación estándar
- Puntos: valores individuales (opcional)
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

# Función para obtener datos de un grupo
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

for idx, signal_name in enumerate(signals):
    ax = axes[idx]
    
    for grupo, nombre, color in [(high_performers, 'High (n=4)', colors['Alto']),
                                   (low_performers, 'Low (n=6)', colors['Bajo'])]:
        if len(grupo) == 0:
            continue
        
        data = get_group_data(grupo, signal_name)
        
        # Calcular promedio y std por examen
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        
        # Puntos individuales (jitter para no superponer)
        x_positions = np.arange(len(exam_labels))
        for i, sujeto_data in enumerate(data):
            # Desplazamiento lateral según grupo
            offset = -0.1 if nombre == 'High' else 0.1
            ax.scatter(x_positions + offset, sujeto_data, alpha=0.3, s=20, 
                      color=color, edgecolors='none')
        
        # Línea de promedio
        ax.plot(x_positions, mean_vals, 'o-', color=color, linewidth=2,
               markersize=6, markerfacecolor='white', markeredgewidth=1.5,
               label=f'{nombre} (mean)')
        
        # Banda de dispersión (std)
        ax.fill_between(x_positions, mean_vals - std_vals, mean_vals + std_vals,
                        color=color, alpha=0.15)
        
        # Añadir valores numéricos del promedio
        for i, (x, y) in enumerate(zip(x_positions, mean_vals)):
            if not np.isnan(y):
                ax.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 5),
                           textcoords='offset points', ha='center', fontsize=7)
    
    ax.set_xlabel('Examination')
    ax.set_ylabel(signal_name)
    ax.set_xticks(np.arange(len(exam_labels)))
    ax.set_xticklabels(exam_labels)
    ax.set_title(f'{signal_name} evolution')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Physiological signal evolution by performance group (mean ± std)', fontsize=12)
plt.tight_layout()
plt.savefig('conference/figures/fig07_group_evolution_with_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Figura 7 guardada: conference/figures/fig07_group_evolution_with_dispersion.png")
print("   Muestra evolución de EDA, HR y TEMP para grupos Alto (n=4) y Bajo (n=6)")
print("   - Línea: promedio del grupo")
print("   - Banda: ± desviación estándar")
print("   - Puntos: valores individuales (con desplazamiento lateral)")
