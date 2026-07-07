"""
FIGURA 7 MODIFICADA: Evolución de señales para 3 sujetos representativos
- S1: Alto rendimiento consistente
- S3: Mejora progresiva
- S8: Alto rendimiento (o S7 para contraste)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_grades_individual

# Cargar calificaciones
grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = ['Midterm 1', 'Midterm 2', 'Final']

# Seleccionar 3 sujetos representativos
selected = [
    {'id': 'S1', 'name': 'S1 - Alto rendimiento consistente', 'color': 'blue', 'marker': 'o'},
    {'id': 'S3', 'name': 'S3 - Mejora progresiva', 'color': 'green', 'marker': 's'},
    {'id': 'S8', 'name': 'S8 - Mejor rendimiento', 'color': 'red', 'marker': '^'}
]

# Configuración estilo paper
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

for idx, signal in enumerate(['EDA', 'HR', 'TEMP']):
    ax = axes[idx]
    
    for s in selected:
        p = s['id']
        values = []
        
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                values.append(np.nan)
                continue
            
            if signal == 'EDA':
                sig = load_signal(p, exam, 'EDA')
                if sig is not None:
                    eda = sig['data'] * 10
                    eda_clean = eda[eda > 0.01]
                    val = np.mean(eda_clean) if len(eda_clean) > 0 else np.nan
                else:
                    val = np.nan
            elif signal == 'HR':
                from scripts.utils import load_ibi
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
        
        # Graficar línea
        ax.plot(exam_labels, values, 'o-', color=s['color'], linewidth=1.5, 
                markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                label=s['name'])
        
        # Añadir valores numéricos
        for i, (x, y) in enumerate(zip(exam_labels, values)):
            if not np.isnan(y):
                ax.annotate(f'{y:.1f}', xy=(i, y), xytext=(0, 5),
                           textcoords='offset points', ha='center', fontsize=7)
    
    ax.set_xlabel('Examination')
    ax.set_ylabel(signal)
    ax.set_title(f'{signal} evolution')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Physiological signal evolution for representative subjects', fontsize=12)
plt.tight_layout()
plt.savefig('conference/figures/fig07_three_subjects_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Figura 7 modificada guardada: conference/figures/fig07_three_subjects_evolution.png")
print("   Muestra evolución de EDA, HR y TEMP para S1, S3 y S8")
