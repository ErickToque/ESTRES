import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import get_student_grades

grades = get_student_grades()
participants = list(grades.keys())
grade_values = list(grades.values())

print('=' * 80)
print('ANÁLISIS DE LA VARIABLE OBJETIVO (GRADES)')
print('=' * 80)

print(f'\n Estadísticas:')
print(f'   Min: {min(grade_values):.1f}')
print(f'   Max: {max(grade_values):.1f}')
print(f'   Mean: {np.mean(grade_values):.1f}')
print(f'   Std: {np.std(grade_values):.1f}')
print(f'   Rango: {max(grade_values) - min(grade_values):.1f}')

print(f'\n Participantes ordenados:')
sorted_idx = np.argsort(grade_values)
for i, idx in enumerate(sorted_idx):
    p = participants[idx]
    g = grade_values[idx]
    print(f'   {i+1:2d}. {p}: {g:.1f}')

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(grade_values, bins=8, color='blue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Grade')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Calificaciones')

axes[1].barh(participants, grade_values, color='green', alpha=0.7)
axes[1].set_xlabel('Grade')
axes[1].set_title('Calificaciones por Participante')
axes[1].axvline(np.mean(grade_values), color='red', linestyle='--', label=f'Mean: {np.mean(grade_values):.1f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/grades_distribution.png', dpi=150)
print('\n Gráfico guardado: results/figures/grades_distribution.png')
