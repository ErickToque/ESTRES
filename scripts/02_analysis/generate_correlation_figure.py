import matplotlib.pyplot as plt
import numpy as np

# Datos
subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
eda_corr = [-0.773, -0.996, 0, 0.800, 0.996, -0.480, -0.970, 1.000, -0.544, 0.877]
hr_corr = [-0.987, 0, -1.000, 0.097, -0.879, -0.403, 0.660, 0.559, 0.689, 0.999]
temp_corr = [-0.443, 0, 0.165, -0.434, 0.879, -0.124, 0.770, -0.174, -0.235, -0.733]

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))

x = np.arange(len(subjects))
width = 0.25

bars1 = ax.bar(x - width, eda_corr, width, label='EDA', color='black', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x, hr_corr, width, label='HR', color='gray', alpha=0.7, edgecolor='black')
bars3 = ax.bar(x + width, temp_corr, width, label='TEMP', color='lightgray', alpha=0.7, edgecolor='black')

ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Correlation (r)')
ax.set_ylim(-1.1, 1.1)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/intra_subject_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Figura guardada: figures/intra_subject_correlations.png')
