import matplotlib.pyplot as plt
import numpy as np

# Datos
windows = [1, 2, 3, 4, 5, 15, 30]
accuracies = [75.0, 71.67, 65.0, 58.33, 68.33, 65.0, 53.33]
stds = [30.96, 25.87, 33.71, 35.94, 32.02, 33.71, 37.12]

# Configuración estilo IEEE
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Ancho de columna IEEE ~ 3.5 in

# Gráfico con líneas y barras de error
ax.errorbar(windows, accuracies, yerr=stds, fmt='o-', 
            capsize=4, capthick=1, markersize=6, 
            linewidth=1.5, color='black', ecolor='gray',
            markerfacecolor='white', markeredgewidth=1.5)

# Línea de azar (50%)
ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Configuración de ejes
ax.set_xscale('log')
ax.set_xlabel('Window Size (minutes)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(windows)
ax.set_xticklabels([str(w) for w in windows])
ax.set_xlim(0.8, 35)
ax.set_ylim(40, 95)
ax.grid(True, alpha=0.3, linestyle='--')

# Añadir etiquetas de valor en puntos clave
for i, (w, acc, std) in enumerate(zip(windows, accuracies, stds)):
    if w in [1, 5, 30]:
        ax.annotate(f'{acc:.0f}%', xy=(w, acc), 
                   xytext=(5, 5 if i != 2 else -10),
                   textcoords='offset points', fontsize=8,
                   ha='center', va='bottom')

# Leyenda
ax.text(0.97, 0.03, 'Chance level (50%)', transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
        color='red', style='italic')

plt.tight_layout()
plt.savefig('window_optimization.png', dpi=300, bbox_inches='tight')
plt.close()