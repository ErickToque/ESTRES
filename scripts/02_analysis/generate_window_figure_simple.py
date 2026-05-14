import matplotlib.pyplot as plt
import numpy as np

# Datos
windows = [1, 2, 3, 4, 5, 15, 30]
accuracies = [75.0, 71.67, 65.0, 58.33, 68.33, 65.0, 53.33]

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Línea con puntos
ax.plot(windows, accuracies, 'o-', color='black', linewidth=1.5, 
        markersize=6, markerfacecolor='white', markeredgewidth=1.5)

# Destacar punto máximo (Tw=1)
ax.plot(1, 75.0, 'o', color='red', markersize=8, markeredgecolor='black')

# Línea de azar
ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Configuración
ax.set_xscale('log')
ax.set_xlabel('Window Size (minutes)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(windows)
ax.set_xticklabels([str(w) for w in windows])
ax.set_xlim(0.8, 35)
ax.set_ylim(40, 85)
ax.grid(True, alpha=0.2, linestyle='--')

# Etiqueta del máximo
ax.annotate('Best: 1 min (75%)', xy=(1, 75), xytext=(2, 78),
            fontsize=8, ha='left')

plt.tight_layout()
plt.savefig('figures/window_optimization_simple.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Figura guardada: figures/window_optimization_simple.png')
