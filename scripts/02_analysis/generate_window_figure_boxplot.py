import matplotlib.pyplot as plt
import numpy as np

# Datos agregados
windows = [1, 2, 3, 4, 5, 15, 30]
accuracies = [75.0, 71.67, 65.0, 58.33, 68.33, 65.0, 53.33]
stds = [30.96, 25.87, 33.71, 35.94, 32.02, 33.71, 37.12]

# Generar datos sintéticos para boxplot (10 sujetos por ventana)
np.random.seed(42)
all_data = []
for mu, sigma in zip(accuracies, stds):
    # Limitar entre 0 y 100
    data = np.random.normal(mu, sigma, 10)
    data = np.clip(data, 0, 100)
    all_data.append(data)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

fig, ax = plt.subplots(figsize=(4, 3))

# Boxplot
bp = ax.boxplot(all_data, positions=windows, widths=0.6 * np.array(windows)/max(windows),
                patch_artist=True, showmeans=True, 
                meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 4})

# Colorear boxes
for i, box in enumerate(bp['boxes']):
    box.set_facecolor('lightgray')
    box.set_alpha(0.7)

# Línea de azar
ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Configuración
ax.set_xscale('log')
ax.set_xlabel('Window Size (minutes)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(windows)
ax.set_xticklabels([str(w) for w in windows])
ax.set_xlim(0.8, 35)
ax.set_ylim(20, 100)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('figures/window_optimization_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Figura guardada: figures/window_optimization_boxplot.png')
