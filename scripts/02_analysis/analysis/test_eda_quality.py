"""
Análisis de calidad de la señal EDA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scripts.utils import load_signal

# Cargar EDA de S1 midterm 1
eda_df = load_signal('S1', 'midterm_1', 'EDA')
eda = eda_df['eda'].values
fs = 4.0
tiempo = np.arange(len(eda)) / fs / 60  # en minutos

print("=== ANÁLISIS DE CALIDAD DE EDA ===\n")

# 1. Estadísticas básicas
print(f"1. ESTADÍSTICAS:")
print(f"   Media: {np.mean(eda):.4f} μS")
print(f"   Std: {np.std(eda):.4f} μS")
print(f"   Min: {np.min(eda):.4f} μS")
print(f"   Max: {np.max(eda):.4f} μS")
print(f"   Rango dinámico: {np.max(eda) - np.min(eda):.4f} μS")

# 2. Detección de saturación
zeros = (eda == 0).sum()
saturation = (eda == eda.max()).sum()
print(f"\n2. SATURACIÓN:")
print(f"   Valores en cero: {zeros} ({zeros/len(eda)*100:.2f}%)")
print(f"   Valores en máximo: {saturation} ({saturation/len(eda)*100:.2f}%)")

# 3. Ruido y artefactos (derivada alta)
deriv = np.abs(np.diff(eda))
high_deriv = deriv > np.percentile(deriv, 99)
print(f"\n3. RUIDO Y ARTEFACTOS:")
print(f"   Picos de derivada >99%: {high_deriv.sum()} eventos")
print(f"   Sugiere: {'MUCHOS artefactos' if high_deriv.sum() > 100 else 'Pocos artefactos'}")

# 4. Relación señal/ruido aproximada
snr = 20 * np.log10(np.std(eda) / (np.median(np.abs(deriv)) + 1e-6))
print(f"\n4. SNR APROXIMADA: {snr:.1f} dB")
print(f"   Interpretación: {'Buena (>20dB)' if snr > 20 else 'Regular (10-20dB)' if snr > 10 else 'Mala (<10dB)'}")

# 5. ¿Necesita filtrado?
print(f"\n5. RECOMENDACIÓN:")
if zeros > len(eda) * 0.01:
    print("    Muchos ceros - posible pérdida de contacto")
if high_deriv.sum() > 500:
    print("    Muchos artefactos - usar filtro mediana")
else:
    print("    Señal aparentemente válida")

# Graficar
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(tiempo, eda, 'b-', linewidth=0.5)
axes[0].set_ylabel('EDA (μS)')
axes[0].set_title('Señal EDA Cruda - S1 Midterm 1')
axes[0].grid(True, alpha=0.3)

# Histograma
axes[1].hist(eda, bins=50, color='blue', alpha=0.7)
axes[1].set_xlabel('EDA (μS)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de valores EDA')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/eda_quality_check.png', dpi=150)
print(f"\n Gráfico guardado en results/figures/eda_quality_check.png")
