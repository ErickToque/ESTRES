"""
Clasificación correcta de estudiantes por su RENDIMIENTO PROMEDIO
No por examen individual
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# Cargar datos
df = pd.read_csv('analysis/estadistica_robusta/datos_completos.csv')

# Calcular promedio de cada estudiante
promedios = df.groupby('participant')['grade'].mean().sort_values()

print("=" * 60)
print("CLASIFICACIÓN POR RENDIMIENTO PROMEDIO")
print("=" * 60)

print("\nPromedios por estudiante:")
for p, prom in promedios.items():
    print(f"  {p}: {prom:.1f}")

# Usar MEDIANA para dividir en dos grupos (más robusto que percentiles)
mediana = promedios.median()
print(f"\nMediana de promedios: {mediana:.1f}")

# Clasificar estudiantes
alto_rendimiento = [p for p in promedios.index if promedios[p] >= mediana]
bajo_rendimiento = [p for p in promedios.index if promedios[p] < mediana]

print(f"\nAlto rendimiento (≥ mediana): {alto_rendimiento}")
print(f"  Promedios: {[promedios[p] for p in alto_rendimiento]}")
print(f"\nBajo rendimiento (< mediana): {bajo_rendimiento}")
print(f"  Promedios: {[promedios[p] for p in bajo_rendimiento]}")

# Añadir clasificación al dataframe
df['grupo_estudiante'] = df['participant'].apply(
    lambda x: 'Alto' if promedios[x] >= mediana else 'Bajo'
)

print("\n" + "=" * 60)
print("COMPARACIÓN ENTRE GRUPOS (por promedio del estudiante)")
print("=" * 60)

for senal in ['eda', 'hr', 'temp']:
    alto_vals = df[df['grupo_estudiante'] == 'Alto'][senal].dropna()
    bajo_vals = df[df['grupo_estudiante'] == 'Bajo'][senal].dropna()
    
    from scipy.stats import mannwhitneyu
    stat, p_valor = mannwhitneyu(alto_vals, bajo_vals, alternative='two-sided')
    
    print(f"\n{senal.upper()}:")
    print(f"  Alto: n={len(alto_vals)}, media={np.mean(alto_vals):.2f} ± {np.std(alto_vals):.2f}")
    print(f"  Bajo: n={len(bajo_vals)}, media={np.mean(bajo_vals):.2f} ± {np.std(bajo_vals):.2f}")
    print(f"  Mann-Whitney U test: p={p_valor:.4f}")
    if p_valor < 0.05:
        print(f"  → Diferencia significativa entre grupos")

# Guardar
df.to_csv('analysis/clasificacion_correcta/datos_con_grupo.csv', index=False)
print("\n✅ Guardado: analysis/clasificacion_correcta/datos_con_grupo.csv")
