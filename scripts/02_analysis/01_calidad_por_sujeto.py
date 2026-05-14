"""
Análisis de calidad de señales por sujeto
Identifica qué sujetos tienen datos utilizables
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from scripts.utils import load_signal, get_student_grades

participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

print("=" * 70)
print("ANÁLISIS DE CALIDAD POR SUJETO")
print("=" * 70)

calidad_data = []

for p in participants:
    print(f"\n📌 {p}:")
    
    for exam in exams:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None:
            print(f"   {exam}: ❌ Sin datos")
            continue
        
        eda = eda_df['eda'].values
        zeros = (eda == 0).sum()
        pct_zeros = zeros / len(eda) * 100
        
        # Determinar calidad
        if pct_zeros < 5:
            calidad = "✅ BUENA"
        elif pct_zeros < 20:
            calidad = "⚠️ ACEPTABLE"
        else:
            calidad = "❌ MALA"
        
        print(f"   {exam}: {pct_zeros:5.1f}% ceros → {calidad}")
        
        calidad_data.append({
            'participant': p,
            'exam': exam,
            'pct_zeros': pct_zeros,
            'calidad': calidad
        })

# Resumen
print("\n" + "=" * 70)
print("RESUMEN POR SUJETO (promedio de ceros)")
print("=" * 70)

df_calidad = pd.DataFrame(calidad_data)
for p in participants:
    p_data = df_calidad[df_calidad['participant'] == p]
    avg_zeros = p_data['pct_zeros'].mean()
    
    if avg_zeros < 5:
        recomendacion = "USAR"
    elif avg_zeros < 20:
        recomendacion = "USAR CON CUIDADO"
    else:
        recomendacion = "DESCARTAR"
    
    print(f"{p}: promedio {avg_zeros:.1f}% ceros → {recomendacion}")

# Guardar resultados
df_calidad.to_csv('analysis/intra_sujeto/calidad_por_sujeto.csv', index=False)
print("\n✅ Guardado: analysis/intra_sujeto/calidad_por_sujeto.csv")
