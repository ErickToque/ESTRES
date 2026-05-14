"""
Análisis de cómo cambian las características fisiológicas
cuando cambian las calificaciones (mismo sujeto)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

# Cargar datos
df = pd.read_csv('results/features_individual_grades.csv')

# Orden correcto de exámenes
exam_order = {'midterm_1': 0, 'midterm_2': 1, 'Final': 2}

participants = df['participant'].unique()
feature_cols = [c for c in df.columns if c.startswith(('eda_', 'hrv_', 'temp_'))]

print("=" * 70)
print("ANÁLISIS DE TENDENCIAS INTRA-SUJETO")
print("=" * 70)

resultados = []

for p in participants:
    print(f"\n{'='*50}")
    print(f"SUJETO: {p}")
    print(f"{'='*50}")
    
    # Datos del sujeto ordenados cronológicamente
    subject_data = df[df['participant'] == p].copy()
    subject_data['exam_num'] = subject_data['exam'].map(exam_order)
    subject_data = subject_data.sort_values('exam_num')
    
    grades = subject_data['grade'].values
    exams = subject_data['exam'].values
    
    print(f"\nCalificaciones:")
    for i, (e, g) in enumerate(zip(exams, grades)):
        print(f"   {e}: {g:.1f}")
    
    # Calcular cambios
    if len(grades) >= 2:
        print(f"\nCambios en calificaciones:")
        for i in range(1, len(grades)):
            delta = grades[i] - grades[i-1]
            arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
            print(f"   {exams[i-1]} → {exams[i]}: {arrow} {delta:+.1f} puntos")
    
    # Analizar cada característica
    print(f"\nCaracterísticas que cambian con las calificaciones:")
    
    for col in feature_cols:
        values = subject_data[col].values
        
        # Verificar que hay variabilidad
        if len(np.unique(values)) < 2:
            continue
        
        # Correlación
        corr = np.corrcoef(grades, values)[0, 1]
        
        # Interpretación
        if abs(corr) > 0.5:
            if corr > 0:
                direction = "↑ AUMENTA con grade"
            else:
                direction = "↓ DISMINUYE con grade"
            print(f"   {col}: r={corr:.2f} {direction}")
            
            resultados.append({
                'participant': p,
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })

# Resumen global
print("\n" + "=" * 70)
print("RESUMEN: Características más consistentes")
print("=" * 70)

if resultados:
    df_res = pd.DataFrame(resultados)
    
    # Promedio por característica
    feature_avg = df_res.groupby('feature')['abs_correlation'].mean().sort_values(ascending=False)
    
    print("\nCaracterísticas con mayor correlación promedio (en valor absoluto):")
    for feat, avg_corr in feature_avg.head(10).items():
        print(f"   {feat}: {avg_corr:.3f}")
    
    # Sujetos con más correlaciones
    subject_avg = df_res.groupby('participant')['abs_correlation'].mean().sort_values(ascending=False)
    print("\nSujetos con mejor consistencia:")
    for subj, avg_corr in subject_avg.items():
        print(f"   {subj}: {avg_corr:.3f}")
    
    # Guardar
    df_res.to_csv('analysis/intra_sujeto/tendencias_intra_sujeto.csv', index=False)
    print("\n✅ Guardado: analysis/intra_sujeto/tendencias_intra_sujeto.csv")
else:
    print("⚠️ No se encontraron correlaciones significativas")
