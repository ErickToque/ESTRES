"""
Compara rendimiento de modelos por sujeto: unimodal vs multimodal
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')

df = pd.read_csv('results/features_individual_grades.csv')

# Orden de exámenes
exam_order = {'midterm_1': 0, 'midterm_2': 1, 'Final': 2}

participants = df['participant'].unique()

# Definir grupos de características
eda_cols = [c for c in df.columns if c.startswith('eda_')]
hrv_cols = [c for c in df.columns if c.startswith('hrv_')]
temp_cols = [c for c in df.columns if c.startswith('temp_')]
all_cols = eda_cols + hrv_cols + temp_cols

print("=" * 70)
print("COMPARACIÓN POR SUJETO: Unimodal vs Multimodal")
print("=" * 70)

resultados = []

for p in participants:
    print(f"\n📌 {p}:")
    
    # Datos del sujeto
    subject_data = df[df['participant'] == p].copy()
    subject_data['exam_num'] = subject_data['exam'].map(exam_order)
    subject_data = subject_data.sort_values('exam_num')
    
    if len(subject_data) < 3:
        print(f"   ⚠️ Solo {len(subject_data)} exámenes, saltando")
        continue
    
    grades = subject_data['grade'].values
    
    # Para cada tipo de modelo
    models = {
        'EDA': eda_cols,
        'HRV': hrv_cols,
        'TEMP': temp_cols,
        'MULTIMODAL': all_cols
    }
    
    for model_name, cols in models.items():
        if len(cols) == 0 or len([c for c in cols if c in subject_data.columns]) < 2:
            print(f"   {model_name}: insuficientes features")
            continue
        
        X = subject_data[cols].fillna(0).values
        y = grades
        
        # Si solo tenemos 3 puntos, usar correlación simple
        if len(X) >= 3:
            # Correlación entre cada feature y grade
            correlations = []
            for i in range(X.shape[1]):
                if len(np.unique(X[:, i])) > 1:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    correlations.append(abs(corr))
            
            avg_corr = np.mean(correlations) if correlations else 0
            
            resultados.append({
                'participant': p,
                'model': model_name,
                'avg_correlation': avg_corr,
                'n_features': len(cols)
            })
            
            print(f"   {model_name}: correlación media = {avg_corr:.3f}")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN POR MODELO")
print("=" * 70)

df_res = pd.DataFrame(resultados)
for model in df_res['model'].unique():
    model_data = df_res[df_res['model'] == model]
    print(f"\n{model}:")
    print(f"   Correlación media: {model_data['avg_correlation'].mean():.3f}")
    print(f"   Mejor sujeto: {model_data.loc[model_data['avg_correlation'].idxmax(), 'participant']} "
          f"({model_data['avg_correlation'].max():.3f})")

df_res.to_csv('analysis/intra_sujeto/comparacion_modelos.csv', index=False)
print("\n✅ Guardado: analysis/intra_sujeto/comparacion_modelos.csv")
