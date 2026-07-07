"""
Pipeline CORRECTO: EDA + Normalización + Modelado
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('results/features_all.csv')

print('=' * 60)
print('PIPELINE CORRECTO CON NORMALIZACIÓN')
print('=' * 60)

# 1. Separar features y target
feat_cols = [c for c in df.columns if c not in ['participant', 'exam', 'grade']]
X_raw = df[feat_cols].fillna(0)
y = df['grade'].values
groups = df['participant'].values

print(f'\n1. DATOS CRUDOS:')
print(f'   Muestras: {X_raw.shape[0]}, Features: {X_raw.shape[1]}')
print(f'   Rango de valores: [{X_raw.min().min():.2f}, {X_raw.max().max():.2f}]')

# 2. Detectar y tratar outliers (winsorización)
print(f'\n2. TRATAMIENTO DE OUTLIERS:')
for col in X_raw.columns:
    # Winsorizar al percentil 1 y 99
    lower = np.percentile(X_raw[col], 1)
    upper = np.percentile(X_raw[col], 99)
    X_raw[col] = X_raw[col].clip(lower, upper)
print('    Outliers tratados (winsorización 1-99%)')

# 3. Normalización (RobustScaler es mejor para outliers)
print(f'\n3. NORMALIZACIÓN:')
scaler = RobustScaler()  # Mejor que StandardScaler para datos fisiológicos
X_scaled = scaler.fit_transform(X_raw)
print(f'   Escaler: RobustScaler (mediana y IQR)')
print(f'   Rango después: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]')

# 4. Reducción de dimensionalidad (opcional, si hay overfitting)
print(f'\n4. ANÁLISIS DE SOBREAJUSTE:')
print(f'   Muestras/Features ratio: {X_raw.shape[0]/X_raw.shape[1]:.2f}')
if X_raw.shape[0]/X_raw.shape[1] < 2:
    print('    Ratio bajo -> aplicando PCA')
    pca = PCA(n_components=min(10, X_raw.shape[0]-1))
    X_scaled = pca.fit_transform(X_scaled)
    print(f'   PCA: {X_scaled.shape[1]} componentes')
    print(f'   Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}')
else:
    print('    Ratio aceptable')

# 5. Modelado con validación adecuada
print(f'\n5. ENTRENAMIENTO (LOSO):')
logo = LeaveOneGroupOut()

models = {
    'Random Forest (default)': RandomForestRegressor(random_state=42),
    'Random Forest (regularized)': RandomForestRegressor(
        n_estimators=50, max_depth=5, min_samples_split=5, random_state=42
    ),
}

results = []
for name, model in models.items():
    mae_scores = []
    r2_scores = []
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    
    results.append({
        'model': name,
        'MAE': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'R2': np.mean(r2_scores),
        'R2_std': np.std(r2_scores)
    })

# 6. Resultados
print('\n' + '=' * 60)
print('RESULTADOS FINALES')
print('=' * 60)
for r in results:
    print(f"\n{r['model']}:")
    print(f"   MAE = {r['MAE']:.2f} ± {r['MAE_std']:.2f}")
    print(f"   R²  = {r['R2']:.3f} ± {r['R2_std']:.3f}")

# 7. Conclusión
print('\n' + '=' * 60)
if results[0]['R2'] > 0.3:
    print(' MODELO VÁLIDO - Las características predicen las calificaciones')
elif results[0]['R2'] > 0.1:
    print(' MODELO DÉBIL - Señal detectable pero ruidosa')
else:
    print(' MODELO NO VÁLIDO - Las características NO predicen las calificaciones')
    print('   Posibles causas:')
    print('   1. Las características no son relevantes para predecir grades')
    print('   2. Demasiado ruido en las señales')
    print('   3. Necesitas más muestras (N=30 es pequeño para 18 features)')
print('=' * 60)
