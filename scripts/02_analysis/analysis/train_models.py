"""
Entrenamiento y comparación de modelos unimodales vs multimodales
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

def load_features():
    """Cargar características previamente extraídas"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import DATA_PATH
    
    features_path = Path(__file__).parent.parent / 'results' / 'features_all.csv'
    df = pd.read_csv(features_path)
    return df

def prepare_model_data(df):
    """Preparar datos para modelos"""
    # Definir grupos de características
    eda_cols = [c for c in df.columns if c.startswith('eda_')]
    hrv_cols = [c for c in df.columns if c.startswith('hrv_')]
    temp_cols = [c for c in df.columns if c.startswith('temp_')]
    
    X_eda = df[eda_cols].fillna(0)
    X_hrv = df[hrv_cols].fillna(0)
    X_temp = df[temp_cols].fillna(0)
    X_all = pd.concat([X_eda, X_hrv, X_temp], axis=1)
    
    y = df['grade'].values
    groups = df['participant'].values  # LOSO por participante
    
    return {
        'X_eda': X_eda, 'X_hrv': X_hrv, 'X_temp': X_temp, 'X_all': X_all,
        'y': y, 'groups': groups,
        'cols': {'eda': eda_cols, 'hrv': hrv_cols, 'temp': temp_cols}
    }

def train_and_evaluate(X, y, groups, model_name):
    """Entrenar con LOSO y retornar métricas"""
    logo = LeaveOneGroupOut()
    mae_scores = []
    r2_scores = []
    predictions = []
    true_values = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelo
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        true_values.extend(y_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    
    return {
        'model': model_name,
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'predictions': predictions,
        'true_values': true_values
    }

def main():
    print("=" * 50)
    print("COMPARACIÓN UNIMODAL VS MULTIMODAL")
    print("=" * 50)
    
    df = load_features()
    data = prepare_model_data(df)
    
    print(f"\nDatos: {len(df)} muestras, {len(df['participant'].unique())} participantes")
    print(f"Features: EDA({data['X_eda'].shape[1]}), HRV({data['X_hrv'].shape[1]}), TEMP({data['X_temp'].shape[1]})")
    
    # Definir modelos
    models = [
        ('M1: EDA-only', data['X_eda']),
        ('M2: HRV-only', data['X_hrv']),
        ('M3: TEMP-only', data['X_temp']),
        ('M4: Multimodal Early (EDA+HRV+TEMP)', data['X_all']),
        ('M5: EDA+HRV', pd.concat([data['X_eda'], data['X_hrv']], axis=1)),
    ]
    
    results = []
    for name, X in models:
        print(f"\n▶ Ejecutando {name}...")
        result = train_and_evaluate(X, data['y'], data['groups'], name)
        results.append(result)
        print(f"   MAE = {result['mae_mean']:.2f} ± {result['mae_std']:.2f}")
        print(f"   R²  = {result['r2_mean']:.3f} ± {result['r2_std']:.3f}")
    
    # Tabla resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE RESULTADOS")
    print("=" * 50)
    results_df = pd.DataFrame(results)
    print(results_df[['model', 'mae_mean', 'mae_std', 'r2_mean']].to_string(index=False))
    
    # Guardar resultados
    from pathlib import Path
    output_path = Path(__file__).parent.parent / 'results' / 'results_all.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Resultados guardados en {output_path}")
    
    # Calcular mejora del multimodal vs EDA
    eda_mae = results_df[results_df['model'].str.contains('EDA-only')]['mae_mean'].values[0]
    multi_mae = results_df[results_df['model'].str.contains('Multimodal Early')]['mae_mean'].values[0]
    improvement = ((eda_mae - multi_mae) / eda_mae) * 100
    print(f"\n Mejora Multimodal vs EDA-only: {improvement:.1f}%")
    
    return results_df

if __name__ == "__main__":
    main()
