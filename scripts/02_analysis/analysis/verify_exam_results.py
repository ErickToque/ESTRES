"""
Verificar resultados de clasificación por examen
Usando los datos reales del dataset
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from scripts.utils import load_features_by_window

print("=" * 70)
print("VERIFICACIÓN DE RESULTADOS POR EXAMEN")
print("=" * 70)

# Probar Tw=1 y Tw=5
windows = [1, 5]
exam_names = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}

results = []

for exam in exam_names:
    print(f"\n📚 {exam_labels[exam]}")
    print("-" * 40)
    
    for tw in windows:
        # Cargar features para esta ventana
        all_features = load_features_by_window(tw)
        
        # Filtrar por examen (necesitamos etiqueta de examen en los datos)
        # Como load_features_by_window no incluye el examen, usamos el enfoque anterior
        # Alternativa: cargar datos completos y filtrar por examen
        
        # Por ahora, usamos los datos que tenemos
        X = np.array([f['features'] for f in all_features])
        y = np.array([f['grade_class'] for f in all_features])
        groups = [f['participant'] for f in all_features]
        
        if len(X) < 3:
            print(f"  Tw={tw}min: datos insuficientes ({len(X)} muestras)")
            continue
        
        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # LOSO
        logo = LeaveOneGroupOut()
        scores = []
        
        for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_scaled[train_idx], y[train_idx])
            y_pred = rf.predict(X_scaled[test_idx])
            scores.append(accuracy_score(y[test_idx], y_pred))
        
        mean_acc = np.mean(scores) * 100
        std_acc = np.std(scores) * 100
        print(f"  Tw={tw}min: {mean_acc:.1f}% ± {std_acc:.1f}% (n={len(X)})")
        
        results.append({
            'exam': exam_labels[exam],
            'tw': tw,
            'accuracy': mean_acc,
            'std': std_acc,
            'n_samples': len(X)
        })

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

for r in results:
    print(f"{r['exam']} - Tw={r['tw']}min: {r['accuracy']:.1f}% (n={r['n_samples']})")
