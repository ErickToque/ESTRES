"""
Comparación del rendimiento del clasificador:
- Intra-sujeto: entrenar con 2 exámenes, predecir el tercero
- Inter-sujeto: 10-fold CV estándar
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

# Cargar datos del método Amin
from analysis.amin_method.01_reproducir_amin import load_all_features_amin

data = load_all_features_amin()

# Organizar por sujeto
subject_data = {}
for d in data:
    p = d['participant']
    if p not in subject_data:
        subject_data[p] = []
    subject_data[p].append(d)

print("=" * 70)
print("COMPARACIÓN INTRA-SUJETO vs INTER-SUJETO")
print("=" * 70)

# ============================================================================
# 1. INTER-SUJETO (10-fold CV estándar)
# ============================================================================
X = np.array([d['features'] for d in data])
y = np.array([d['grade_class'] for d in data])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=min(10, len(y)), shuffle=True, random_state=42)

inter_scores = []
for train_idx, test_idx in cv.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    inter_scores.append(accuracy_score(y_test, y_pred))

print(f"\n📊 INTER-SUJETO (10-fold CV):")
print(f"   Accuracy media: {np.mean(inter_scores):.2%} ± {np.std(inter_scores):.2%}")

# ============================================================================
# 2. INTRA-SUJETO (dejar un examen fuera por sujeto)
# ============================================================================
intra_scores = []

for p, samples in subject_data.items():
    if len(samples) < 2:
        continue
    
    for i in range(len(samples)):
        # Test: un examen, Train: los otros
        X_test = np.array([samples[i]['features']])
        y_test = np.array([samples[i]['grade_class']])
        
        X_train = np.array([samples[j]['features'] for j in range(len(samples)) if j != i])
        y_train = np.array([samples[j]['grade_class'] for j in range(len(samples)) if j != i])
        
        if len(X_train) == 0:
            continue
        
        # Normalizar por sujeto (escalamiento intra-sujeto)
        scaler_intra = StandardScaler()
        X_train_scaled = scaler_intra.fit_transform(X_train)
        X_test_scaled = scaler_intra.transform(X_test)
        
        clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        intra_scores.append(accuracy_score(y_test, y_pred))

print(f"\n📊 INTRA-SUJETO (dejar un examen fuera):")
print(f"   Accuracy media: {np.mean(intra_scores):.2%} ± {np.std(intra_scores):.2%}")
print(f"   N predicciones: {len(intra_scores)}")

# ============================================================================
# 3. ANÁLISIS DE FALLAS POR SUJETO
# ============================================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE FALLAS POR SUJETO")
print("=" * 70)

for p, samples in subject_data.items():
    if len(samples) < 2:
        continue
    
    correctas = 0
    total = 0
    
    for i in range(len(samples)):
        X_test = np.array([samples[i]['features']])
        y_test = np.array([samples[i]['grade_class']])
        
        X_train = np.array([samples[j]['features'] for j in range(len(samples)) if j != i])
        y_train = np.array([samples[j]['grade_class'] for j in range(len(samples)) if j != i])
        
        if len(X_train) == 0:
            continue
        
        scaler_intra = StandardScaler()
        X_train_scaled = scaler_intra.fit_transform(X_train)
        X_test_scaled = scaler_intra.transform(X_test)
        
        clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        if y_pred[0] == y_test[0]:
            correctas += 1
        total += 1
        
        # Mostrar fallas específicas
        if y_pred[0] != y_test[0]:
            grade_actual = samples[i]['grade']
            print(f"   {p} - {samples[i]['exam']}: grade={grade_actual:.0f}, "
                  f"predicción={'Alta' if y_pred[0]==1 else 'Baja'}, "
                  f"real={'Alta' if y_test[0]==1 else 'Baja'} ❌")
    
    if total > 0:
        print(f"\n{p}: {correctas}/{total} correctas ({correctas/total:.0%})")

# ============================================================================
# CONCLUSIÓN
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSIONES")
print("=" * 70)

if np.mean(inter_scores) > np.mean(intra_scores) + 0.1:
    print("⚠️ El modelo inter-sujeto funciona MEJOR que intra-sujeto")
    print("   → Esto sugiere que las señales capturan diferencias ENTRE sujetos,")
    print("     pero NO cambios DENTRO del mismo sujeto.")
elif np.mean(intra_scores) > np.mean(inter_scores) + 0.1:
    print("✅ El modelo intra-sujeto funciona MEJOR que inter-sujeto")
    print("   → Cada sujeto tiene su propia 'firma fisiológica'")
    print("   → Necesitan modelos personalizados")
else:
    print("➡️ Rendimiento similar entre intra e inter-sujeto")
    print("   → Las señales no son consistentes ni entre ni dentro de sujetos")

print("\n" + "=" * 70)
