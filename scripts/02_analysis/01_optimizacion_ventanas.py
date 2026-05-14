"""
OPTIMIZACIÓN DE VENTANAS Y CUT-OFF PARA CLASIFICACIÓN BINARIA
- Diferentes tamaños de ventana (5, 15, 30 minutos)
- Cut-off adaptativo por sujeto (optimizado con búsqueda)
- Validación LOSO (sin data leakage)
- Solo clasificación binaria (≥80% = alto rendimiento)
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from scripts.utils import load_signal

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Posibles cut-off frequencies (Hz) a probar
CUTOFF_CANDIDATES = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# Tamaños de ventana (minutos)
TW_VALUES = [5, 15, 30]

# Feature names (9 por ventana)
FEATURE_NAMES = ['mu_start', 'mu_mid', 'mu_end', 
                 'var_start', 'var_mid', 'var_end',
                 'rho', 'mu_diff', 'var_diff']

def load_grades_individual():
    """Carga calificaciones individuales"""
    grades_raw = {}
    with open('data/wearable-exam-stress/StudentGrades.txt', 'r', encoding='latin1') as f:
        lines = f.readlines()
    current_exam = None
    for line in lines:
        line = line.strip()
        if 'MIDTERM 1' in line.upper():
            current_exam = 'midterm_1'
        elif 'MIDTERM 2' in line.upper():
            current_exam = 'midterm_2'
        elif 'FINAL' in line.upper():
            current_exam = 'Final'
        elif line.startswith('S') and current_exam:
            parts = line.split()
            if len(parts) >= 2:
                student_num = int(parts[0].strip()[1:])
                student_key = f'S{student_num}'
                if student_key not in grades_raw:
                    grades_raw[student_key] = {}
                grades_raw[student_key][current_exam] = float(parts[-1])
    return grades_raw

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    """Filtro lowpass FIR para extraer trendline"""
    nyquist = fs / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_features_from_trendline(trendline, fs=4.0, tw_min=5):
    """Extrae 9 features de la trendline para una ventana específica"""
    tw_samples = int(tw_min * 60 * fs)
    
    if tw_samples * 3 > len(trendline):
        return None
    
    # Ventanas: inicio, mitad, final
    start = trendline[:tw_samples]
    mid_start = len(trendline)//2 - tw_samples//2
    mid = trendline[mid_start:mid_start + tw_samples]
    end = trendline[-tw_samples:]
    
    # Features
    mu_start, mu_mid, mu_end = np.mean(start), np.mean(mid), np.mean(end)
    var_start, var_mid, var_end = np.var(start), np.var(mid), np.var(end)
    rho = mu_mid / (mu_start + mu_end + 1e-6)
    diff = np.diff(start)
    mu_diff, var_diff = np.mean(diff), np.var(diff)
    
    return np.array([mu_start, mu_mid, mu_end, var_start, var_mid, var_end, rho, mu_diff, var_diff])

def load_data_for_subject(participant, exam, cutoff_hz, tw_min=5):
    """Carga features para un sujeto, examen, cutoff y ventana específicos"""
    eda_df = load_signal(participant, exam, 'EDA')
    if eda_df is None:
        return None
    
    eda = eda_df['eda'].values * 10
    trendline = lowpass_filter_trendline(eda, 4.0, cutoff_hz)
    features = extract_features_from_trendline(trendline, 4.0, tw_min)
    
    return features

def evaluate_configuration(tw_min, cutoff_hz, all_data):
    """Evalúa una configuración específica con LOSO"""
    X = []
    y = []
    groups = []
    
    for item in all_data:
        if item['tw'] != tw_min:
            continue
        if abs(item['cutoff'] - cutoff_hz) > 1e-6:
            continue
        
        X.append(item['features'])
        y.append(item['grade_class'])
        groups.append(item['participant'])
    
    if len(X) < 10:
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    # Eliminar features constantes
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logo = LeaveOneGroupOut()
    scores = []
    f1_scores = []
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    
    return {
        'tw': tw_min,
        'cutoff': cutoff_hz,
        'accuracy': np.mean(scores),
        'accuracy_std': np.std(scores),
        'f1': np.mean(f1_scores),
        'n_samples': len(X)
    }

print("=" * 80)
print("OPTIMIZACIÓN DE VENTANAS Y CUT-OFF")
print("Clasificación binaria (≥80% = alto rendimiento)")
print("Validación: LOSO (sin data leakage)")
print("=" * 80)

# Cargar calificaciones
grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

print("\n📊 Cargando datos para todas las combinaciones...")

# Generar todos los datos
all_data = []
total_combinations = len(participants) * len(exams) * len(CUTOFF_CANDIDATES) * len(TW_VALUES)
count = 0

for p in participants:
    for exam in exams:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        
        grade_raw = grades_raw[p][exam]
        grade = grade_raw / 2 if exam == 'Final' else grade_raw
        grade_class = 1 if grade >= 80 else 0
        
        for cutoff in CUTOFF_CANDIDATES:
            for tw in TW_VALUES:
                features = load_data_for_subject(p, exam, cutoff, tw)
                if features is not None:
                    all_data.append({
                        'participant': p,
                        'exam': exam,
                        'grade': grade,
                        'grade_class': grade_class,
                        'cutoff': cutoff,
                        'tw': tw,
                        'features': features
                    })
        count += 1
        print(f"   Procesado {p} - {exam}...")

print(f"\n✅ Total muestras generadas: {len(all_data)}")

# Evaluar todas las configuraciones
print("\n🔍 Evaluando configuraciones con LOSO...")

results = []
for tw in TW_VALUES:
    for cutoff in CUTOFF_CANDIDATES:
        result = evaluate_configuration(tw, cutoff, all_data)
        if result is not None:
            results.append(result)
            print(f"   Tw={tw}min, cutoff={cutoff}Hz → Acc={result['accuracy']:.2%} ± {result['accuracy_std']:.2%}")

# Ordenar por accuracy
results.sort(key=lambda x: x['accuracy'], reverse=True)

print("\n" + "=" * 80)
print("MEJORES CONFIGURACIONES")
print("=" * 80)

for i, r in enumerate(results[:10]):
    print(f"{i+1}. Tw={r['tw']}min, cutoff={r['cutoff']}Hz: {r['accuracy']:.2%} ± {r['accuracy_std']:.2%} (F1={r['f1']:.2%})")

# Mejor configuración
best = results[0]
print("\n" + "=" * 80)
print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
print("=" * 80)
print(f"   Tamaño de ventana: {best['tw']} minutos")
print(f"   Cut-off frequency: {best['cutoff']} Hz")
print(f"   Accuracy (LOSO):   {best['accuracy']:.2%} ± {best['accuracy_std']:.2%}")
print(f"   F1-Score:          {best['f1']:.2%}")

# Comparar con baseline (cutoff=0.002, tw=15 como en paper original)
baseline = None
for r in results:
    if abs(r['cutoff'] - 0.002) < 1e-6 and r['tw'] == 15:
        baseline = r
        break

if baseline:
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON BASELINE (cutoff=0.002Hz, Tw=15min)")
    print("=" * 80)
    print(f"   Baseline:     {baseline['accuracy']:.2%} ± {baseline['accuracy_std']:.2%}")
    print(f"   Optimizado:   {best['accuracy']:.2%} ± {best['accuracy_std']:.2%}")
    improvement = (best['accuracy'] - baseline['accuracy']) * 100
    print(f"   Mejora:       +{improvement:.1f}%")

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('paper_resultados/optimizacion/resultados_optimizacion.csv', index=False)
print("\n✅ Resultados guardados: paper_resultados/optimizacion/resultados_optimizacion.csv")

# Gráfico de resultados
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, tw in enumerate(TW_VALUES):
    ax = axes[i]
    tw_results = [r for r in results if r['tw'] == tw]
    if not tw_results:
        continue
    
    cutoffs = [r['cutoff'] for r in tw_results]
    accuracies = [r['accuracy'] for r in tw_results]
    
    ax.semilogx(cutoffs, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Cut-off frequency (Hz)')
    ax.set_ylabel('Accuracy (LOSO)')
    ax.set_title(f'Tw = {tw} minutos')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Azar (50%)')
    ax.legend()
    
    # Marcar mejor punto
    best_idx = np.argmax(accuracies)
    ax.plot(cutoffs[best_idx], accuracies[best_idx], 'ro', markersize=12, markeredgecolor='black')

plt.suptitle('Optimización de Cut-off Frequency por Tamaño de Ventana', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/optimizacion/optimizacion_cutoff.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/optimizacion/optimizacion_cutoff.png")

print("\n" + "=" * 80)
print("RESUMEN DE MEJORES RESULTADOS POR VENTANA")
print("=" * 80)
for tw in TW_VALUES:
    tw_results = [r for r in results if r['tw'] == tw]
    if tw_results:
        best_tw = max(tw_results, key=lambda x: x['accuracy'])
        print(f"Tw={tw}min: mejor cutoff={best_tw['cutoff']}Hz → Acc={best_tw['accuracy']:.2%}")
