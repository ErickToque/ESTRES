"""
VENTANAS MÁS PEQUEÑAS (Tw < 5 minutos)
- Probar Tw = 1, 2, 3, 4 minutos
- Cut-off óptimo basado en resultados anteriores (0.0002 Hz)
- Validación LOSO
- Comparar con Tw=5 (mejor hasta ahora)
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from scripts.utils import load_signal

# Configuración
CUTOFF = 0.0002  # Hz (mejor encontrado)
TW_VALUES = [1, 2, 3, 4, 5]  # minutos

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    nyquist = fs / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_features_from_trendline(trendline, fs=4.0, tw_min=5):
    tw_samples = int(tw_min * 60 * fs)
    
    # Verificar que la ventana no sea demasiado grande
    if tw_samples * 3 > len(trendline):
        return None
    
    start = trendline[:tw_samples]
    mid_start = len(trendline)//2 - tw_samples//2
    mid = trendline[mid_start:mid_start + tw_samples]
    end = trendline[-tw_samples:]
    
    mu_start, mu_mid, mu_end = np.mean(start), np.mean(mid), np.mean(end)
    var_start, var_mid, var_end = np.var(start), np.var(mid), np.var(end)
    rho = mu_mid / (mu_start + mu_end + 1e-6)
    diff = np.diff(start)
    mu_diff, var_diff = np.mean(diff), np.var(diff)
    
    return np.array([mu_start, mu_mid, mu_end, var_start, var_mid, var_end, rho, mu_diff, var_diff])

def load_grades_individual():
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

print("=" * 80)
print("VENTANAS MÁS PEQUEÑAS (Tw = 1, 2, 3, 4, 5 minutos)")
print(f"Cut-off fijo: {CUTOFF} Hz")
print("Validación: LOSO - Clasificación binaria (≥80% = alto)")
print("=" * 80)

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']

# Generar datos para cada Tw
results = []

for tw in TW_VALUES:
    print(f"\n📊 Procesando Tw = {tw} min...")
    
    X = []
    y = []
    groups = []
    skipped = 0
    
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            # Cargar EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                skipped += 1
                continue
            
            eda = eda_df['eda'].values * 10
            trendline = lowpass_filter_trendline(eda, 4.0, CUTOFF)
            features = extract_features_from_trendline(trendline, 4.0, tw)
            
            if features is not None:
                X.append(features)
                y.append(grade_class)
                groups.append(p)
            else:
                skipped += 1
    
    if len(X) < 10:
        print(f"   ❌ Datos insuficientes: {len(X)} muestras")
        continue
    
    X = np.array(X)
    y = np.array(y)
    
    # Eliminar features constantes
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
    
    results.append({
        'tw': tw,
        'accuracy': np.mean(scores),
        'accuracy_std': np.std(scores),
        'f1': np.mean(f1_scores),
        'n_samples': len(X),
        'n_features': X.shape[1]
    })
    
    print(f"   ✅ Acc={np.mean(scores):.2%} ± {np.std(scores):.2%} (F1={np.mean(f1_scores):.2%})")
    print(f"      Muestras: {len(X)}, Features tras filtro: {X.shape[1]}")

# Ordenar por accuracy
results.sort(key=lambda x: x['accuracy'], reverse=True)

print("\n" + "=" * 80)
print("RANKING DE VENTANAS (mejor a peor)")
print("=" * 80)
for i, r in enumerate(results, 1):
    tw_color = "🏆" if i == 1 else "  "
    print(f"{tw_color} {i}. Tw={r['tw']}min: {r['accuracy']:.2%} ± {r['accuracy_std']:.2%} (F1={r['f1']:.2%})")

# Mejor resultado
best = results[0]
print("\n" + "=" * 80)
print("🏆 MEJOR VENTANA ENCONTRADA")
print("=" * 80)
print(f"   Tamaño de ventana: {best['tw']} minutos")
print(f"   Accuracy (LOSO):   {best['accuracy']:.2%} ± {best['accuracy_std']:.2%}")
print(f"   F1-Score:          {best['f1']:.2%}")

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('paper_resultados/optimizacion/resultados_ventanas_pequenas.csv', index=False)
print("\n✅ Guardado: paper_resultados/optimizacion/resultados_ventanas_pequenas.csv")

# Gráfico comparativo
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
tw_vals = [r['tw'] for r in results]
accuracies = [r['accuracy'] for r in results]
acc_stds = [r['accuracy_std'] for r in results]

bars = ax.bar([str(t) for t in tw_vals], accuracies, yerr=acc_stds, capsize=5,
              color=['#2ecc71' if i==0 else '#3498db' for i in range(len(results))],
              edgecolor='black')
ax.axhline(y=0.5, color='red', linestyle='--', label='Azar (50%)')
ax.axhline(y=0.6833, color='orange', linestyle=':', label='Tw=5 (referencia)')
ax.set_ylabel('Accuracy (LOSO)')
ax.set_xlabel('Tamaño de ventana (minutos)')
ax.set_title('Comparación de ventanas pequeñas (Tw = 1-5 min)')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('paper_resultados/optimizacion/comparacion_ventanas_pequenas.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/optimizacion/comparacion_ventanas_pequenas.png")

# Análisis adicional: ¿Qué ventana es más estable?
print("\n" + "=" * 80)
print("ANÁLISIS DE ESTABILIDAD")
print("=" * 80)
for r in results:
    cv = r['accuracy_std'] / r['accuracy'] if r['accuracy'] > 0 else 0
    print(f"Tw={r['tw']}min: CV = {cv:.2f} (menor = más estable)")
