"""
COMBINACIÓN DE VENTANAS Tw=1 y Tw=5
Análisis SEPARADO por cada examen (midterm_1, midterm_2, Final)
- Validación LOSO (sin data leakage)
- Clasificación binaria (≥80% = alto rendimiento)
- Comparar: Tw=1 sola, Tw=5 sola, Tw=1+5 combinadas
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

# Configuración óptima
CUTOFF = 0.0002  # Hz
TW_VALUES = [1, 5]

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    nyquist = fs / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_features_from_trendline(trendline, fs=4.0, tw_min=5):
    tw_samples = int(tw_min * 60 * fs)
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

def get_features_for_exam(participant, exam, tw_min, cutoff):
    """Obtiene features para un participante, examen, ventana específica"""
    eda_df = load_signal(participant, exam, 'EDA')
    if eda_df is None:
        return None
    
    eda = eda_df['eda'].values * 10
    trendline = lowpass_filter_trendline(eda, 4.0, cutoff)
    features = extract_features_from_trendline(trendline, 4.0, tw_min)
    
    return features

print("=" * 80)
print("COMBINACIÓN Tw=1 y Tw=5 POR EXAMEN")
print(f"Cut-off: {CUTOFF} Hz")
print("Validación: LOSO - Clasificación binaria (≥80% = alto)")
print("=" * 80)

grades_raw = load_grades_individual()
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}

# Almacenar resultados por examen
all_results = []

for exam in exams:
    print(f"\n{'='*60}")
    print(f"📚 {exam_labels[exam]}")
    print(f"{'='*60}")
    
    # Cargar datos para este examen
    X_tw1 = []
    X_tw5 = []
    y = []
    groups = []
    
    for p in participants:
        if p not in grades_raw or exam not in grades_raw[p]:
            continue
        
        grade_raw = grades_raw[p][exam]
        grade = grade_raw / 2 if exam == 'Final' else grade_raw
        grade_class = 1 if grade >= 80 else 0
        
        features_tw1 = get_features_for_exam(p, exam, 1, CUTOFF)
        features_tw5 = get_features_for_exam(p, exam, 5, CUTOFF)
        
        if features_tw1 is not None and features_tw5 is not None:
            X_tw1.append(features_tw1)
            X_tw5.append(features_tw5)
            y.append(grade_class)
            groups.append(p)
    
    if len(X_tw1) == 0:
        print("   ❌ Datos insuficientes")
        continue
    
    X_tw1 = np.array(X_tw1)
    X_tw5 = np.array(X_tw5)
    X_combined = np.hstack([X_tw1, X_tw5])
    y = np.array(y)
    
    print(f"   Muestras: {len(y)}")
    print(f"   Clase Alta (≥80): {sum(y)}/{len(y)}")
    
    # Escalar y evaluar cada configuración
    configs = [
        ('Tw=1 sola', X_tw1),
        ('Tw=5 sola', X_tw5),
        ('Tw=1+5 combinadas', X_combined)
    ]
    
    for name, X in configs:
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
        
        all_results.append({
            'exam': exam_labels[exam],
            'config': name,
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'f1': np.mean(f1_scores),
            'n_samples': len(y)
        })
        
        print(f"   {name}: Acc={np.mean(scores):.2%} ± {np.std(scores):.2%} (F1={np.mean(f1_scores):.2%})")

# ============================================================================
# RESUMEN GENERAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN GENERAL POR EXAMEN Y CONFIGURACIÓN")
print("=" * 80)

df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))

# ============================================================================
# FIGURA: Comparación por examen
# ============================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {'Tw=1 sola': '#3498db', 'Tw=5 sola': '#2ecc71', 'Tw=1+5 combinadas': '#e74c3c'}

for idx, exam in enumerate(['Midterm 1', 'Midterm 2', 'Final']):
    ax = axes[idx]
    exam_data = df_results[df_results['exam'] == exam]
    
    if len(exam_data) == 0:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
        continue
    
    configs = exam_data['config'].values
    accuracies = exam_data['accuracy'].values
    acc_stds = exam_data['accuracy_std'].values
    
    bars = ax.bar(configs, accuracies, yerr=acc_stds, capsize=5,
                  color=[colors.get(c, 'gray') for c in configs],
                  edgecolor='black')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Azar (50%)')
    ax.set_ylabel('Accuracy (LOSO)')
    ax.set_title(f'{exam}')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Comparación de Configuraciones por Examen', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/optimizacion/comparacion_por_examen.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura guardada: paper_resultados/optimizacion/comparacion_por_examen.png")

# ============================================================================
# TABLA RESUMEN
# ============================================================================
print("\n" + "=" * 80)
print("TABLA RESUMEN PARA EL PAPER")
print("=" * 80)

pivot = df_results.pivot(index='exam', columns='config', values='accuracy')
print(pivot.to_string())

# Guardar resultados
df_results.to_csv('paper_resultados/optimizacion/resultados_por_examen.csv', index=False)
print("\n✅ Guardado: paper_resultados/optimizacion/resultados_por_examen.csv")

# ============================================================================
# CONCLUSIÓN
# ============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONES")
print("=" * 80)

# Mejor por examen
for exam in ['Midterm 1', 'Midterm 2', 'Final']:
    exam_data = df_results[df_results['exam'] == exam]
    if len(exam_data) > 0:
        best = exam_data.loc[exam_data['accuracy'].idxmax()]
        print(f"{exam}: Mejor = {best['config']} ({best['accuracy']:.2%})")

# Mejor global
best_global = df_results.loc[df_results['accuracy'].idxmax()]
print(f"\n🏆 GLOBAL: {best_global['exam']} - {best_global['config']}: {best_global['accuracy']:.2%}")
