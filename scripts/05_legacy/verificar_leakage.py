"""
Verificación de data leakage en el método de Amin et al.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
sys.path.append('.')

from scripts.utils import load_signal

# Configuración
CUTOFF_FREQS = {
    'S1': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S2': {'midterm_1': 0.002, 'midterm_2': 0.001, 'Final': 0.002},
    'S3': {'midterm_1': 0.0002, 'midterm_2': 0.0002, 'Final': 0.001},
    'S4': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S5': {'midterm_1': 0.001, 'midterm_2': 0.001, 'Final': 0.001},
    'S6': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S7': {'midterm_1': 0.001, 'midterm_2': 0.0002, 'Final': 0.0002},
    'S8': {'midterm_1': 0.0002, 'midterm_2': 0.002, 'Final': 0.002},
    'S9': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S10': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
}

def load_features_direct():
    """Carga features directamente sin usar el otro script"""
    from scipy import signal
    
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    
    # Cargar calificaciones
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
    
    def lowpass_filter(eda, fs=4.0, cutoff_hz=0.002):
        nyquist = fs / 2
        normalized_cutoff = cutoff_hz / nyquist
        order = 100
        b = signal.firwin(order, normalized_cutoff, window='hamming')
        return signal.filtfilt(b, 1, eda)
    
    def extract_features(trendline, fs=4.0, tw_min=5):
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
    
    TW_VALUES = [5, 15, 30]
    all_data = []
    
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                continue
            
            eda = eda_df['eda'].values * 10
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            trendline = lowpass_filter(eda, 4.0, cutoff)
            
            features = []
            for tw in TW_VALUES:
                feat = extract_features(trendline, 4.0, tw)
                if feat is not None:
                    features.extend(feat)
            
            if len(features) > 0:
                all_data.append({
                    'participant': p,
                    'grade_class': grade_class,
                    'features': np.array(features)
                })
    
    return all_data

print("=" * 70)
print("VERIFICACIÓN DE DATA LEAKAGE")
print("=" * 70)

# Cargar datos
print("\nCargando datos...")
data = load_features_direct()
X = np.array([d['features'] for d in data])
y = np.array([d['grade_class'] for d in data])
participants = [d['participant'] for d in data]

print(f"Muestras: {len(data)}")
print(f"Participantes: {len(set(participants))}")
print(f"Clase Alta: {sum(y)}/{len(y)}")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 1. MÉTODO ORIGINAL (10-fold CV aleatorio)
# ============================================================================
print("\n1. MÉTODO ORIGINAL (10-fold CV aleatorio):")
cv_random = StratifiedKFold(n_splits=min(10, len(y)), shuffle=True, random_state=42)
scores_random = []

for train_idx, test_idx in cv_random.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    scores_random.append(accuracy_score(y_test, y_pred))

print(f"   Accuracy: {np.mean(scores_random):.2%} ± {np.std(scores_random):.2%}")

# ============================================================================
# 2. MÉTODO SIN LEAKAGE (Leave-One-Subject-Out)
# ============================================================================
print("\n2. MÉTODO SIN LEAKAGE (Leave-One-Subject-Out):")
logo = LeaveOneGroupOut()
scores_logo = []

for train_idx, test_idx in logo.split(X_scaled, y, groups=participants):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    scores_logo.append(accuracy_score(y_test, y_pred))

print(f"   Accuracy: {np.mean(scores_logo):.2%} ± {np.std(scores_logo):.2%}")

# ============================================================================
# 3. DIFERENCIA
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)

diff = np.mean(scores_random) - np.mean(scores_logo)
print(f"Diferencia: {diff:.2%}")

if diff > 0.1:
    print("\n⚠️ ¡DATA LEAKAGE DETECTADO! (diferencia >10%)")
    print("   El método original sobreestima el rendimiento")
    print("   Recomendación: Reportar la validación LOSO como resultado principal")
elif diff > 0.05:
    print("\n⚠️ Posible data leakage moderado (5-10%)")
    print("   Recomendación: Reportar ambos resultados y discutir la diferencia")
else:
    print("\n✅ No hay evidencia de data leakage significativo")
    print("   El método original es robusto y generalizable")

print(f"\n📊 RESUMEN:")
print(f"   10-fold CV (con posible leakage): {np.mean(scores_random):.2%}")
print(f"   LOSO (sin leakage): {np.mean(scores_logo):.2%}")
