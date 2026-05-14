"""
Reproducción del método de Amin et al. 2022 (HI-POCT)
- Extracción de trendline con lowpass filter (cut-off manual)
- Features por ventanas (Tw=5,15,30 min)
- Clasificación binaria (grade ≥80% = alto)
- Validación 10-fold cross-validation
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cut-off frequencies según Tabla I del paper (convertidas de Hz a rad/s para filtro)
# Nota: Los autores usaron filtros FIR lowpass con cut-offs entre 0.0002 y 0.002 Hz
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

# Ventanas de tiempo (minutos)
TW_VALUES = [5, 15, 30]

# ============================================================================
# FUNCIONES DEL PAPER
# ============================================================================

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    """
    Extrae trendline usando lowpass filter (método de Amin et al.)
    """
    # Diseño de filtro FIR (como en el paper)
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    order = 100  # Orden alto para filtro muy suave
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    trendline = signal.filtfilt(b, 1, eda)
    return trendline

def extract_features_by_windows(trendline, fs=4.0, tw_min=5):
    """
    Extrae las 9 features por ventana (como en el paper)
    
    Features:
    1. µ_start: mean of start window
    2. µ_mid: mean of middle window
    3. µ_end: mean of end window
    4. σ²_start: variance of start window
    5. σ²_mid: variance of middle window
    6. σ²_end: variance of end window
    7. ρ = µ_mid / (µ_start + µ_end)
    8. µ_diff: mean of difference signal
    9. σ²_diff: variance of difference signal
    """
    T_seconds = len(trendline) / fs
    T_minutes = T_seconds / 60
    
    tw_samples = int(tw_min * 60 * fs)
    
    if tw_samples * 3 > len(trendline):
        return None
    
    # Ventanas: inicio, mitad, final
    start = trendline[:tw_samples]
    
    mid_start = len(trendline)//2 - tw_samples//2
    mid = trendline[mid_start:mid_start + tw_samples]
    
    end = trendline[-tw_samples:]
    
    # Features
    mu_start = np.mean(start)
    mu_mid = np.mean(mid)
    mu_end = np.mean(end)
    
    var_start = np.var(start)
    var_mid = np.var(mid)
    var_end = np.var(end)
    
    # Ratio ρ
    rho = mu_mid / (mu_start + mu_end + 1e-6)
    
    # Difference signal (para ventana start, como en el paper)
    diff = np.diff(start)
    mu_diff = np.mean(diff)
    var_diff = np.var(diff)
    
    features = [mu_start, mu_mid, mu_end, var_start, var_mid, var_end, rho, mu_diff, var_diff]
    
    return np.array(features)

def load_all_features_amin():
    """
    Carga todas las features para todos los participantes y exámenes
    Usando el método de Amin et al.
    """
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    
    # Cargar calificaciones originales
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
                try:
                    grades_raw[student_key][current_exam] = float(parts[-1])
                except:
                    pass
    
    all_data = []
    
    for p in participants:
        for exam in exams:
            # Obtener calificación
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            if exam == 'Final':
                grade = grade_raw / 2
            else:
                grade = grade_raw
            
            # Clase binaria (≥80 = alto)
            grade_class = 1 if grade >= 80 else 0
            
            # Cargar EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                continue
            
            eda = eda_df['eda'].values * 10  # corrección x10
            fs = 4.0
            
            # Cut-off específico para este sujeto/examen
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            
            # Extraer trendline
            trendline = lowpass_filter_trendline(eda, fs, cutoff)
            
            # Extraer features para cada Tw
            features = []
            for tw in TW_VALUES:
                feat = extract_features_by_windows(trendline, fs, tw)
                if feat is not None:
                    features.extend(feat)
            
            if len(features) == 0:
                continue
            
            all_data.append({
                'participant': p,
                'exam': exam,
                'grade': grade,
                'grade_class': grade_class,
                'features': np.array(features)
            })
    
    return all_data

# ============================================================================
# CLASIFICACIÓN
# ============================================================================

def run_classification(data, classifier_name, tw_min=None):
    """
    Ejecuta clasificación con 10-fold cross-validation
    """
    # Filtrar por Tw si se especifica
    # (en el paper entrenaron clasificadores separados para cada Tw)
    
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clasificador
    if classifier_name == 'kNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'SVM':
        clf = SVC(kernel='rbf', C=1.0)
    elif classifier_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == 'BaggedTrees':
        clf = BaggingClassifier(random_state=42)
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
    
    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=min(10, len(y)), shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    
    return {
        'classifier': classifier_name,
        'accuracy_mean': np.mean(scores),
        'accuracy_std': np.std(scores),
        'scores': scores
    }

# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("REPRODUCCIÓN DEL MÉTODO DE AMIN ET AL. (2022)")
print("=" * 70)

# Cargar datos
print("\n1. Cargando datos con método Amin et al...")
data = load_all_features_amin()
print(f"   Muestras totales: {len(data)}")

# Ver distribución de clases
classes = [d['grade_class'] for d in data]
print(f"   Clase Alta (≥80): {sum(classes)} muestras")
print(f"   Clase Baja (<80): {len(classes) - sum(classes)} muestras")

# Clasificadores a probar (como en Tablas II-IV del paper)
classifiers = ['kNN', 'SVM', 'RandomForest', 'BaggedTrees']

print("\n2. Ejecutando clasificación con 10-fold CV...")
print("-" * 50)

results = []
for clf_name in classifiers:
    result = run_classification(data, clf_name)
    results.append(result)
    print(f"\n{clf_name}:")
    print(f"   Accuracy: {result['accuracy_mean']:.2%} ± {result['accuracy_std']:.2%}")

# Mejor resultado
best = max(results, key=lambda x: x['accuracy_mean'])
print("\n" + "=" * 70)
print(f"MEJOR RESULTADO: {best['classifier']} con {best['accuracy_mean']:.2%} accuracy")
print(f"Rango reportado por Amin et al.: 70-80%")
print("=" * 70)

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('analysis/amin_method/resultados_amin.csv', index=False)
print("\n✅ Resultados guardados: analysis/amin_method/resultados_amin.csv")
