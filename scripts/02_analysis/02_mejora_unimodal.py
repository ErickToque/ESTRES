"""
MEJORA DEL MÉTODO DE AMIN ET AL. (2022)
- Limpieza de datos (excluir señales con >20% ceros)
- Optimización de hiperparámetros
- Validación más robusta
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
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

TW_VALUES = [5, 15, 30]

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_features_by_windows(trendline, fs=4.0, tw_min=5):
    T_minutes = len(trendline) / fs / 60
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

def load_data_with_quality_filter(quality_threshold=20):
    """Carga datos excluyendo señales con >quality_threshold% ceros"""
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
    
    all_data = []
    excluded_count = 0
    
    for p in participants:
        for exam in exams:
            # Verificar calidad EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                excluded_count += 1
                continue
            
            eda = eda_df['eda'].values
            zeros_pct = (eda == 0).sum() / len(eda) * 100
            if zeros_pct > quality_threshold:
                excluded_count += 1
                continue
            
            # Calificación
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            # Procesar EDA
            eda_corrected = eda * 10
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            trendline = lowpass_filter_trendline(eda_corrected, 4.0, cutoff)
            
            features = []
            for tw in TW_VALUES:
                feat = extract_features_by_windows(trendline, 4.0, tw)
                if feat is not None:
                    features.extend(feat)
            
            if len(features) > 0:
                all_data.append({
                    'participant': p, 'exam': exam, 'grade': grade,
                    'grade_class': grade_class, 'features': np.array(features),
                    'zeros_pct': zeros_pct
                })
    
    print(f"   Excluidas {excluded_count} muestras por mala calidad (> {quality_threshold}% ceros)")
    print(f"   Muestras finales: {len(all_data)}")
    return all_data

def optimize_and_evaluate(data):
    """Optimización de hiperparámetros y evaluación"""
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    
    # Definir grids de hiperparámetros
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf']
        },
        'kNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'BaggedTrees': {
            'n_estimators': [10, 50, 100]
        }
    }
    
    results = {}
    best_models = {}
    
    for clf_name, param_grid in param_grids.items():
        print(f"\n🔍 Optimizando {clf_name}...")
        
        if clf_name == 'RandomForest':
            base_clf = RandomForestClassifier(random_state=42)
        elif clf_name == 'SVM':
            base_clf = SVC(random_state=42)
        elif clf_name == 'kNN':
            base_clf = KNeighborsClassifier()
        elif clf_name == 'BaggedTrees':
            base_clf = BaggingClassifier(random_state=42)
        
        grid_search = GridSearchCV(base_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        results[clf_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_scores': grid_search.cv_results_['mean_test_score']
        }
        best_models[clf_name] = grid_search.best_estimator_
        
        print(f"   Mejor accuracy: {grid_search.best_score_:.2%}")
        print(f"   Mejores params: {grid_search.best_params_}")
    
    return results, best_models

print("=" * 70)
print("HALLAZGO 2: MEJORA DEL MÉTODO ORIGINAL")
print("Con limpieza de datos y optimización de hiperparámetros")
print("=" * 70)

# Sin filtro de calidad (baseline original)
print("\n1. BASELINE (sin filtro de calidad):")
data_all = load_data_with_quality_filter(quality_threshold=100)
results_all, _ = optimize_and_evaluate(data_all)

# Con filtro de calidad (solo señales buenas)
print("\n2. CON FILTRO DE CALIDAD (<20% ceros):")
data_clean = load_data_with_quality_filter(quality_threshold=20)
results_clean, best_models_clean = optimize_and_evaluate(data_clean)

# Comparación
print("\n" + "=" * 70)
print("COMPARACIÓN DE RESULTADOS")
print("=" * 70)

comparison = []
for clf in ['RandomForest', 'SVM', 'kNN', 'BaggedTrees']:
    comparison.append({
        'classifier': clf,
        'baseline_accuracy': results_all[clf]['best_score'] * 100,
        'clean_accuracy': results_clean[clf]['best_score'] * 100,
        'improvement': (results_clean[clf]['best_score'] - results_all[clf]['best_score']) * 100
    })

df_comp = pd.DataFrame(comparison)
print(df_comp.to_string(index=False))

print("\n✅ Mejora máxima: {:.1f}%".format(df_comp['improvement'].max()))
df_comp.to_csv('analysis/enfoque_mejorado/mejora_unimodal.csv', index=False)
