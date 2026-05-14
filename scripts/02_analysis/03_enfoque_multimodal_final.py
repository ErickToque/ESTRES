"""
ENFOQUE MULTIMODAL FINAL: Combinación de EDA, HRV y TEMP
Forzando longitud fija de features
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

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

def extract_eda_features(trendline, fs=4.0, tw_min=5):
    """Devuelve 9 features por ventana"""
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

def extract_hrv_features_fixed(ibi_df):
    """Devuelve 4 features HRV (o ceros si no hay datos)"""
    if ibi_df is None:
        return np.zeros(4)
    
    if 'ibi' in ibi_df.columns:
        ibi = ibi_df['ibi'].values
    else:
        ibi = ibi_df.iloc[:, 1].values
    
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    if len(ibi) < 10:
        return np.zeros(4)
    
    hr_mean = 60.0 / np.mean(ibi)
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2)) * 1000
    sdnn = np.std(ibi) * 1000
    pnn50 = np.mean(np.abs(np.diff(ibi)) > 0.050) * 100
    
    return np.array([hr_mean, rmssd, sdnn, pnn50])

def extract_temp_features_fixed(temp_df):
    """Devuelve 3 features de temperatura (o ceros si no hay datos)"""
    if temp_df is None:
        return np.zeros(3)
    
    temp = temp_df['temp'].values
    return np.array([np.mean(temp), np.std(temp), temp[-1] - temp[0]])

def load_data_multimodal_fixed():
    """Carga datos con features de longitud fija"""
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
    
    for p in participants:
        for exam in exams:
            # Calificación
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            # EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                continue
            
            eda = eda_df['eda'].values * 10
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            trendline = lowpass_filter_trendline(eda, 4.0, cutoff)
            
            # Features EDA (9 features por Tw * 3 Tw = 27 features)
            eda_features = []
            for tw in TW_VALUES:
                feat = extract_eda_features(trendline, 4.0, tw)
                if feat is not None:
                    eda_features.extend(feat)
                else:
                    eda_features.extend([0] * 9)  # padding
            
            # HRV (4 features)
            ibi_df = load_ibi(p, exam)
            hrv_features = extract_hrv_features_fixed(ibi_df)
            
            # TEMP (3 features)
            temp_df = load_signal(p, exam, 'TEMP')
            temp_features = extract_temp_features_fixed(temp_df)
            
            # Combinar (27 + 4 + 3 = 34 features fijas)
            all_features = np.array(eda_features + list(hrv_features) + list(temp_features))
            
            all_data.append({
                'participant': p,
                'exam': exam,
                'grade': grade,
                'grade_class': grade_class,
                'features': all_features
            })
    
    return all_data

print("=" * 70)
print("HALLAZGO 3: ENFOQUE MULTIMODAL FINAL")
print("=" * 70)

print("\n1. Cargando datos multimodales (features fijas)...")
data = load_data_multimodal_fixed()
print(f"   Muestras totales: {len(data)}")
if len(data) > 0:
    print(f"   Features por muestra: {len(data[0]['features'])}")

# Separar por tipo de modelo
print("\n2. Preparando datasets...")

# EDA-only (primeras 27 features)
X_eda = np.array([d['features'][:27] for d in data])
# Multimodal (todas las 34 features)
X_multi = np.array([d['features'] for d in data])

y = np.array([d['grade_class'] for d in data])

print(f"   EDA-only: {X_eda.shape}")
print(f"   Multimodal: {X_multi.shape}")
print(f"   Target: {y.shape}")

def evaluate_model(X, y, model_name):
    """Evalúa modelo con validación cruzada"""
    # Eliminar features constantes
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"\n{model_name}:")
    print(f"   Accuracy: {grid_search.best_score_:.2%}")
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Features tras filtro: {X.shape[1]}")
    
    return grid_search.best_score_

print("\n3. Evaluando modelos...")
acc_eda = evaluate_model(X_eda, y, "EDA-only")
acc_multi = evaluate_model(X_multi, y, "MULTIMODAL (EDA+HRV+TEMP)")

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)
print(f"EDA-only:                 {acc_eda:.2%}")
print(f"Multimodal:               {acc_multi:.2%}")
improvement = (acc_multi - acc_eda) * 100
print(f"Diferencia:               {improvement:+.1f}%")

if improvement > 0:
    print(f"\n✅ El enfoque MULTIMODAL mejora en {improvement:.1f}% al EDA-only")
else:
    print(f"\n⚠️ El enfoque MULTIMODAL no mejora significativamente")

# Guardar resultados
results_df = pd.DataFrame([
    {'model': 'EDA-only (Amin et al. baseline)', 'accuracy': acc_eda * 100},
    {'model': 'Multimodal (EDA+HRV+TEMP)', 'accuracy': acc_multi * 100}
])
results_df.to_csv('analysis/enfoque_mejorado/resultados_finales.csv', index=False)
print("\n✅ Guardado: analysis/enfoque_mejorado/resultados_finales.csv")
