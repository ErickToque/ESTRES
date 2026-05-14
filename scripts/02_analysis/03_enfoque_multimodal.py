"""
ENFOQUE MULTIMODAL: Combinación de EDA, HRV y TEMP
Comparación con unimodal (solo EDA)
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Configuración (igual que antes)
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

def extract_hrv_features_from_ibi(ibi_df):
    """Extrae features de HRV (frecuencia cardíaca media, RMSSD, etc.)"""
    if ibi_df is None:
        return None
    
    if 'ibi' in ibi_df.columns:
        ibi = ibi_df['ibi'].values
    else:
        ibi = ibi_df.iloc[:, 1].values
    
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    if len(ibi) < 10:
        return None
    
    hr_mean = 60.0 / np.mean(ibi)
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2)) * 1000
    sdnn = np.std(ibi) * 1000
    pnn50 = np.mean(np.abs(np.diff(ibi)) > 0.050) * 100
    
    return np.array([hr_mean, rmssd, sdnn, pnn50])

def extract_temp_features(temp_df):
    """Extrae features de temperatura"""
    if temp_df is None:
        return None
    
    temp = temp_df['temp'].values
    return np.array([np.mean(temp), np.std(temp), temp[-1] - temp[0]])

def load_multimodal_data(quality_threshold=20):
    """Carga datos multimodales (EDA + HRV + TEMP)"""
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
            # EDA (con filtro de calidad)
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is None:
                continue
            
            eda = eda_df['eda'].values
            zeros_pct = (eda == 0).sum() / len(eda) * 100
            if zeros_pct > quality_threshold:
                continue
            
            # Calificación
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            # Features EDA
            eda_corrected = eda * 10
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            trendline = lowpass_filter_trendline(eda_corrected, 4.0, cutoff)
            
            eda_features = []
            for tw in TW_VALUES:
                feat = extract_eda_features(trendline, 4.0, tw)
                if feat is not None:
                    eda_features.extend(feat)
            
            # Features HRV
            ibi_df = load_ibi(p, exam)
            hrv_features = extract_hrv_features_from_ibi(ibi_df)
            
            # Features TEMP
            temp_df = load_signal(p, exam, 'TEMP')
            temp_features = extract_temp_features(temp_df)
            
            # Combinar features
            multimodal_features = []
            multimodal_features.extend(eda_features)
            if hrv_features is not None:
                multimodal_features.extend(hrv_features)
            if temp_features is not None:
                multimodal_features.extend(temp_features)
            
            all_data.append({
                'participant': p, 'exam': exam, 'grade': grade,
                'grade_class': grade_class,
                'features_eda': np.array(eda_features),
                'features_multimodal': np.array(multimodal_features)
            })
    
    return all_data

def evaluate_model(data, feature_type='eda'):
    """Evalúa modelo con validación cruzada"""
    if feature_type == 'eda':
        X = np.array([d['features_eda'] for d in data])
    else:
        X = np.array([d['features_multimodal'] for d in data])
    
    y = np.array([d['grade_class'] for d in data])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    
    # Optimización de Random Forest
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    return {
        'accuracy': grid_search.best_score_,
        'best_params': grid_search.best_params_
    }

print("=" * 70)
print("HALLAZGO 3: ENFOQUE MULTIMODAL")
print("Comparación EDA-only vs Multimodal (EDA+HRV+TEMP)")
print("=" * 70)

print("\n1. Cargando datos con filtro de calidad (<20% ceros)...")
data = load_multimodal_data(quality_threshold=20)
print(f"   Muestras totales: {len(data)}")

print("\n2. Evaluando modelo EDA-only:")
result_eda = evaluate_model(data, 'eda')
print(f"   Accuracy: {result_eda['accuracy']:.2%}")
print(f"   Best params: {result_eda['best_params']}")

print("\n3. Evaluando modelo MULTIMODAL (EDA+HRV+TEMP):")
result_multimodal = evaluate_model(data, 'multimodal')
print(f"   Accuracy: {result_multimodal['accuracy']:.2%}")
print(f"   Best params: {result_multimodal['best_params']}")

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)
improvement = (result_multimodal['accuracy'] - result_eda['accuracy']) * 100
print(f"EDA-only:      {result_eda['accuracy']:.2%}")
print(f"Multimodal:    {result_multimodal['accuracy']:.2%}")
print(f"Mejora:        +{improvement:.1f}%")

# Guardar resultados
results_df = pd.DataFrame([
    {'model': 'EDA-only', 'accuracy': result_eda['accuracy']},
    {'model': 'Multimodal (EDA+HRV+TEMP)', 'accuracy': result_multimodal['accuracy']}
])
results_df.to_csv('analysis/enfoque_mejorado/resultados_multimodal.csv', index=False)
print("\n✅ Guardado: analysis/enfoque_mejorado/resultados_multimodal.csv")
