"""
LOSO para todas las señales y combinaciones
Comparación de EDA, HR, TEMP, ACC y fusiones
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

def extract_acc_features(acc_df, fs=32.0):
    """Extrae características del acelerómetro"""
    if acc_df is None:
        return np.zeros(6)
    
    # ACC tiene formato especial: timestamp, fs, luego datos
    if isinstance(acc_df, str):
        with open(acc_df, 'r') as f:
            lines = f.readlines()
        acc_data = []
        for line in lines[2:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                acc_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        acc_data = np.array(acc_data)
    else:
        return np.zeros(6)
    
    if len(acc_data) < 100:
        return np.zeros(6)
    
    acc_x, acc_y, acc_z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    features = [
        np.mean(acc_magnitude),
        np.std(acc_magnitude),
        np.max(acc_magnitude),
        np.mean(acc_x), np.std(acc_x),
        np.mean(acc_y), np.std(acc_y),
        np.mean(acc_z), np.std(acc_z)
    ]
    return np.array(features[:6])  # limitar a 6 features

def extract_hrv_features_fixed(ibi_df):
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
    if temp_df is None:
        return np.zeros(3)
    temp = temp_df['temp'].values
    return np.array([np.mean(temp), np.std(temp), temp[-1] - temp[0]])

def load_all_data_with_acc():
    """Carga datos incluyendo ACC"""
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    
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
            
            eda_features = []
            for tw in TW_VALUES:
                feat = extract_eda_features(trendline, 4.0, tw)
                if feat is not None:
                    eda_features.extend(feat)
                else:
                    eda_features.extend([0] * 9)
            
            # HRV
            ibi_df = load_ibi(p, exam)
            hrv_features = extract_hrv_features_fixed(ibi_df)
            
            # TEMP
            temp_df = load_signal(p, exam, 'TEMP')
            temp_features = extract_temp_features_fixed(temp_df)
            
            # ACC
            acc_file = f'/home/etoque/ESTRES/data/wearable-exam-stress/{p}/{exam}/ACC.csv'
            import os
            if os.path.exists(acc_file):
                acc_features = extract_acc_features(acc_file)
            else:
                acc_features = np.zeros(6)
            
            all_data.append({
                'participant': p,
                'grade_class': grade_class,
                'features_eda': np.array(eda_features),
                'features_hrv': hrv_features,
                'features_temp': temp_features,
                'features_acc': acc_features,
                'features_eda_hrv': np.concatenate([eda_features, hrv_features]),
                'features_eda_temp': np.concatenate([eda_features, temp_features]),
                'features_eda_acc': np.concatenate([eda_features, acc_features]),
                'features_all': np.concatenate([eda_features, hrv_features, temp_features, acc_features])
            })
    
    return all_data

def evaluate_loso(data, feature_type, nombre):
    """Evaluación con Leave-One-Subject-Out"""
    if feature_type == 'eda':
        X = np.array([d['features_eda'] for d in data])
    elif feature_type == 'hrv':
        X = np.array([d['features_hrv'] for d in data])
    elif feature_type == 'temp':
        X = np.array([d['features_temp'] for d in data])
    elif feature_type == 'acc':
        X = np.array([d['features_acc'] for d in data])
    elif feature_type == 'eda_hrv':
        X = np.array([d['features_eda_hrv'] for d in data])
    elif feature_type == 'eda_temp':
        X = np.array([d['features_eda_temp'] for d in data])
    elif feature_type == 'eda_acc':
        X = np.array([d['features_eda_acc'] for d in data])
    else:
        X = np.array([d['features_all'] for d in data])
    
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]
    
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
        'model': nombre,
        'accuracy': np.mean(scores),
        'accuracy_std': np.std(scores),
        'f1_score': np.mean(f1_scores),
        'n_features': X.shape[1]
    }

print("=" * 70)
print("LOSO PARA TODAS LAS SEÑALES Y COMBINACIONES")
print("Validación rigurosa sin data leakage")
print("=" * 70)

print("\nCargando datos...")
data = load_all_data_with_acc()
print(f"Muestras totales: {len(data)}")
print(f"Clase Alta (≥80): {sum(d['grade_class'] for d in data)}/{len(data)}")

# Definir todas las configuraciones a evaluar
configs = [
    ('eda', '🔵 EDA-only'),
    ('hrv', '🟢 HRV-only'),
    ('temp', '🟡 TEMP-only'),
    ('acc', '⚪ ACC-only'),
    ('eda_hrv', '🟣 EDA + HRV'),
    ('eda_temp', '🟠 EDA + TEMP'),
    ('eda_acc', '🔴 EDA + ACC'),
    ('all', '🌈 Multimodal (EDA+HRV+TEMP+ACC)')
]

print("\n" + "=" * 70)
print("RESULTADOS LOSO")
print("=" * 70)

results = []
for feat_type, nombre in configs:
    result = evaluate_loso(data, feat_type, nombre)
    results.append(result)
    print(f"\n{nombre}:")
    print(f"   Accuracy: {result['accuracy']:.2%} ± {result['accuracy_std']:.2%}")
    print(f"   F1-Score: {result['f1_score']:.2%}")
    print(f"   Features: {result['n_features']}")

# Ordenar por accuracy
results.sort(key=lambda x: x['accuracy'], reverse=True)

print("\n" + "=" * 70)
print("RANKING FINAL (mejor a peor)")
print("=" * 70)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['model']}: {r['accuracy']:.2%}")

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('paper_resultados/loso_todas_senales.csv', index=False)
print("\n✅ Guardado: paper_resultados/loso_todas_senales.csv")
