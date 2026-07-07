"""
ABLATION STUDY - Basado en la misma lógica del script de ventanas
- Mismo filtro: cutoff=0.0002 Hz
- Misma validación: LOSO
- Misma ventana: Tw=1 minuto (óptima)
- Mismo RandomForest (random_state=42, max_depth=5, n_estimators=100)
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

from scripts.utils import load_signal, load_ibi

# ============================================================================
# CONFIGURACIÓN (idéntica al script de ventanas)
# ============================================================================
CUTOFF = 0.0002  # Hz
TW = 1  # minuto óptimo
SEED = 42

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    nyquist = fs / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_eda_features(trendline, fs=4.0, tw_min=1):
    """Extrae 9 features de EDA (mismo que en script de ventanas)"""
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

def extract_hrv_features(ibi_df):
    """Extrae características de HRV"""
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

def extract_temp_features(temp_signal):
    """Extrae características de temperatura (load_signal devuelve dict con 'data')"""
    if temp_signal is None:
        return np.zeros(3)
    
    temp = temp_signal['data']
    return np.array([np.mean(temp), np.std(temp), temp[-1] - temp[0]])

def extract_acc_features(acc_file_path):
    """Extrae características de acelerómetro"""
    try:
        with open(acc_file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            return np.zeros(6)
        
        # Leer datos desde línea 2
        acc_data = []
        for line in lines[2:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    acc_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    pass
        
        if len(acc_data) < 100:
            return np.zeros(6)
        
        acc_data = np.array(acc_data)
        acc_x, acc_y, acc_z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        return np.array([np.mean(acc_magnitude), np.std(acc_magnitude),
                        np.max(acc_magnitude), np.mean(acc_x), np.std(acc_x), np.mean(acc_y)])
    except:
        return np.zeros(6)

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

def load_all_data():
    """Carga todos los datos con todas las modalidades"""
    grades_raw = load_grades_individual()
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    all_data = []
    
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            grade_class = 1 if grade >= 80 else 0
            
            # ========== EDA ==========
            eda_signal = load_signal(p, exam, 'EDA')
            eda_features = None
            if eda_signal is not None:
                eda = eda_signal['data'] * 10
                trendline = lowpass_filter_trendline(eda, 4.0, CUTOFF)
                eda_features = extract_eda_features(trendline, 4.0, TW)
            
            # ========== HRV ==========
            ibi_df = load_ibi(p, exam)
            hrv_features = extract_hrv_features(ibi_df)
            
            # ========== TEMP ==========
            temp_signal = load_signal(p, exam, 'TEMP')
            temp_features = extract_temp_features(temp_signal)
            
            # ========== ACC ==========
            acc_path = f'data/wearable-exam-stress/{p}/{exam}/ACC.csv'
            acc_features = extract_acc_features(acc_path)
            
            if eda_features is not None:
                all_data.append({
                    'participant': p,
                    'grade_class': grade_class,
                    'features_eda': eda_features,
                    'features_hrv': hrv_features,
                    'features_temp': temp_features,
                    'features_acc': acc_features
                })
    
    return all_data

def evaluate_model(data, feature_keys, model_name):
    """Evalúa modelo con LOSO"""
    if len(data) < 3:
        return None, None, None
    
    X_list = []
    for d in data:
        feats = np.concatenate([d[k] for k in feature_keys])
        X_list.append(feats)
    
    X = np.array(X_list)
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]
    
    # Eliminar features constantes
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logo = LeaveOneGroupOut()
    scores = []
    f1_scores = []
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
        rf.fit(X_scaled[train_idx], y[train_idx])
        y_pred = rf.predict(X_scaled[test_idx])
        scores.append(accuracy_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred, zero_division=0))
    
    return np.mean(scores)*100, np.std(scores)*100, np.mean(f1_scores)

print("=" * 80)
print("ABLATION STUDY (misma metodología que ventanas)")
print(f"Cut-off: {CUTOFF} Hz | Ventana: {TW} minuto | Validación: LOSO")
print("=" * 80)

# Cargar datos
data = load_all_data()
print(f"\nMuestras totales: {len(data)}")
print(f"Clase Alta (≥80): {sum(d['grade_class'] for d in data)}/{len(data)}")
print()

# Definir configuraciones
configs = [
    (['features_eda'], "EDA-only"),
    (['features_hrv'], "HRV-only"),
    (['features_temp'], "TEMP-only"),
    (['features_acc'], "ACC-only"),
    (['features_eda', 'features_acc'], "EDA + ACC"),
    (['features_eda', 'features_hrv', 'features_temp', 'features_acc'], "Multimodal")
]

results = []

for feature_keys, model_name in configs:
    acc, std, f1 = evaluate_model(data, feature_keys, model_name)
    if acc is not None:
        results.append({
            'model': model_name,
            'accuracy': acc,
            'std': std,
            'f1': f1
        })
        print(f"{model_name}:")
        print(f"  Accuracy: {acc:.1f}% ± {std:.1f}%")
        print(f"  F1-Score: {f1:.2f}")
        print()

print("=" * 80)
print("TABLA LATEX PARA EL PAPER")
print("=" * 80)
print("\\begin{table}[htbp]")
print("\\caption{Ablation study results using leave-one-subject-out validation with a one-minute window}")
print("\\centering")
print("\\footnotesize")
print("\\setlength{\\tabcolsep}{3pt}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1-Score} \\\\")
print("\\midrule")
for r in results:
    print(f"{r['model']} & {r['accuracy']:.1f}\\% & {r['f1']:.2f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:ablation}")
print("\\end{table}")

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('ablation_results_tw1.csv', index=False)
print("\n Resultados guardados en ablation_results_tw1.csv")
