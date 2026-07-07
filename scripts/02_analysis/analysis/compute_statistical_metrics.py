"""
Cálculo de métricas estadísticas robustas para el paper
- Balanced Accuracy
- Bootstrap Confidence Intervals
- Permutation Test
- Correlation Confidence Intervals (Fisher Z)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils import resample
from scipy.stats import norm, pearsonr
from scipy import signal
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

sys.path.append('.')
from scripts.utils import load_signal, load_grades_individual

DATA_PATH = Path('/home/etoque/ESTRES/data/wearable-exam-stress')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CUTOFF = 0.0002
TW = 1  # minuto óptimo

def lowpass_filter(eda, fs=4.0, cutoff=0.0002):
    nyquist = fs / 2
    norm_cutoff = min(cutoff / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, norm_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_features(trendline, fs=4.0, tw=1):
    tw_samples = int(tw * 60 * fs)
    if tw_samples * 3 > len(trendline):
        return None
    start = trendline[:tw_samples]
    mid_start = len(trendline)//2 - tw_samples//2
    mid = trendline[mid_start:mid_start + tw_samples]
    end = trendline[-tw_samples:]
    mu_s, mu_m, mu_e = np.mean(start), np.mean(mid), np.mean(end)
    var_s, var_m, var_e = np.var(start), np.var(mid), np.var(end)
    rho = mu_m / (mu_s + mu_e + 1e-6)
    diff = np.diff(start)
    mu_diff, var_diff = np.mean(diff), np.var(diff)
    return np.array([mu_s, mu_m, mu_e, var_s, var_m, var_e, rho, mu_diff, var_diff])

def load_all_data():
    """Carga todos los datos con Tw=1 minuto"""
    grades_raw = load_grades_individual()
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    all_data = []
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            grade = grades_raw[p][exam]
            if exam == 'Final':
                grade = grade / 2
            grade_class = 1 if grade >= 80 else 0
            sig = load_signal(p, exam, 'EDA')
            if sig is None:
                continue
            eda = sig['data'] * 10
            trendline = lowpass_filter(eda, 4.0, CUTOFF)
            features = extract_features(trendline, 4.0, TW)
            if features is not None:
                all_data.append({
                    'participant': p,
                    'grade_class': grade_class,
                    'features': features
                })
    return all_data

def evaluate_model_loso(data):
    """Evalúa modelo con LOSO y retorna predicciones por fold"""
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logo = LeaveOneGroupOut()
    scores = []
    balanced_scores = []
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        balanced_scores.append(balanced_accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    return np.mean(scores), np.std(scores), np.mean(balanced_scores), y_true_all, y_pred_all

# ============================================================================
# 1. Bootstrap Confidence Intervals
# ============================================================================
def bootstrap_ci(y_true, y_pred, n_bootstrap=5000, ci=95):
    """Calcula intervalos de confianza bootstrap para accuracy"""
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(y_true)), n_samples=len(y_true), replace=True)
        scores.append(accuracy_score(np.array(y_true)[idx], np.array(y_pred)[idx]))
    lower = np.percentile(scores, (100-ci)/2)
    upper = np.percentile(scores, 100 - (100-ci)/2)
    return lower, upper, np.mean(scores)

# ============================================================================
# 2. Permutation Test
# ============================================================================
def permutation_test(y_true, y_pred, n_permutations=5000):
    """Test de permutación para comparar con azar (50%)"""
    observed_acc = accuracy_score(y_true, y_pred)
    null_acc = []
    # Generar predicciones aleatorias (azar)
    for _ in range(n_permutations):
        y_shuffled = np.random.permutation(y_true)
        null_acc.append(accuracy_score(y_true, y_shuffled))  # comparar con azar
    p_value = np.mean(np.array(null_acc) >= observed_acc)
    return p_value, null_acc

# ============================================================================
# 3. Correlation Confidence Intervals (Fisher Z)
# ============================================================================
def correlation_ci(r, n, ci=0.95):
    """Intervalo de confianza para correlación usando transformación Fisher Z"""
    if abs(r) >= 0.99:
        return np.nan, np.nan  # No calcular para correlaciones perfectas
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - (1-ci)/2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)
    return ci_low, ci_high

# ============================================================================
# MAIN
# ============================================================================
print("=" * 70)
print("CÁLCULO DE MÉTRICAS ESTADÍSTICAS PARA EL PAPER")
print("=" * 70)

# Cargar datos
print("\n1. Cargando datos con Tw=1 minuto...")
data = load_all_data()
print(f"   Muestras totales: {len(data)}")
print(f"   Clase Alta (≥80): {sum(d['grade_class'] for d in data)}/{len(data)}")

# Evaluar modelo EDA-only
print("\n2. Evaluando modelo EDA-only con LOSO...")
mean_acc, std_acc, balanced_acc, y_true, y_pred = evaluate_model_loso(data)
print(f"   Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
print(f"   Balanced Accuracy: {balanced_acc:.2%}")

# 1. Bootstrap Confidence Interval
print("\n3. Bootstrap Confidence Interval (95% CI)...")
ci_lower, ci_upper, ci_mean = bootstrap_ci(y_true, y_pred, n_bootstrap=5000)
print(f"   95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
print(f"   Media bootstrap: {ci_mean:.2%}")

# 2. Permutation Test (vs chance)
print("\n4. Permutation Test (EDA-only vs chance level 50%)...")
p_value, null_dist = permutation_test(y_true, y_pred, n_permutations=5000)
print(f"   P-value: {p_value:.4f}")
if p_value < 0.05:
    print("    Estadísticamente significativo (p < 0.05)")
else:
    print("    No significativo (p > 0.05)")

# 3. Correlation Confidence Intervals
print("\n5. Correlation Confidence Intervals (Fisher Z)...")
# Datos de correlaciones intra-sujeto
correlations = [
    ('S1', 'EDA', -0.773, 3), ('S1', 'HR', -0.987, 3), ('S1', 'TEMP', -0.443, 3),
    ('S2', 'EDA', -0.996, 2), ('S3', 'HR', -1.000, 3), ('S3', 'TEMP', 0.165, 3),
    ('S4', 'EDA', 0.800, 3), ('S4', 'HR', 0.097, 3), ('S4', 'TEMP', -0.434, 3),
    ('S5', 'EDA', 0.996, 3), ('S5', 'HR', -0.879, 3), ('S5', 'TEMP', 0.879, 3),
    ('S6', 'EDA', -0.480, 3), ('S6', 'HR', -0.403, 3), ('S6', 'TEMP', -0.124, 3),
    ('S7', 'EDA', -0.970, 3), ('S7', 'HR', 0.660, 3), ('S7', 'TEMP', 0.770, 3),
    ('S8', 'EDA', 1.000, 3), ('S8', 'HR', 0.559, 3), ('S8', 'TEMP', -0.174, 3),
    ('S9', 'EDA', -0.544, 3), ('S9', 'HR', 0.689, 3), ('S9', 'TEMP', -0.235, 3),
    ('S10', 'EDA', 0.877, 3), ('S10', 'HR', 0.999, 3), ('S10', 'TEMP', -0.733, 3)
]

for subj, signal, r, n in correlations:
    if abs(r) < 0.99 and n >= 3:
        ci_low, ci_high = correlation_ci(r, n)
        print(f"   {subj} - {signal}: r={r:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")
    elif abs(r) >= 0.99:
        print(f"   {subj} - {signal}: r={r:.3f} (CI no calculable para |r|≈1)")
    else:
        print(f"   {subj} - {signal}: r={r:.3f} (n={n}, insuficiente para CI)")

# ============================================================================
# TABLA PARA LATEX
# ============================================================================
print("\n" + "=" * 70)
print("TABLA RESUMEN PARA EL PAPER")
print("=" * 70)
print("\\begin{table}[htbp]")
print("\\caption{Statistical metrics for EDA-only model (Tw=1 min, LOSO)}")
print("\\centering")
print("\\footnotesize")
print("\\setlength{\\tabcolsep}{3pt}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Metric} & \\textbf{Value} & \\textbf{95\\% CI / p-value} \\\\")
print("\\midrule")
print(f"Standard Accuracy & {mean_acc:.1%} & [{ci_lower:.1%}, {ci_upper:.1%}] \\\\")
print(f"Balanced Accuracy & {balanced_acc:.1%} & --- \\\\")
print(f"Permutation Test (vs chance) & p = {p_value:.4f} & {'Significant' if p_value < 0.05 else 'Not significant'} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:statistical_metrics}")
print("\\end{table}")

print("\n Cálculos completados. Copia los valores a tu paper.")
