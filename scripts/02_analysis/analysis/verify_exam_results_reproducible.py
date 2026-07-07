"""
Verificar resultados de clasificación por examen
Con semilla fija para reproducibilidad
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import signal
from pathlib import Path
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_grades_individual

# Fijar semillas para reproducibilidad
np.random.seed(42)

DATA_PATH = Path('/home/etoque/ESTRES/data/wearable-exam-stress')


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

    mu_s = np.mean(start)
    mu_m = np.mean(mid)
    mu_e = np.mean(end)
    var_s = np.var(start)
    var_m = np.var(mid)
    var_e = np.var(end)
    rho = mu_m / (mu_s + mu_e + 1e-6)
    diff = np.diff(start)
    mu_diff = np.mean(diff)
    var_diff = np.var(diff)

    return np.array([mu_s, mu_m, mu_e, var_s, var_m, var_e, rho, mu_diff, var_diff])


def load_data_for_exam(exam, tw=1):
    grades_raw = load_grades_individual()
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    all_data = []

    for p in participants:
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
        trendline = lowpass_filter(eda, 4.0, 0.0002)
        features = extract_features(trendline, 4.0, tw)

        if features is not None:
            all_data.append({
                'participant': p,
                'grade_class': grade_class,
                'features': features
            })

    return all_data


print("=" * 70)
print("VERIFICACIÓN DE RESULTADOS POR EXAMEN (SEMILLA FIJA)")
print("RandomForest con random_state=42")
print("=" * 70)

exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}
windows = [1, 5]

results = []

for exam in exams:
    print(f"\n {exam_labels[exam]}")
    print("-" * 40)

    for tw in windows:
        data = load_data_for_exam(exam, tw)

        if len(data) < 3:
            print(f"  Tw={tw}min: datos insuficientes ({len(data)} muestras)")
            continue

        X = np.array([d['features'] for d in data])
        y = np.array([d['grade_class'] for d in data])
        groups = [d['participant'] for d in data]

        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LOSO
        logo = LeaveOneGroupOut()
        scores = []
        fold_details = []

        for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_scaled[train_idx], y[train_idx])
            y_pred = rf.predict(X_scaled[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            scores.append(acc)
            fold_details.append((groups[test_idx[0]], acc))

        mean_acc = np.mean(scores) * 100
        std_acc = np.std(scores) * 100
        print(f"  Tw={tw}min: {mean_acc:.1f}% ± {std_acc:.1f}% (n={len(data)})")
        print(f"    Detalle por fold: {fold_details}")

        results.append({
            'exam': exam_labels[exam],
            'tw': tw,
            'accuracy': mean_acc,
            'std': std_acc,
            'n_samples': len(data)
        })

print("\n" + "=" * 70)
print("RESUMEN FINAL - MEJOR POR EXAMEN")
print("=" * 70)

for exam in exam_labels.values():
    exam_results = [r for r in results if r['exam'] == exam]
    if exam_results:
        best = max(exam_results, key=lambda x: x['accuracy'])
        print(f"{exam}: Tw={best['tw']}min → {best['accuracy']:.1f}% (n={best['n_samples']})")

print("\n" + "=" * 70)
print("TABLA PARA LATEX (ACTUALIZADA CON RESULTADOS REPRODUCIBLES)")
print("=" * 70)
print("\\begin{table}[htbp]")
print("\\caption{Best performance by examination using leave-one-subject-out validation}")
print("\\centering")
print("\\footnotesize")
print("\\setlength{\\tabcolsep}{3pt}")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("\\textbf{Examination} & \\textbf{Best Window} & \\textbf{Accuracy} & \\textbf{N samples} \\\\")
print("\\midrule")
for exam in ['Midterm 1', 'Midterm 2', 'Final']:
    exam_results = [r for r in results if r['exam'] == exam]
    if exam_results:
        best = max(exam_results, key=lambda x: x['accuracy'])
        print(f"{exam} & {best['tw']} minute{'s' if best['tw']>1 else ''} & {best['accuracy']:.1f}\\% & {best['n_samples']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:by_exam}")
print("\\end{table}")
