"""
Verificar resultados de clasificación por examen CORRECTAMENTE
Filtrando los datos por examen específico
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

from scripts.utils import load_signal, load_grades_individual, lowpass_filter_trendline, extract_window_features

def load_data_for_exam(exam, tw=1):
    """Carga datos específicos para un examen y ventana"""
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
        trendline = lowpass_filter_trendline(eda, 4.0, 0.0002)
        features = extract_window_features(trendline, 4.0, tw)
        
        if features is not None:
            all_data.append({
                'participant': p,
                'grade_class': grade_class,
                'features': features,
                'grade': grade
            })
    
    return all_data

print("=" * 70)
print("VERIFICACIÓN DE RESULTADOS POR EXAMEN (CORREGIDO)")
print("=" * 70)

exams = ['midterm_1', 'midterm_2', 'Final']
exam_labels = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}
windows = [1, 5]

results = []

for exam in exams:
    print(f"\n📚 {exam_labels[exam]}")
    print("-" * 40)
    
    # Mostrar distribución de clases
    for tw in windows:
        data = load_data_for_exam(exam, tw)
        if len(data) > 0:
            classes = [d['grade_class'] for d in data]
            n_high = sum(classes)
            n_low = len(classes) - n_high
            print(f"  Clases (Tw={tw}): Alta={n_high}, Baja={n_low}, Total={len(data)}")
    
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
        
        for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_scaled[train_idx], y[train_idx])
            y_pred = rf.predict(X_scaled[test_idx])
            scores.append(accuracy_score(y[test_idx], y_pred))
        
        mean_acc = np.mean(scores) * 100
        std_acc = np.std(scores) * 100
        print(f"  Tw={tw}min: {mean_acc:.1f}% ± {std_acc:.1f}% (n={len(data)})")
        
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
print("TABLA PARA LATEX (ACTUALIZADA)")
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
for exam in exam_labels.values():
    exam_results = [r for r in results if r['exam'] == exam]
    if exam_results:
        best = max(exam_results, key=lambda x: x['accuracy'])
        print(f"{exam} & {best['tw']} minute{'s' if best['tw']>1 else ''} & {best['accuracy']:.1f}\\% & {best['n_samples']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:by_exam}")
print("\\end{table}")
