"""
Utilidades comunes para el análisis
"""
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

DATA_PATH = Path('/home/etoque/ESTRES/data/wearable-exam-stress')


def load_signal(participant, exam, signal_name):
    """
    Carga señales en formato especial del dataset
    """
    file_path = DATA_PATH / participant / exam / f'{signal_name}.csv'
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    fs = float(lines[1].strip())
    data = np.array([float(l.strip()) for l in lines[2:]])

    return {'data': data, 'fs': fs}


def load_ibi(participant, exam):
    """
    Carga IBI (formato diferente)
    """
    file_path = DATA_PATH / participant / exam / 'IBI.csv'
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    if 'ibi' in df.columns:
        return df
    elif len(df.columns) >= 2:
        df.columns = ['timestamp', 'ibi']
        return df

    return None


def load_grades_individual():
    """Carga calificaciones individuales"""
    grades_raw = {}
    with open(DATA_PATH / 'StudentGrades.txt', 'r', encoding='latin1') as f:
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
                except ValueError:
                    pass
    return grades_raw


def get_student_average_grades():
    """Calcula el promedio de calificaciones por estudiante"""
    grades_raw = load_grades_individual()
    averages = {}
    for student, exams in grades_raw.items():
        grades = []
        for exam, grade in exams.items():
            if exam == 'Final':
                grade = grade / 2
            grades.append(grade)
        if grades:
            averages[student] = np.mean(grades)
    return averages


def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.0002):
    """Filtro lowpass FIR para extraer trendline"""
    nyquist = fs / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)


def extract_window_features(trendline, fs=4.0, tw_min=1):
    """Extrae 9 features de la trendline para una ventana específica"""
    tw_samples = int(tw_min * 60 * fs)

    if tw_samples * 3 > len(trendline):
        return None

    start = trendline[:tw_samples]
    mid_start = len(trendline) // 2 - tw_samples // 2
    mid = trendline[mid_start:mid_start + tw_samples]
    end = trendline[-tw_samples:]

    mu_start = np.mean(start)
    mu_mid = np.mean(mid)
    mu_end = np.mean(end)

    var_start = np.var(start)
    var_mid = np.var(mid)
    var_end = np.var(end)

    rho = mu_mid / (mu_start + mu_end + 1e-6)

    diff = np.diff(start)
    mu_diff = np.mean(diff)
    var_diff = np.var(diff)

    return np.array([mu_start, mu_mid, mu_end, var_start, var_mid, var_end,
                     rho, mu_diff, var_diff])


def load_features_by_window(tw):
    """Carga features para una ventana específica"""
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
            trendline = lowpass_filter_trendline(eda, 4.0, 0.0002)
            features = extract_window_features(trendline, 4.0, tw)

            if features is not None:
                all_data.append({
                    'participant': p,
                    'grade_class': grade_class,
                    'features': features
                })

    return all_data


def load_features_optimal():
    """Carga features con configuración óptima (Tw=1 min)"""
    return load_features_by_window(1)


def load_features_amin():
    """Carga features según método de Amin et al. (2022) (Tw=15 min, cutoff variable)"""
    # Implementación simplificada usando Tw=15
    return load_features_by_window(15)


if __name__ == "__main__":
    print("Testing utils...")
    grades = load_grades_individual()
    print(f"Loaded grades for {len(grades)} participants")

    averages = get_student_average_grades()
    print(f"Averages: {averages}")

    features = load_features_by_window(1)
    print(f"Loaded {len(features)} samples with Tw=1 min")