"""
Extracción de características CON CALIFICACIONES INDIVIDUALES POR EXAMEN
EDA corregida x10 según documentación Empatica E4
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from utils import DATA_PATH, PARTICIPANTS, load_signal, load_ibi, load_grades

def trapz(y, dx=1.0):
    return np.sum((y[1:] + y[:-1]) * dx / 2)

def preprocess_eda(eda_raw, fs=4.0):
    """Preprocesamiento de EDA con corrección de escala x10"""
    eda_corrected = eda_raw * 10.0
    eda_median = median_filter(eda_corrected, size=5)
    b, a = signal.butter(4, 1.0, btype='low', fs=fs)
    eda_filtered = signal.filtfilt(b, a, eda_median)
    return eda_filtered

def detect_robust_scrs(eda, fs=4.0):
    """Detección robusta de SCRs"""
    deriv = np.abs(np.diff(eda))
    deriv_smooth = uniform_filter1d(deriv, size=int(fs))
    threshold = np.mean(deriv_smooth) + 2 * np.std(deriv_smooth)
    min_distance = int(fs * 2)
    
    scr_onsets = []
    i = 0
    while i < len(deriv_smooth):
        if deriv_smooth[i] > threshold:
            scr_onsets.append(i)
            i += min_distance
        else:
            i += 1
    
    scr_amplitudes = []
    for onset in scr_onsets:
        peak_window = slice(onset, min(onset + int(fs*3), len(eda)))
        if len(eda[peak_window]) > 0:
            peak_value = np.max(eda[peak_window])
            baseline = eda[max(0, onset - int(fs))] if onset > int(fs) else eda[0]
            amplitude = peak_value - baseline
            scr_amplitudes.append(max(0, amplitude))
    
    return scr_onsets, scr_amplitudes

def extract_eda_features_corrected(eda_df):
    """Extraer características de EDA con corrección"""
    if eda_df is None or len(eda_df) < 100:
        return None
    
    eda_raw = eda_df['eda'].values
    fs = 4.0
    dt = 1.0 / fs
    eda_clean = preprocess_eda(eda_raw, fs)
    
    features = {
        'mean': np.mean(eda_clean),
        'std': np.std(eda_clean),
        'min': np.min(eda_clean),
        'max': np.max(eda_clean),
        'range': np.max(eda_clean) - np.min(eda_clean),
        'mean_deriv': np.mean(np.abs(np.diff(eda_clean) / dt)),
        'std_deriv': np.std(np.diff(eda_clean) / dt),
        'slope': (eda_clean[-1] - eda_clean[0]) / (len(eda_clean) * dt),
    }
    
    scr_onsets, scr_amplitudes = detect_robust_scrs(eda_clean, fs)
    features['num_scr'] = len(scr_onsets)
    features['scr_density'] = len(scr_onsets) / (len(eda_clean) / fs / 60)
    
    if len(scr_amplitudes) > 0:
        features['scr_mean_amplitude'] = np.mean(scr_amplitudes)
    else:
        features['scr_mean_amplitude'] = 0
    
    window = int(fs * 30)
    if len(eda_clean) > window:
        tonic = uniform_filter1d(eda_clean, size=window, mode='nearest')
        features['tonic_mean'] = np.mean(tonic)
    else:
        features['tonic_mean'] = np.mean(eda_clean)
    
    features['auc'] = trapz(eda_clean, dt)
    return features

def extract_hrv_features(ibi_df):
    """Extraer características de HRV"""
    if ibi_df is None or len(ibi_df) < 10:
        return None
    
    if 'ibi' in ibi_df.columns:
        ibi_sec = ibi_df['ibi'].values
    else:
        ibi_sec = ibi_df.iloc[:, 1].values
    
    ibi_sec = ibi_sec[(ibi_sec > 0.3) & (ibi_sec < 2.0)]
    if len(ibi_sec) < 10:
        return None
    
    features = {
        'mean_ibi': np.mean(ibi_sec),
        'std_ibi': np.std(ibi_sec),
        'mean_hr': 60.0 / np.mean(ibi_sec),
        'rmssd': np.sqrt(np.mean(np.diff(ibi_sec)**2)) * 1000,
        'sdnn': np.std(ibi_sec) * 1000,
        'pnn50': np.mean(np.abs(np.diff(ibi_sec)) > 0.050) * 100,
    }
    return features

def extract_temp_features(temp_df):
    """Extraer características de temperatura"""
    if temp_df is None or len(temp_df) < 100:
        return None
    
    temp = temp_df['temp'].values
    fs = 4.0
    dt = 1.0 / fs
    n = len(temp)
    
    features = {
        'mean': np.mean(temp),
        'std': np.std(temp),
        'min': np.min(temp),
        'max': np.max(temp),
        'slope': (temp[-1] - temp[0]) / (n * dt),
    }
    
    window = min(int(fs * 60 * 5), n // 4)
    if window > 0:
        features['temp_drop'] = np.mean(temp[:window]) - np.mean(temp[-window:])
    else:
        features['temp_drop'] = 0
    
    return features

def extract_all_features():
    """Extraer todas las características con calificaciones INDIVIDUALES"""
    # Cargar calificaciones individuales por examen
    grades_raw = load_grades()
    
    exam_types = ['midterm_1', 'midterm_2', 'Final']
    all_data = []
    
    print("=== EXTRACCIÓN CON CALIFICACIONES INDIVIDUALES ===")
    print("EDA corregida x10 (rango 0.01-100 μS según Empatica)")
    print("-" * 50)
    
    for participant in PARTICIPANTS:
        for exam in exam_types:
            print(f"  {participant} - {exam}...")
            
            # Obtener calificación INDIVIDUAL de este examen
            if participant not in grades_raw:
                print(f"     {participant} no tiene calificaciones")
                continue
            
            if exam not in grades_raw[participant]:
                print(f"     {participant} - {exam} sin calificación")
                continue
            
            grade_raw = grades_raw[participant][exam]
            
            # Convertir Final (sobre 200) a escala sobre 100
            if exam == 'Final':
                grade = grade_raw / 2.0
            else:
                grade = grade_raw
            
            # Cargar señales
            eda_df = load_signal(participant, exam, 'EDA')
            ibi_df = load_ibi(participant, exam)
            temp_df = load_signal(participant, exam, 'TEMP')
            
            eda_feat = extract_eda_features_corrected(eda_df) if eda_df is not None else None
            hrv_feat = extract_hrv_features(ibi_df)
            temp_feat = extract_temp_features(temp_df)
            
            row = {
                'participant': participant,
                'exam': exam,
                'grade': grade,  # ¡AHORA ES INDIVIDUAL!
            }
            
            if eda_feat:
                for k, v in eda_feat.items():
                    row[f'eda_{k}'] = v
            if hrv_feat:
                for k, v in hrv_feat.items():
                    row[f'hrv_{k}'] = v
            if temp_feat:
                for k, v in temp_feat.items():
                    row[f'temp_{k}'] = v
            
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    df = extract_all_features()
    print(f"\n Extraídas {len(df)} muestras")
    
    print("\n VERIFICACIÓN DE CALIFICACIONES INDIVIDUALES:")
    for p in ['S1', 'S3', 'S7']:
        for exam in ['midterm_1', 'midterm_2', 'Final']:
            row = df[(df['participant'] == p) & (df['exam'] == exam)]
            if len(row) > 0:
                print(f"   {p} - {exam}: grade={row['grade'].values[0]:.1f}")
    
    output_path = Path(__file__).parent.parent / 'results' / 'features_individual_grades.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n Guardado en {output_path}")
