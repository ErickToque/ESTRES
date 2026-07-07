"""
Extracción de características CON CORRECCIÓN DE ESCALA EDA
Factor x10 para llevar a rango fisiológico normal (Boucsein, 2012)
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
from utils import DATA_PATH, PARTICIPANTS, load_signal, load_ibi, get_student_grades

def trapz(y, dx=1.0):
    return np.sum((y[1:] + y[:-1]) * dx / 2)

def preprocess_eda(eda_raw, fs=4.0):
    """Preprocesamiento de EDA con corrección de escala x10"""
    # CORRECCIÓN DE ESCALA: multiplicar por 10 para rango fisiológico
    eda_corrected = eda_raw * 10.0
    
    # Filtro mediana para eliminar artefactos
    eda_median = median_filter(eda_corrected, size=5)
    
    # Filtro paso bajo (Butterworth, 1 Hz)
    b, a = signal.butter(4, 1.0, btype='low', fs=fs)
    eda_filtered = signal.filtfilt(b, a, eda_median)
    
    return eda_filtered

def detect_robust_scrs(eda, fs=4.0):
    """Detección robusta de SCRs"""
    deriv = np.diff(eda)
    deriv = np.abs(deriv)
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
    """Extraer características de EDA con corrección de escala"""
    if eda_df is None or len(eda_df) < 100:
        return None
    
    eda_raw = eda_df['eda'].values
    fs = 4.0
    dt = 1.0 / fs
    
    # Preprocesar (incluye corrección x10)
    eda_clean = preprocess_eda(eda_raw, fs)
    
    features = {
        'mean': np.mean(eda_clean),
        'std': np.std(eda_clean),
        'min': np.min(eda_clean),
        'max': np.max(eda_clean),
        'range': np.max(eda_clean) - np.min(eda_clean),
    }
    
    # Derivada y pendiente
    eda_deriv = np.diff(eda_clean) / dt
    features['mean_deriv'] = np.mean(np.abs(eda_deriv))
    features['std_deriv'] = np.std(eda_deriv)
    features['slope'] = (eda_clean[-1] - eda_clean[0]) / (len(eda_clean) * dt)
    
    # SCRs robustos
    scr_onsets, scr_amplitudes = detect_robust_scrs(eda_clean, fs)
    features['num_scr'] = len(scr_onsets)
    features['scr_density'] = len(scr_onsets) / (len(eda_clean) / fs / 60)
    
    if len(scr_amplitudes) > 0:
        features['scr_mean_amplitude'] = np.mean(scr_amplitudes)
        features['scr_std_amplitude'] = np.std(scr_amplitudes)
    else:
        features['scr_mean_amplitude'] = 0
        features['scr_std_amplitude'] = 0
    
    # Tónica
    window = int(fs * 30)
    if len(eda_clean) > window:
        tonic = uniform_filter1d(eda_clean, size=window, mode='nearest')
        features['tonic_mean'] = np.mean(tonic)
        features['tonic_slope'] = (tonic[-1] - tonic[0]) / (len(tonic) * dt)
    else:
        features['tonic_mean'] = np.mean(eda_clean)
        features['tonic_slope'] = features['slope']
    
    # Área total
    features['auc'] = trapz(eda_clean, dt)
    
    return features

def extract_hrv_features_advanced(ibi_df):
    """Extraer características de HRV (sin cambios)"""
    if ibi_df is None or len(ibi_df) < 10:
        return None
    
    if 'ibi' in ibi_df.columns:
        ibi_sec = ibi_df['ibi'].values
    else:
        ibi_sec = ibi_df.iloc[:, 1].values
    
    ibi_sec = ibi_sec[(ibi_sec > 0.3) & (ibi_sec < 2.0)]
    
    if len(ibi_sec) < 10:
        return None
    
    features = {}
    features['mean_ibi'] = np.mean(ibi_sec)
    features['std_ibi'] = np.std(ibi_sec)
    features['mean_hr'] = 60.0 / features['mean_ibi']
    
    diffs = np.diff(ibi_sec)
    features['rmssd'] = np.sqrt(np.mean(diffs**2)) * 1000
    features['sdnn'] = np.std(ibi_sec) * 1000
    features['pnn50'] = np.mean(np.abs(diffs) > 0.050) * 100
    
    if len(ibi_sec) >= 30:
        try:
            time_ibi = np.cumsum(ibi_sec)
            time_uniform = np.arange(0, time_ibi[-1], 0.25)
            if len(time_uniform) > 10:
                from scipy.interpolate import interp1d
                f_interp = interp1d(time_ibi, ibi_sec, kind='cubic', fill_value='extrapolate')
                ibi_interp = f_interp(time_uniform)
                ibi_detrend = signal.detrend(ibi_interp)
                nperseg = min(128, len(ibi_detrend) // 4)
                if nperseg >= 8:
                    frequencies, psd = signal.welch(ibi_detrend, fs=4.0, nperseg=nperseg)
                    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
                    hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)
                    features['lf_power'] = np.trapz(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
                    features['hf_power'] = np.trapz(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
                    features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-6)
                else:
                    features['lf_power'] = features['hf_power'] = features['lf_hf_ratio'] = 0
            else:
                features['lf_power'] = features['hf_power'] = features['lf_hf_ratio'] = 0
        except:
            features['lf_power'] = features['hf_power'] = features['lf_hf_ratio'] = 0
    else:
        features['lf_power'] = features['hf_power'] = features['lf_hf_ratio'] = 0
    
    return features

def extract_temp_features_advanced(temp_df):
    """Extraer características de temperatura (sin cambios)"""
    if temp_df is None or len(temp_df) < 100:
        return None
    
    temp = temp_df['temp'].values
    fs = 4.0
    dt = 1.0 / fs
    
    features = {
        'mean': np.mean(temp),
        'std': np.std(temp),
        'min': np.min(temp),
        'max': np.max(temp),
    }
    
    n = len(temp)
    features['slope'] = (temp[-1] - temp[0]) / (n * dt)
    
    window = min(int(fs * 60 * 5), n // 4)
    if window > 0:
        features['temp_drop'] = np.mean(temp[:window]) - np.mean(temp[-window:])
    else:
        features['temp_drop'] = 0
    
    return features

def extract_all_features():
    """Extraer todas las características con corrección de EDA"""
    grades = get_student_grades()
    exam_types = ['midterm_1', 'midterm_2', 'Final']
    
    all_data = []
    
    print("=== EXTRACCIÓN CON CORRECCIÓN DE ESCALA EDA (x10) ===")
    print("-" * 50)
    
    for participant in PARTICIPANTS:
        if participant not in grades:
            continue
        
        grade = grades[participant]
        
        for exam in exam_types:
            print(f"  {participant} - {exam}...")
            
            eda_df = load_signal(participant, exam, 'EDA')
            ibi_df = load_ibi(participant, exam)
            temp_df = load_signal(participant, exam, 'TEMP')
            
            eda_feat = extract_eda_features_corrected(eda_df) if eda_df is not None else None
            hrv_feat = extract_hrv_features_advanced(ibi_df)
            temp_feat = extract_temp_features_advanced(temp_df)
            
            row = {'participant': participant, 'exam': exam, 'grade': grade}
            
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
    print(f"EDA mean (corregida): {df['eda_mean'].mean():.2f} μS")
    
    output_path = Path(__file__).parent.parent / 'results' / 'features_all_corrected.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Guardado en {output_path}")
