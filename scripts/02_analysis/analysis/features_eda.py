import numpy as np
import pandas as pd
from scipy import signal

def compute_eda_features(eda_signal, fs=4):
    """
    Calcular características básicas de EDA
    fs = frecuencia de muestreo (Hz) - default 4 Hz para Empatica E4
    """
    # Derivada
    eda_deriv = np.diff(eda_signal)
    
    # Características
    features = {
        'mean': np.mean(eda_signal),
        'std': np.std(eda_signal),
        'min': np.min(eda_signal),
        'max': np.max(eda_signal),
        'range': np.max(eda_signal) - np.min(eda_signal),
        'mean_deriv': np.mean(eda_deriv),
        'std_deriv': np.std(eda_deriv),
        'max_deriv': np.max(eda_deriv),
        'slope': (eda_signal[-1] - eda_signal[0]) / len(eda_signal)
    }
    
    # Picos (SCRs) - método simple
    peaks, _ = signal.find_peaks(eda_signal, height=np.std(eda_signal))
    features['num_scr'] = len(peaks)
    
    return features

def compute_hrv_features(ibi_ms):
    """
    Calcular características HRV desde IBI (ms)
    """
    ibi = np.array(ibi_ms) / 1000.0  # convertir a segundos
    
    # Diferencias entre intervalos
    diffs = np.diff(ibi)
    
    features = {
        'mean_ibi': np.mean(ibi),
        'std_ibi': np.std(ibi),
        'rmssd': np.sqrt(np.mean(diffs**2)),
        'pnn50': np.mean(np.abs(diffs) > 0.05) * 100,
        'hr_mean': 60 / np.mean(ibi),
        'hr_std': 60 * np.std(ibi) / np.mean(ibi)**2
    }
    
    return features
