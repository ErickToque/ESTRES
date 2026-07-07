"""
Análisis SEÑAL por SEÑAL, INDIVIDUO por INDIVIDUO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi, get_student_grades

participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
grades = get_student_grades()

print('=' * 80)
print('ANÁLISIS INDIVIDUAL POR SEÑAL Y PARTICIPANTE')
print('=' * 80)

analysis_results = []

for p in participants:
    print(f'\n{"="*40}')
    print(f'PARTICIPANTE: {p} - Grade: {grades.get(p, 0):.1f}')
    print(f'{"="*40}')
    
    for exam in exams:
        print(f'\n   {exam}:')
        
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None or len(eda_df) == 0:
            print(f'      EDA: No data o vacío')
            continue
        
        eda = eda_df['eda'].values
        fs = 4.0
        duration_min = len(eda) / fs / 60
        
        print(f'     EDA: {len(eda)} muestras, {duration_min:.1f} minutos')
        print(f'         Min: {np.min(eda):.4f} μS, Max: {np.max(eda):.4f} μS')
        print(f'         Mean: {np.mean(eda):.4f} μS, Std: {np.std(eda):.4f} μS')
        
        zeros = (eda == 0).sum()
        print(f'         Ceros: {zeros} ({zeros/len(eda)*100:.2f}%)')
        
        deriv = np.abs(np.diff(eda))
        high_deriv = (deriv > np.percentile(deriv, 99)).sum()
        print(f'         Picos extremos: {high_deriv}')
        
        ibi_df = load_ibi(p, exam)
        if ibi_df is not None:
            if 'ibi' in ibi_df.columns:
                ibi_vals = ibi_df['ibi'].values
            else:
                ibi_vals = ibi_df.iloc[:, 1].values
            ibi_vals = ibi_vals[(ibi_vals > 0.3) & (ibi_vals < 2.0)]
            hr_mean = 60.0 / np.mean(ibi_vals) if len(ibi_vals) > 0 else 0
            print(f'     HRV: {len(ibi_vals)} latidos, HR: {hr_mean:.1f} bpm')
        
        temp_df = load_signal(p, exam, 'TEMP')
        if temp_df is not None:
            temp = temp_df['temp'].values
            print(f'     TEMP: Mean={np.mean(temp):.1f}°C, Range={np.max(temp)-np.min(temp):.1f}°C')
        
        analysis_results.append({
            'participant': p, 'exam': exam, 'grade': grades.get(p, 0),
            'eda_mean': np.mean(eda), 'eda_std': np.std(eda),
            'eda_zeros_pct': zeros/len(eda)*100, 'duration_min': duration_min
        })

print('\n' + '=' * 80)
print('RESUMEN GLOBAL')
print('=' * 80)
results_df = pd.DataFrame(analysis_results)
print(f'\n EDA media global: {results_df["eda_mean"].mean():.3f} ± {results_df["eda_mean"].std():.3f} μS')
