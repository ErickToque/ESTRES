"""
ANÁLISIS INTRA-SUJETO
Correlaciones entre señales fisiológicas y calificaciones por individuo
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import sys

sys.path.append('.')

# ============================================================================
# FUNCIONES LOCALES (para evitar dependencias externas)
# ============================================================================

DATA_PATH = Path('/home/etoque/ESTRES/data/wearable-exam-stress')

def load_signal_local(participant, exam, signal_name):
    """Carga señales en formato especial del dataset"""
    file_path = DATA_PATH / participant / exam / f'{signal_name}.csv'
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        return None
    
    data = np.array([float(l.strip()) for l in lines[2:]])
    
    return {'data': data, 'fs': 4.0}

def load_ibi_local(participant, exam):
    """Carga IBI (formato diferente)"""
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

def load_grades_individual_local():
    """Carga calificaciones individuales"""
    grades_raw = {}
    grades_file = DATA_PATH / 'StudentGrades.txt'
    
    if not grades_file.exists():
        return grades_raw
    
    with open(grades_file, 'r', encoding='latin1') as f:
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
                except:
                    pass
    
    return grades_raw


def main():
    print("=" * 70)
    print("ANÁLISIS INTRA-SUJETO")
    print("=" * 70)
    
    grades_raw = load_grades_individual_local()
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    
    results = []
    
    for p in participants:
        print(f"\n📌 {p}:")
        
        grades = []
        eda_vals = []
        hr_vals = []
        temp_vals = []
        
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade = grades_raw[p][exam]
            if exam == 'Final':
                grade = grade / 2
            grades.append(grade)
            
            # EDA
            eda_signal = load_signal_local(p, exam, 'EDA')
            if eda_signal is not None:
                eda = eda_signal['data'] * 10
                eda_no_ceros = eda[eda > 0.01]
                if len(eda_no_ceros) > 0:
                    eda_vals.append(np.mean(eda_no_ceros))
                else:
                    eda_vals.append(np.nan)
            else:
                eda_vals.append(np.nan)
            
            # HR (desde IBI)
            ibi_df = load_ibi_local(p, exam)
            if ibi_df is not None:
                if 'ibi' in ibi_df.columns:
                    ibi = ibi_df['ibi'].values
                else:
                    ibi = ibi_df.iloc[:, 1].values
                ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                if len(ibi) > 5:
                    hr_vals.append(60.0 / np.mean(ibi))
                else:
                    hr_vals.append(np.nan)
            else:
                hr_vals.append(np.nan)
            
            # TEMP
            temp_signal = load_signal_local(p, exam, 'TEMP')
            if temp_signal is not None:
                temp_vals.append(np.mean(temp_signal['data']))
            else:
                temp_vals.append(np.nan)
        
        if len(grades) < 2:
            print(f"   Datos insuficientes para correlación")
            continue
        
        grades_arr = np.array(grades)
        
        for name, vals in [('EDA', eda_vals), ('HR', hr_vals), ('TEMP', temp_vals)]:
            vals_arr = np.array(vals)
            mask = ~np.isnan(vals_arr) & ~np.isnan(grades_arr)
            if mask.sum() >= 3:
                corr, p_val = pearsonr(grades_arr[mask], vals_arr[mask])
                results.append({
                    'participant': p, 
                    'signal': name, 
                    'correlation': corr, 
                    'p_value': p_val
                })
                sig = "" if p_val < 0.05 else ""
                print(f"   {name}: r={corr:.3f} (p={p_val:.4f}) {sig}")
            else:
                print(f"   {name}: datos insuficientes (n={mask.sum()})")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN INTRA-SUJETO")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        significativas = df[df['p_value'] < 0.05]
        print(f"\nCorrelaciones significativas (p<0.05): {len(significativas)} de {len(df)}")
        
        for _, row in significativas.iterrows():
            print(f"   {row['participant']} - {row['signal']}: r={row['correlation']:.3f} (p={row['p_value']:.4f})")
        
        # Guardar resultados
        import os
        os.makedirs('results/tables', exist_ok=True)
        df.to_csv('results/tables/intra_subject_correlations.csv', index=False)
        print(f"\n Guardado: results/tables/intra_subject_correlations.csv")
    else:
        print("No se encontraron correlaciones válidas")
    
    return df


if __name__ == "__main__":
    main()