"""
Matriz de correlación entre features de señales y grade por sujeto
Permite identificar qué features son más predictivas para cada individuo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi
from scipy import signal

# Configuración
CUTOFF_FREQS = {
    'S1': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S2': {'midterm_1': 0.002, 'midterm_2': 0.001, 'Final': 0.002},
    'S3': {'midterm_1': 0.0002, 'midterm_2': 0.0002, 'Final': 0.001},
    'S4': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S5': {'midterm_1': 0.001, 'midterm_2': 0.001, 'Final': 0.001},
    'S6': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S7': {'midterm_1': 0.001, 'midterm_2': 0.0002, 'Final': 0.0002},
    'S8': {'midterm_1': 0.0002, 'midterm_2': 0.002, 'Final': 0.002},
    'S9': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
    'S10': {'midterm_1': 0.002, 'midterm_2': 0.002, 'Final': 0.002},
}

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

def lowpass_filter_trendline(eda, fs=4.0, cutoff_hz=0.002):
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    order = 100
    b = signal.firwin(order, normalized_cutoff, window='hamming')
    return signal.filtfilt(b, 1, eda)

def extract_all_features_for_subject(participant, exam, cutoff):
    """Extrae todas las features para un sujeto y examen"""
    eda_df = load_signal(participant, exam, 'EDA')
    if eda_df is None:
        return None
    
    eda = eda_df['eda'].values * 10
    trendline = lowpass_filter_trendline(eda, 4.0, cutoff)
    
    features = {}
    
    # Estadísticas básicas EDA
    features['eda_mean'] = np.mean(trendline)
    features['eda_std'] = np.std(trendline)
    features['eda_max'] = np.max(trendline)
    features['eda_min'] = np.min(trendline)
    features['eda_range'] = features['eda_max'] - features['eda_min']
    
    # Derivada
    deriv = np.diff(trendline)
    features['eda_mean_deriv'] = np.mean(np.abs(deriv))
    features['eda_std_deriv'] = np.std(deriv)
    
    # HRV
    ibi_df = load_ibi(participant, exam)
    if ibi_df is not None:
        if 'ibi' in ibi_df.columns:
            ibi = ibi_df['ibi'].values
        else:
            ibi = ibi_df.iloc[:, 1].values
        ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
        if len(ibi) > 5:
            features['hr_mean'] = 60.0 / np.mean(ibi)
            features['hr_std'] = np.std(60.0 / ibi)
            diffs = np.diff(ibi)
            features['hr_rmssd'] = np.sqrt(np.mean(diffs**2)) * 1000
        else:
            features['hr_mean'] = np.nan
            features['hr_std'] = np.nan
            features['hr_rmssd'] = np.nan
    else:
        features['hr_mean'] = np.nan
        features['hr_std'] = np.nan
        features['hr_rmssd'] = np.nan
    
    # Temperatura
    temp_df = load_signal(participant, exam, 'TEMP')
    if temp_df is not None:
        temp = temp_df['temp'].values
        features['temp_mean'] = np.mean(temp)
        features['temp_std'] = np.std(temp)
        features['temp_slope'] = (temp[-1] - temp[0]) / len(temp)
    else:
        features['temp_mean'] = np.nan
        features['temp_std'] = np.nan
        features['temp_slope'] = np.nan
    
    return features

def load_features_all_subjects():
    """Carga features para todos los sujetos y exámenes"""
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    grades_raw = load_grades_individual()
    
    all_data = []
    
    for p in participants:
        for exam in exams:
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            
            cutoff = CUTOFF_FREQS.get(p, {}).get(exam, 0.001)
            features = extract_all_features_for_subject(p, exam, cutoff)
            
            if features:
                row = {'participant': p, 'exam': exam, 'grade': grade}
                row.update(features)
                all_data.append(row)
    
    return pd.DataFrame(all_data)

print("=" * 70)
print("MATRIZ DE CORRELACIÓN: FEATURES vs GRADE POR SUJETO")
print("=" * 70)

# Cargar datos
df = load_features_all_subjects()
print(f"Muestras totales: {len(df)}")

# Lista de features a analizar
feature_cols = ['eda_mean', 'eda_std', 'eda_max', 'eda_min', 'eda_range',
                'eda_mean_deriv', 'eda_std_deriv', 'hr_mean', 'hr_std', 'hr_rmssd',
                'temp_mean', 'temp_std', 'temp_slope']

# Nombres amigables para las figuras
feature_names = {
    'eda_mean': 'EDA Mean', 'eda_std': 'EDA Std', 'eda_max': 'EDA Max',
    'eda_min': 'EDA Min', 'eda_range': 'EDA Range', 'eda_mean_deriv': 'EDA Mean Deriv',
    'eda_std_deriv': 'EDA Std Deriv', 'hr_mean': 'HR Mean', 'hr_std': 'HR Std',
    'hr_rmssd': 'HR RMSSD', 'temp_mean': 'Temp Mean', 'temp_std': 'Temp Std',
    'temp_slope': 'Temp Slope'
}

participants = df['participant'].unique()

# ============================================================================
# FIGURA 1: MATRIZ DE CORRELACIÓN COMPLETA (todos los sujetos)
# ============================================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, p in enumerate(participants):
    ax = axes[idx]
    sujeto_df = df[df['participant'] == p].dropna(subset=feature_cols)
    
    if len(sujeto_df) < 2:
        ax.text(0.5, 0.5, f'{p}\nDatos insuficientes', ha='center', va='center')
        ax.set_visible(True)
        continue
    
    correlations = []
    for col in feature_cols:
        if col in sujeto_df.columns:
            valid = sujeto_df[[col, 'grade']].dropna()
            if len(valid) >= 3:
                corr, p_val = pearsonr(valid['grade'], valid[col])
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        else:
            correlations.append(0)
    
    # Gráfico de barras
    colors_bar = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations]
    bars = ax.barh(range(len(feature_cols)), correlations, color=colors_bar, edgecolor='black')
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_names.get(c, c) for c in feature_cols], fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_title(f'{p}\n(grade avg={sujeto_df["grade"].mean():.1f})')
    ax.set_xlabel('Correlación con Grade')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(bar.get_width() + 0.02 * np.sign(corr), bar.get_y() + bar.get_height()/2,
                f'{corr:.2f}', ha='left' if corr > 0 else 'right', va='center', fontsize=7)

plt.suptitle('Correlación entre Features Fisiológicas y Calificaciones por Sujeto', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/matrices/correlaciones_features_por_sujeto.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/matrices/correlaciones_features_por_sujeto.png")

# ============================================================================
# FIGURA 2: HEATMAP DE CORRELACIONES (matriz sujetos x features)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Crear matriz de correlaciones
corr_matrix = []
for p in participants:
    sujeto_df = df[df['participant'] == p].dropna(subset=feature_cols)
    row = []
    for col in feature_cols:
        if col in sujeto_df.columns:
            valid = sujeto_df[[col, 'grade']].dropna()
            if len(valid) >= 3:
                corr, _ = pearsonr(valid['grade'], valid[col])
                row.append(corr if not np.isnan(corr) else 0)
            else:
                row.append(0)
        else:
            row.append(0)
    corr_matrix.append(row)

corr_matrix = np.array(corr_matrix)

# Heatmap
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([feature_names.get(c, c) for c in feature_cols], rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(participants)))
ax.set_yticklabels(participants, fontsize=10)
ax.set_xlabel('Features Fisiológicas', fontsize=12)
ax.set_ylabel('Sujeto', fontsize=12)
ax.set_title('Matriz de Correlación: Sujetos vs Features (color = correlación con Grade)', 
             fontsize=14, fontweight='bold')

# Añadir valores en el heatmap
for i in range(len(participants)):
    for j in range(len(feature_cols)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black", fontsize=8)

plt.colorbar(im, label='Correlación con Grade')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/matrices/heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/matrices/heatmap_correlaciones.png")

# ============================================================================
# FIGURA 3: TOP FEATURES POR SUJETO (las 3 más correlacionadas)
# ============================================================================
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for idx, p in enumerate(participants):
    ax = axes[idx]
    sujeto_df = df[df['participant'] == p].dropna(subset=feature_cols)
    
    if len(sujeto_df) < 2:
        ax.text(0.5, 0.5, f'{p}\nDatos insuficientes', ha='center', va='center')
        ax.set_visible(True)
        continue
    
    correlations = []
    for col in feature_cols:
        if col in sujeto_df.columns:
            valid = sujeto_df[[col, 'grade']].dropna()
            if len(valid) >= 3:
                corr, _ = pearsonr(valid['grade'], valid[col])
                correlations.append((col, corr if not np.isnan(corr) else 0))
            else:
                correlations.append((col, 0))
        else:
            correlations.append((col, 0))
    
    # Ordenar por |correlación| y tomar top 3
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = correlations[:3]
    
    names = [feature_names.get(c, c) for c, _ in top3]
    values = [v for _, v in top3]
    colors_bar = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(names, values, color=colors_bar, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_title(f'{p}\n(avg grade={sujeto_df["grade"].mean():.1f})')
    ax.set_xlabel('Correlación con Grade')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02 * np.sign(val), bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)

plt.suptitle('Top 3 Features más Correlacionadas con Grade por Sujeto', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/matrices/top3_features_por_sujeto.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/matrices/top3_features_por_sujeto.png")

# ============================================================================
# TABLA RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("TOP 3 FEATURES POR SUJETO (|r| más alta)")
print("=" * 70)

for p in participants:
    sujeto_df = df[df['participant'] == p].dropna(subset=feature_cols)
    if len(sujeto_df) < 2:
        continue
    
    correlations = []
    for col in feature_cols:
        if col in sujeto_df.columns:
            valid = sujeto_df[[col, 'grade']].dropna()
            if len(valid) >= 3:
                corr, _ = pearsonr(valid['grade'], valid[col])
                correlations.append((col, corr if not np.isnan(corr) else 0))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = correlations[:3]
    
    print(f"\n{p} (promedio grade={sujeto_df['grade'].mean():.1f}):")
    for col, corr in top3:
        print(f"   {feature_names.get(col, col)}: r={corr:+.3f}")

print("\n✅ Todas las matrices guardadas en paper_resultados/figuras/matrices/")
