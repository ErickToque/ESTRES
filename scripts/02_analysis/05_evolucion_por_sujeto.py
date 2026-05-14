"""
Evolución de señales fisiológicas y calificaciones por sujeto
Muestra la trayectoria de EDA, HR, TEMP y Grades a lo largo de los 3 exámenes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

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

def load_data_for_evolution():
    """Carga datos de evolución para todos los participantes"""
    participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    exams = ['midterm_1', 'midterm_2', 'Final']
    exam_labels = ['Midterm 1', 'Midterm 2', 'Final']
    grades_raw = load_grades_individual()
    
    all_data = []
    
    for p in participants:
        for idx, exam in enumerate(exams):
            if p not in grades_raw or exam not in grades_raw[p]:
                continue
            
            grade_raw = grades_raw[p][exam]
            grade = grade_raw / 2 if exam == 'Final' else grade_raw
            
            # EDA
            eda_df = load_signal(p, exam, 'EDA')
            if eda_df is not None:
                eda = eda_df['eda'].values * 10
                eda_no_ceros = eda[eda > 0.01]
                eda_mean = np.mean(eda_no_ceros) if len(eda_no_ceros) > 0 else np.nan
            else:
                eda_mean = np.nan
            
            # HR
            ibi_df = load_ibi(p, exam)
            if ibi_df is not None:
                if 'ibi' in ibi_df.columns:
                    ibi = ibi_df['ibi'].values
                else:
                    ibi = ibi_df.iloc[:, 1].values
                ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
                hr = 60.0 / np.mean(ibi) if len(ibi) > 5 else np.nan
            else:
                hr = np.nan
            
            # TEMP
            temp_df = load_signal(p, exam, 'TEMP')
            if temp_df is not None:
                temp_mean = np.mean(temp_df['temp'].values)
            else:
                temp_mean = np.nan
            
            all_data.append({
                'participant': p,
                'exam_num': idx,
                'exam_label': exam_labels[idx],
                'grade': grade,
                'eda': eda_mean,
                'hr': hr,
                'temp': temp_mean
            })
    
    return pd.DataFrame(all_data)

print("=" * 70)
print("GENERANDO GRÁFICOS DE EVOLUCIÓN POR SUJETO")
print("=" * 70)

df = load_data_for_evolution()
print(f"Muestras totales: {len(df)}")

# Definir sujetos y colores por rendimiento
participants = df['participant'].unique()
promedios = df.groupby('participant')['grade'].mean().to_dict()
grupo_color = {p: ('green' if promedios[p] >= 80 else 'red') for p in participants}
grupo_nombre = {p: ('Alto' if promedios[p] >= 80 else 'Bajo') for p in participants}

# ============================================================================
# FIGURA 1: EVOLUCIÓN COMPLETA (3x4 grid)
# ============================================================================
fig, axes = plt.subplots(4, 3, figsize=(15, 18))
axes = axes.flatten()

for idx, p in enumerate(participants):
    ax = axes[idx]
    sujeto_data = df[df['participant'] == p].sort_values('exam_num')
    
    if len(sujeto_data) == 0:
        ax.set_visible(False)
        continue
    
    exams = sujeto_data['exam_label'].values
    grades = sujeto_data['grade'].values
    eda = sujeto_data['eda'].values
    hr = sujeto_data['hr'].values
    temp = sujeto_data['temp'].values
    
    # Crear gráfico con dos ejes
    ax_twin = ax.twinx()
    
    # Barras de EDA
    x = np.arange(len(exams))
    width = 0.35
    bars = ax.bar(x - width/2, eda, width, alpha=0.7, color='blue', label='EDA')
    ax.set_ylabel('EDA (μS)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Línea de notas
    line, = ax_twin.plot(x, grades, 'o-', color='black', linewidth=2, 
                          markersize=8, label='Grade')
    ax_twin.set_ylabel('Grade', color='black')
    ax_twin.tick_params(axis='y', labelcolor='black')
    ax_twin.set_ylim(0, 100)
    
    # Configuración
    ax.set_xticks(x)
    ax.set_xticklabels(exams, rotation=45, ha='right')
    ax.set_title(f'{p} - {grupo_nombre[p]} rendimiento (prom={promedios[p]:.1f})', 
                 color=grupo_color[p])
    ax.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, val in zip(bars, eda):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='blue')
    
    # Añadir valores de notas
    for i, (g, e) in enumerate(zip(grades, exams)):
        ax_twin.text(i, g + 2, f'{g:.0f}', ha='center', va='bottom', fontsize=8, color='black')

plt.suptitle('Evolución de EDA y Calificaciones por Sujeto', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/evolucion_eda_grades.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/evolucion_eda_grades.png")

# ============================================================================
# FIGURA 2: EVOLUCIÓN DE LAS 3 SEÑALES + NOTAS (para casos seleccionados)
# ============================================================================
casos_destacados = ['S3', 'S7', 'S8', 'S10']  # Casos más interesantes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, p in enumerate(casos_destacados):
    ax = axes[idx]
    sujeto_data = df[df['participant'] == p].sort_values('exam_num')
    
    if len(sujeto_data) == 0:
        ax.set_visible(False)
        continue
    
    exams = sujeto_data['exam_label'].values
    x = np.arange(len(exams))
    
    # Normalizar señales para comparación visual
    grades = sujeto_data['grade'].values
    eda = sujeto_data['eda'].values
    hr = sujeto_data['hr'].values
    temp = sujeto_data['temp'].values
    
    # Normalizar (min-max)
    def normalize(vals):
        if np.all(np.isnan(vals)):
            return vals
        vals_clean = vals[~np.isnan(vals)]
        if len(vals_clean) == 0:
            return vals
        min_v, max_v = np.min(vals_clean), np.max(vals_clean)
        if max_v - min_v == 0:
            return vals - min_v
        return (vals - min_v) / (max_v - min_v)
    
    grades_norm = normalize(grades)
    eda_norm = normalize(eda)
    hr_norm = normalize(hr)
    temp_norm = normalize(temp)
    
    ax.plot(x, grades_norm, 'o-', color='black', linewidth=2, markersize=8, label='Grade')
    ax.plot(x, eda_norm, 's-', color='blue', linewidth=1.5, markersize=6, label='EDA')
    ax.plot(x, hr_norm, '^-', color='red', linewidth=1.5, markersize=6, label='HR')
    ax.plot(x, temp_norm, 'd-', color='orange', linewidth=1.5, markersize=6, label='TEMP')
    
    ax.set_xticks(x)
    ax.set_xticklabels(exams, rotation=45, ha='right')
    ax.set_ylabel('Valor normalizado')
    ax.set_title(f'{p} - Evolución (promedio grade={promedios[p]:.1f})', 
                 color=grupo_color[p])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

plt.suptitle('Evolución de Señales Fisiológicas vs Calificaciones (Casos Destacados)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/evolucion_casos_destacados.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/evolucion_casos_destacados.png")

# ============================================================================
# FIGURA 3: MATRIZ DE CORRELACIÓN POR SUJETO (mapa de calor)
# ============================================================================
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for idx, p in enumerate(participants):
    ax = axes[idx]
    sujeto_data = df[df['participant'] == p].sort_values('exam_num')
    
    if len(sujeto_data) < 3:
        ax.set_visible(False)
        continue
    
    grades = sujeto_data['grade'].values
    eda = sujeto_data['eda'].values
    hr = sujeto_data['hr'].values
    temp = sujeto_data['temp'].values
    
    # Calcular correlaciones
    corr_data = {}
    for name, vals in [('Grade', grades), ('EDA', eda), ('HR', hr), ('TEMP', temp)]:
        if len(vals) >= 3 and not np.all(np.isnan(vals)):
            vals_clean = vals[~np.isnan(vals)]
            grades_clean = grades[~np.isnan(vals)]
            if len(vals_clean) >= 3:
                corr = np.corrcoef(grades_clean, vals_clean)[0, 1]
                corr_data[name] = corr if not np.isnan(corr) else 0
            else:
                corr_data[name] = 0
        else:
            corr_data[name] = 0
    
    # Gráfico de barras
    names = list(corr_data.keys())
    values = list(corr_data.values())
    colors_bar = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    bars = ax.bar(names, values, color=colors_bar, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f'{p}\n(prom={promedios[p]:.1f})', color=grupo_color[p])
    ax.set_ylabel('Correlación con Grade')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * np.sign(val),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

plt.suptitle('Correlación entre Señales Fisiológicas y Calificaciones por Sujeto', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/correlaciones_por_sujeto.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/correlaciones_por_sujeto.png")

# ============================================================================
# DATOS RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("RESUMEN DE CORRELACIONES POR SUJETO")
print("=" * 70)

for p in participants:
    sujeto_data = df[df['participant'] == p].sort_values('exam_num')
    if len(sujeto_data) >= 3:
        grades = sujeto_data['grade'].values
        eda = sujeto_data['eda'].values
        hr = sujeto_data['hr'].values
        temp = sujeto_data['temp'].values
        
        def safe_corr(a, b):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() >= 3:
                return np.corrcoef(a[mask], b[mask])[0, 1]
            return np.nan
        
        print(f"\n{p} (prom={promedios[p]:.1f}):")
        print(f"   r(EDA,Grade) = {safe_corr(eda, grades):+.3f}")
        print(f"   r(HR,Grade)  = {safe_corr(hr, grades):+.3f}")
        print(f"   r(TEMP,Grade)= {safe_corr(temp, grades):+.3f}")

print("\n✅ Todas las figuras guardadas en paper_resultados/figuras/")
