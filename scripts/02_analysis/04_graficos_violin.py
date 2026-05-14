"""
Gráficos de violín para señales fisiológicas
Separación por alto (≥80) y bajo rendimiento (<80)
Permite selección manual de casos específicos
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from scripts.utils import load_signal, load_ibi

# Configuración de estilo para paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = {'Alto': '#2ecc71', 'Bajo': '#e74c3c'}

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

def load_data_for_plots():
    """Carga datos para todos los participantes y exámenes"""
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
            grade_class = 'Alto' if grade >= 80 else 'Bajo'
            
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
                'exam': exam,
                'grade': grade,
                'grade_class': grade_class,
                'eda': eda_mean,
                'hr': hr,
                'temp': temp_mean
            })
    
    return pd.DataFrame(all_data)

print("=" * 70)
print("GENERANDO GRÁFICOS DE VIOLÍN")
print("=" * 70)

# Cargar datos
df = load_data_for_plots()
df = df.dropna(subset=['eda', 'hr', 'temp'])

print(f"Muestras totales: {len(df)}")
print(f"Alto rendimiento (≥80): {len(df[df['grade_class'] == 'Alto'])}")
print(f"Bajo rendimiento (<80): {len(df[df['grade_class'] == 'Bajo'])}")

# ============================================================================
# FIGURA 1: VIOLIN PLOTS PARA TODAS LAS SEÑALES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# 1. EDA
ax = axes[0]
data_to_plot = [df[df['grade_class'] == 'Alto']['eda'].dropna().values,
                df[df['grade_class'] == 'Bajo']['eda'].dropna().values]
parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
for i, color in enumerate([colors['Alto'], colors['Bajo']]):
    parts['bodies'][i].set_facecolor(color)
    parts['bodies'][i].set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Alto\n(≥80)', 'Bajo\n(<80)'])
ax.set_ylabel('EDA (μS)')
ax.set_title('Electrodermal Activity (EDA)')
ax.grid(True, alpha=0.3)

# 2. HR
ax = axes[1]
data_to_plot = [df[df['grade_class'] == 'Alto']['hr'].dropna().values,
                df[df['grade_class'] == 'Bajo']['hr'].dropna().values]
parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
for i, color in enumerate([colors['Alto'], colors['Bajo']]):
    parts['bodies'][i].set_facecolor(color)
    parts['bodies'][i].set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Alto\n(≥80)', 'Bajo\n(<80)'])
ax.set_ylabel('Heart Rate (bpm)')
ax.set_title('Heart Rate (HR)')
ax.grid(True, alpha=0.3)

# 3. TEMP
ax = axes[2]
data_to_plot = [df[df['grade_class'] == 'Alto']['temp'].dropna().values,
                df[df['grade_class'] == 'Bajo']['temp'].dropna().values]
parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
for i, color in enumerate([colors['Alto'], colors['Bajo']]):
    parts['bodies'][i].set_facecolor(color)
    parts['bodies'][i].set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Alto\n(≥80)', 'Bajo\n(<80)'])
ax.set_ylabel('Temperature (°C)')
ax.set_title('Skin Temperature (TEMP)')
ax.grid(True, alpha=0.3)

plt.suptitle('Distribución de Señales Fisiológicas por Rendimiento', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/violin_todas_senales.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura guardada: paper_resultados/figuras/violin_todas_senales.png")

# ============================================================================
# FIGURA 2: CASOS ESPECÍFICOS SELECCIONADOS MANUALMENTE
# ============================================================================
# Aquí puedes seleccionar manualmente los casos que quieres destacar
# Por ejemplo, S3 (bajo rendimiento que mejoró) y S8 (alto rendimiento consistente)

casos_especiales = [
    {'participant': 'S3', 'nombre': 'S3 - Mejora dramática (77→94)', 'color': 'blue'},
    {'participant': 'S8', 'nombre': 'S8 - Alto rendimiento consistente (92→92)', 'color': 'green'},
    {'participant': 'S7', 'nombre': 'S7 - Caída severa (64→33→55)', 'color': 'red'},
    {'participant': 'S10', 'nombre': 'S10 - Descenso progresivo (89→64→58)', 'color': 'orange'},
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, caso in enumerate(casos_especiales[:4]):
    ax = axes[idx]
    p = caso['participant']
    
    # Datos del sujeto
    sujeto_data = df[df['participant'] == p].copy()
    exam_order = {'midterm_1': 0, 'midterm_2': 1, 'Final': 2}
    sujeto_data['exam_num'] = sujeto_data['exam'].map(exam_order)
    sujeto_data = sujeto_data.sort_values('exam_num')
    
    # Evolución de EDA y Grade
    exams = sujeto_data['exam'].values
    grades = sujeto_data['grade'].values
    eda_vals = sujeto_data['eda'].values
    
    # Gráfico con dos ejes
    ax_twin = ax.twinx()
    
    # Barras de EDA
    bars = ax.bar(exams, eda_vals, alpha=0.6, color=caso['color'], label='EDA')
    ax.set_ylabel('EDA (μS)', color=caso['color'])
    ax.tick_params(axis='y', labelcolor=caso['color'])
    
    # Línea de grades
    ax_twin.plot(exams, grades, 'o-', color='black', linewidth=2, markersize=8, label='Grade')
    ax_twin.set_ylabel('Grade', color='black')
    ax_twin.tick_params(axis='y', labelcolor='black')
    ax_twin.set_ylim(0, 100)
    
    ax.set_title(caso['nombre'])
    ax.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, val in zip(bars, eda_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=caso['color'])

plt.suptitle('Casos Especiales: Evolución de EDA vs Calificaciones', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/casos_especiales.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/casos_especiales.png")

# ============================================================================
# FIGURA 3: BOXPLOT POR SUJETO PARA SELECCIÓN MANUAL
# ============================================================================
# Esta figura te ayuda a seleccionar qué casos destacar
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# EDA por sujeto
ax = axes[0]
sns.boxplot(data=df, x='participant', y='eda', hue='grade_class', ax=ax, palette=colors)
ax.set_ylabel('EDA (μS)')
ax.set_title('EDA por Sujeto')
ax.legend(title='Rendimiento')
ax.grid(True, alpha=0.3)

# HR por sujeto
ax = axes[1]
sns.boxplot(data=df, x='participant', y='hr', hue='grade_class', ax=ax, palette=colors)
ax.set_ylabel('Heart Rate (bpm)')
ax.set_title('HR por Sujeto')
ax.legend(title='Rendimiento')
ax.grid(True, alpha=0.3)

# TEMP por sujeto
ax = axes[2]
sns.boxplot(data=df, x='participant', y='temp', hue='grade_class', ax=ax, palette=colors)
ax.set_ylabel('Temperature (°C)')
ax.set_title('TEMP por Sujeto')
ax.legend(title='Rendimiento')
ax.grid(True, alpha=0.3)

plt.suptitle('Distribución de Señales por Sujeto y Grupo de Rendimiento', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_resultados/figuras/boxplot_por_sujeto.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: paper_resultados/figuras/boxplot_por_sujeto.png")

# ============================================================================
# DATOS RESUMEN PARA SELECCIÓN MANUAL
# ============================================================================
print("\n" + "=" * 70)
print("DATOS RESUMEN POR SUJETO (para selección manual)")
print("=" * 70)

for p in df['participant'].unique():
    sujeto = df[df['participant'] == p]
    grades = sujeto['grade'].values
    eda_mean = sujeto['eda'].mean()
    hr_mean = sujeto['hr'].mean()
    temp_mean = sujeto['temp'].mean()
    rendimiento = 'Alto' if np.mean(grades) >= 80 else 'Bajo'
    
    print(f"\n{p}:")
    print(f"   Grades: {grades}")
    print(f"   Promedio: {np.mean(grades):.1f} → {rendimiento}")
    print(f"   EDA: {eda_mean:.2f} μS")
    print(f"   HR: {hr_mean:.1f} bpm")
    print(f"   TEMP: {temp_mean:.1f} °C")

print("\n✅ Datos listos para selección manual de casos")
