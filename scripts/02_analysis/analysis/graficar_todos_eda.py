"""
Genera gráficas de EDA para todos los participantes y exámenes
Con rangos normales superpuestos (Boucsein, 2012)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from scripts.utils import load_signal, get_student_grades

# Configuración
data_path = Path('/home/etoque/ESTRES/data/wearable-exam-stress')
participants = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
exams = ['midterm_1', 'midterm_2', 'Final']
exam_names = {'midterm_1': 'Midterm 1', 'midterm_2': 'Midterm 2', 'Final': 'Final'}

# Rangos normales EDA (μS) según Boucsein (2012)
ranges = {
    'reposo_profundo': (0.5, 2.0, 'lightgreen', 'Reposo profundo (0.5-2.0)'),
    'reposo_normal': (1.0, 5.0, 'lightblue', 'Reposo normal (1.0-5.0)'),
    'estres_moderado': (3.0, 12.0, 'salmon', 'Estrés moderado (3.0-12.0)'),
    'estres_intenso': (5.0, 20.0, 'lightcoral', 'Estrés intenso (5.0-20.0)')
}

# Cargar calificaciones
grades = get_student_grades()

# Crear directorio para figuras
fig_dir = Path('results/figuras_eda')
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERANDO GRÁFICAS EDA PARA TODOS LOS PARTICIPANTES")
print("=" * 60)

for p in participants:
    for exam in exams:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is None or len(eda_df) == 0:
            print(f" {p} - {exam}: Sin datos EDA")
            continue
        
        eda = eda_df['eda'].values
        fs = 4.0
        tiempo_min = np.arange(len(eda)) / fs / 60  # minutos
        
        # Calcular estadísticas
        ceros_pct = (eda == 0).sum() / len(eda) * 100
        eda_mean = np.mean(eda[eda > 0.001]) if (eda > 0.001).any() else np.mean(eda)
        eda_max = eda.max()
        
        # Obtener calificación
        grade = grades.get(p, 0)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Graficar EDA
        ax.plot(tiempo_min, eda, 'b-', linewidth=0.8, alpha=0.7, label='EDA')
        
        # Resaltar ceros
        zero_mask = eda == 0
        if zero_mask.any():
            ax.scatter(tiempo_min[zero_mask], eda[zero_mask], 
                      c='red', s=2, alpha=0.5, label=f'Ceros ({ceros_pct:.1f}%)')
        
        # Añadir rangos normales (como áreas sombreadas)
        for name, (low, high, color, label) in ranges.items():
            ax.axhspan(low, high, alpha=0.15, color=color, label=label if name == 'reposo_normal' else "")
        
        # Solo mostrar leyenda de rangos una vez
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.3, label='Reposo profundo (0.5-2.0)'),
            Patch(facecolor='lightblue', alpha=0.3, label='Reposo normal (1.0-5.0)'),
            Patch(facecolor='salmon', alpha=0.3, label='Estrés moderado (3.0-12.0)'),
            Patch(facecolor='lightcoral', alpha=0.3, label='Estrés intenso (5.0-20.0)')
        ]
        
        # Configuración del gráfico
        ax.set_xlabel('Tiempo (minutos)', fontsize=12)
        ax.set_ylabel('EDA (μS)', fontsize=12)
        ax.set_title(f'{p} - {exam_names[exam]} | Grade: {grade:.1f} | '
                    f'Media: {eda_mean:.3f} μS | Ceros: {ceros_pct:.1f}%', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Ajustar límites Y
        y_max = min(25, eda_max + 2) if eda_max > 0 else 10
        ax.set_ylim(-0.5, y_max)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'EDA_{p}_{exam}.png', dpi=150)
        plt.close()
        
        print(f" {p} - {exam}: media={eda_mean:.3f} μS, ceros={ceros_pct:.1f}%, grade={grade:.1f}")

print("\n" + "=" * 60)
print(f" {len(participants) * 3} gráficas generadas en {fig_dir}")
print("=" * 60)

# Resumen final
print("\n RESUMEN POR PARTICIPANTE (media EDA sin ceros):")
for p in participants:
    medias = []
    for exam in exams:
        eda_df = load_signal(p, exam, 'EDA')
        if eda_df is not None and len(eda_df) > 0:
            eda = eda_df['eda'].values
            eda_no_ceros = eda[eda > 0.001]
            if len(eda_no_ceros) > 0:
                medias.append(np.mean(eda_no_ceros))
    if medias:
        media_total = np.mean(medias)
        if media_total < 0.5:
            estado = " Anormalmente bajo (<0.5 μS)"
        elif media_total < 1.0:
            estado = " Muy bajo (0.5-1.0 μS)"
        elif media_total < 3.0:
            estado = " Reposo normal (1.0-3.0 μS)"
        elif media_total < 5.0:
            estado = " Reposo activo (3.0-5.0 μS)"
        else:
            estado = " Estrés elevado (>5.0 μS)"
        print(f"   {p}: media={media_total:.3f} μS → {estado}")
