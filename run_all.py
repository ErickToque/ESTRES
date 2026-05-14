#!/usr/bin/env python3
"""
Pipeline completo para el analisis de estres en examenes
Ejecuta los scripts principales en orden
"""
import subprocess
import sys
import os

# Directorios
SCRIPTS_BASE = "scripts"
OUT_DIR = "out"

os.makedirs(f"{OUT_DIR}/tables", exist_ok=True)
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

# Scripts principales (por orden de ejecucion)
scripts = [
    "scripts/01_preprocessing/extract_features.py",
    "scripts/02_analysis/01_intra_subject.py",
    "scripts/02_analysis/04_window_optimization.py",
    "scripts/02_analysis/02_reproduce_baseline.py",
    "scripts/03_models/train_models.py",
    "scripts/04_visualization/visualize.py"
]

print("=" * 60)
print("PIPELINE COMPLETO - ESTRES EN EXAMENES")
print("=" * 60)

for script_path in scripts:
    if not os.path.exists(script_path):
        print(f"  {script_path} no encontrado (omitido)")
        continue
    
    print(f"\n  Ejecutando {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"  Error en {script_path}")
        break

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO")
print("Resultados en out/tables/ y out/figures/")
print("=" * 60)
