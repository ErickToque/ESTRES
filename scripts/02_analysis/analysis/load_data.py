"""
Módulo para cargar el dataset Wearable Exam Stress
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

class ExamStressDataset:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.grades = self._load_grades()
        self.participants = self._get_participants()
    
    def _load_grades(self):
        """Cargar calificaciones de StudentGrades.txt"""
        grades_file = self.data_path / 'StudentGrades.txt'
        return pd.read_csv(grades_file, sep='\t')
    
    def _get_participants(self):
        """Obtener lista de participantes (S1, S2, ...)"""
        return [d.name for d in self.data_path.iterdir() 
                if d.is_dir() and d.name.startswith('S')]
    
    def load_signal(self, participant, exam, signal):
        """
        Cargar una señal específica
        
        Parámetros:
        - participant: 'S1', 'S2', ...
        - exam: 'Midterm 1', 'Midterm 2', 'Final'
        - signal: 'EDA', 'HR', 'IBI', 'ACC', 'BVP', 'TEMP'
        """
        file_path = self.data_path / participant / exam / f'{signal}.csv'
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"No se encontró: {file_path}")
    
    def load_all_signals(self, participant, exam):
        """Cargar todas las señales para un participante/examen"""
        signals = {}
        for sig in ['EDA', 'HR', 'IBI', 'ACC', 'BVP', 'TEMP']:
            try:
                signals[sig] = self.load_signal(participant, exam, sig)
            except FileNotFoundError:
                print(f"Advertencia: {sig} no disponible para {participant}/{exam}")
        return signals

# Ejemplo de uso
if __name__ == "__main__":
    dataset = ExamStressDataset('/home/etoque/ESTRES/data/wearable-exam-stress')
    print(f"Participantes: {dataset.participants}")
    print(f"Calificaciones:\n{dataset.grades}")
