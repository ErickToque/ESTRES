"""
ANALISIS DE FEATURE IMPORTANCE
Que caracteristicas fisiologicas son mas importantes?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')

from scripts.utils import load_features_amin

def main():
    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    data = load_features_amin()
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar modelo final con LOSO
    feature_importances = []
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        feature_importances.append(rf.feature_importances_)

    mean_importance = np.mean(feature_importances, axis=0)

    # Top 10 features
    top_idx = np.argsort(mean_importance)[-10:]
    top_names = [f'F{i}' for i in top_idx]
    top_values = mean_importance[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_values, color='steelblue')
    plt.xlabel('Importance')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=150)
    print("Figure saved: results/figures/feature_importance.png")

if __name__ == "__main__":
    main()