"""
REPRODUCCIÓN DEL MÉTODO DE AMIN ET AL. (2022)
Comparacion 10-fold CV vs LOSO
Deteccion de data leakage
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
sys.path.append('.')

from scripts.utils import load_features_amin

def main():
    print("=" * 70)
    print("REPRODUCCION DEL METODO DE AMIN ET AL. (2022)")
    print("=" * 70)

    data = load_features_amin()
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 10-fold CV (como en el paper original)
    cv_random = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores_random = []
    for train_idx, test_idx in cv_random.split(X_scaled, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        y_pred = rf.predict(X_scaled[test_idx])
        scores_random.append(accuracy_score(y[test_idx], y_pred))

    # LOSO (sin leakage)
    logo = LeaveOneGroupOut()
    scores_loso = []
    for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled[train_idx], y[train_idx])
        y_pred = rf.predict(X_scaled[test_idx])
        scores_loso.append(accuracy_score(y[test_idx], y_pred))

    print(f"\nRESULTADOS:")
    print(f"10-fold CV (original): {np.mean(scores_random):.2%} +- {np.std(scores_random):.2%}")
    print(f"LOSO (corregido): {np.mean(scores_loso):.2%} +- {np.std(scores_loso):.2%}")
    print(f"Diferencia: {(np.mean(scores_random)-np.mean(scores_loso)):.2%} (Data Leakage)")

    results = pd.DataFrame({
        'method': ['10-fold CV (original)', 'LOSO (corrected)'],
        'accuracy': [np.mean(scores_random), np.mean(scores_loso)],
        'std': [np.std(scores_random), np.std(scores_loso)]
    })
    results.to_csv('results/tables/baseline_comparison.csv', index=False)
    return results

if __name__ == "__main__":
    main()