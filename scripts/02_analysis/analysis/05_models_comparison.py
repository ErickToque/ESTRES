"""
COMPARACIÓN DE MODELOS (Ensemble)

Random Forest vs SVM vs kNN vs Gradient Boosting

LOSO validation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append('.')

from scripts.utils import load_features_optimal

def main():
    print("=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)

    data = load_features_optimal()
    X = np.array([d['features'] for d in data])
    y = np.array([d['grade_class'] for d in data])
    groups = [d['participant'] for d in data]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    logo = LeaveOneGroupOut()
    results = []

    for name, model in models.items():
        scores = []
        f1s = []

        for train_idx, test_idx in logo.split(X_scaled, y, groups=groups):
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred = model.predict(X_scaled[test_idx])
            scores.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))

        results.append({
            'model': name,
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'f1_score': np.mean(f1s)
        })
        print(f"\n{name}:")
        print(f" Accuracy: {np.mean(scores):.2%} ± {np.std(scores):.2%}")
        print(f" F1-Score: {np.mean(f1s):.2%}")

    df = pd.DataFrame(results)
    df.to_csv('results/tables/models_comparison.csv', index=False)

    # Gráfico
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['model'], df['accuracy'], yerr=df['accuracy_std'],
                   capsize=5, color='steelblue', edgecolor='black')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Azar (50%)')
    plt.ylabel('Accuracy (LOSO)')
    plt.title('Comparación de Modelos')
    plt.ylim(0, 1)
    plt.legend()

    for bar, acc in zip(bars, df['accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/models_comparison.png', dpi=150)
    print("\nFigura guardada: results/figures/models_comparison.png")

    return df


if __name__ == "__main__":
    main()
