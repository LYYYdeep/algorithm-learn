import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    X = np.array([
        [1.0, 5, 18, 0],
        [1.5, 8, 19, 0],
        [2.0, 10, 20, 0],
        [2.5, 15, 21, 0],
        [3.0, 20, 20, 1],
        [3.5, 25, 22, 1],
        [4.0, 30, 23, 1],
        [4.5, 35, 24, 1],
        [5.0, 40, 22, 1],
        [5.5, 45, 21, 1],
        [6.0, 50, 25, 0],
        [6.5, 55, 26, 1],
        [7.0, 60, 27, 1],
        [7.5, 65, 28, 1],
        [8.0, 70, 29, 0],
        [8.5, 75, 30, 1],
        [1.2, 6, 18, 0],
        [2.2, 12, 19, 0],
        [3.2, 22, 22, 1],
        [4.2, 32, 23, 1],
    ], dtype=float)

    y = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1,
        0,
        1,
        1, 1,
        0,
        1,
        0, 0, 1, 1,
    ], dtype=int)

    feature_names = [
        "study_hours",
        "exercises",
        "age",
        "has_tutor",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_

    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs = sorted(
        feature_importance_pairs,
        key=lambda x: x[1],
        reverse=True,
    )

    print("Feature Importances:")
    for feature_name, importance in feature_importance_pairs:
        print(f"{feature_name}: {importance:.4f}")

    print()
    print("Sum of importances:", importances.sum())


if __name__ == "__main__":
    main()
