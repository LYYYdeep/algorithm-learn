import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    X = np.array([
        [1.0, 5],
        [1.5, 8],
        [2.0, 10],
        [2.5, 15],
        [3.0, 20],
        [3.5, 25],
        [4.0, 30],
        [4.5, 35],
        [5.0, 40],
        [5.5, 45],
        [6.0, 50],
        [6.5, 55],
        [7.0, 60],
        [7.5, 65],
        [8.0, 70],
        [8.5, 75],
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
    ], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    tree_model = DecisionTreeClassifier(random_state=42)

    forest_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )

    tree_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    tree_pred = tree_model.predict(X_test)
    forest_pred = forest_model.predict(X_test)

    tree_acc = accuracy_score(y_test, tree_pred)
    forest_acc = accuracy_score(y_test, forest_pred)

    print("Decision Tree predictions:", tree_pred)
    print("Random Forest predictions:", forest_pred)
    print("y_test:", y_test)

    print()
    print("Decision Tree accuracy:", tree_acc)
    print("Random Forest accuracy:", forest_acc)

    print()
    print("Random Forest feature importances:")
    print(forest_model.feature_importances_)


if __name__ == "__main__":
    main()
