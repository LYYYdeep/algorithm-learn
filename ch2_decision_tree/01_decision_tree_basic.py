import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    X = np.array([
        [1.0, 5],
        [2.0, 10],
        [3.0, 20],
        [4.0, 30],
        [1.5, 8],
        [3.5, 25],
    ], dtype=float)

    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)

    feature_names = ["study_hours", "exercises"]

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,
        random_state=42,
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print("Predictions:", y_pred)
    print("Accuracy:", accuracy)

    tree_rules = export_text(
        model,
        feature_names=feature_names,
    )

    print()
    print("Decision Tree Rules:")
    print(tree_rules)

    new_students = np.array([
        [1.0, 6],
        [2.5, 12],
        [4.0, 20],
    ], dtype=float)

    new_pred = model.predict(new_students)
    print("New Predictions:", new_pred)


if __name__ == "__main__":
    main()
