import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def evaluate_tree(name, model, X_train, X_test, y_train, y_test, feature_names):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print()
    print(name)
    print("-" * len(name))
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)

    rules = export_text(model, feature_names=feature_names)
    print("Rules:")
    print(rules)


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
        0,  # 异常点
        1,
        1, 1,
        0,  # 异常点
        1,
    ], dtype=int)

    feature_names = ["study_hours", "exercises"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    models = [
        (
            "Default Tree",
            DecisionTreeClassifier(random_state=42),
        ),
        (
            "Max Depth 2",
            DecisionTreeClassifier(max_depth=2, random_state=42),
        ),
        (
            "Min Samples Leaf 2",
            DecisionTreeClassifier(min_samples_leaf=2, random_state=42),
        ),
        (
            "Entropy Tree",
            DecisionTreeClassifier(criterion="entropy", random_state=42),
        ),
    ]

    for name, model in models:
        evaluate_tree(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
        )


if __name__ == "__main__":
    main()
