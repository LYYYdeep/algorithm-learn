import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


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
    ], dtype=float)

    y = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1,
        0,  # 故意加入一个异常点：学习很多但没通过
        1,
    ], dtype=int)

    feature_names = ["study_hours", "exercises"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )

    deep_tree = DecisionTreeClassifier(
        random_state=42,
    )

    shallow_tree = DecisionTreeClassifier(
        max_depth=2,
        random_state=42,
    )

    deep_tree.fit(X_train, y_train)
    shallow_tree.fit(X_train, y_train)

    for name, model in [
        ("Deep Tree", deep_tree),
        ("Shallow Tree", shallow_tree),
    ]:
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


if __name__ == "__main__":
    main()
