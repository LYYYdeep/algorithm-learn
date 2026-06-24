import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    X = np.array([
        [0.5, 5],
        [1.0, 8],
        [1.5, 10],
        [2.0, 15],
        [2.5, 18],
        [3.0, 25],
        [3.5, 30],
        [4.0, 35],
        [4.5, 40],
        [5.0, 45],
        [1.0, 3],
        [1.5, 6],
        [2.0, 8],
        [3.0, 12],
        [4.0, 20],
        [5.0, 30],
    ], dtype=float)

    y = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        1, 1, 0, 0,
        0, 1, 1, 1,
    ], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()

    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print("Scaler mean:", scaler.mean_)
    print("Scaler scale:", scaler.scale_)

    print()
    print("X_test original:")
    print(X_test)

    print()
    print("X_test scaled:")
    print(X_test_scaled)

    print()
    print("y_test:", y_test)
    print("y_prob:", y_prob)
    print("y_pred:", y_pred)

    print()
    print("Metrics")
    print("-------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(matrix)

    new_students = np.array([
        [1.0, 5],
        [3.0, 20],
        [5.0, 40],
    ], dtype=float)

    new_students_scaled = scaler.transform(new_students)

    new_prob = model.predict_proba(new_students_scaled)[:, 1]
    new_pred = model.predict(new_students_scaled)

    print()
    print("New Students original:")
    print(new_students)

    print()
    print("New Students scaled:")
    print(new_students_scaled)

    print()
    print("Pass probabilities:", new_prob)
    print("Predictions:", new_pred)


if __name__ == "__main__":
    main()
