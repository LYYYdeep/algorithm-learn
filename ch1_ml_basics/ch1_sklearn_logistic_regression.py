import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([0, 0, 0, 1], dtype=int)

    model = LogisticRegression(C=1e6)

    model.fit(X, y)

    probabilities = model.predict_proba(X)
    positive_probabilities = probabilities[:, 1]
    labels = model.predict(X)

    accuracy = accuracy_score(y, labels)
    precision = precision_score(y, labels, zero_division=0)
    recall = recall_score(y, labels, zero_division=0)
    f1 = f1_score(y, labels, zero_division=0)

    print("coef:", model.coef_)
    print("intercept:", model.intercept_)
    print("probabilities:")
    print(probabilities)
    print("positive probabilities:", positive_probabilities)
    print("labels:", labels)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)


if __name__ == "__main__":
    main()
