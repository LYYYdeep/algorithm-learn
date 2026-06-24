import numpy as np


def sigmoid(z):
    """
    计算 sigmoid 函数。
    """
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w, b):
    """
    预测正类概率。
    """
    z = X @ w + b
    return sigmoid(z)


def predict_label(X, w, b, threshold=0.5):
    """
    根据概率和阈值预测类别标签。
    """
    probabilities = predict_proba(X, w, b)
    labels = (probabilities >= threshold).astype(int)
    
    return labels

def main():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
    ], dtype=float)

    w = np.array([1.0, -0.5], dtype=float)
    b = 0.0

    probabilities = predict_proba(X, w, b)
    labels = predict_label(X, w, b, threshold=0.5)

    print("Probabilities:", probabilities)
    print("Labels:", labels)


if __name__ == "__main__":
    main()