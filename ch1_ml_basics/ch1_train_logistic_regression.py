import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w, b):
    z = X @ w + b
    probabilities = sigmoid(z)
    return probabilities


def predict_label(X, w, b, threshold=0.5):
    probabilities = predict_proba(X, w, b)
    labels = (probabilities >= threshold).astype(int)
    return labels


def binary_cross_entropy(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    losses = -(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    return np.mean(losses)


def compute_gradients(X, y_true, y_prob):
    """
    计算逻辑回归 BCE loss 对 w 和 b 的梯度。

    Args:
        X: shape = (num_samples, num_features)
        y_true: shape = (num_samples,)
        y_prob: shape = (num_samples,)

    Returns:
        dw: shape = (num_features,)
        db: 标量
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    losses = -(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )
    
    return np.mean(losses)


def compute_gradients(X, y_true, y_prob):
    """
    计算逻辑回归 BCE loss 对 w 和 b 的梯度。

    Args:
        X: shape = (num_samples, num_features)
        y_true: shape = (num_samples,)
        y_prob: shape = (num_samples,)

    Returns:
        dw: shape = (num_features,)
        db: 标量
    """
    num_samples = X.shape[0]

    errors = y_prob - y_true

    dw = (1 / num_samples) * (X.T @ errors)
    db = (1 / num_samples) * np.sum(errors)

    return dw, db

def update_params(w, b, dw, db, learning_rate):
    """
    梯度下降更新参数。
    """
    w= w - learning_rate * dw
    b = b - learning_rate * db
    
    return w, b


def train_logistic_regression(X, y, learning_rate, num_epochs):
    """
    使用梯度下降训练逻辑回归。
    """
    num_features = X.shape[1]
    
    w = np.zeros(num_features)
    b = 0.0
    
    for epoch in range(num_epochs):
        y_prob = predict_proba(X, w, b)
        loss = binary_cross_entropy(y, y_prob)
        dw, db = compute_gradients(X, y, y_prob)
        w, b = update_params(w, b, dw, db, learning_rate)

        if epoch % 200 == 0:
            labels = predict_label(X, w, b)
            accuracy = np.mean(labels == y)
            print(
                f"epoch={epoch}, "
                f"loss={loss:.6f}, "
                f"accuracy={accuracy:.4f}, "
                f"w={w}, "
                f"b={b:.6f}"
            )

    return w, b


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([0, 0, 0, 1], dtype=float)

    learning_rate = 0.5
    num_epochs = 2000

    w, b = train_logistic_regression(X, y, learning_rate, num_epochs)

    probabilities = predict_proba(X, w, b)
    labels = predict_label(X, w, b)

    print("Final w:", w)
    print("Final b:", b)
    print("Probabilities:", probabilities)
    print("Labels:", labels)


if __name__ == "__main__":
    main()
