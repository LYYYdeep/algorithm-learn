import numpy as np


def predict(X, w, b):
    y_pred = X @ w + b
    return y_pred


def mean_squared_error(y_true, y_pred):
    errors = y_pred - y_true
    mse = np.mean(errors ** 2)
    return mse


def compute_gradients(X, y_true, y_pred):
    """
    计算线性回归 MSE loss 对 w 和 b 的梯度。

    Args:
        X: 特征矩阵，shape = (num_samples, num_features)
        y_true: 真实值，shape = (num_samples,)
        y_pred: 预测值，shape = (num_samples,)

    Returns:
        dw: w 的梯度，shape = (num_features,)
        db: b 的梯度，标量
    """
    num_samples = X.shape[0]
    errors = y_pred - y_true
    dw = (2 / num_samples) * (X.T @ errors)
    db = (2 / num_samples) * np.sum(errors)
    
    return dw, db


def update_params(w, b, dw, db, learning_rate):
    """
    使用梯度下降更新参数。
    """
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    return w, b


def main():
    X = np.array([
        [1, 2],
        [3, 4],
    ], dtype=float)

    y_true = np.array([5, 11], dtype=float)

    w = np.array([0, 0], dtype=float)
    b = 0.0

    learning_rate = 0.01

    y_pred = predict(X, w, b)
    loss = mean_squared_error(y_true, y_pred)
    dw, db = compute_gradients(X, y_true, y_pred)

    print("Before update")
    print("y_pred:", y_pred)
    print("loss:", loss)
    print("dw:", dw)
    print("db:", db)
    print("w:", w)
    print("b:", b)

    w, b = update_params(w, b, dw, db, learning_rate)

    print()
    print("After update")
    print("w:", w)
    print("b:", b)


if __name__ == "__main__":
    main()
