import numpy as np


def predict(X, w, b):
    """
    使用线性回归模型进行预测。
    """
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵")

    if w.ndim != 1:
        raise ValueError("w 必须是一维向量")

    if X.shape[1] != w.shape[0]:
        raise ValueError("X 的特征数必须等于 w 的长度")

    y_pred = X @ w + b

    return y_pred


def mean_squared_error(y_true, y_pred):
    """
    计算 MSE 损失。
    """
    if y_true.ndim != 1:
        raise ValueError("y_true 必须是一维数组")

    if y_pred.ndim != 1:
        raise ValueError("y_pred 必须是一维数组")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 和 y_pred 的 shape 必须相同")

    if y_true.shape[0] == 0:
        raise ValueError("输入数组不能为空")

    errors = y_pred - y_true
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)

    return mse


def forward(X, w, b, y_true):
    """
    完成线性回归的一次前向计算。

    Args:
        X: 特征矩阵，shape = (num_samples, num_features)
        w: 权重向量，shape = (num_features,)
        b: 偏置项，标量
        y_true: 真实值，shape = (num_samples,)

    Returns:
        y_pred: 预测值
        loss: MSE 损失
    """
    y_pred = predict(X, w, b)
    loss = mean_squared_error(y_true, y_pred)

    return y_pred, loss


def main():
    X = np.array([
        [80, 3, 2.5],
        [100, 4, 1.0],
        [60, 2, 5.0],
    ], dtype=float)

    y_true = np.array([140, 170, 100], dtype=float)

    w = np.array([0.8, 10.0, -2.0], dtype=float)
    b = 50.0

    y_pred, loss = forward(X, w, b, y_true)

    print("y_pred:", y_pred)
    print("loss:", loss)


if __name__ == "__main__":
    main()
