import numpy as np


def predict(X, w, b):
    """
    使用线性回归模型进行预测。

    Args:
        X: 特征矩阵，shape = (num_samples, num_features)
        w: 权重向量，shape = (num_features,)
        b: 偏置项，标量

    Returns:
        y_pred: 预测值，shape = (num_samples,)
    """
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵， shape = (num_samples, num_features)")
    
    if w.ndim != 1:
        raise ValueError("w 必须是一维向量， shape = (num_samples,)")
    
    if X.shape[1] != w.shape[0]:
        raise ValueError("X 的特征数必须等于 w 的长度")
    
    y_pred = X @ w + b
    
    return y_pred


def main():
    X = np.array([
        [80, 3, 2.5],
        [100, 4, 1.0],
        [60, 2, 5.0],
    ], dtype=float)

    w = np.array([0.8, 10.0, -2.0], dtype=float)
    b = 50.0

    y_pred = predict(X, w, b)

    print("Predictions:", y_pred)
    print("Prediction shape:", y_pred.shape)


if __name__ == "__main__":
    main()