import numpy as np


def predict(X, w, b):
    y_pred = X @ w + b
    return y_pred


def mean_squared_error(y_true, y_pred):
    errors = y_pred - y_true
    mse = np.mean(errors ** 2)
    return mse


def compute_gradients(X, y_true, y_pred):
    num_samples = X.shape[0]
    errors = y_pred - y_true

    dw = (2 / num_samples) * (X.T @ errors)
    db = (2 / num_samples) * np.sum(errors)

    return dw, db


def update_params(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


def train_linear_regression(X, y, learning_rate, num_epochs):
    """
    使用梯度下降训练线性回归模型。

    Args:
        X: 特征矩阵，shape = (num_samples, num_features)
        y: 真实值，shape = (num_samples,)
        learning_rate: 学习率
        num_epochs: 训练轮数

    Returns:
        w: 训练后的权重
        b: 训练后的偏置
    """
    num_features = X.shape[1]
    
    
    w = np.zeros(num_features)
    b = 0.0
    
    for epoch in range(num_epochs):
        y_pred = predict(X, w, b)
        loss = mean_squared_error(y, y_pred)
        dw, db = compute_gradients(X, y, y_pred)
        w, b = update_params(w, b, dw, db, learning_rate)

        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss={loss:.6f}, w={w}, b={b:.6f}")

    return w, b


def main():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
    ], dtype=float)

    y = np.array([3, 5, 7, 9, 11], dtype=float)

    learning_rate = 0.01
    num_epochs = 5000

    w, b = train_linear_regression(X, y, learning_rate, num_epochs)

    print("Final w:", w)
    print("Final b:", b)

    test_X = np.array([
        [6],
        [7],
    ], dtype=float)

    predictions = predict(test_X, w, b)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
