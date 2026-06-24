import numpy as np



def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 MSE。

    Args:
        y_true: 真实值，shape = (num_samples,)
        y_pred: 预测值，shape = (num_samples,)

    Returns:
        mse: 均方误差，标量
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


def main():
    y_true = np.array([100, 150, 200], dtype=float)
    y_pred = np.array([110, 140, 210], dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    print("MSE:", mse)


if __name__ == "__main__":
    main()