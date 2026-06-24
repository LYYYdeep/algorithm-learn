import numpy as np


def binary_cross_entropy(y_true, y_prob):
    if y_true.ndim != 1:
        raise ValueError("y_true 必须是一维数组")

    if y_prob.ndim != 1:
        raise ValueError("y_prob 必须是一维数组")

    if y_true.shape != y_prob.shape:
        raise ValueError("y_true 和 y_prob 的 shape 必须相同")

    if y_true.shape[0] == 0:
        raise ValueError("输入数组不能为空")
    
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    losses = -(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )
    
    loss = np.mean(losses)
    
    return loss


def main():
    y_true = np.array([1, 0, 1, 0], dtype=float)
    y_prob = np.array([0.9, 0.1, 0.8, 0.2], dtype=float)

    loss = binary_cross_entropy(y_true, y_prob)

    print("BCE Loss:", loss)


if __name__ == "__main__":
    main()