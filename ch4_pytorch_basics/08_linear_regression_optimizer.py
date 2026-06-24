import torch


def predict(X, w, b):
    y_pred = X @ w + b
    return y_pred


def mean_squared_error(y_true, y_pred):
    loss = torch.mean((y_pred - y_true) ** 2)
    return loss


def main():
    X = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ], dtype=torch.float32)

    y = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0], dtype=torch.float32)

    w = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    learning_rate = 0.01
    num_epochs = 1000

    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    for epoch in range(num_epochs):
        y_pred = predict(X, w, b)
        loss = mean_squared_error(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"epoch={epoch}, "
                f"loss={loss.item():.6f}, "
                f"w={w.item():.6f}, "
                f"b={b.item():.6f}"
            )

    print()
    print("Final w:", w.item())
    print("Final b:", b.item())

    test_X = torch.tensor([
        [6.0],
        [7.0],
    ], dtype=torch.float32)

    test_pred = predict(test_X, w, b)

    print("Test predictions:", test_pred)


if __name__ == "__main__":
    main()
