import torch


def main():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)

    print("a:", a)
    print("b:", b)

    print()
    print("a + b:", a + b)
    print("a - b:", a - b)
    print("a * b:", a * b)
    print("a / b:", a / b)

    print()
    print("dot product:", torch.dot(a, b))

    X = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=torch.float32)

    w = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    y = X @ w

    print()
    print("X:")
    print(X)
    print("w:", w)
    print("X @ w:", y)
    print("y shape:", y.shape)

    b_scalar = 0.5
    scores = y + b_scalar

    print()
    print("scores:", scores)


if __name__ == "__main__":
    main()
