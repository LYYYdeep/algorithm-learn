import torch


def main():
    x = torch.tensor([1, 2, 3])
    X = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])

    print("x:", x)
    print("x shape:", x.shape)
    print("x ndim:", x.ndim)
    print("x dtype:", x.dtype)
    print("x device:", x.device)

    print()
    print("X:")
    print(X)
    print("X shape:", X.shape)
    print("X ndim:", X.ndim)
    print("X dtype:", X.dtype)
    print("X device:", X.device)


if __name__ == "__main__":
    main()