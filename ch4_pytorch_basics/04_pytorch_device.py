import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    device = get_device()

    print("Using device:", device)

    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = x.to(device)

    print("x:", x)
    print("x device:", x.device)

    y = torch.tensor([4, 5, 6], dtype=torch.float32, device=device)

    print("y:", y)
    print("y device:", y.device)

    z = x + y

    print("z:", z)
    print("z device:", z.device)


if __name__ == "__main__":
    main()
