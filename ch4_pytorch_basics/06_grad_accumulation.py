import torch


def main():
    x = torch.tensor(2.0, requires_grad=True)

    y = x ** 2
    y.backward()

    print("After first backward:")
    print("x.grad:", x.grad)

    y = x ** 2
    y.backward()

    print()
    print("After second backward:")
    print("x.grad:", x.grad)

    x.grad.zero_()

    print()
    print("After zero grad:")
    print("x.grad:", x.grad)

    y = x ** 2
    y.backward()

    print()
    print("After third backward:")
    print("x.grad:", x.grad)


if __name__ == "__main__":
    main()
