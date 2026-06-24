import torch


def main():
    x = torch.tensor(3.0, requires_grad=True)

    y = x ** 2

    y.backward()

    print("x:", x)
    print("y:", y)
    print("x.grad:", x.grad)

def example_two():
    x = torch.tensor(2.0, requires_grad=True)

    y = 3 * x ** 2 + 2 * x + 1

    y.backward()

    print("Example two")
    print("x:", x)
    print("y:", y)
    print("x.grad:", x.grad)



if __name__ == "__main__":
    main()
    example_two()