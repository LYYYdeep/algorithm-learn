import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
    
def main():
    X = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ], dtype=torch.float32)

    y = torch.tensor([
        [3.0],
        [5.0],
        [7.0],
        [9.0],
        [11.0],
    ], dtype=torch.float32)

    model = LinearRegressionModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            weight = model.linear.weight.item()
            bias = model.linear.bias.item()

            print(
                f"epoch={epoch}, "
                f"loss={loss.item():.6f}, "
                f"weight={weight:.6f}, "
                f"bias={bias:.6f}"
            )

    print()
    print("Learned weight:", model.linear.weight.item())
    print("Learned bias:", model.linear.bias.item())

    test_X = torch.tensor([
        [6.0],
        [7.0],
    ], dtype=torch.float32)

    test_pred = model(test_X)

    print("Test predictions:", test_pred)


if __name__ == "__main__":
    main()
    