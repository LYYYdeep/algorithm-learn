import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class StudentDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ], dtype=torch.float32)

        self.y = torch.tensor([
            [3.0],
            [5.0],
            [7.0],
            [9.0],
            [11.0],
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, X):
        y_pred = self.linear(X)
        return y_pred


def main():
    dataset = StudentDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    model = LinearRegressionModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_X, batch_y in dataloader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss = epoch_loss / len(dataset)

        if epoch % 100 == 0:
            weight = model.linear.weight.item()
            bias = model.linear.bias.item()

            print(
                f"epoch={epoch}, "
                f"loss={epoch_loss:.6f}, "
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
