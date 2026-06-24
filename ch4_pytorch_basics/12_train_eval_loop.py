import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class StudentDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
        ], dtype=torch.float32)

        self.y = 2 * self.X + 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, X):
        return self.linear(X)


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples

    return avg_loss


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples

    return avg_loss


def main():
    torch.manual_seed(42)

    dataset = StudentDataset()

    train_size = 8
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
    )

    model = LinearRegressionModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
        )

        if epoch % 100 == 0:
            weight = model.linear.weight.item()
            bias = model.linear.bias.item()

            print(
                f"epoch={epoch}, "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"weight={weight:.6f}, "
                f"bias={bias:.6f}"
            )

    print()
    print("Learned weight:", model.linear.weight.item())
    print("Learned bias:", model.linear.bias.item())


if __name__ == "__main__":
    main()
