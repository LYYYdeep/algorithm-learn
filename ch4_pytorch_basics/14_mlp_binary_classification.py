import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class StudentPassDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [0.5, 5],
            [1.0, 8],
            [1.5, 10],
            [2.0, 15],
            [2.5, 18],
            [3.0, 25],
            [3.5, 30],
            [4.0, 35],
            [4.5, 40],
            [5.0, 45],
            [1.0, 3],
            [1.5, 6],
            [2.0, 8],
            [3.0, 12],
            [4.0, 20],
            [5.0, 30],
        ], dtype=torch.float32)

        self.y = torch.tensor([
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class MLPBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=2, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, X):
        logits = self.network(X)
        return logits


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_X.size(0)
        total_loss += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        total_correct += (predictions == batch_y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    torch.manual_seed(42)

    dataset = StudentPassDataset()

    train_size = 12
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
    )

    model = MLPBinaryClassifier()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
        )

        if epoch % 100 == 0:
            print(
                f"epoch={epoch}, "
                f"train_loss={train_loss:.6f}, "
                f"train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.6f}, "
                f"val_acc={val_acc:.4f}"
            )

    new_students = torch.tensor([
        [1.0, 5],
        [3.0, 20],
        [5.0, 40],
    ], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(new_students)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

    print()
    print("New students probabilities:")
    print(probabilities)
    print("New students predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
