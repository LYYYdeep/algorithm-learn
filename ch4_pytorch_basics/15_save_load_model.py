import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in dataloader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss={avg_loss:.6f}")


def predict(model, X):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

    return probabilities, predictions


def main():
    torch.manual_seed(42)

    dataset = StudentPassDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    model = MLPBinaryClassifier()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=500,
    )

    test_X = torch.tensor([
        [1.0, 5],
        [3.0, 20],
        [5.0, 40],
    ], dtype=torch.float32)

    probabilities, predictions = predict(model, test_X)

    print()
    print("Before saving:")
    print("Probabilities:", probabilities)
    print("Predictions:", predictions)

    save_path = "ch4_pytorch_basics/mlp_student_pass.pt"

    torch.save(model.state_dict(), save_path)

    print()
    print(f"Model saved to: {save_path}")

    loaded_model = MLPBinaryClassifier()
    loaded_model.load_state_dict(torch.load(save_path))

    loaded_probabilities, loaded_predictions = predict(loaded_model, test_X)

    print()
    print("After loading:")
    print("Probabilities:", loaded_probabilities)
    print("Predictions:", loaded_predictions)


if __name__ == "__main__":
    main()
