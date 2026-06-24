import torch
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


def main():
    dataset = StudentDataset()

    print("Dataset length:", len(dataset))

    x0, y0 = dataset[0]

    print("First sample X:", x0)
    print("First sample y:", y0)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    print()
    print("Batches:")

    for batch_X, batch_y in dataloader:
        print("batch_X:", batch_X)
        print("batch_X shape:", batch_X.shape)
        print("batch_y:", batch_y)
        print("batch_y shape:", batch_y.shape)
        print()


if __name__ == "__main__":
    main()
