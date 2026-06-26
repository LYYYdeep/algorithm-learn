import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def build_vocab(tokenized_texts):
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
    }

    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def encode(tokens, vocab):
    unk_id = vocab["[UNK]"]
    return [vocab.get(token, unk_id) for token in tokens]


def pad_or_truncate(ids, max_len, pad_id=0):
    if len(ids) > max_len:
        ids = ids[:max_len]

    num_padding = max_len - len(ids)

    return ids + [pad_id] * num_padding


class TextClassificationDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len):
        self.input_ids = []
        self.labels = []

        for tokens, label in zip(tokenized_texts, labels):
            ids = encode(tokens, vocab)
            padded_ids = pad_or_truncate(
                ids,
                max_len=max_len,
                pad_id=vocab["[PAD]"],
            )

            self.input_ids.append(padded_ids)
            self.labels.append(label)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]


class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        output, hidden = self.rnn(embedded)

        sentence_vector = hidden.squeeze(0)

        logits = self.classifier(sentence_vector)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, labels in dataloader:
        logits = model(input_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def predict_texts(model, tokenized_texts, vocab, max_len):
    input_ids_list = []

    for tokens in tokenized_texts:
        ids = encode(tokens, vocab)
        padded_ids = pad_or_truncate(
            ids,
            max_len=max_len,
            pad_id=vocab["[PAD]"],
        )
        input_ids_list.append(padded_ids)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

    return probabilities, predictions


def main():
    torch.manual_seed(42)

    tokenized_texts = [
        ["我", "喜欢", "机器", "学习"],
        ["深度", "学习", "很", "有趣"],
        ["这个", "模型", "效果", "很好"],
        ["我", "讨厌", "这个", "错误"],
        ["这个", "结果", "很", "糟糕"],
        ["模型", "训练", "失败"],
        ["我", "喜欢", "NLP"],
        ["这个", "方法", "很", "有效"],
        ["结果", "不好"],
        ["错误", "很多"],
    ]

    labels = [
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
    ]

    vocab = build_vocab(tokenized_texts)

    max_len = 6

    dataset = TextClassificationDataset(
        tokenized_texts=tokenized_texts,
        labels=labels,
        vocab=vocab,
        max_len=max_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    model = RNNTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=8,
        hidden_size=16,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 200

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            dataloader,
            criterion,
            optimizer,
        )

        if epoch % 20 == 0:
            print(
                f"epoch={epoch}, "
                f"loss={train_loss:.6f}, "
                f"accuracy={train_acc:.4f}"
            )

    test_texts = [
        ["我", "喜欢", "这个", "模型"],
        ["错误", "很多"],
        ["这个", "方法", "很", "糟糕"],
    ]

    probabilities, predictions = predict_texts(
        model=model,
        tokenized_texts=test_texts,
        vocab=vocab,
        max_len=max_len,
    )

    print()
    print("Test probabilities:")
    print(probabilities)
    print("Test predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
