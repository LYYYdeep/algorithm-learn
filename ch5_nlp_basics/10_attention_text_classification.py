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
    padded_ids = ids + [pad_id] * num_padding
    attention_mask = [1] * len(ids) + [0] * num_padding

    return padded_ids, attention_mask


class TextClassificationDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for tokens, label in zip(tokenized_texts, labels):
            ids = encode(tokens, vocab)
            padded_ids, attention_mask = pad_or_truncate(
                ids,
                max_len=max_len,
                pad_id=vocab["[PAD]"],
            )

            self.input_ids.append(padded_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(label)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.attention_masks[index],
            self.labels[index],
        )


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.attention_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
        )

    def forward(self, embedded, attention_mask):
        scores = self.attention_layer(embedded)
        scores = scores.squeeze(-1)

        scores = scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=1)

        sentence_vector = (
            embedded * attention_weights.unsqueeze(-1)
        ).sum(dim=1)

        return sentence_vector, attention_weights


class AttentionTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.attention_pooling = AttentionPooling(
            embedding_dim=embedding_dim,
        )

        self.classifier = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)

        sentence_vector, attention_weights = self.attention_pooling(
            embedded,
            attention_mask,
        )

        logits = self.classifier(sentence_vector)

        return logits, attention_weights


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, attention_mask, labels in dataloader:
        logits, _ = model(input_ids, attention_mask)
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
    attention_masks = []

    for tokens in tokenized_texts:
        ids = encode(tokens, vocab)
        padded_ids, attention_mask = pad_or_truncate(
            ids,
            max_len=max_len,
            pad_id=vocab["[PAD]"],
        )

        input_ids_list.append(padded_ids)
        attention_masks.append(attention_mask)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)

    model.eval()

    with torch.no_grad():
        logits, attention_weights = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

    return probabilities, predictions, attention_weights


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

    model = AttentionTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=8,
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

    probabilities, predictions, attention_weights = predict_texts(
        model=model,
        tokenized_texts=test_texts,
        vocab=vocab,
        max_len=max_len,
    )

    print()
    print("Test probabilities:")
    print(probabilities)

    print()
    print("Test predictions:")
    print(predictions)

    print()
    print("Attention weights:")
    print(attention_weights)

    print()
    print("Test texts:")
    for tokens in test_texts:
        print(tokens)


if __name__ == "__main__":
    main()
