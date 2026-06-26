import torch
from torch import nn


def main():
    torch.manual_seed(42)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "我": 2,
        "喜欢": 3,
        "机器": 4,
        "学习": 5,
        "NLP": 6,
    }

    vocab_size = len(vocab)
    embedding_dim = 4

    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=0,
    )

    input_ids = torch.tensor([
        [2, 3, 4, 5],
        [2, 3, 6, 0],
    ], dtype=torch.long)

    embedded = embedding(input_ids)

    print("Input ids:")
    print(input_ids)
    print("Input ids shape:", input_ids.shape)

    print()
    print("Embedding weight shape:")
    print(embedding.weight.shape)

    print()
    print("Embedded:")
    print(embedded)
    print("Embedded shape:", embedded.shape)

    print()
    print("Embedding for token id 2:")
    print(embedding.weight[2])


if __name__ == "__main__":
    main()
