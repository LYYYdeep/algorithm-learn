import torch
from torch import nn


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    vocab_size = 10
    embedding_dim = 4
    hidden_size = 6

    input_ids = torch.tensor([
        [2, 3, 4, 5, 0],
        [2, 6, 7, 0, 0],
    ], dtype=torch.long)

    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=0,
    )

    rnn = nn.RNN(
        input_size=embedding_dim,
        hidden_size=hidden_size,
        batch_first=True,
    )

    embedded = embedding(input_ids)

    output, hidden = rnn(embedded)

    print("input_ids shape:", input_ids.shape)
    print("embedded shape:", embedded.shape)
    print("output shape:", output.shape)
    print("hidden shape:", hidden.shape)

    print()
    print("Last hidden from output:")
    print(output[:, -1, :])

    print()
    print("Hidden squeezed:")
    print(hidden.squeeze(0))


if __name__ == "__main__":
    main()
