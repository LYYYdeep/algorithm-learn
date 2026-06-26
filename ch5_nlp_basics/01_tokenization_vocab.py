def build_vocab(tokenized_texts):
    """
    根据已经分好词的文本构建词表。

    Args:
        tokenized_texts: list[list[str]]
            例如 [["我", "喜欢", "机器", "学习"], ["我", "喜欢", "NLP"]]

    Returns:
        vocab: dict[str, int]
    """
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
    """
    把 token 列表转成 id 列表。
    """
    unk_id = vocab["[UNK]"]

    ids = []

    for token in tokens:
        token_id = vocab.get(token, unk_id)
        ids.append(token_id)

    return ids


def pad_or_truncate(ids, max_len, pad_id=0):
    """
    把 id 序列补齐或截断到 max_len。
    """
    if len(ids) > max_len:
        return ids[:max_len]

    num_padding = max_len - len(ids)

    return ids + [pad_id] * num_padding


def main():
    tokenized_texts = [
        ["我", "喜欢", "机器", "学习"],
        ["我", "喜欢", "NLP"],
        ["深度", "学习", "很", "有趣"],
    ]

    vocab = build_vocab(tokenized_texts)

    print("Vocab:")
    print(vocab)

    tokens = ["我", "喜欢", "深度", "学习"]
    ids = encode(tokens, vocab)

    print()
    print("Tokens:", tokens)
    print("Ids:", ids)

    unknown_tokens = ["我", "喜欢", "大模型"]
    unknown_ids = encode(unknown_tokens, vocab)

    print()
    print("Unknown tokens:", unknown_tokens)
    print("Unknown ids:", unknown_ids)

    max_len = 6

    padded_ids = pad_or_truncate(ids, max_len=max_len)

    print()
    print("Padded ids:", padded_ids)

    long_ids = [2, 3, 4, 5, 6, 7, 8]
    truncated_ids = pad_or_truncate(long_ids, max_len=max_len)

    print()
    print("Long ids:", long_ids)
    print("Truncated ids:", truncated_ids)


if __name__ == "__main__":
    main()
