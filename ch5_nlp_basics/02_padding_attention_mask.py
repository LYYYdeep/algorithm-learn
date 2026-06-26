def pad_or_truncate_with_mask(ids, max_len, pad_id=0):
    """
    对 token ids 进行 padding / truncation，并生成 attention mask。

    attention_mask:
        1 表示真实 token
        0 表示 padding token
    """
    if len(ids) > max_len:
        truncated_ids = ids[:max_len]
        attention_mask = [1] * max_len
        return truncated_ids, attention_mask

    num_padding = max_len - len(ids)

    padded_ids = ids + [pad_id] * num_padding
    attention_mask = [1] * len(ids) + [0] * num_padding

    return padded_ids, attention_mask


def main():
    examples = [
        [2, 3, 4, 5],
        [2, 3],
        [7, 8, 9, 10, 11, 12],
    ]

    max_len = 4

    for ids in examples:
        padded_ids, attention_mask = pad_or_truncate_with_mask(
            ids,
            max_len=max_len,
            pad_id=0,
        )

        print("Original ids:", ids)
        print("Padded ids:", padded_ids)
        print("Attention mask:", attention_mask)
        print()


if __name__ == "__main__":
    main()
