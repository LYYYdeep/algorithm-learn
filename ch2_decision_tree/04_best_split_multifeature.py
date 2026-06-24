from collections import Counter


def gini_impurity(labels):
    if len(labels) == 0:
        return 0.0

    total = len(labels)
    counts = Counter(labels)

    sum_squared_probs = 0.0

    for count in counts.values():
        prob = count / total
        sum_squared_probs += prob ** 2

    return 1 - sum_squared_probs


def weighted_gini(left_labels, right_labels):
    total = len(left_labels) + len(right_labels)

    if total == 0:
        return 0.0

    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    return (
        left_weight * gini_impurity(left_labels)
        + right_weight * gini_impurity(right_labels)
    )


def find_best_split_for_feature(feature_values, labels):
    """
    寻找单个特征的最佳阈值。
    """
    if len(feature_values) != len(labels):
        raise ValueError("feature_values 和 labels 长度必须相同")

    sorted_pairs = sorted(zip(feature_values, labels), key=lambda x: x[0])

    sorted_values = [pair[0] for pair in sorted_pairs]
    sorted_labels = [pair[1] for pair in sorted_pairs]

    candidate_thresholds = []

    for i in range(len(sorted_values) - 1):
        if sorted_values[i] != sorted_values[i + 1]:
            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
            candidate_thresholds.append(threshold)

    best_threshold = None
    best_gini = float("inf")

    for threshold in candidate_thresholds:
        left_labels = []
        right_labels = []

        for value, label in zip(sorted_values, sorted_labels):
            if value <= threshold:
                left_labels.append(label)
            else:
                right_labels.append(label)

        current_gini = weighted_gini(left_labels, right_labels)

        if current_gini < best_gini:
            best_gini = current_gini
            best_threshold = threshold

    return best_threshold, best_gini


def find_best_split(X, y):
    """
    在所有特征中寻找最佳分裂。

    Args:
        X: 二维列表，shape = (num_samples, num_features)
        y: 标签列表

    Returns:
        best_feature_index: 最佳特征下标
        best_threshold: 最佳阈值
        best_gini: 最小加权 Gini
    """
    if len(X) != len(y):
        raise ValueError("X 和 y 的样本数量必须相同")

    if len(X) == 0:
        raise ValueError("X 不能为空")

    num_features = len(X[0])

    best_feature_index = None
    best_threshold = None
    best_gini = float("inf")

    for feature_index in range(num_features):
        feature_values = []

        for row in X:
            feature_values.append(row[feature_index])

        threshold, gini = find_best_split_for_feature(feature_values, y)

        print(
            f"feature_index={feature_index}, "
            f"best_threshold={threshold}, "
            f"best_gini={gini:.4f}"
        )

        if gini < best_gini:
            best_gini = gini
            best_threshold = threshold
            best_feature_index = feature_index

    return best_feature_index, best_threshold, best_gini


def main():
    X = [
        [1.0, 5],
        [1.5, 8],
        [2.0, 10],
        [3.0, 20],
        [3.5, 25],
        [4.0, 30],
    ]

    y = [0, 0, 0, 1, 1, 1]

    feature_names = ["study_hours", "exercises"]

    best_feature_index, best_threshold, best_gini = find_best_split(X, y)

    print()
    print("Best feature:", feature_names[best_feature_index])
    print("Best threshold:", best_threshold)
    print("Best weighted Gini:", best_gini)


if __name__ == "__main__":
    main()
