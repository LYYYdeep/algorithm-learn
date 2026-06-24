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


def find_best_split(feature_values, labels):
    """
    给定一维特征和标签，寻找最佳划分阈值。

    Args:
        feature_values: 一维特征列表，例如 [1.0, 1.5, 2.0, 3.0]
        labels: 标签列表，例如 [0, 0, 1, 1]

    Returns:
        best_threshold: 最佳阈值
        best_gini: 最小加权 Gini
    """
    if len(feature_values) != len(labels):
        raise ValueError("feature_values and labels must have the same length.")
    
    if len(feature_values) < 2:
        raise ValueError("At least two samples are required to find a split.")
    
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
        
        print(
            f"threshold={threshold:.2f}, "
            f"left={left_labels}, "
            f"right={right_labels}, "
            f"weighted_gini={current_gini:.4f}"
        )

        if current_gini < best_gini:
            best_gini = current_gini
            best_threshold = threshold

    return best_threshold, best_gini


def main():
    feature_values = [1.0, 1.5, 2.0, 3.0, 3.5, 4.0]
    labels = [0, 0, 0, 1, 1, 1]

    best_threshold, best_gini = find_best_split(feature_values, labels)

    print("Best threshold:", best_threshold)
    print("Best weighted Gini:", best_gini)


if __name__ == "__main__":
    main()
