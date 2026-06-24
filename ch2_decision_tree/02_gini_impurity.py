from collections import Counter


def gini_impurity(labels):
    """
    计算一个节点的 Gini impurity。

    Args:
        labels: 当前节点中的类别标签列表，例如 [0, 0, 1, 1]

    Returns:
        gini: Gini 不纯度
    """
    if len(labels) == 0:
        return 0.0
    
    total = len(labels)
    counts = Counter(labels)
    
    sum_squared_probs = 0.0
    
    for count in counts.values():
        prob = count / total
        sum_squared_probs += prob ** 2
        
    geni = 1 - sum_squared_probs
    return geni


def weighted_gini(left_labels, right_labels):
    """
    计算一次二叉划分后的加权 Gini。
    """
    total = len(left_labels) + len(right_labels)
    
    if total == 0:
        return 0.0
    
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    
    geni = (
        left_weight * gini_impurity(left_labels)
        + right_weight * gini_impurity(right_labels)
    )
    
    return geni


def main():
    labels = [0, 0, 0, 1, 1, 1]
    print("Gini:", gini_impurity(labels))

    pure_labels = [1, 1, 1, 1]
    print("Pure Gini:", gini_impurity(pure_labels))

    left_labels = [0, 0, 0]
    right_labels = [1, 1, 1]
    print("Weighted Gini perfect split:", weighted_gini(left_labels, right_labels))

    left_labels = [0, 1, 1]
    right_labels = [0, 0, 1]
    print("Weighted Gini bad split:", weighted_gini(left_labels, right_labels))


if __name__ == "__main__":
    main()
