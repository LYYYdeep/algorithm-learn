import math
from collections import Counter


def entropy(labels):
    """
    计算信息熵。
    """
    if len(labels) == 0:
        return 0
    
    total = len(labels)
    counts = Counter(labels)
    
    result = 0.0
    
    for count in counts.values():
        prob = count / total
        
        if prob > 0:
            result -= prob * math.log2(prob)
            
    return result


def weighted_entropy(left_labels, right_labels):
    """
    计算划分后的加权 entropy。
    """
    total = len(left_labels) + len(right_labels)

    if total == 0:
        return 0.0

    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    return (
        left_weight * entropy(left_labels)
        + right_weight * entropy(right_labels)
    )


def information_gain(parent_labels, left_labels, right_labels):
    """
    计算信息增益。
    """
    parent_entropy = entropy(parent_labels)
    child_entropy = weighted_entropy(left_labels, right_labels)
    
    gain = parent_entropy - child_entropy
    
    return gain


def main():
    labels = [0, 0, 0, 1, 1, 1]
    print("Parent entropy:", entropy(labels))

    pure_labels = [1, 1, 1, 1]
    print("Pure entropy:", entropy(pure_labels))

    left_labels = [0, 0, 0]
    right_labels = [1, 1, 1]
    print("Information gain perfect split:",
          information_gain(labels, left_labels, right_labels))

    left_labels = [0, 1, 1]
    right_labels = [0, 0, 1]
    print("Information gain bad split:",
          information_gain(labels, left_labels, right_labels))


if __name__ == "__main__":
    main()
