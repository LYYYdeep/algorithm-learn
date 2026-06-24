def calculate_accuracy(y_true, y_pred):
    """
    计算分类准确率。

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        accuracy: 准确率
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的长度必须相同。")
    
    if len(y_true) == 0:
        raise ValueError("y_true 和 y_pred 不能为空。")
    
    correct = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct += 1
            
    accuracy = correct / len(y_true)
    
    return accuracy


def build_confusion_matrix(y_true, y_pred, num_classes):
    """
    构建多分类混淆矩阵。

    Args:
        y_true: 真实标签列表，标签范围为 0 到 num_classes - 1
        y_pred: 预测标签列表，标签范围为 0 到 num_classes - 1
        num_classes: 类别数量

    Returns:
        matrix: num_classes x num_classes 的二维列表
                matrix[i][j] 表示真实类别为 i、预测类别为 j 的样本数
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的长度必须相同。")
    
    if len(y_true) == 0:
        raise ValueError("y_true 和 y_pred 不能为空。")
    
    if num_classes <= 0:
        raise ValueError("num_classes 必须为正整数。")
    
    matrix = []

    for _ in range(num_classes):
        row = []
        for _ in range(num_classes):
            row.append(0)
        matrix.append(row)
        
    for true_label, pred_label in zip(y_true, y_pred):
        if not 0 <= true_label < num_classes:
            raise ValueError(f"真实标签 {true_label} 超出范围 [0, {num_classes - 1}]。")
        
        if not 0 <= pred_label < num_classes:
            raise ValueError(f"预测标签 {pred_label} 超出范围 [0, {num_classes - 1}]。")

        matrix[true_label][pred_label] += 1
        
    return matrix

def calculate_multiclass_metrics(matrix):
    """
    根据混淆矩阵计算每个类别的 precision、recall、f1，
    以及 macro_f1、micro_f1。

    Args:
        matrix: 混淆矩阵，matrix[true_label][pred_label]

    Returns:
        result: 一个字典，包含：
            - per_class: 每个类别的指标
            - macro_f1
            - micro_f1
    """
    num_classes = len(matrix)

    per_class = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_id in range(num_classes):
        tp = matrix[class_id][class_id]

        fp = 0
        for row in range(num_classes):
            if row != class_id:
                fp += matrix[row][class_id]

        fn = 0
        for col in range(num_classes):
            if col != class_id:
                fn += matrix[class_id][col]

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class.append({
            "class_id": class_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = 0.0
    for item in per_class:
        macro_f1 += item["f1"]
    macro_f1 /= num_classes

    if total_tp + total_fp == 0:
        micro_precision = 0.0
    else:
        micro_precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        micro_recall = 0.0
    else:
        micro_recall = total_tp / (total_tp + total_fn)

    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (
            micro_precision + micro_recall
        )

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


def print_confusion_matrix(matrix):
    """
    打印混淆矩阵。
    """
    for row in matrix:
        print(row)


def main():
    y_true = [0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0, 2]
    num_classes = 3

    accuracy = calculate_accuracy(y_true, y_pred)
    matrix = build_confusion_matrix(y_true, y_pred, num_classes)
    metrics = calculate_multiclass_metrics(matrix)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print_confusion_matrix(matrix)

    print("Per-class Metrics:")
    for item in metrics["per_class"]:
        print(
            f"class {item['class_id']}: "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"f1={item['f1']:.4f}"
        )

    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Micro-F1: {metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
