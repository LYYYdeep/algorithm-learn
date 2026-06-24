def calculate_binary_metrics(y_true, y_pred):
    """
    计算二分类任务的 accuracy、precision、recall、f1.

    Args:
        y_true (_type_): 真实标签列表，只包含0和1
        y_pred (_type_): 预测标签列表，只包含0和1
        
    Returns:
        一个字典，包含 accuracy、precision、recall、f1、tp、fp、tn、fn
    """
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的长度必须相同")
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true 和 y_pred 不能为空")
    
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
            
    total = len(y_true)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    
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
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]
    
    metrics = calculate_binary_metrics(y_pred=y_pred, y_true=y_true)
    
    print("Evaluation Metrics")
    print("------------------")
    print(f"TP: {metrics['tp']}")
    print(f"FP: {metrics['fp']}")
    print(f"TN: {metrics['tn']}")
    print(f"FN: {metrics['fn']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    
    
    
    
if __name__ == "__main__":
    main()