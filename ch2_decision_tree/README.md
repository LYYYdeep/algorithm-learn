# Decision Tree Notes

## 核心思想

决策树通过一系列 if-else 规则递归划分特征空间。

预测时：

```text
样本从根节点开始，根据每个节点的特征和阈值判断走左子树还是右子树，直到叶子节点。
```

分类树通常用叶子节点中的多数类作为预测结果。

## 分裂目标

每次分裂选择一个特征和阈值，使划分后的子节点尽量纯。

常用标准：

- Gini impurity
- Entropy / Information Gain

## Gini impurity

```text
Gini = 1 - Σp_k^2
```

二分类：

```text
Gini = 1 - p0^2 - p1^2
```

值越小，节点越纯。

## Entropy

```text
Entropy = -Σp_k log2(p_k)
```

值越小，节点越纯。

## Information Gain

```text
Gain = Entropy(parent) - weighted_entropy(children)
```

选择信息增益最大的划分。

## 连续特征候选阈值

通常先对某个特征排序，然后取相邻不同取值的中点作为候选阈值。

## 过拟合

决策树容易过拟合，因为树太深时会把训练样本划分得过细，甚至记住噪声和异常点。

常用控制参数：

- `max_depth`：最大深度
- `min_samples_split`：节点至少多少样本才允许分裂
- `min_samples_leaf`：每个叶子节点至少多少样本
- `max_leaf_nodes`：最多叶子节点数

## 为什么不需要标准化

决策树基于单个特征阈值划分。标准化只改变数值尺度，不改变样本在该特征上的排序关系，因此通常不影响树的分裂。

## 单棵树为什么不稳定

单棵决策树采用贪心递归分裂。训练数据的小扰动可能改变上层节点分裂，一旦上层节点变了，整棵树结构都可能变化。
