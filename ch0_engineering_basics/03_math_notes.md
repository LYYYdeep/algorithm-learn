# Math Notes for Machine Learning

## 1. 学习目标

算法工程师实习不要求一开始数学特别深，但需要理解机器学习中常用的数学概念：

- 向量和矩阵
- 点积和矩阵乘法
- 范数
- 均值、方差、标准差
- 概率和条件概率
- 最大似然直觉
- 导数、偏导、梯度
- 链式法则
- 梯度下降

---

# Part 1: 线性代数

## 2. 向量 Vector

向量可以表示一个样本的多个特征：

```text
x = [age, income, study_hours]
```

在机器学习中，一个样本通常表示为一个向量。

---

## 3. 矩阵 Matrix

多个样本组成矩阵：

```text
X.shape = (num_samples, num_features)
```

例如：

```text
X = [
  [18, 1.0, 5],
  [20, 3.0, 20],
  [22, 4.0, 30],
]
```

行表示样本，列表示特征。

---

## 4. 点积 Dot Product

两个向量：

```text
a = [1, 2, 3]
b = [4, 5, 6]
```

点积：

```text
a · b = 1*4 + 2*5 + 3*6 = 32
```

机器学习中，线性模型就是：

```text
y = w^T x + b
```

也就是特征和权重做点积。

---

## 5. 矩阵乘法

如果：

```text
X.shape = (n, d)
w.shape = (d,)
```

那么：

```text
X @ w -> shape = (n,)
```

含义：一次性计算 n 个样本的预测值。

---

## 6. 转置 Transpose

```text
X.shape = (n, d)
X.T.shape = (d, n)
```

线性回归梯度中：

```text
dw = (2 / n) * X.T @ errors
```

使用 `X.T` 是为了按特征维度聚合所有样本的误差贡献。

---

## 7. 范数 Norm

范数可以理解为向量大小。

常见 L2 范数：

```text
||x||_2 = sqrt(x1^2 + x2^2 + ... + xd^2)
```

机器学习中常用于：

- 正则化
- 距离度量
- 梯度裁剪
- 向量归一化

---

# Part 2: 概率统计

## 8. 均值 Mean

```text
mean = 所有值之和 / 样本数
```

用于描述数据中心位置。

---

## 9. 方差 Variance

```text
variance = mean((x - mean)^2)
```

用于描述数据波动程度。

---

## 10. 标准差 Standard Deviation

```text
std = sqrt(variance)
```

标准化常用：

```text
x_scaled = (x - mean) / std
```

---

## 11. 概率 Probability

概率表示事件发生的可能性：

```text
0 <= P(A) <= 1
```

分类模型输出概率：

```text
P(y = 1 | x)
```

---

## 12. 条件概率 Conditional Probability

```text
P(A | B)
```

表示在 B 已经发生的条件下，A 发生的概率。

逻辑回归输出可以理解为：

```text
P(y = 1 | x)
```

---

## 13. 最大似然 Maximum Likelihood

最大似然的直觉：

```text
选择一组参数，让当前观测到的数据出现的概率尽可能大。
```

逻辑回归使用 BCE loss，本质上和最大化伯努利分布似然有关。

---

# Part 3: 微积分和优化

## 14. 导数 Derivative

导数表示函数变化率。

例如：

```text
f(x) = x^2
f'(x) = 2x
```

---

## 15. 偏导 Partial Derivative

多变量函数中，对某一个变量求导。

例如：

```text
f(w, b) = loss
```

需要计算：

```text
∂loss/∂w
∂loss/∂b
```

---

## 16. 梯度 Gradient

梯度是所有偏导数组成的向量：

```text
gradient = [∂loss/∂w1, ∂loss/∂w2, ...]
```

梯度方向表示函数增长最快的方向。

---

## 17. 梯度下降 Gradient Descent

目标：最小化 loss。

更新公式：

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

因为梯度指向 loss 增长最快方向，所以要沿反方向更新。

---

## 18. 学习率 Learning Rate

学习率控制每次参数更新步长。

- 太大：loss 可能震荡或发散
- 太小：收敛速度慢

---

## 19. 链式法则 Chain Rule

复合函数求导需要链式法则。

深度学习中的反向传播本质上就是链式法则的大规模应用。

例如：

```text
z = wx + b
p = sigmoid(z)
loss = BCE(y, p)
```

计算参数梯度时，需要从 loss 一层层往回传。

---

# Part 4: 和前面代码的对应关系

## 20. 线性回归

模型：

```text
y_pred = Xw + b
```

损失：

```text
MSE = mean((y_pred - y)^2)
```

梯度：

```text
dw = (2 / n) * X.T @ (y_pred - y)
db = (2 / n) * sum(y_pred - y)
```

---

## 21. 逻辑回归

模型：

```text
z = Xw + b
p = sigmoid(z)
```

损失：

```text
BCE = -mean(y log p + (1-y) log(1-p))
```

梯度：

```text
dw = (1 / n) * X.T @ (p - y)
db = (1 / n) * sum(p - y)
```

---

## 22. 面试高频

### 为什么梯度下降是减去梯度？

因为梯度方向是 loss 增长最快的方向，而训练目标是最小化 loss，所以要沿梯度反方向更新。

### MSE 为什么平方？

避免正负误差抵消，对大误差惩罚更强，并且连续可导方便优化。

### MSE 缺点？

对异常值敏感，因为大误差会被平方放大。

### 为什么要标准化？

对逻辑回归、SVM、KNN、神经网络等尺度敏感模型，标准化能让不同特征处于相近尺度，优化更稳定。
