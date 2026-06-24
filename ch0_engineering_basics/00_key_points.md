# Key Points Before PyTorch

本文件记录进入 PyTorch 前已经学过的重点，后续用到时可回来复习。

## 1. 工程基础

### Linux / Shell

重点掌握：

- `pwd`：查看当前目录
- `ls -lah`：查看目录内容
- `cd`：切换目录
- `mkdir -p`：递归创建目录
- `cp` / `mv` / `rm`：复制、移动、删除
- `cat` / `less` / `head` / `tail -f`：查看文件和日志
- `grep`：搜索文件内容
- `find`：搜索文件路径
- `ps aux | grep python`：查看 Python 进程
- `kill PID`：终止进程
- `df -h` / `du -sh`：查看磁盘空间
- `ssh` / `scp`：远程连接和传文件
- `nohup` / `tmux`：后台运行训练任务
- `nvidia-smi`：查看 GPU 状态
- `CUDA_VISIBLE_DEVICES=0`：指定 GPU

最重要的工程习惯：

- 删除前确认路径，谨慎使用 `rm -rf`
- 长时间训练要保存日志
- 服务器训练推荐用 `tmux` 或 `nohup`

### Git / GitHub

重点掌握：

- `git status`：查看当前修改状态
- `git add`：加入暂存区
- `git commit -m "message"`：提交版本
- `git diff`：查看改动
- `git log --oneline`：查看提交历史
- `git branch` / `git switch`：分支管理
- `git push` / `git pull`：推送和拉取
- `.gitignore`：忽略缓存、数据、模型权重、密钥等不该提交的文件

常用工作流：

```bash
git status
git add .
git commit -m "update notes"
git push
```

---

## 2. 数学基础

### 线性代数

- 一个样本通常表示为向量 `x`
- 多个样本组成矩阵 `X`
- 机器学习中常用约定：`X.shape = (num_samples, num_features)`
- 线性模型：`y_pred = Xw + b`
- 点积：`w^T x`
- 矩阵乘法：`X @ w`
- 转置：`X.T`

### 概率统计

- 均值：描述中心位置
- 方差 / 标准差：描述波动程度
- 标准化：`x_scaled = (x - mean) / std`
- 分类模型输出可理解为条件概率：`P(y=1|x)`

### 微积分和优化

- 导数：函数变化率
- 偏导：多变量函数对某个变量求导
- 梯度：所有偏导组成的向量
- 梯度方向是 loss 增长最快方向
- 梯度下降沿负梯度方向更新参数：

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

---

## 3. 评估指标

### 二分类混淆矩阵

- TP：真实为正，预测为正
- FP：真实为负，预测为正，误报
- TN：真实为负，预测为负
- FN：真实为正，预测为负，漏报

### 指标

```text
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

重点：

- Accuracy 在类别不平衡时可能误导
- Precision 关注误报
- Recall 关注漏报
- 医疗筛查、风控召回等更关注 Recall
- F1 综合 Precision 和 Recall

### 多分类

- Confusion Matrix：行通常是真实类别，列通常是预测类别
- Macro-F1：先算每类 F1，再平均；更关注少数类
- Micro-F1：先汇总 TP/FP/FN，再算 F1；单标签多分类中常等于 accuracy

---

## 4. 传统机器学习模型

### 线性回归

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

重点：

- 用于回归任务
- MSE 对异常值敏感
- 梯度下降需要 learning rate
- sklearn `LinearRegression` 通常直接求最小二乘解，不需要 epoch

### 逻辑回归

模型：

```text
z = Xw + b
p = sigmoid(z)
```

其中：

```text
sigmoid(z) = 1 / (1 + exp(-z))
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

重点：

- 用于二分类
- `z = Xw + b` 叫 logit
- sigmoid 把 logit 映射到 `(0, 1)`
- 阈值降低通常提高 recall，降低 precision
- 决策边界：`Xw + b = 0`
- 普通逻辑回归不能直接解决 XOR，因为 XOR 非线性可分

### 决策树

重点：

- 通过 if-else 规则递归划分特征空间
- 分类树叶子节点通常输出多数类
- 分裂目标：让子节点更纯
- Gini：`1 - Σp_k^2`
- Entropy：`-Σp_k log2(p_k)`
- Information Gain：父节点 entropy - 子节点加权 entropy
- 连续特征候选阈值：排序后相邻不同取值的中点
- 容易过拟合，因为树太深会记住噪声
- 不需要标准化，因为标准化不改变排序关系

常用参数：

- `max_depth`：最大深度
- `min_samples_split`：节点至少多少样本才允许分裂
- `min_samples_leaf`：叶子节点至少多少样本
- `criterion`：`gini` 或 `entropy`

### 随机森林

重点：

- 多棵决策树集成
- 分类投票，回归平均
- 随机性来自：bootstrap 样本随机 + 节点分裂时特征随机
- Bagging = Bootstrap Aggregating
- 相比单棵树更稳定，主要降低方差

常用参数：

- `n_estimators`：树的数量
- `max_features`：每个节点分裂时考虑的特征数
- `max_depth`：每棵树最大深度
- `bootstrap=True`：有放回采样
- `oob_score=True`：使用袋外样本估计性能

特征重要性：

- `feature_importances_` 通常基于不纯度下降
- 重要性之和为 1
- 只能说明相关性，不代表因果关系
- 可能偏向连续值多或类别基数高的特征

---

## 5. 数据处理和 sklearn 工程流程

### pandas 基础

常用：

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df[features]
df[target]
```

### 缺失值

- 数值特征：均值 / 中位数填充
- 类别特征：众数 / Unknown 填充
- 缺失本身可能有信息，可增加 missing indicator

### 类别编码

- One-Hot Encoding：无序类别，如城市
- Ordinal Encoding：有序类别，如学历、满意度

### 标准化

```text
x_scaled = (x - mean) / std
```

训练集：`fit_transform`
测试集：`transform`

原因：避免测试集分布信息泄露。

### Pipeline

- `Pipeline`：把预处理和模型串起来
- `ColumnTransformer`：不同列使用不同预处理方式
- `OneHotEncoder(handle_unknown="ignore")`：预测时遇到未知类别不报错，已知类别列全为 0

---

## 6. 下一步 PyTorch 衔接

前面手写的训练循环：

```text
y_pred = model(X)
loss = criterion(y_pred, y)
计算梯度
更新参数
```

PyTorch 会自动完成梯度计算：

```python
loss.backward()
optimizer.step()
```

PyTorch 重点学习路线：

1. Tensor 基础
2. dtype / shape / device
3. Tensor 运算和矩阵乘法
4. autograd 自动求导
5. 用 PyTorch 重写线性回归
6. nn.Module
7. Dataset / DataLoader
8. MLP 分类器
