# PyTorch Basics

本目录用于学习 PyTorch 基础。

## Learning Path

1. Tensor 基础
2. Tensor 运算
3. Tensor 和 NumPy 转换
4. device：CPU / CUDA / MPS
5. autograd 自动求导
6. 用 PyTorch 重写线性回归
7. `nn.Module`
8. `Dataset` / `DataLoader`
9. MLP 分类器

## Why PyTorch

PyTorch 的核心能力：

- Tensor 多维数组计算
- GPU / MPS 加速
- autograd 自动求导
- 神经网络模块 `torch.nn`
- 优化器 `torch.optim`
- 数据加载工具 `torch.utils.data`

前面用 NumPy 手写过的：

```text
预测 -> loss -> 梯度 -> 参数更新
```

在 PyTorch 中会变成：

```python
y_pred = model(X)
loss = criterion(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
