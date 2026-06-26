# Qwen2-0.5B-Instruct LoRA 指令微调

## 项目介绍

基于 Qwen2-0.5B-Instruct 大模型，使用 LoRA 进行中文指令微调。

本项目展示了：
- 大模型参数高效微调（LoRA）完整流程
- 中文指令数据准备
- 训练代码 & 推理代码
- 显存占用优化，T4 16GB 可训练

## 技术要点

- 基座模型：Qwen2-0.5B-Instruct（Hugging Face 格式）
- 微调方法：LoRA (Low-Rank Adaptation)
- 框架：Transformers + PEFT
- 仅训练 ~0.4% 参数，显存占用 ~6GB

## 环境安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

训练默认参数：
- LoRA rank r = 8
- 学习率 = 1e-4
- 训练 epoch = 3
- batch size = 4 / 梯度累积 = 4

## 推理

```bash
python inference.py
```

## 效果展示

```
用户: 介绍一下什么是机器学习
助理: ...
```

## 面试要点（可写进简历）

- 理解 LoRA 原理，实现参数高效微调，大幅降低显存占用
- 实践过中文大模型指令微调，熟悉 Hugging Face + PEFT 工作流
- 掌握大模型微调工程实践，能够在单张 T4 上完成 0.5B 模型微调

## 参考

- Qwen2: https://github.com/QwenLM/Qwen2
- PEFT: https://github.com/huggingface/peft
