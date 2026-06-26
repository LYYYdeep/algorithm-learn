import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
)

# ===== 1. 从 Hugging Face 加载完整 ChnSentiCorp 数据集 =====
dataset = load_dataset("lansinuote/ChnSentiCorp")
print(f"Dataset loaded:")
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
print("\nFirst sample:")
print(dataset['train'][0])

# ===== 2. 加载 tokenizer =====
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ===== 3. 预处理 =====
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== 4. 加载模型 =====
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# ===== 5. 评估指标 =====
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# ===== 6. 训练参数 =====
training_args = TrainingArguments(
    output_dir="./bert-chinese-sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    push_to_hub=False,
)

# ===== 7. Trainer 训练 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== 8. 开始训练 =====
print("\n" + "="*60)
print("Start training with Trainer...")
trainer.train()

# ===== 9. 测试集评估 =====
print("\n" + "="*60)
print("Test set evaluation:")
eval_result = trainer.evaluate(tokenized_dataset["test"])
print(eval_result)

# ===== 10. 测试推理 =====
print("\n" + "="*60)
print("Test inference on custom texts:")
test_texts = [
    "这家酒店位置很好，房间干净整洁，服务也很热情，非常满意。",
    "房间很脏，气味难闻，服务也差，再也不会来了。",
    "价格便宜，交通方便，性价比很高，值得推荐。",
    "菜品味道一般，价格还贵，不推荐大家来。",
    "位置很好找，就在市中心，逛街吃饭都很方便。",
    "晚上噪音很大，根本睡不着觉，体验很差。"
]

inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)

print("\n预测结果:")
for text, pred, prob in zip(test_texts, predictions, probs):
    label = id2label[pred.item()]
    print(f"\n文本: {text}")
    print(f"预测: {label}, 负类: {prob[0]:.4f}, 正类: {prob[1]:.4f}")

# 保存最好的模型
trainer.save_model("./bert-chinese-sentiment-best")
print("\nModel saved to ./bert-chinese-sentiment-best")
