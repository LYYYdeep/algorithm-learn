import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from tqdm import tqdm

# ===== 1. 我们直接用内置的小型中文情感数据集 =====
# 这是一份简化版的中文酒店评论情感分类数据
train_data = [
    {"text": "这家酒店位置很好，房间干净整洁，服务也很热情，非常满意。", "label": 1},
    {"text": "房间宽敞明亮，床很舒服，早餐也不错，下次还来。", "label": 1},
    {"text": "位置便利，离地铁站近，周边吃饭方便。", "label": 1},
    {"text": "价格便宜，性价比很高，推荐入住。", "label": 1},
    {"text": "服务员态度很好，打扫卫生也很及时。", "label": 1},
    {"text": "酒店装修很新，设施齐全，体验不错。", "label": 1},
    {"text": "房间很大，景观很好，值得这个价格。", "label": 1},
    {"text": "交通方便，离景点近，购物也方便。", "label": 1},
    {"text": "整体不错，挺满意的，会推荐给朋友。", "label": 1},
    {"text": "环境安静，睡觉很舒服，很不错。", "label": 1},
    {"text": "房间很脏，气味难闻，服务也差，再也不会来了。", "label": 0},
    {"text": "价格贵，房间小，设施老旧，不推荐。", "label": 0},
    {"text": "噪音很大，睡不好觉，体验很差。", "label": 0},
    {"text": "卫生间不干净，有异味，很失望。", "label": 0},
    {"text": "位置偏僻，找不到，周边也不安全。", "label": 0},
    {"text": "服务态度很差，爱答不理，很生气。", "label": 0},
    {"text": "照片和实物差距太大，被骗了。", "label": 0},
    {"text": "空调坏了也不修，晚上很冷。", "label": 0},
    {"text": "早餐种类少，味道也不好。", "label": 0},
    {"text": "WiFi 信号很差，根本用不了。", "label": 0},
    {"text": "非常好的入住体验，房间干净，服务周到。", "label": 1},
    {"text": "地理位置绝佳，出门就是地铁站。", "label": 1},
    {"text": "性价比超高，会再来入住。", "label": 1},
    {"text": "房间太小了，转个身都难。", "label": 0},
    {"text": "卫生条件太差了，被子上还有污渍。", "label": 0},
    {"text": "酒店很不错，推荐给大家。", "label": 1},
    {"text": "这个价格能住到这样的酒店很不错了。", "label": 1},
    {"text": "服务员很有礼貌，有问题都及时解决。", "label": 1},
    {"text": "房间有一股霉味，很难闻。", "label": 0},
    {"text": "洗澡水忽冷忽热，不舒服。", "label": 0},
    {"text": "退房排队太久，浪费了半小时。", "label": 0},
    {"text": "总体来说很满意，性价比不错。", "label": 1},
    {"text": "床太硬了，睡的腰酸背痛。", "label": 0},
    {"text": "房间隔音不好，隔壁说话都听得见。", "label": 0},
    {"text": "大堂很漂亮，房间也很宽敞。", "label": 1},
    {"text": "游泳池很干净，人也不多。", "label": 1},
    {"text": "停车很方便，免费停车场。", "label": 1},
    {"text": "电梯太慢了，等半天。", "label": 0},
    {"text": "前台办理入住很快，效率高。", "label": 1},
    {"text": "浴室漏水，地板都湿了。", "label": 0},
]

val_data = [
    {"text": "酒店环境优美，服务热情，值得推荐。", "label": 1},
    {"text": "太差劲了，一辈子不会再来。", "label": 0},
    {"text": "位置好，房间大，很满意。", "label": 1},
    {"text": "价格太高，不值这个价。", "label": 0},
    {"text": "干净卫生，服务到位，很好。", "label": 1},
    {"text": "设施老化，需要翻新了。", "label": 0},
]

test_data = [
    {"text": "酒店位置很好找，就在市中心，逛街方便。", "label": 1},
    {"text": "房间异味很重，前台也不处理。", "label": 0},
    {"text": "服务员态度非常好，帮我们提行李。", "label": 1},
    {"text": "房间比图片小很多，失望。", "label": 0},
    {"text": "早餐丰富，味道不错。", "label": 1},
    {"text": "晚上很吵，根本睡不着。", "label": 0},
]

# 转换成 Hugging Face Dataset
def create_dataset(data_list):
    texts = [item["text"] for item in data_list]
    labels = [item["label"] for item in data_list]
    return HFDataset.from_dict({"text": texts, "label": labels})

dataset = DatasetDict({
    "train": create_dataset(train_data),
    "validation": create_dataset(val_data),
    "test": create_dataset(test_data),
})

print(f"Dataset loaded locally!")
print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")
print("\n第一个训练样本:")
print(dataset['train'][0])

# ===== 设置设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ===== 2. 加载 tokenizer =====
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ===== 3. 数据预处理 =====
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

# data collator 自动 padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 创建 DataLoader
train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=8,
    collate_fn=data_collator
)

test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=8,
    collate_fn=data_collator
)

# ===== 4. 加载模型 =====
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

model = model.to(device)

# ===== 5. 优化器 =====
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# ===== 训练函数 =====
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch["input_ids"].size(0)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["input_ids"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ===== 评估函数 =====
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        total_loss += loss.item() * batch["input_ids"].size(0)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["input_ids"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ===== 6. 开始训练 =====
num_epochs = 3
best_val_acc = 0

print("\n" + "="*60)
print("Start training...")
print(f"Train batch size: 8, Learning rate: 2e-5, Epochs: {num_epochs}")
print("="*60 + "\n")

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)

    print(f"\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"New best validation accuracy: {best_val_acc:.4f}, saving model...")
        import os
        os.makedirs("./bert-chinese-sentiment-best", exist_ok=True)
        torch.save(model.state_dict(), "./bert-chinese-sentiment-best/pytorch_model.bin")

# ===== 7. 测试集评估 =====
print("\n" + "="*60)
print("Test set evaluation (best model):")
test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# ===== 8. 测试推理 =====
print("\n" + "="*60)
print("Test inference on custom texts:")
test_texts = [
    "这家酒店位置很好，房间干净整洁，服务也很热情，非常满意。",
    "房间很脏，气味难闻，服务也差，再也不会来了。",
    "价格便宜，交通方便，性价比很高。",
    "菜品味道一般，价格还贵，不推荐。"
]

# tokenize
inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}

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
    print(f"预测: {label}, 负类概率: {prob[0]:.4f}, 正类概率: {prob[1]:.4f}")

print("\n" + "="*60)
print(f"Training finished! Best validation accuracy: {best_val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print("Model saved to ./bert-chinese-sentiment-best/pytorch_model.bin")
