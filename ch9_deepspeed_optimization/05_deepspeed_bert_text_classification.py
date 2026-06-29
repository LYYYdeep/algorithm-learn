import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
import deepspeed
from tqdm import tqdm

# ===== 1. 内置小数据集 =====
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

def create_dataset(data_list):
    texts = [item["text"] for item in data_list]
    labels = [item["label"] for item in data_list]
    return HFDataset.from_dict({"text": texts, "label": labels})

dataset = DatasetDict({
    "train": create_dataset(train_data),
    "validation": create_dataset(val_data),
    "test": create_dataset(test_data),
})

# ===== 2. 加载 tokenizer 和数据 =====
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# ===== 3. 加载模型 =====
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# ===== 4. DeepSpeed 初始化 =====
# DeepSpeed 会帮你处理 ZeRO，配置从 json 读
deepspeed_config = "ch9_deepspeed_optimization/04_deepspeed_zero_config.json"
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=deepspeed_config,
)

print("\n===== DeepSpeed + ZeRO 初始化完成 =====")
print(f"Stage: {model.zero_optimization_stage()}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# ===== 5. 训练循环 =====
def train_one_epoch(model, dataloader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        # DeepSpeed 自动处理 backward 和 step
        model.backward(loss)
        model.step()

        total_loss += loss.item() * batch["input_ids"].size(0)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["input_ids"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}

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
print("Start training BERT text classification with DeepSpeed ZeRO stage 1")
print("="*60 + "\n")

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    train_loss, train_acc = train_one_epoch(model, train_loader, model.device)
    val_loss, val_acc = evaluate(model, val_loader, model.device)

    print(f"\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"New best validation accuracy: {best_val_acc:.4f}")

print("\n" + "="*60)
print(f"Training finished! Best validation accuracy: {best_val_acc:.4f}")
print("DeepSpeed ZeRO reduced memory usage, allows larger batch/models on same GPU")
