import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载预训练 tokenizer 和模型
# 这里我们加载 bert-base-chinese，它是 BERT 在中文语料上预训练的
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# 告诉模型我们要做二分类，所以 num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 看一下模型结构
print(f"Model total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print()

# 测试单样本推理
text = "我非常喜欢这部电影"
encoded = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=32
)

print(f"输入文本: {text}")
print(f"input_ids shape: {encoded['input_ids'].shape}")
print(f"tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")

# 推理
model.eval()
with torch.no_grad():
    outputs = model(**encoded)

logits = outputs.logits
print(f"\nlogits: {logits}")
predicted_class = torch.argmax(logits, dim=1)
print(f"预测类别: {predicted_class.item()}")
probabilities = torch.softmax(logits, dim=1)
print(f"概率分布: {probabilities}")

# ===== 批量推理测试 =====
print("\n" + "="*50)
print("批量推理测试:")
texts = [
    "我非常喜欢这部电影，剧情很精彩",
    "这个电影真的太难看了，浪费时间",
    "服务很好，菜品也很美味，下次再来",
    "味道太差了，价格还贵，不会再来了"
]

encoded_batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**encoded_batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)

print("\n批量预测结果:")
for text, pred, prob in zip(texts, predictions, probs):
    print(f"文本: {text}")
    print(f"预测类别: {pred.item()}, 概率: 负类={prob[0]:.4f}, 正类={prob[1]:.4f}")
    print()
