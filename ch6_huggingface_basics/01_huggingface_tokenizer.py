from transformers import AutoTokenizer

# 加载中文 BERT 预训练的 tokenizer
# 会自动下载 vocab 等文件
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# ===== 例子 1：单句编码 =====
text = "我喜欢机器学习"
print("原始文本:", text)

# 编码
encoded = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=10,
    return_tensors="pt"  # 返回 PyTorch tensor
)

print("\n=== 单句编码结果 ===")
print(f"input_ids: {encoded['input_ids']}")
print(f"attention_mask: {encoded['attention_mask']}")
print(f"tokenized: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")

# ===== 例子 2：批量编码 =====
texts = [
    "我喜欢机器学习",
    "深度学习很有趣",
    "这个模型效果很好"
]

print("\n\n=== 批量编码结果 ===")
encoded_batch = tokenizer(
    texts,
    padding=True,  # 按 batch 里最长的填充
    truncation=True,
    max_length=10,
    return_tensors="pt"
)

print(f"input_ids shape: {encoded_batch['input_ids'].shape}")
print(f"attention_mask shape: {encoded_batch['attention_mask'].shape}")
print("\n第一个样本分词:")
print(tokenizer.convert_ids_to_tokens(encoded_batch['input_ids'][0]))
print("\n第二个样本分词:")
print(tokenizer.convert_ids_to_tokens(encoded_batch['input_ids'][1]))

# ===== 例子 3：成对句子（问答/文本匹配）=====
text1 = "什么是机器学习"
text2 = "机器学习是人工智能的一个分支"

print("\n\n=== 成对句子编码（BERT 输入）===")
encoded_pair = tokenizer(
    text1, text2,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"input_ids shape: {encoded_pair['input_ids'].shape}")
tokens = tokenizer.convert_ids_to_tokens(encoded_pair['input_ids'][0])
print(f"tokens: {tokens}")
print(f"注意：开头有 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>，两句之间有 [SEP]")

# ===== 例子 4：手动构建分词过程 =====
print("\n\n=== 手动分词演示 ===")
tokens = tokenizer.tokenize("我爱自然语言处理")
print(f"tokenize 结果: {tokens}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"转id: {ids}")
# 添加 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 和 [SEP]
full_ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
print(f"加上 special tokens: {full_ids}")
print(f"cls_token = {tokenizer.cls_token}, cls_id = {tokenizer.cls_token_id}")
print(f"sep_token = {tokenizer.sep_token}, sep_id = {tokenizer.sep_token_id}")