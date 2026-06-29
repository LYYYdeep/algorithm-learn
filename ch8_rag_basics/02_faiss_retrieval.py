import numpy as np
import faiss
import torch
from transformers import AutoModel, AutoTokenizer

# ===== 加载编码模型 =====
model_name = "shibing624/text2vec-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def encode(sentences, normalize_embeddings=True):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)
        # mean pooling
        attention_mask = encoded["attention_mask"]
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# ===== 构建知识库 =====
documents = [
    "机器学习是人工智能的一个分支，研究如何让计算机从数据中学习模式。",
    "深度学习是机器学习的一个子领域，基于神经网络进行特征学习。",
    "自然语言处理是人工智能的一个方向，研究计算机理解人类语言。",
    "BERT是一种基于Transformer的预训练语言模型，由Google提出。",
    "Transformer是Google在2017年提出的注意力架构，现在是NLP主流架构。",
    "LoRA是一种大模型参数高效微调技术，可以大幅减少训练参数量。",
    "RAG是检索增强生成技术，结合检索和大模型生成减少幻觉。",
    "苹果是一种常见的水果，味道酸甜可口。",
    "香蕉是热带水果，富含钾元素。",
    "西瓜夏天吃很解渴，水分很多。",
]

print(f"知识库共有 {len(documents)} 篇文档")

# 编码所有文档
doc_embeddings = encode(documents)
print(f"文档 embeddings shape: {doc_embeddings.shape}")

# ===== 构建 FAISS 索引 =====
dimension = doc_embeddings.shape[1]
# 如果我们已经 L2 归一化了，用 IndexFlatIP（内积）就是余弦相似度
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)
print(f"FAISS 索引构建完成，包含 {index.ntotal} 个向量")

# ===== 检索测试 =====
def search(query, top_k=3):
    q_emb = encode([query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": score,
            "document": documents[idx],
        })
    return results

# 测试
query = "什么是机器学习"
print(f"\n查询: {query}")
print(f"Top 3 检索结果:")
results = search(query, top_k=3)
for i, res in enumerate(results):
    print(f"{i+1}. score={res['score']:.4f}, document={res['document']}")

query = "大模型参数高效微调有什么方法"
print(f"\n查询: {query}")
print(f"Top 3 检索结果:")
results = search(query, top_k=3)
for i, res in enumerate(results):
    print(f"{i+1}. score={res['score']:.4f}, document={res['document']}")

query = "夏天吃什么水果好"
print(f"\n查询: {query}")
print(f"Top 3 检索结果:")
results = search(query, top_k=3)
for i, res in enumerate(results):
    print(f"{i+1}. score={res['score']:.4f}, document={res['document']}")

print("\n✅ FAISS 检索测试完成！")
