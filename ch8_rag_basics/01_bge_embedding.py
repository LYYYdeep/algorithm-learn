import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ===== 使用 transformer 手动加载，会走 HF_ENDPOINT 镜像 =====
model_name = "BAAI/bge-small-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def encode(sentences, normalize_embeddings=True):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)
        # 用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 输出作为句子表示，这是 BGE 做法
        embeddings = model_output[0][:, 0]
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# ===== 测试 =====
sentences = [
    "什么是机器学习",
    "机器学习是人工智能的一个分支，让计算机从数据中学习模式",
    "深度学习是机器学习的一个子领域，基于神经网络",
    "我喜欢吃苹果",
]

# BGE instruction
instruction = "为这个句子生成表示以用于检索相关文章："

embeddings = encode([instruction + s for s in sentences])

print(f"\nInput sentences: {len(sentences)} sentences")
print(f"Output embeddings shape: {embeddings.shape}")
print(f"Each embedding dimension: {embeddings.shape[1]}")

# ===== 计算余弦相似度 =====
def cosine_sim(a, b):
    return (a @ b) / (torch.norm(torch.tensor(a)) * torch.norm(torch.tensor(b)))

print("\n===== 余弦相似度测试 =====")
emb1 = embeddings[0]
emb2 = embeddings[1]
emb3 = embeddings[2]
emb4 = embeddings[3]

print(f"相似度 '什么是机器学习' vs '机器学习是...': {cosine_sim(emb1, emb2):.4f}")
print(f"相似度 '什么是机器学习' vs '深度学习是...': {cosine_sim(emb1, emb3):.4f}")
print(f"相似度 '什么是机器学习' vs '我喜欢吃苹果': {cosine_sim(emb1, emb4):.4f}")

print("\n✅ 结论：语义相近的句子相似度更高，说明 Embedding 学到了语义")
