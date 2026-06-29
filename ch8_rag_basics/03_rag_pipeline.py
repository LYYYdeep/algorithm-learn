import numpy as np
import torch
from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForCausalLM,
)

# =================== 1.  embedding 模型  ===================
emb_model_name = "shibing624/text2vec-base-chinese"
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
emb_model = AutoModel.from_pretrained(emb_model_name)

def encode_sentences(sentences, normalize=True):
    encoded = emb_tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = emb_model(**encoded)
        # mean pooling
        attn_mask = encoded["attention_mask"]
        token_embeds = outputs[0]
        mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        sum_embeds = torch.sum(token_embeds * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeds / sum_mask
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# =================== 2. 知识库构建  ===================
knowledge_base = [
    {
        "content": "机器学习是人工智能的一个分支，研究如何让计算机从数据中学习模式。",
    },
    {
        "content": "深度学习是机器学习的一个子领域，基于神经网络自动提取特征学习。",
    },
    {
        "content": "自然语言处理是人工智能的一个方向，研究计算机理解处理人类语言。",
    },
    {
        "content": "BERT是Google在2018年提出的基于Transformer的预训练语言模型，支持双向编码。",
    },
    {
        "content": "Transformer是Google在2017年提出的纯注意力架构，现在是NLP领域主流架构。",
    },
    {
        "content": "LoRA是大模型参数高效微调技术，冻结原始权重只训练低秩增量矩阵，大幅减少训练参数量，节省显存。",
    },
    {
        "content": "RAG全称检索增强生成，是先检索相关知识再让大模型生成回答，可以减少大模型幻觉，能方便更新知识不用重新微调模型。",
    },
    {
        "content": "苹果是一种常见温带水果，味道酸甜可口，富含维生素。",
    },
    {
        "content": "香蕉是热带水果，形状长条弯曲，富含钾元素。",
    },
    {
        "content": "西瓜是夏季常见水果，水分很多，含糖量高，夏天吃可以解渴。",
    },
]

# 编码所有文档
documents = [doc["content"] for doc in knowledge_base]
doc_embeddings = encode_sentences(documents)
print(f"知识库构建完成，共 {len(documents)} 篇文档，embedding 维度 {doc_embeddings.shape[1]}")

# =================== 3. 检索  ===================
def retrieve(query, top_k=3):
    q_emb = encode_sentences([query])[0]
    # 余弦相似度 = 点积（因为归一化了）
    scores = doc_embeddings @ q_emb
    # top-k 降序
    top_indices = scores.argsort()[-top_k:][::-1]
    results = [knowledge_base[idx] for idx in top_indices]
    return results

# =================== 4. 拼 Prompt + 生成  ===================
# 加载 Qwen2-0.5B-Instruct 大模型
# 如果本地跑不动，可以用更小的模型，这里演示流程
llm_model_name = "Qwen/Qwen2-0.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def build_prompt(query, retrieved_docs):
    # RAG prompt 模板
    context = "\n".join([f"- {doc['content']}" for doc in retrieved_docs])
    prompt = f"""请根据以下背景信息回答用户的问题。如果背景信息里没有答案，就说不知道。

背景信息:
{context}

用户问题: {query}

回答:
"""
    return prompt

def rag_generate(query, top_k=3, max_new_tokens=256):
    # 第一步检索
    retrieved = retrieve(query, top_k=top_k)
    # 第二步拼 prompt
    prompt = build_prompt(query, retrieved)
    # 第三步大模型生成
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取回答部分
    answer = answer.split("回答:")[-1].strip()
    return answer, retrieved

# =================== 测试  ===================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG 流水线测试")
    print("="*60 + "\n")
    
    query = "什么是 RAG，它有什么好处？"
    print(f"用户问题: {query}\n")
    
    answer, retrieved = rag_generate(query, top_k=3)
    
    print("检索到的背景:")
    for i, doc in enumerate(retrieved):
        print(f"{i+1}. {doc['content']}")
    print()
    print(f"模型回答: {answer}")
