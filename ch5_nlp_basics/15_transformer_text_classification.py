import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算好位置编码，存在缓冲区
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算 10000^(2i/d_model) 的分母部份
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-2 * torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # sin
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 就是从索引 0 开始，每隔 2 个取一个 → 所有偶数索引
        # cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个 batch 维度
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，模型保存时会保存该缓冲区buffer
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        # 将位置编码加到输入上
        x = x + self.pe[:, :seq_len].detach()
        return self.dropout(x)
    
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 实现
    
    参数:
        d_model: 总维度
        n_heads: 多头数量
    
    要求: d_model 必须能被 n_heads 整除
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 初始化线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        """
        输入: [batch, seq_len, d_model]
        输出: [batch, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        # 把 d_model 分成 n_heads * d_k
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # 交换维度， 把 n_heads 放前面 -> [batch, n_heads, seq_len, d_k]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, attention_mask=None):
        """
        输入:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len], 1 是真实 token
        
        输出:
            out: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 Q, K, V
        q = self.w_q(x)  # [batch, seq_len, d_model]
        k = self.w_k(x)  # [batch, seq_len, d_model]
        v = self.w_v(x)  # [batch, seq_len, d_model]
        
        # 2. 分割成多头
        q = self.split_heads(q)  # [batch, n_heads, seq_len, d_k]
        k = self.split_heads(k)  # [batch, n_heads, seq_len, d_k]
        v = self.split_heads(v)  # [batch, n_heads, seq_len, d_k]
        
        # 3. 计算注意力分数: Q * K^T / sqrt(d_k)
        # k.transpose(-2, -1) -> [batch, n_heads, d_k, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch, n_heads, seq_len, seq_len]
        # scores: [batch, n_heads, seq_len, seq_len]
        
        # 4. 如果有 mask, 将 padding 位置的分数设置为 -inf
        if attention_mask is not None:
            # attention_mask [batch, seq_len] -> [batch, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        # 5. softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, seq_len, seq_len]
        
        # 6. 用注意力权重对 V 进行加权求和
        out = torch.matmul(attention_weights, v)  # [batch, n_heads, seq_len, d_k]
        
        # 7. 拼接多头
        # [batch, n_heads, seq_len, d_k] → [batch, seq_len, n_heads, d_k]
        out = out.permute(0, 2, 1, 3).contiguous()
        # 合并最后两个维度 → [batch, seq_len, d_model]
        out = out.view(batch_size, seq_len, self.d_model)
        
        # 8. 最终投影
        out = self.w_o(out)
        
        return out, attention_weights
    
    

class FeedForward(nn.Module):
    """Feed Forward Network: Linear -> GELU -> Linear"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block (Pre-LN 版本)
    
    结构:
        输入 x
        → LayerNorm
        → Multi-Head Attention
        → Dropout
        → 残差连接 x = x + attn_out
        → LayerNorm
        → Feed Forward
        → Dropout
        → 残差连接 x = x + ffn_out
        → 输出
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        if d_ff is None:
            d_ff = 4 * d_model  # 默认是 4 * d_model，原始论文就是这么来的
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attention_mask=None):
        """
        输入:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len], 1 表示真实 token
        
        输出:
            out: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 第一步: Multi-Head Attention + 残差
        residual = x
        x_norm = self.norm1(x)
        attn_out, attention_weights = self.attention(x_norm, attention_mask)
        x = residual + self.dropout(attn_out)
        
        # 第二步: FFN + 残差
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)
        
        return x, attention_weights


def build_vocab(tokenized_texts):
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
    }
    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def encode(tokens, vocab):
    unk_id = vocab["[UNK]"]
    return [vocab.get(token, unk_id) for token in tokens]

def pad_or_truncate(ids, max_len, pad_id=0):
    if len(ids) > max_len:
        return ids[:max_len]
    num_padding = max_len - len(ids)
    padded_ids = ids + [pad_id] * num_padding
    attention_mask = [1] * len(ids) + [0] * num_padding
    return padded_ids, attention_mask


class TextClassificationDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for tokens, label in zip(tokenized_texts, labels):
            ids = encode(tokens, vocab)
            padded_ids, attention_mask = pad_or_truncate(
                ids, max_len=max_len, pad_id=vocab["[PAD]"]
            )
            self.input_ids.append(padded_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(label)
        
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.attention_masks[index],
            self.labels[index],
        )
        

class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff=None,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1.词嵌入
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )
        
        # 2.位置编码
        self.pos_encoding = PositionEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # 3.堆叠多个 Transformer Encoder Block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 4.分类头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)  # 二分类任务，输出一个 logit
        
    def forward(self, input_ids, attention_mask):
        """
        输入:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        输出:
            logits: [batch_size, 1]
            all_attention_weights: list of attention weights, 每层一个
        """
        batch_size, seq_len = input_ids.shape
        
        # 1.词嵌入 + 位置编码
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)  # 加上位置编码
        
        # 2.过每个 transformer block
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask)
            all_attention_weights.append(attn_weights)
            
        # 3. 平均池化 (只对真实 token 平均)
        # attention_mask [batch, seq_len] → [batch, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = x * mask_expanded
        # 求和除以真实 token 数
        sum_x = x.sum(dim=1)  # [batch, d_model]
        lengths = mask_expanded.sum(dim=1)  # [batch, d_model]
        pooled = sum_x / lengths  # [batch, d_model]
        
        # 4. 分类
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        
        return logits, all_attention_weights
        
def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def predict_texts(model, tokenized_texts, vocab, max_len, device="cpu"):
    input_ids_list = []
    attention_masks = []
    
    for tokens in tokenized_texts:
        ids = encode(tokens, vocab)
        padded_ids, attention_mask = pad_or_truncate(
            ids, max_len=max_len, pad_id=vocab["[PAD]"]
        )
        input_ids_list.append(padded_ids)
        attention_masks.append(attention_mask)
    
    input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
    
    model.eval()
    logits, all_attn_weights = model(input_ids, attention_mask)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).int()
    
    return probabilities, predictions, all_attn_weights


def main():
    torch.manual_seed(42)
    device = "cpu"
    
    # 样例数据: 中文情感分类，1 正面，0 负面
    tokenized_texts = [
        ["我", "喜欢", "机器", "学习"],
        ["深度", "学习", "很", "有趣"],
        ["这个", "模型", "效果", "很好"],
        ["我", "讨厌", "这个", "错误"],
        ["这个", "结果", "很", "糟糕"],
        ["模型", "训练", "失败"],
        ["我", "喜欢", "NLP"],
        ["这个", "方法", "很", "有效"],
        ["结果", "不好"],
        ["错误", "很多"],
    ]
    
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    
    vocab = build_vocab(tokenized_texts)
    max_len = 6
    
    dataset = TextClassificationDataset(
        tokenized_texts=tokenized_texts,
        labels=labels,
        vocab=vocab,
        max_len=max_len,
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 模型超参数
    model = TransformerTextClassifier(
        vocab_size=len(vocab),
        d_model=16,       # 模型维度
        n_heads=4,        # 多头数
        n_layers=2,       # 几层 transformer
        d_ff=64,          # 4 * 16 = 64
        max_len=max_len,
        dropout=0.1,
    )
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, dataloader, criterion, optimizer, device
        )
        
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={train_loss:.6f}, accuracy={train_acc:.4f}")
    
    # 测试
    test_texts = [
        ["我", "喜欢", "这个", "模型"],
        ["错误", "很多"],
        ["这个", "方法", "很", "糟糕"],
    ]
    
    probabilities, predictions, all_attn_weights = predict_texts(
        model=model,
        tokenized_texts=test_texts,
        vocab=vocab,
        max_len=max_len,
        device=device
    )
    
    print()
    print("Test probabilities:")
    print(probabilities)
    print()
    print("Test predictions:")
    print(predictions)
    print()
    print(f"Number of transformer layers: {len(all_attn_weights)}")
    print(f"Attention weights shape per layer: {all_attn_weights[0].shape}")
    print()
    print("Test texts:")
    for tokens in test_texts:
        print(tokens)


if __name__ == "__main__":
    main()