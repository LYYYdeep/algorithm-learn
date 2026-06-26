import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    基础的 Self-Attention 实现
    
    直觉：
        每个位置在输出表示的时候，需要关注输入序列中其他位置的信息，
        并根据相关性分配不同的权重。
    
    输入：
        x: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len], 1 表示真实 token，0 表示 padding
        
    输出：
        out: [batch_size, seq_len, d_model]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"
        
        # 1.计算 Q, K, V
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 2.计算注意力分数: Q * K^T / sqrt(d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)  # [batch_size, seq_len, seq_len]
        
        # 3.如果有 mask, 将 padding 位置的分数设置为 -inf
        if attention_mask is not None:
            # attention_mask [batch, seq_len] → [batch, 1, seq_len]
            # 每个 query 位置都需要 mask 同一个 key 位置
            attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 4.sotfmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 5. 用注意力权重对 V 进行加权求和
        out = torch.matmul(attention_weights, v)  # [batch_size, seq_len, d_model]
        
        return out, attention_weights
    
    
def test_self_attention():
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 5
    d_model = 8
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # attention mask
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],  # 第一个序列有 3 个有效 token
        [1, 1, 1, 1, 1],  # 第二个序列有 4 个有效 token
    ], dtype=torch.long)
    
    # 初始化模型
    self_attn = SelfAttention(d_model=d_model)
    
    # 前向传播
    out, attention_weights = self_attn(x, attention_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()
    print("第一个句子的注意力权重（第一个 query 行）:")
    print(attention_weights[0, 0, :])  # 第一个query对各个key的权重
    
    # 检查 padding 位置权重是否接近 0
    print()
    print(f"Padding位置（索引3,4）的权重之和: {attention_weights[0, 0, 3:].sum():.6f}")
    
    # 检查每行权重和是否接近 1
    print(f"第一个query权重和: {attention_weights[0, 0, :].sum():.6f}")


if __name__ == "__main__":
    test_self_attention()