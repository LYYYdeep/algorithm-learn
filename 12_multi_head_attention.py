import torch
from torch import nn
import torch.nn.functional as F


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
    

def test_multi_head_attention():
    """测试 Multi-Head Attention 形状"""
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 6
    d_model = 16
    n_heads = 4
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # attention mask
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],  # 长度 4
        [1, 1, 1, 1, 1, 1],  # 长度 6
    ], dtype=torch.long)
    
    # 模型
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # 前向
    out, attention_weights = mha(x, attention_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()
    print(f"Expected output shape: [batch={batch_size}, seq_len={seq_len}, d_model={d_model}]")
    print(f"Expected attention shape: [batch={batch_size}, n_heads={n_heads}, seq_len={seq_len}, seq_len={seq_len}]")
    
    # 验证形状正确
    assert out.shape == (batch_size, seq_len, d_model), f"Wrong output shape {out.shape}"
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Wrong attn shape {attention_weights.shape}"
    print()
    print("✅ Shape test passed!")
    
    # 检查 padding 位置权重
    print()
    print("第一个 batch，第一个 head，第一个 query 的注意力权重:")
    print(attention_weights[0, 0, 0, :])
    print(f"Padding位置权重之和: {attention_weights[0, 0, 0, 4:].sum():.6f} (应该接近 0)")


if __name__ == "__main__":
    test_multi_head_attention()