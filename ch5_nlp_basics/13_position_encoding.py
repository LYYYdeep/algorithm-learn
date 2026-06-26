import torch
from torch import nn


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
    
    

def test_position_encoding():
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 10
    d_model = 16
    
    x = torch.zeros(batch_size, seq_len, d_model)
    
    pe = PositionEncoding(d_model=d_model, max_len=512, dropout=0.1)
    
    out = pe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print()
    print(f"PE shape stored in model: {pe.pe.shape}")
    print(f"First position (pos=0):")
    print(pe.pe[0, 0, :8])  # 前8维
    print()
    print(f"Second position (pos=1):")
    print(pe.pe[0, 1, :8])
    
    # 验证形状
    assert out.shape == (batch_size, seq_len, d_model), "Wrong shape"
    print()
    print("✅ Shape test passed!")
    
    # 检查正弦余弦规律
    print()
    print("Check sin/cos pattern for pos=2:")
    print(f"PE[2, 0] = {pe.pe[0, 2, 0]:.6f} = sin(2 / 10000^(0/16)) = {torch.sin(torch.tensor(2 * torch.exp(-torch.log(torch.tensor(10000.0)) * 0 / 16))):.6f}")
    print(f"PE[2, 1] = {pe.pe[0, 2, 1]:.6f} = cos(2 / 10000^(0/16)) = {torch.cos(torch.tensor(2 * torch.exp(-torch.log(torch.tensor(10000.0)) * 0 / 16))):.6f}")


if __name__ == "__main__":
    test_position_encoding()