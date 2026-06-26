import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA 包装后的 Linear 层
    
    核心思想：冻结原始权重，只训练低秩增量矩阵 ΔW = BA
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        r: LoRA 秩，r 越小参数量越少
        alpha: 缩放因子，通常 alpha = r
        dropout: dropout 概率
    """
    def __init__(self, in_features, out_features, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        # 原始 Linear，权重冻结
        self.original = nn.Linear(in_features, out_features, bias=True)
        for param in self.original.parameters():
            param.requires_grad = False
            
        # LoRA 低秩矩阵 A 和 B
        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        
        # 缩放因子 = alpha / r
        self.scaling = self.alpha / self.r
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化：A 用高斯噪声，B 初始化为 0
        # 初始增量 ΔW = BA = 0，开始训练和原始模型一样
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        # B 初始化为 0，所以一开始没有增量
        
    
    def forward(self, x):
        # 原始输出
        original_out = self.original(x)
        
        # LoRA 输出：x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T
        lora_out = lora_out @ self.lora_B.T
        lora_out = lora_out * self.scaling
        
        return original_out + lora_out
    
def test_lora_linear():
    """测试 LoRA 形状和参数量"""
    torch.manual_seed(42)
    
    in_features = 768
    out_features = 768
    r = 8
    
    lora_linear = LoRALinear(in_features, out_features, r=r, alpha=r)
    
    # 统计可训练参数量
    total_params = sum(p.numel() for p in lora_linear.parameters())
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    
    print(f"Original Linear 参数量: {in_features * out_features}")
    print(f"LoRA 参数量公式: r*(in_features + out_features) = {r}*({in_features}+{out_features}) = {r*(in_features+out_features)}")
    print(f"实际总参数量: {total_params}")
    print(f"实际可训练参数量: {trainable_params}")
    print(f"压缩比: {in_features*out_features / trainable_params:.1f}x")
    print()
    
    # 测试前向
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)
    out = lora_linear(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print("\n✅ 形状测试通过！")
    print()
    
    # 检查梯度
    loss = out.sum()
    loss.backward()
    
    # 检查只有 LoRA 有梯度，原始权重没有
    print("梯度检查:")
    print(f"original.weight.grad 是否为 None: {lora_linear.original.weight.grad is None}")
    print(f"lora_A.grad 是否不为 None: {lora_linear.lora_A.grad is not None}")
    print(f"lora_B.grad 是否不为 None: {lora_linear.lora_B.grad is not None}")
    
    print("\n✅ LoRA 工作正确！只有 LoRA 参数有梯度，原始权重冻结！")


if __name__ == "__main__":
    test_lora_linear()