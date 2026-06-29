import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class ModelWithCheckpoint(nn.Module):
    def __init__(self, num_layers, hidden_size, use_gradient_checkpointing=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden_size) for _ in range(num_layers)])
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    def forward(self, x):
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                # 🔥 修复：use_reentrant=False 才不会没梯度
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def test_memory_compare_mps():
    if not torch.backends.mps.is_available():
        print("MPS not available, this example needs Apple GPU")
        return
    
    num_layers = 48
    hidden_size = 1024
    batch_size = 8
    seq_len = 512
    
    print(f"Testing on MPS: num_layers={num_layers}, hidden_size={hidden_size}, batch={batch_size}, seq_len={seq_len}")
    
    # ========== 不用梯度检查点 ==========
    model = ModelWithCheckpoint(num_layers, hidden_size, use_gradient_checkpointing=False)
    model = model.to("mps")
    x = torch.randn(batch_size, seq_len, hidden_size).to("mps")
    
    mem_before = torch.mps.current_allocated_memory() / 1024**2
    print(f"\nWithout gradient checkpointing:")
    print(f"Memory allocated before forward: {mem_before:.2f} MB")
    
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    mem_after = torch.mps.current_allocated_memory() / 1024**2
    print(f"Memory allocated after forward+backward: {mem_after:.2f} MB")
    print(f"Increase: {mem_after - mem_before:.2f} MB")
    
    del model, x, out, loss
    torch.mps.empty_cache()
    
    # ========== 用梯度检查点 ==========
    model = ModelWithCheckpoint(num_layers, hidden_size, use_gradient_checkpointing=True)
    model = model.to("mps")
    x = torch.randn(batch_size, seq_len, hidden_size).to("mps")
    
    mem_before = torch.mps.current_allocated_memory() / 1024**2
    print(f"\nWith gradient checkpointing:");
    print(f"Memory allocated before forward: {mem_before:.2f} MB");
    
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    mem_after = torch.mps.current_allocated_memory() / 1024**2
    print(f"Memory allocated after forward+backward: {mem_after:.2f} MB");
    print(f"Increase: {mem_after - mem_before:.2f} MB");
    
    print("\n✅ Done! You should see MUCH lower memory usage with gradient checkpointing");
    print("Gradient checkpointing trades computation for memory: slower but uses much less memory");


if __name__ == "__main__":
    test_memory_compare_mps()
