import torch
import torch.nn as nn

def test_memory_compare():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return
    
    print("Testing FP32 vs BF16 memory usage on MPS")
    
    # 大模型对比
    batch_size = 16
    seq_len = 512
    hidden_size = 1024
    num_layers = 12
    
    # ========== FP32 ==========
    from collections import OrderedDict
    layers = OrderedDict()
    for i in range(num_layers):
        layers[f"layer{i}"] = nn.Linear(hidden_size, hidden_size)
    model = nn.Sequential(layers).to("mps")
    x = torch.randn(batch_size, seq_len, hidden_size).to("mps")
    
    mem_before = torch.mps.current_allocated_memory() / 1024**2
    print(f"\nFP32:")
    print(f"Memory before: {mem_before:.2f} MB")
    
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    mem_after = torch.mps.current_allocated_memory() / 1024**2
    print(f"Memory after forward+backward: {mem_after:.2f} MB")
    print(f"Increase: {mem_after - mem_before:.2f} MB")
    
    del model, x, out, loss
    torch.mps.empty_cache()
    
    # ========== BF16 ==========
    from collections import OrderedDict
    layers = OrderedDict()
    for i in range(num_layers):
        layers[f"layer{i}"] = nn.Linear(hidden_size, hidden_size)
    model = nn.Sequential(layers).to("mps")
    x = torch.randn(batch_size, seq_len, hidden_size).to("mps")
    
    mem_before = torch.mps.current_allocated_memory() / 1024**2
    print(f"\nBF16 mixed precision:")
    print(f"Memory before: {mem_before:.2f} MB")
    
    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        out = model(x)
        loss = out.mean()
        loss.backward()
    
    mem_after = torch.mps.current_allocated_memory() / 1024**2
    print(f"Memory after forward+backward: {mem_after:.2f} MB")
    print(f"Increase: {mem_after - mem_before:.2f} MB")
    
    print("\n✅ Done! BF16 should use about half memory compared to FP32")

if __name__ == "__main__":
    test_memory_compare()
