import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from peft import PeftModel, PeftConfig

# ===== 加载 =====
base_model_name = "Qwen/Qwen2-0.5B-Instruct"
lora_model_path = "./qwen2-0.5b-lora/final"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

print(f"Loaded model and LoRA successfully!")

# ===== 对话生成 =====
def generate_response(
    instruction,
    input_text="",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    # Qwen 聊天模板
    if input_text.strip() != "":
        prompt = f"<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取助手回答
    if "<|im_start|>assistant\n" in output_text:
        output_text = output_text.split("<|im_start|>assistant\n")[-1]
    
    return output_text.strip()

# ===== 交互式对话 =====
print("\n" + "="*60)
print("Qwen2-0.5B LoRA 对话演示 (输入 'quit' 退出)")
print("="*60 + "\n")

while True:
    instruction = input("用户: ")
    if instruction.lower() == "quit":
        break
    response = generate_response(instruction)
    print(f"助理: {response}\n")
