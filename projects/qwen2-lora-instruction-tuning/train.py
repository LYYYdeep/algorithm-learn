import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== 配置 ======
model_name = "Qwen/Qwen2-0.5B-Instrict"
dataset_name = "silk-road/alpaca-data-gpt4-chinese"
output_dir = "./qwen2-0.5b-lora"

# LoRA 配置
lora_r = 8
lora_alpha = 16
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_dropout = 0.05

# 训练配置
num_epochs = 3
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 1e-4
max_seq_length = 512
logging_steps = 10
save_total_limit = 3

# ===== 设置设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token
tokenizer.padding_side = "right"  # 设置 padding 方向为右侧

# ===== 加载数据集 =====
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")
print(f"Dataset loaded with {len(dataset)} samples.")

# 格式化对话数据：用 Qwen 聊天模版
def format_and_tokenize(examples):
    texts = []
    for instruction, input_text in zip(examples["instruction"], examples["input"]):
        # Qwen 聊天模板
        if input_text.strip() != "":
            prompt = f"<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        texts.append(prompt)
        
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=dataset.column_names,
)

# data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ===== 加载模型 =====
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 准备模型进行 k-bit 训练（我们这里不用量化，但是 prepare_model_for_kbit_training 也能帮我们处理好梯度）
model = prepare_model_for_kbit_training(model)

# ===== 配置 LoRA =====
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数

# ===== 训练参数 =====
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    logging_steps=logging_steps,
    save_strategy="epoch",
    save_total_limit=save_total_limit,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ===== 开始训练 =====
print("Start training...")
model.config.use_cache = False  # 关闭cache训练更稳定
trainer.train()

# 保存最后的 LoRA
model.save_pretrained(output_dir + "/final")
tokenizer.save_pretrained(output_dir + "/final")

print(f"\nTraining finished! LoRA saved to {output_dir}/final")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")