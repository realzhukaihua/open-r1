import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from src.open_r1.sft_article_quality import make_conversation_for_sft_trainer

# ✅ 加载模型和 Tokenizer
model_name = "data/Qwen2.5-7B"  # 替换为你的模型路径或模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# ✅ 构造输入 Prompt
dataset = load_dataset("json", data_files={"train": "data/train.s02.all.jsonl", "test": "data/dev.all.jsonl"})
model_name = "data/Qwen2.5-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('data/Qwen2.5-7B')
tokenizer.pad_token = tokenizer.eos_token
# Initialize SFTTrainer (for data processing only)
    # Format into conversation

dataset['test'] = dataset['test'].select(range(100))
dataset['test'] = dataset['test'].map(lambda example: make_conversation_for_sft_trainer(example, tokenizer=tokenizer))

# ✅ 使用 ChatML 模板（如果模型支持）
for example in dataset['test']:
    input_chat = example["input_text"]
    # input_text = tokenizer.apply_chat_template(input_chat, tokenize=False)
    input_ids = tokenizer(input_chat, return_tensors="pt").input_ids.to(model.device)

    # ✅ 生成文本并获取每个 Token 的概率
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,  # 控制生成长度
            return_dict_in_generate=True,  # 让输出是一个字典
            output_scores=True,  # 获取每个 Token 的概率
            do_sample=False  # 关闭随机采样，使用贪心或 Beam Search
        )

    # ✅ 解码输出文本
    generated_tokens = output.sequences[0][input_ids.shape[1]:]  # 只取新生成的部分
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generated Text:", generated_text)

    # ✅ 解析每个 Token 的概率
    logits = torch.stack(output.scores, dim=1)  # (batch_size, seq_len, vocab_size)
    probs = torch.nn.functional.softmax(logits, dim=-1)  # 转换为概率

    # ✅ 打印每个 Token 及其概率
    print("\nToken Probabilities:")
    for i, token_id in enumerate(generated_tokens):
        token_str = tokenizer.decode([token_id])  # 解码 Token
        token_prob = probs[0, i, token_id].item()  # 取该 Token 的概率
        print(f"Token: {token_str}, Probability: {token_prob:.6f}")
