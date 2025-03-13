import json
import requests
from transformers import AutoTokenizer
from trl import SFTTrainer
from datasets import load_dataset
from transformers import set_seed
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
# from open_r1.configs import SFTConfig, GRPOConfig
# from open_r1.rewards import accuracy_reward_simple
import pandas as pd
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
# from open_r1.configs import GRPOConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.  The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
def process_example(example, VLLM_SERVER_URL, tokenizer):
    """处理单个示例并返回结果"""
    # chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": example['problem']}]
    prompt = example['prompt']
    payload = {
        "prompt": example['prompt'],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "model": "gpttest",
    }

    try:
        response = requests.post(VLLM_SERVER_URL, json=payload)
        if response.status_code == 200:
            outputs = response.json().get("choices", [])
            for output in outputs:
                print(prompt, output['text'])
                # reward = accuracy_reward_simple(output['text'], example['solution'])
                # result = [prompt, output['text'], reward]
                # print(result)
                # return result
        else:
            print(f"Error: {response.text}")
            return [prompt, None, None]  # 处理失败情况
    except Exception as e:
        print(f"Request failed: {e}")
        return [prompt, None, None]  # 处理异常情况

def main():
    # Set seed for reproducibility
    # Define vLLM server endpoint
    VLLM_SERVER_URL = "http://localhost:8018/v1/completions"  # Change if different

    # Load tokenizer
    # model_name = "data/Qwen2.5-1.5B-Open-R1-Distill"  # Example model
    dataset = load_dataset("json", data_files={"train": "data/train.s02.all.jsonl", "test": "data/dev.all.jsonl"})
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained('data/Qwen2.5-7B')
    tokenizer.pad_token = tokenizer.eos_token
    # Initialize SFTTrainer (for data processing only)
        # Format into conversation
    def make_conversation_for_sft_trainer(example):
        # Build the prompt
        # Truncate the article content to the top 7000 tokens
        content_tokens = tokenizer.tokenize(example['seg_content'])
        truncated_content = tokenizer.convert_tokens_to_string(content_tokens[:7000])

        prompt = (
            f"This is article title:{example['seg_title']}\n"
            f"This is article content:{truncated_content}\n"
            f"This is article domain:{example['domain']}\n"
            f"This is article category:{example['category']}\n"
            f"You should help to judge this article quality with reason and final label. "
            "Please output with json format like {\"reason\":***,\"moderation\":***,\"label\":***}"
        )
        # Create the label as a JSON string
        reason = ''
        moderation = ''
        for item in ['ok','detrimental','poor']:
            if example["reason"].get(item) is not None:
                reason = example["reason"].get(item)[0]
                moderation = item

        # print(example)
        label = json.dumps({"reason": reason, 'moderation':moderation, "label": example["status"]})
        # Construct the conversation structure
        input_chat = [
            {"role": "system", "content": "You are a helpful AI Assistant and good at content moderation."},
            {"role": "user", "content": prompt}
        ]
        output_chat = [{"role": "assistant", "content": label}]
        # Convert conversation to text using the chat template
        # full_text = tokenizer.apply_chat_template(input_chat+output_chat, tokenize=False)
        input_text = tokenizer.apply_chat_template(input_chat, tokenize=False)
        return {
            "prompt":input_text
        }
    dataset['test'] = dataset['test'].select(range(100))
    dataset['test'] = dataset['test'].map(make_conversation_for_sft_trainer)

    for example in dataset['test']:
        process_example(example, VLLM_SERVER_URL, tokenizer)
    # results = []
    # max_threads = min(10, len(dataset["test"]))  # 限制最大线程数，避免服务器过载

    # with ThreadPoolExecutor(max_threads) as executor:
    #     future_to_example = {
    #         executor.submit(process_example, example, VLLM_SERVER_URL, tokenizer): example
    #         for example in dataset["test"]
    #     }

    #     for future in as_completed(future_to_example):
    #         result = future.result()
    #         if result:
    #             results.append(result)
    # df = pd.DataFrame(results, columns=['prompt','output','reward'])
    # print(df['reward'].mean())
    # df.to_csv('tests/vllm_inference.csv',index=False)


if __name__ == "__main__":
    main()