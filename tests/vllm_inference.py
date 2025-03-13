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
from open_r1.configs import SFTConfig, GRPOConfig
from open_r1.rewards import accuracy_reward_simple
import pandas as pd
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.configs import GRPOConfig
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
    chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": example['problem']}]
    prompt = tokenizer.apply_chat_template(conversation=chat, tokenize=False, add_generation_prompt=True)

    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 3500,
        "model": "gpttest",
    }

    try:
        response = requests.post(VLLM_SERVER_URL, json=payload)
        if response.status_code == 200:
            outputs = response.json().get("choices", [])
            for output in outputs:
                reward = accuracy_reward_simple(output['text'], example['solution'])
                result = [prompt, output['text'], reward]
                print(result)
                return result
        else:
            print(f"Error: {response.text}")
            return [prompt, None, None]  # 处理失败情况
    except Exception as e:
        print(f"Request failed: {e}")
        return [prompt, None, None]  # 处理异常情况

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    # Define vLLM server endpoint
    VLLM_SERVER_URL = "http://localhost:8018/v1/completions"  # Change if different

    # Load tokenizer
    # model_name = "data/Qwen2.5-1.5B-Open-R1-Distill"  # Example model
    model_name = "Qwen/Qwen2.5-Math-7B"
    tokenizer = AutoTokenizer.from_pretrained('data/Qwen2.5-7B-v2')

    # Load dataset
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", name=script_args.dataset_config)
    dataset["train"] = dataset["train"].select(range(1000))  # 仅限于 "train" 集
    if "train" in dataset:
        # 按 90% 训练，10% 评估拆分
        dataset = dataset["train"].train_test_split(test_size=0.1,seed=2047)
        dataset["test"] = dataset["test"].select(range(20))
    # Initialize SFTTrainer (for data processing only)
        # Format into conversation
    def make_conversation_for_grpo_trainer(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        chat = [{"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"},
                {"role": "user", "content": example["problem"]}
                ]
        example['prompt'] = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False,
            add_generation_prompt=True
        )
        return example
    
    def make_conversation_for_sft_trainer(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        chat = [{"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"}
                ]+example['messages']
        example['text'] = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False
        )
        return example

    # dataset = dataset.map(make_conversation_for_sft_trainer)
    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")
            
    # trainer = SFTTrainer(
    #     model=model_args.model_name_or_path,
    #     args=training_args,
    #     train_dataset=dataset[script_args.dataset_train_split],
    #     eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    #     processing_class=tokenizer,
    #     peft_config=get_peft_config(model_args)
    #     # compute_metrics=compute_metrics  # 添加自定义 metrics
    # )
    # Get DataLoader from trainer
    # train_dataloader = trainer.get_train_dataloader()
    # for item in train_dataloader:
    #     # print(item)
    #     input_ids = item["input_ids"][0]  # Get the first example from the batch
    #     restored_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)  # Convert tokens back to text
    #     print("Restored Prompt:", restored_prompt)
    #     break  # Only print one example
    # return
    # train_dataloader = trainer.get_eval_dataloader()

    # Iterate through batches and send requests to vLLM
        # **多线程并发请求**
    results = []
    max_threads = min(10, len(dataset["test"]))  # 限制最大线程数，避免服务器过载

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_example = {
            executor.submit(process_example, example, VLLM_SERVER_URL, tokenizer): example
            for example in dataset["test"]
        }

        for future in as_completed(future_to_example):
            result = future.result()
            if result:
                results.append(result)
    df = pd.DataFrame(results, columns=['prompt','output','reward'])
    print(df['reward'].mean())
    df.to_csv('tests/vllm_inference.csv',index=False)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)