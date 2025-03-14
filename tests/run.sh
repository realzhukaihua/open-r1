export CUDA_VISIBLE_DEVICES="3"
# vn=v8_hotel_shopping_query_first_v2
# model_name=mt_qisa_${vn}_merged

vn="v8"
model_name="v11_fulltrain_"${vn}

#vn="v1"
#model_name="text"_${vn}

port=9112
sudo docker run --gpus '"device=3"' -v /mnt/share16t/zhaozhenyu/models:/models --shm-size 4g -p ${port}:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /models/${model_name} --dtype bfloat16 --max-total-tokens 4096 --cuda-memory-fraction 0.4 --max-input-length 3000 --sharded false


# sudo docker run --gpus '"device=4"' -v /mnt/share16t/zhaozhenyu/models:/models --shm-size 4g -p ${port}:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /models/${model_name} --dtype bfloat16 --max-total-tokens 4096 --cuda-memory-fraction 0.4 --max-input-length 3000 --sharded false

sudo docker run --name dpo0130 --gpus '"device=1"' -v $PWD:/data --shm-size 1g -p 8301:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0130_0.1_dpo_his_0.2 --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --name dpo0130 --gpus '"device=1"' -v $PWD:/data --shm-size 1g -p 8301:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0130_0.1_dpo_his_0.01 --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --name dpo0130_0.05 --gpus '"device=6"' -v $PWD:/data --shm-size 1g -p 8303:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0130_0.1_dpo_his_0.05 --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --name dpo0130_0.01_kto --gpus '"device=1"' -v $PWD:/data --shm-size 1g -p 8301:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0130_0.1_dpo_his_0.01_kto --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --name dpo0131_0.03_2e6 --gpus '"device=5"' -v $PWD:/data --shm-size 1g -p 8301:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0201_dpo_0.03 --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --name llama_13bf_0203_0.1_kto --gpus '"device=7"' -v $PWD:/data --shm-size 1g -p 8301:80 ghcr.io/huggingface/text-generation-inference:sha-9c320e2 --model-id /data/exp/llama_13bf_0202_0.1_kto --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --rm --name geotagseq2seq --gpus '"device=1"' -v /mnt/share16t/kaihua/geotag_llm:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:1.3.0 --model-id /data/geotag_llm_0406_seq2seq --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --rm --name geotagseq2seq --gpus '"device=6"' -v /mnt/share16t/kaihua/geotag_llm:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:1.4.5 --model-id /data/geotag_llm_0408_seq2seq_gemma --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --rm --name fact_check --gpus '"device=7"' -v /mnt/share16t/kaihua/fact_check:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:1.3.0 --model-id /data/fact_check_0407 --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095
sudo docker run --rm --name geotagseq2seq --gpus '"device=7"' -v /mnt/share16t/kaihua/geotag_llm:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:1.4.5 --model-id /data/geotag_llm_0410_cls_gemma --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095

sudo docker run --rm --name geotag_cls --gpus '"device=7"' -v /mnt/share16t/kaihua/geotag_llm:/data --shm-size 1g -p 8300:80 ghcr.io/huggingface/text-generation-inference:2.0.0 --model-id /data/geotag_llm_0410_cls_gemma --dtype bfloat16 --max-total-tokens 4096 --sharded false --max-input-length 4095

# best kto:llama_13bf_0202_0.1_kto dpo:llama_13bf_0130_0.1_dpo_his_0.05
# local container service test 
curl 127.0.0.1:8300/generate -X POST  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'


# TEI
sudo docker run --rm --name geotag_cls --gpus '"device=6"' -p  830:80 -v /mnt/share16t/kaihua/geotag_llm:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.2 --model-id /data/geotag_llm_0410_cls_gemma 

# claims extraction
sudo docker run --rm   --name claims --gpus '"device=5"' -v /data5/kaihua/models/claims-extraction-allin1/exp/20241219changepadtoken:/data --shm-size 4g -p 8002:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000
# 49
sudo docker run --rm  -d --name claims --gpus '"device=7"' -v /data5/kaihua/models/claims-extraction-allin1/exp/1130/checkpoint-2479:/data --shm-size 4g -p 8300:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000
sudo docker run --rm  -d --name claimsp2 --gpus '"device=6"' -v /data5/kaihua/models/claims-extraction-allin1/exp/1130/checkpoint-4958:/data --shm-size 4g -p 8301:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000
sudo docker run --rm  -d --name claimsp3 --gpus '"device=5"' -v /data5/kaihua/models/claims-extraction-allin1/exp/1130/checkpoint-7437:/data --shm-size 4g -p 8302:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000
sudo docker run --rm  --name claims --gpus '"device=3,4"' -e CUDA_VISIBLE_DEVICES=0,1  -v /data5/kaihua/models/claims-extraction-allin1/exp/20241204_newvalnewtrain:/data --shm-size 64g -p 8300:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000
# same-event
sudo docker run --rm  --name same-event --gpus '"device=3,4"' -e CUDA_VISIBLE_DEVICES=0,1  -v /data5/claims/models/event-tracking-allin1/exp/20241204_newvalnewtrain:/data --shm-size 64g -p 8300:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /data --dtype bfloat16 --max-total-tokens 8192 --cuda-memory-fraction 0.8 --sharded false --max-input-length 8000 --max-batch-prefill-tokens 8000

-e CUDA_VISIBLE_DEVICES=3,4



# vllm docker deployment
docker pull vllm/vllm-openai:v0.6.4.post1
sudo docker run  --rm --gpus '"device=6,7"'  -p 8002:8000 -v /mnt/share/kaihua/20250105:/models vllm/vllm-openai --model /models --served-model-name claims_extraction --tensor-parallel-size 2
sudo docker run  --rm --gpus '"device=5"'  -p 8001:8000 -v /data6/kaihua/claims/models/event-tracking-allin1/exp/20241216:/models vllm/vllm-openai --model /models --served-model-name same_event

# test
curl http://localhost:8018/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "gpttest",
    "prompt": "San Francisco is a",
    "max_tokens": 50,
    "temperature": 0.7
}'

curl http://localhost:8019/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "gpttest",
    "prompt": "San Francisco is a",
    "max_tokens": 50,
    "temperature": 0.7
}'

#deploy mul
sudo docker run  --rm --gpus '"device=6,7"'  -p 8002:8000 -v /mnt/share/kaihua/20250105:/models vllm/vllm-openai --model /models --served-model-name claims_extraction --tensor-parallel-size 2


CUDA_VISIBLE_DEVICES=7 vllm serve data/Qwen2.5-7B-v2 --port 8018 --served-model-name gpttest

CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-Math-7B --port 8019 --served-model-name gpttest

python /home/kaihua/open-r1/tests/vllm_inference.py     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml


CUDA_VISIBLE_DEVICES=4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml  src/open_r1/sft_article_quality.py --config recipes/article_quality/sft/config.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml  src/open_r1/sft_article_quality.py --config recipes/article_quality/sft/config.yaml

CUDA_VISIBLE_DEVICES=4,5   ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml     src/open_r1/sft.py     --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml