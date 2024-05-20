"""
Dockerize the Huggingface TGI Inference Container:

%%bash
# model=$PWD/{args.output_dir} # path to model
model=$(pwd)/code-llama-7b-text-to-sql # path to model
num_shard=1             # number of shards
max_input_length=1024   # max input length
max_total_tokens=2048   # max total tokens

docker run -d --name tgi --gpus all -ti -p 8080:80 \
  -e MODEL_ID=/workspace \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -v $model:/workspace \
  ghcr.io/huggingface/text-generation-inference:latest

"""
### Ref: dockerizing and running finetuning/text_generation/sql_from_text_finetuned.py


import requests as r
from transformers import AutoTokenizer
from datasets import load_dataset
from random import randint

# Load our test dataset and Tokenizer again
tokenizer = AutoTokenizer.from_pretrained("code-llama-7b-text-to-sql")
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))

# generate the same prompt as for the first local test
prompt = tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
request= {"inputs":prompt,"parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}

# send request to inference server
resp = r.post("http://127.0.0.1:8080/generate", json=request)

output = resp.json()["generated_text"].strip()
time_per_token = resp.headers.get("x-time-per-token")
time_prompt_tokens = resp.headers.get("x-prompt-tokens")

# Print results
print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{output}")
print(f"Latency per token: {time_per_token}ms")
print(f"Latency prompt encoding: {time_prompt_tokens}ms")