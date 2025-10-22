#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=3,6

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --served-model-name llama \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --host 127.0.0.1 --port 8002
