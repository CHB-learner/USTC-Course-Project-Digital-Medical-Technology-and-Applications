#!/usr/bin/env bash
set -e

# 仅使用 GPU: 0,1
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --host 127.0.0.1 --port 8001

