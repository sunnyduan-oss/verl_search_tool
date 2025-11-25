#!/bin/bash
set -e

# -----------------------------
# 配置参数
# -----------------------------
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen2.5-VL-3B-Instruct"
QUESTION="Who is the character in the image?"
IMAGE_URL="https://i.imgur.com/5bGzZi7.jpg"
API_KEY="9e1d4a5257c55bf7ca3c2bd8232157419fa3136740dd2048ed2ffb59cb39ddd8"

echo "Searching with $MODEL_NAME ..."
python search/main_search.py \
    model.path="$MODEL_PATH" \
    model.name="$MODEL_NAME" \
    data.question="$QUESTION" \
    data.image_url="$IMAGE_URL" \
    data.api_key="$API_KEY" \

echo "Searching finished."