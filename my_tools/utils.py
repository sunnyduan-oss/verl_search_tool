import os
import json
import torch
import numpy as np
from PIL import Image
import io
import requests
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def postprocess_serpapi_results(serp_results, max_items=3, max_images_per_item=1):
    """
    将 SerpApi 搜索结果转成模型可用的图文上下文

    Args:
        serp_results (list): SerpApi 返回的列表，每个 item 是一个 dict
        max_items (int): 最多保留的网页数量
        max_images_per_item (int): 每条网页最多保留的图片数量

    Returns:
        list of dict: 每个 dict 代表一个图文片段
            {
                "text": "<title>\n<snippet>\n<displayed_link>",
                "images": ["https://...", ...]  # 最多 max_images_per_item
            }
    """
    processed = []

    for item in serp_results[:max_items]:
        # ------------------------
        # 文本部分
        # ------------------------
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("displayed_link") or item.get("link", "")
        text = f"{title}\n{snippet}\n{link}".strip()

        # ------------------------
        # 图片部分
        # ------------------------
        images = []
        if "images" in item:
            images = item["images"][:max_images_per_item]
        elif "thumbnail" in item:
            images = [item["thumbnail"]]
        elif "favicon" in item:
            images = [item["favicon"]]

        processed.append({
            "text": text,
            "images": images
        })

    return processed


def build_multimodal_prompt(text_question, original_image_url, base_knowledge_list):
    """
    构建 VERL 格式的多模态提示消息

    Args:
        text_question (str): 用户提问
        original_image_url (str): 用户提供的主要图片 URL
        base_knowledge_list (list of dict): 每个 dict 包含 'text' 和 'images'，作为检索知识

    Returns:
        list of dict: VERL 格式的消息
    """
    
    # ============================
    # 0. System Instructions
    # ============================
    instructions = (
        "You are a multimodal question-answering assistant. Please answer the question in one sentence.\n"
        "You will receive:\n"
        "1. A user image\n"
        "2. Retrieved webpage information (each contains text and/or images)\n"
        "3. A user question\n\n"
        "Your task:\n"
        "- Understand the user image\n"
        "- Read all retrieved texts and examine all retrieved images\n"
        "- Use *only* these provided materials to answer the question\n"
        "- If the answer cannot be determined from the given sources, reply: "
        "\"I cannot determine based on the given information.\"\n"
        "- Keep the answer concise and factual.\n"
    )

   # ============================
    # 1. Start building content list
    # ============================
    content = []

    # ============================
    # Add system instructions + question
    # ============================
    final_text = instructions
    final_text += "\n### User Question\n" + text_question.strip()
    content.append({"type": "text", "text": final_text})
    
    # ----------------------------
    # Add user main image
    # ----------------------------
    if original_image_url:
        content.append({"type": "image", "image": original_image_url})

   # ============================
    # 2. Add retrieved knowledge
    # ============================
    if base_knowledge_list:
        for bk in base_knowledge_list:
            bk_text = bk.get("text", "").strip()
            bk_images = bk.get("images", [])

            if bk_text:
                content.append({"type": "text", "text": bk_text})

            for img_url in bk_images:
                content.append({"type": "image", "image": img_url})

    # ============================
    # 3. Construct VERL-format messages
    # ============================
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    return messages
