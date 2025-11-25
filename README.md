<div align="center">

# ✨VERL-based Retrieval-Augmented Multimodal QA Tool✨

<p align="center">
    <a href="https://github.com/">Duan Sangni</a><sup>1</sup>
    <br>
    <sup>1</sup>The University of Hong Kong
</p>

</div>

## 📘 Overview

This project provides a **Retrieval-Augmented Reasoning (RAR) tool** built on the **VERL** framework to enhance multimodal large language models (MLLMs).  
When an MLLM receives only an **(image, text)** input, it may lack the necessary external knowledge to answer accurately.  
This tool solves that problem through retrieval augmentation.

### 🔍 What the tool does

1. Takes an **image + text query** from the multimodal model.  
2. Performs **image-based web search** using Google Search API (via SerpAPI).  
3. Retrieves webpage thumbnails, captions, and titles as external evidence.  
4. Feeds the retrieved evidence back to the model to produce a more accurate answer.

### 🌟 Highlights

- Fully compatible with **a wide range of multimodal foundation models** (Qwen, InternVL, LLaVA, etc.)
- Drop-in enhancement module for **improved reasoning & factual grounding**.
- Clean design + modular tool interface for VERL.

---

## 📂 Project Structure
This project is built on the **volcengine/VERL** framework:

- **`search/`**: Contains the core logic, including model queries, web search API calls, and evidence integration.  

- **`my_tools/`**: Contains all custom utilities for this project.  

- **`scripts/`**: One-click shell scripts.  

- **`verl/`**: The VERL submodule.

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/sunnyduan-oss/verl_search.git
cd verl_search
```

### 2. Environment Setup

1. **Create conda environment**

```bash
conda create -n verl_env python=3.11 -y
conda activate verl_env
```

2. **Install required packages for inference and evaluation**

```bash
pip install -r requirements.txt
cd transformers && pip install -e . && cd ..
cd verl && pip install -e . && cd ..
```

## 💻 Run the Tool

1. **Downdload the model**

```bash
modelscope download --model 'Qwen/Qwen2.5-VL-3B-Instruct' --local_dir your-model-path
```
2. **Run the Search Tool**

You can run everything with the provided script:
```bash
bash scripts/run_search.sh \
    --model.path your-model-path \
    --model.name Qwen2.5-VL-3B-Instruct \
    --data.question Who is the character in the image?
    --data.image_url your-image-url \
    --data.api_key your-serpapi-key
```

---

## 📊 Quick Start Diagram

Below is a simple workflow showing how the retrieval-augmented multimodal QA system operates:

```sql
    ┌──────────────┐
    │    Query      │
    │ (Image+Text)  │
    └───────┬──────┘
            │
            ▼
    ┌──────────────┐
    │ Preprocessing │
    └───────┬──────┘
            │
            ▼
    ┌──────────────────┐
    │  Multimodal LLM  │
    │    (Tool Call)   │
    └───────┬─────────┘
            │ triggers web search
            ▼
 ┌──────────────────────────┐
 │   Retrieval Module       │
 │  (Image-based Web Search)│
 └──────────┬──────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Evidence Fusion   │
    │ (titles/snippets) │
    └───────┬──────────┘
            │
            ▼
    ┌──────────────┐
    │   Final QA    │
    └──────────────┘
```

## 📝 Notes

- This repository extends the VERL framework with **custom retrieval tools** to enhance multimodal reasoning.
- The system supports **varies VLM capable of tool calling**. 
- Make sure your SerpAPI key is valid.