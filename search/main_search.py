import sys
import os
import io
import json
import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pprint import pprint

from omegaconf import OmegaConf
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision
)
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.hf_rollout import HFRollout
from verl.utils.distributed import initialize_global_process_group

# 添加项目根目录到 sys.path
tools_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if tools_root not in sys.path:
    sys.path.insert(0, tools_root)

from my_tools.utils import postprocess_serpapi_results, build_multimodal_prompt


os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8268"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =============================================================================
# Main Function
# =============================================================================
@hydra.main(config_path="config", config_name="search", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    hf_config = config.rollout
    hf_config.update({"n": 2, "do_sample": True, "validate": False})
    
    local_rank, rank, world_size = initialize_global_process_group()
    print("world size: ", world_size)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
    ).eval().cuda()

    processor = AutoProcessor.from_pretrained(config.model.path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True)

    # Initialize HFRollout and start generate
    hf_rollout = HFRollout(model, OmegaConf.create(hf_config), tokenizer)

    text_input = config.data.question
    image_url = config.data.image_url
    
    # 调用serpapi
    import serpapi
    raw_results = serpapi.search(
        engine="google_reverse_image",  
        image_url=image_url,
        api_key=config.data.api_key)

    
    print("raw results: ", raw_results['image_results'])
    processed_results = postprocess_serpapi_results(raw_results['image_results'])
    print("processed_results: ", processed_results)
    prompts = build_multimodal_prompt(text_input, image_url, processed_results)
    
    text = processor.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(prompts)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input = DataProto.from_dict(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "position_ids": compute_position_id_with_mask(inputs["attention_mask"]),
        },
        meta_info={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "validate": False,
        },
    )
    input = input.to("cuda")

    # 生成回答
    outputs = hf_rollout.generate_sequences(input)
    
    # check generated batch size is expected
    generated_batch_size = outputs.batch.batch_size[0]
    assert generated_batch_size == input.batch.batch_size[0]

    for i in range(generated_batch_size):
        prompt_tokens = outputs.batch["prompts"][i]
        prompt_mask = prompt_tokens != tokenizer.pad_token_id
        prompt_tokens = prompt_tokens[prompt_mask]
        decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        response_tokens = outputs.batch["responses"][i]
        response_mask = response_tokens != tokenizer.pad_token_id
        response_tokens = response_tokens[response_mask]
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        # print generated text for inspection
        if torch.distributed.get_rank() == 0:
            print(f"prompt: {decoded_prompt}")
            print(f"response: {decoded_response}")
            print("=" * 30)
    

if __name__ == "__main__":
    main()