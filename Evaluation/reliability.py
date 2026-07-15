import os
import shutil
import gc
from typing import Tuple, Union
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    MllamaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import PeftModel
import torch
import pandas as pd
import numpy as np
from utils import resolve_image_path
from ReinforcementLearning.Custom.Config import ModelConfigSection

# Complete mapping from your model names to Hugging Face IDs
MODEL_ID_MAPPING = {
    # Base models from generation_task
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "medgemma-4b-it": "google/medgemma-4b-it",
    "meditron-7b": "epfl-llm/meditron-7b",
    "medical-coding-llm": "Kavyaah/medical-coding-llm",
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "qwen2.5-7b-medical": "haohao12/qwen2.5-7b-medical",
    
    # VLMs from generation_vlm
    "SmolVLM-Instruct": "HuggingFaceTB/SmolVLM-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-11B-Vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
    
    
    # Finetuned models (these are local/adaptor paths)
    "Qwen2.5-7B-Instruct-lora": "Qwen/Qwen2.5-7B-Instruct",
    "meditron-7b-lora": "epfl-llm/meditron-7b",
    "Qwen3-VL-8B-Instruct-lora": "Qwen/Qwen3-VL-8B-Instruct",
    "SmolVLM-Instruct-lora": "HuggingFaceTB/SmolVLM-Instruct",

    "Qwen2.5-7B-Instruct-gen-lora": "Qwen/Qwen2.5-7B-Instruct",
    "meditron-7b-gen-lora": "epfl-llm/meditron-7b",
    "Qwen3-VL-8B-Instruct-gen-lora": "Qwen/Qwen3-VL-8B-Instruct",
    "SmolVLM-Instruct-gen-lora": "HuggingFaceTB/SmolVLM-Instruct", 
    "Qwen2.5-7B-Instruct-gen-lora": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it-gen-lora": "google/gemma-3-4b-it",
    "medgemma-4b-it-gen-lora": "google/medgemma-4b-it",
    "meditron-7b-gen-lora": "epfl-llm/meditron-7b",
    "qwen2.5-7b-medical-gen-lora": "haohao12/qwen2.5-7b-medical",
    
    # Finetuned generation models
    "Qwen2.5-7B-Instruct-finetuned": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it-finetuned": "google/gemma-3-4b-it",
    "medgemma-4b-it-finetuned": "google/medgemma-4b-it",
    "meditron-7b-finetuned": "epfl-llm/meditron-7b",
    "qwen2.5-7b-medical-finetuned": "haohao12/qwen2.5-7b-medical",
    "Qwen2.5-7b-medical": "haohao12/qwen2.5-7b-medical",
    
    # Finetuned VLM generation
    "SmolVLM-Instruct-finetuned": "HuggingFaceTB/SmolVLM-Instruct",
    "Qwen3-VL-8B-Instruct-finetuned": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Instruct-finetuned": "Qwen/Qwen3-VL-4B-Instruct",
    "blip2-opt-2.7b-finetuned": "Salesforce/blip2-opt-2.7b",
    
    
    # RL models (these use base models with adapters)
    "Qwen2.5-7B-Instruct-RLVR_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3-VL-8B-Instruct-RLVR_aligned": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_tanh-tanh_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_pref3_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_custom_GRPO_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_noscore_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_aligned": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-RLAIF_GroupedGRPO_aligned": "Qwen/Qwen2.5-7B-Instruct",
}

_BASE_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}

def is_vlm_hf_id(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower and "vl" in model_id_lower:
        return True
    if "llama" in model_id_lower and "vision" in model_id_lower:
        return True
    if "smolvlm" in model_id_lower or ("smol" in model_id_lower and "vlm" in model_id_lower):
        return True
    if "blip" in model_id_lower:
        return True
    return False


def _load_pretrained_model(model_id: str, model_kwargs: dict) -> PreTrainedModel:
    if "Qwen" in model_id and "VL" in model_id:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if "Llama" in model_id and "Vision" in model_id:
        return MllamaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if "smol" in model_id.lower():
        return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    if "blip" in model_id.lower():
        return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)


def get_hf_model_id(model_name):
    """Get Hugging Face model ID for a given model name."""
    return MODEL_ID_MAPPING.get(model_name)

def get_model_architecture_group(model_name):
    """Get the architecture group for a model (for batching)."""
    hf_id = get_hf_model_id(model_name)
    if hf_id is None:
        return "unknown"
    
    if "Qwen" in hf_id:
        if "VL" in hf_id:
            return "qwen_vl"
        return "qwen"
    elif "gemma" in hf_id.lower():
        return "gemma"
    elif "meditron" in hf_id:
        return "meditron"
    elif "Llama" in hf_id:
        return "llama"
    elif "Phi" in hf_id:
        return "phi"
    elif "SmolVLM" in hf_id:
        return "smolvlm"
    elif "blip2" in hf_id.lower():
        return "blip2"
    else:
        return "other"

def cleanup_model_cache(model_name, cache_dir=None):
    """Delete downloaded model files to free up space."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    hf_model_id = get_hf_model_id(model_name)
    if hf_model_id is None:
        return

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cleared {cache_dir}")
    
def cleanup_base_model_cache():
    """Clear cached base models and tokenizers from memory."""
    global _BASE_MODEL_CACHE, _TOKENIZER_CACHE
    for model_name in _BASE_MODEL_CACHE.keys():
        cleanup_model_cache(model_name)

    _BASE_MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()
    print("Base model cache cleared")


def _build_model_kwargs(config: ModelConfigSection, device: torch.device) -> dict:
    trust_remote_code = config.trust_remote_code
    model_kwargs = {"trust_remote_code": trust_remote_code}

    model_dtype_str = config.dtype
    if model_dtype_str != "auto":
        try:
            model_kwargs["dtype"] = getattr(torch, model_dtype_str)
            print(f"Setting model dtype to {model_dtype_str}.")
        except AttributeError:
            print(f"Invalid dtype '{model_dtype_str}' specified. Falling back to auto.")

    quantization_cfg = config.quantization
    if quantization_cfg:
        print(f"Applying quantization with config: {quantization_cfg}")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization_cfg.load_in_4bit,
            load_in_8bit=quantization_cfg.load_in_8bit,
            bnb_4bit_quant_type=quantization_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, quantization_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=quantization_cfg.bnb_4bit_use_double_quant,
        )
        device_index = device.index if device.index is not None else 0
        model_kwargs["device_map"] = {"": device_index}

    if not is_vlm_hf_id(config.model_name):
        attn_implementation = config.attn_implementation
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            print(f"Using attention implementation: {attn_implementation}.")
        if "phi" in config.model_name.lower() or "medical-coding" in config.model_name.lower():
            model_kwargs["attn_implementation"] = "eager"
            print("Using attention implementation: eager.")

    return model_kwargs


def _load_tokenizer_or_processor(config: ModelConfigSection) -> Union[PreTrainedTokenizerBase, object]:
    if is_vlm_hf_id(config.model_name):
        processor = AutoProcessor.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=config.trust_remote_code,
        )
        if processor.tokenizer.pad_token is None or processor.tokenizer.pad_token_id is None:
            print("Processor tokenizer does not have a pad token. Setting pad token to eos token.")
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        return processor

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer does not have a pad token. Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    device: torch.device, config: ModelConfigSection
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizerBase, object]]:
    cache_key = config.model_name

    if cache_key in _BASE_MODEL_CACHE:
        print(f"Using cached base model {cache_key}")
        return _BASE_MODEL_CACHE[cache_key], _TOKENIZER_CACHE[cache_key]

    print(f"Loading base model {cache_key}")

    tokenizer = _load_tokenizer_or_processor(config)
    model_kwargs = _build_model_kwargs(config, device)
    model = _load_pretrained_model(config.model_name, model_kwargs)

    if not model_kwargs.get("quantization_config"):
        model = model.to(device)

    decode_tokenizer = tokenizer.tokenizer if is_vlm_hf_id(config.model_name) else tokenizer
    if getattr(model.config, "pad_token_id", None) is None and decode_tokenizer.pad_token_id is not None:
        model.config.pad_token_id = decode_tokenizer.pad_token_id

    print(f"Base model loaded with dtype: {model.dtype}")

    _BASE_MODEL_CACHE[cache_key] = model
    _TOKENIZER_CACHE[cache_key] = tokenizer

    return model, tokenizer


def load_adapter(base_model, peft_path):
    if peft_path is None:
        return base_model

    print(f"Loading PEFT adaptor from {peft_path}")
    return PeftModel.from_pretrained(
        base_model,
        peft_path,
        is_trainable=False,
    )

def build_rlvr_prompt(findings: str, impression: str) -> str:
    return (
        "Based ONLY on the following clinical findings, reason through the diagnosis step by step.\n\n"
        "Instructions:\n"
        "1. First, use <think> tags to reason through the findings and consider possible diagnoses\n"
        "2. After reasoning, provide ONLY the diagnosis label (1-4 words), respond with 'normal' if there is no diagnosis\n"
        "3. Do not add explanation, markdown, lists, or extra text after the diagnosis\n\n"
        f"Findings:\n{findings}\n\n"
        "<think>\n"
        "(reason through the findings here)\n"
        "</think>\n\n"
        "Diagnosis:"
    )


def _decode_tokenizer(tokenizer_or_processor):
    return tokenizer_or_processor.tokenizer if hasattr(tokenizer_or_processor, "tokenizer") else tokenizer_or_processor


def _build_vlm_inputs(row, processor, device):
    prompt = build_rlvr_prompt(row["findings"], row["impression"])
    vision_prompt = (
        "You are a radiology assistant. Use the X-ray image(s) and clinical findings.\n"
        f"{prompt}"
    )

    images = []
    for col in ("image_1", "image_2", "image_3"):
        value = row.get(col)
        if value is None or pd.isna(value) or str(value).strip() in ("", "None", "nan"):
            continue
        path = resolve_image_path(value)
        if path:
            images.append(path)

    content = [{"type": "text", "text": vision_prompt}]
    for _ in images:
        content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]

    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    except (ValueError, AttributeError):
        text = vision_prompt

    processor_images = [images] if images else None
    inputs = processor(text=text, images=processor_images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _extract_diagnosis(text: str) -> str:
    return text.split("\n")[-1].strip() if text else ""


def compute_reliability_for_model(
    model,
    tokenizer,
    preds,
    gts,
    prompt_ids,
    device,
    is_vlm=False,
    num_samples=15,
    num_consistency=4,
):
    """Compute reliability using a pre-loaded model."""
    dataset = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
    sample_uids = set(dataset.sample(min(num_samples, len(dataset)))["uid"].values)
    uid_to_row = dataset.set_index("uid")
    reliab_scores = []
    decode_tokenizer = _decode_tokenizer(tokenizer)

    for pred, gt, prompt_id in zip(preds, gts, prompt_ids):
        if prompt_id not in sample_uids or prompt_id not in uid_to_row.index:
            continue
        row = uid_to_row.loc[prompt_id]

        if is_vlm:
            prompt_input = _build_vlm_inputs(row, tokenizer, device)
            input_len = prompt_input["input_ids"].shape[1]
            gen_kwargs = {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": decode_tokenizer.eos_token_id,
                "eos_token_id": decode_tokenizer.eos_token_id,
            }
        else:
            prompt = build_rlvr_prompt(row["findings"], row["impression"])
            prompt_input = decode_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
            prompt_input = {k: v.to(device) for k, v in prompt_input.items()}
            input_len = prompt_input["input_ids"].shape[1]
            gen_kwargs = {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True,
            }

        with torch.no_grad():
            outputs = model.generate(
                **prompt_input,
                num_return_sequences=num_consistency,
                **gen_kwargs,
            )

        diagnoses = []
        for seq in outputs:
            generated_ids = seq[input_len:]
            text = decode_tokenizer.decode(generated_ids, skip_special_tokens=True)
            diagnosis = _extract_diagnosis(text)
            if diagnosis:
                diagnoses.append(diagnosis)

        if diagnoses:
            diagnosis_counts = Counter(diagnoses)
            max_count = max(diagnosis_counts.values()) if diagnosis_counts else 0
            consistency = max_count / len(diagnoses) if len(diagnoses) > 0 else 0.0
            reliab_scores.append(consistency)

    mean_reliability = np.mean(reliab_scores) if reliab_scores else 0.0
    print(f"Reliability score: {mean_reliability:.4f} (based on {len(reliab_scores)} samples)")
    return mean_reliability


def compute_self_consistency(base_model, tokenizer, peft_path, preds, gts, prompt_ids, model_id=None):
    """Load an adapter onto a cached base model, evaluate, then drop the wrapper."""
    device = torch.device("cuda")
    model = load_adapter(base_model, peft_path)
    is_vlm = is_vlm_hf_id(model_id) if model_id else False

    try:
        return compute_reliability_for_model(
            model, tokenizer, preds, gts, prompt_ids, device, is_vlm=is_vlm
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

