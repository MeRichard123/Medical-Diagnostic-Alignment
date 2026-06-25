import os
import shutil
import gc
from typing import Tuple, List, Dict, Optional
from collections import Counter
from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from peft import PeftModel
import torch
import random
import pandas as pd
import numpy as np
from ReinforcementLearning.Custom.Config import ModelConfigSection, QuantizationConfig

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

# Cache for loaded models to avoid reloading
_LOADED_MODEL_CACHE = {}
_LOADED_TOKENIZER_CACHE = {}

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
    
def cleanup_loaded_models(keep_base_model=False):
    """
    Clean up all loaded models and clear cache.
    
    Args:
        keep_base_model: If True, keeps the base model but unloads adaptors
    """
    global _LOADED_MODEL_CACHE, _LOADED_TOKENIZER_CACHE
    
    for key in list(_LOADED_MODEL_CACHE.keys()):
        model = _LOADED_MODEL_CACHE[key]
        if model is not None:
            try:
                # If it's a PeftModel, unload the adaptor
                if isinstance(model, PeftModel):
                    print(f"Unloading PEFT adaptor for {key}")
                    # Unload adaptor but keep base model
                    model.unload_adapter()
                    # If we want to keep base model, we could do:
                    # base_model = model.base_model.model
                    # _LOADED_MODEL_CACHE[key] = base_model
                # Delete the model
                del _LOADED_MODEL_CACHE[key]
            except Exception as e:
                print(f"Error cleaning up model {key}: {e}")
                del _LOADED_MODEL_CACHE[key]
    
    for key in list(_LOADED_TOKENIZER_CACHE.keys()):
        if _LOADED_TOKENIZER_CACHE[key] is not None:
            del _LOADED_TOKENIZER_CACHE[key]
    
    _LOADED_MODEL_CACHE.clear()
    _LOADED_TOKENIZER_CACHE.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    print("✅ Memory cleaned up")

def load_models_and_tokenizer(device: torch.device, config: ModelConfigSection, use_cache=True) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load models with caching support."""
    
    # Check if we can use cached model
    cache_key = f"{config.model_name}_{config.peft_adaptor_path}"
    
    if use_cache and cache_key in _LOADED_MODEL_CACHE:
        print(f"Using cached model for {config.model_name}")
        return _LOADED_MODEL_CACHE[cache_key], _LOADED_TOKENIZER_CACHE.get(cache_key)
    
    print(f"Loading tokenizer: {config.tokenizer_name}")

    trust_remote_code = config.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer does not have a pad token. Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    print(f"Loading actor model: {config.model_name}")

    model_kwargs = {"trust_remote_code": trust_remote_code}
    model_dtype_str = config.dtype
    if model_dtype_str != "auto":
        try:
            model_dtype = getattr(torch, model_dtype_str)
            model_kwargs["dtype"] = model_dtype
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

    attn_implementation = config.attn_implementation
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
        print(f"Using attention implementation: {attn_implementation}.")
    if 'phi' in config.model_name.lower() or 'medical-coding' in config.model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
        print(f"Using attention implementation: eager.")
    
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **model_kwargs
    )

    if not model_kwargs.get("quantization_config"):
        actor_model = actor_model.to(device)
    if actor_model.config.pad_token_id is None:
        print("Actor model does not have a pad token. Setting pad token to eos token.")
        actor_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Actor model loaded with dtype: {actor_model.dtype}")
    # gradient checkpointing can be enabled here if needed, but be cautious with 8-bit models
    print("Enabling gradient checkpointing for actor model.")
    actor_model.gradient_checkpointing_enable()

    if config.peft_adaptor_path:
        print(f"Loading PEFT adaptor from {config.peft_adaptor_path}")
        actor_model = PeftModel.from_pretrained(
            actor_model, 
            config.peft_adaptor_path, 
            is_trainable=True,
            device_map="auto"
        ).to(device)
        print("PEFT adaptor loaded and applied to actor model.")
    else:
        print("No Peft config using base model")
    
    # Cache the loaded model if requested
    if use_cache:
        _LOADED_MODEL_CACHE[cache_key] = actor_model
        _LOADED_TOKENIZER_CACHE[cache_key] = tokenizer
           
    return actor_model, tokenizer

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

def compute_reliability_for_model(model, tokenizer, preds, gts, prompt_ids, device, num_samples=15, num_consistency=4):
    """Compute reliability using a pre-loaded model."""
    dataset = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
    
    # Sample indices
    subset = dataset.sample(min(num_samples, len(dataset)))
    reliab_scores = []
    
    data = zip(preds, gts, prompt_ids)
    for i, (pred, gt, prompt_id) in enumerate(data):
        if i in subset['uid'].values:
            row = subset[subset['uid'] == i]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            prompt = build_rlvr_prompt(row['findings'], row['impression'])
            prompt_input = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            prompt_input = {k: v.to(device) for k, v in prompt_input.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **prompt_input,
                    num_return_sequences=num_consistency,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                )
            
            # Decode completions and extract diagnoses
            diagnoses = []
            for seq in outputs:
                text = tokenizer.decode(seq, skip_special_tokens=True)
                # Extract diagnosis after the last newline
                diagnosis = text.split('\n')[-1].strip() if text else ""
                if diagnosis:
                    diagnoses.append(diagnosis)
            
            # Compute consistency: proportion of samples that match the most common diagnosis
            if diagnoses:
                diagnosis_counts = Counter(diagnoses)
                max_count = max(diagnosis_counts.values()) if diagnosis_counts else 0
                consistency = max_count / len(diagnoses) if len(diagnoses) > 0 else 0.0
                reliab_scores.append(consistency)
    
    mean_reliability = np.mean(reliab_scores) if reliab_scores else 0.0
    print(f"Reliability score: {mean_reliability:.4f} (based on {len(reliab_scores)} samples)")
    return mean_reliability

def compute_self_consistency(config, preds, gts, prompt_ids, use_cache=True, cleanup_after=True):
    """Compute self-consistency with optional caching and cleanup."""
    device = torch.device("cuda")
    model, tokenizer = load_models_and_tokenizer(device, config, use_cache=use_cache)
    
    try:
        reliability = compute_reliability_for_model(
            model, tokenizer, preds, gts, prompt_ids, device
        )
        return reliability
    finally:
        if cleanup_after:
            # Clean up only this model if not using cache
            if not use_cache:
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                if isinstance(model, PeftModel):
                    try:
                        model.unload_adapter()
                        print(f"Unloaded PEFT adaptor")
                    except:
                        pass
                
                # Delete model and tokenizer
                del model
                del tokenizer
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()



# Function to clean all caches after processing
def cleanup_all_caches(model):
    """Clean all model caches and free memory."""
    cleanup_loaded_models()
    
    # Optional: Also clean disk cache
    cleanup_model_cache(model)  # You'd need to implement this carefully
    
    print("All caches cleaned up")

