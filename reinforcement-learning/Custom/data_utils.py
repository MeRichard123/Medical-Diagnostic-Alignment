from transformers import GenerationConfig, PreTrainedTokenizerBase
from .Config import PPOConfig
from datasets import Dataset
import torch.nn as nn
import accelerate
import os
import pandas as pd

accelerator = accelerate.Accelerator()


def load_and_process_dataset(
        cfg: PPOConfig, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    print(f"Loading dataset: {cfg.dataset.dataset_path}")
    data = pd.read_csv(cfg.dataset.dataset_path)
    qwen_results = pd.read_json("results/Qwen2.5-7B-Instruct_generation_results.json", orient="records")
    qwen_ids_set = set(qwen_results["id"].values)
    available_data = data[data["uid"].isin(qwen_ids_set)].reset_index(drop=True)
    print(f"Samples with predictions: {len(available_data)}")

    train_size = int(len(available_data) * 0.8)
    train_df = available_data[:train_size]
    test_df = available_data[train_size:]
    print(f"Train split: {len(train_df)}, Test split: {len(test_df)}")

    def build_rlvr_prompt(findings: str) -> str:
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


    def build_grpo_dataset(dataframe):
        records = []
        for _, row in dataframe.iterrows():
            correct = row["copt"]
            prompt = build_rlvr_prompt(str(row["findings"]))
            # Include all samples for training - GRPO will learn to improve
            records.append({
                "prompt": prompt,
                "solution": correct,
            })
        return Dataset.from_list(records)
    
    def build_eval_dataset(dataframe):
        records = []
        for _, row in dataframe.iterrows():
            correct = row["copt"]
            prompt = build_rlvr_prompt(str(row["findings"]))
            records.append({
                "prompt": prompt,
                "solution": correct,
            })
        return Dataset.from_list(records)

    if cfg.dataset.split == "train":
        raw_dataset = build_grpo_dataset(train_df)
    else:
        raw_dataset = build_eval_dataset(test_df)


    num_samples = cfg.dataset.num_samples
    if num_samples is not None and num_samples < len(raw_dataset):
        raw_dataset = raw_dataset.shuffle(seed=cfg.training.seed).select(range(num_samples))
        print(f"Dataset shuffled and truncated to {num_samples} samples.")

    print(f"Final dataset size: {len(raw_dataset)}")
    raw_dataset = raw_dataset.map(lambda x: tokenizer(x["prompt"], truncation=True, padding="max_length", max_length=512), batched=True)
    return raw_dataset


def save_model(model: nn.Module, tokenizer: PreTrainedTokenizerBase,
               save_path: str):
    """Saves the model and tokenizer."""
    if not accelerator.is_main_process:
        return
    print(f"Saving model checkpoint to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved.")
    except Exception as e:
        print(f"Error saving model: {e}")


def create_generation_config(
        cfg: PPOConfig,
        tokenizer: PreTrainedTokenizerBase) -> GenerationConfig:
    """Creates the GenerationConfig object."""
    return GenerationConfig(max_new_tokens=cfg.generation.max_new_tokens,
                            min_new_tokens=cfg.generation.min_new_tokens,
                            temperature=cfg.generation.temperature,
                            top_k=cfg.generation.top_k,
                            top_p=cfg.generation.top_p,
                            do_sample=cfg.generation.do_sample,
                            pad_token_id=tokenizer.pad_token_id)