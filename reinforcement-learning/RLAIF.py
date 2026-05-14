import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from utils import train_test_split

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID
REWARD_MODEL_PATH = BASE_PATH / "reinforcement-learning" / "RLFF" / "Reward_Models" / "reward_model_manualBT"

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def build_rl_prompt(findings: str) -> str:
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


def load_instruction_dataset() -> Dataset:
    data = pd.read_csv(BASE_PATH / "data" / "processed_iuxray_mcqa_dataset.csv")
    train_df, _ = train_test_split(data, test_size=0.2, random_state=42)
    prompts = [build_rl_prompt(str(findings)) for findings in train_df["findings"].tolist()]
    return Dataset.from_dict({"prompt": prompts})


def main() -> None:
    policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    if getattr(policy_tokenizer, "pad_token_id", None) is None and policy_tokenizer.eos_token_id is not None:
        policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

    policy_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        low_cpu_mem_usage=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
        quantization_config=quantization_config,
        device_map="auto",
    )
    print("Loaded Policy Model Base from:", BASE_MODEL_ID)
    policy_model = PeftModel.from_pretrained(
        policy_base,
        SFT_POLICY_PATH,
        is_trainable=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
    )
    print(f"Loaded Policy Model from PEFT path: {SFT_POLICY_PATH}")
    try:
        if getattr(policy_model.config, "pad_token_id", None) is None and policy_tokenizer.eos_token_id is not None:
            policy_model.config.pad_token_id = policy_tokenizer.eos_token_id
    except Exception:
        pass

    reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    if getattr(reward_tokenizer, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

    reward_base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=1,
        low_cpu_mem_usage=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
        quantization_config=quantization_config,
        device_map="auto",
    )
    print("Loaded Reward Model Base from:", BASE_MODEL_ID)
    reward_model = PeftModel.from_pretrained(
        reward_base,
        REWARD_MODEL_PATH,
        is_trainable=False,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
    )
    print(f"Loaded Reward Model from PEFT path: {REWARD_MODEL_PATH}")
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    try:
        if getattr(reward_model.config, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
            reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
    except Exception:
        pass

    grpo_config = GRPOConfig(
        output_dir=str(BASE_PATH / "reinforcement-learning" / "intermediate" / "grpo_model_manualBT"),
        learning_rate=1.4e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  
        num_train_epochs=3,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy="no",
        beta=0.0,
        max_completion_length=16,
        num_generations=4,
        temperature=0.7,
        report_to="wandb" if _WANDB_AVAILABLE else "none",
    )

    if _WANDB_AVAILABLE:
        try:
            wandb.init(
                project="rl-grpo",
                name="grpo-run-manualBT",
                config={
                    "model_id": MODEL_ID,
                    "learning_rate": 1.4e-5,
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "num_train_epochs": 3,
                    "max_completion_length": 16,
                    "num_generations": 4,
                    "temperature": 0.7,
                },
                reinit=True,
            )
        except Exception:
            pass

    instruction_dataset = load_instruction_dataset()

    grpo_trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward_model],
        reward_processing_classes=[reward_tokenizer],
        train_dataset=instruction_dataset,
        processing_class=policy_tokenizer,
        args=grpo_config,
    )

    grpo_trainer.train(resume_from_checkpoint=True)
    grpo_trainer.save_model(str(BASE_PATH / "reinforcement-learning" / "intermediate" / "grpo_model_manualBT"))


if __name__ == "__main__":
    main()
