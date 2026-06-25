import os
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch
from datasets import Dataset
from .rewards import get_reward_funcs
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel
import wandb

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

wandb_run = None
wandb_run_name = os.getenv("WANDB_RUN_NAME", f"{MODEL}_grpo")
report_to = "none"
if wandb is not None:
    wandb_project = "rl-tuning-medical-model-alignment"
    wandb_mode = os.getenv("WANDB_MODE", "online")
    wandb_run = wandb.init(
        project=wandb_project,
        entity="merichard123-university-of-lincoln",
        name=wandb_run_name,
        mode=wandb_mode,
        config={"model_id": MODEL_ID, "trainer": "GRPO"},
    )
    report_to = "wandb"
    print(f"WandB logging enabled: project={wandb_project}, run={wandb_run_name}, mode={wandb_mode}")
else:
    print("WandB not installed; continuing with terminal logging only.")

BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
GRPO_RESULTS_PATH = BASE_PATH / "reinforcement-learning" / "grpo_results"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

qwen_results = pd.read_json("results/Qwen2.5-7B-Instruct_generation_results.json", orient="records")

# Filter data to only include samples with predictions in qwen_results
qwen_ids_set = set(qwen_results["id"].values)
available_data = data[data["uid"].isin(qwen_ids_set)].reset_index(drop=True)
print(f"Samples with predictions: {len(available_data)}")

# Split the available data: 80% train, 20% test
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

train_dataset = build_grpo_dataset(train_df)
test_dataset = build_eval_dataset(test_df)

print("Train dataset size:", len(train_dataset))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", low_cpu_mem_usage=True,
    offload_buffers=True, offload_folder="./offload",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    dtype=torch.bfloat16    
)

# Load the LoRA fine-tuned model using PeftModel 
# to do further post-training with GRPO
model = PeftModel.from_pretrained(
    model, MODEL_PATH, 
    offload_buffers=True, 
    offload_folder="./offload",
    is_trainable=True
)

model = model.to("cuda")
model.eval()
model.config.tie_word_embeddings = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params}")

try:
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
except:
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

if tokeniser.pad_token is None:
    tokeniser.add_special_tokens({"pad_token": tokeniser.eos_token})
    model.resize_token_embeddings(len(tokeniser))

# Import modular reward functions from rewards module

reward_funcs = get_reward_funcs(use_verifier=True)

training_args = GRPOConfig(
    output_dir=str(GRPO_RESULTS_PATH),
    report_to=report_to,
    run_name=wandb_run_name,

    # core training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  
    learning_rate=1e-5,
    max_steps=500,

    # scheduling
    lr_scheduler_type="cosine",
    warmup_steps=20,  

    # logging/eval
    eval_strategy="steps",
    eval_steps=100,  
    save_strategy="steps",
    save_steps=50,
    logging_strategy="steps",
    logging_steps=5, 

    # precision
    bf16=True,
    gradient_checkpointing=True,

    # RL-specific 
    num_generations=4,
    max_completion_length=256,
    temperature=0.7,  
    top_p=0.9,  

    beta=0.05,  
    log_completions=True, 
    num_completions_to_print=2,

    push_to_hub=False,
)


def resolve_resume_checkpoint(results_path: Path, preferred_step: int = 450) -> Path | None:
    preferred_checkpoint = results_path / f"checkpoint-{preferred_step}"
    if preferred_checkpoint.exists():
        return preferred_checkpoint

    checkpoint_paths = []
    for checkpoint_path in results_path.glob("checkpoint-*"):
        try:
            step = int(checkpoint_path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoint_paths.append((step, checkpoint_path))

    if not checkpoint_paths:
        return None

    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths[-1][1]


trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokeniser,
)
if wandb_run is not None:
    try:
        # Log dataset sizes and key hyperparameters
        wandb.config.update({
            "train_size": len(train_dataset),
            "eval_size": len(test_dataset),
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "num_generations": training_args.num_generations,
            "max_completion_length": training_args.max_completion_length,
            "beta": training_args.beta,
        }, allow_val_change=True)
    except Exception:
        pass

    try:
        # Watch the model for gradient/parameter logging (best-effort)
        wandb.watch(model, log="all", log_freq=100)
    except Exception:
        pass

resume_checkpoint = resolve_resume_checkpoint(GRPO_RESULTS_PATH)
if resume_checkpoint is not None:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint))
else:
    print(f"No checkpoint found under {GRPO_RESULTS_PATH}; starting a fresh run.")
    trainer.train()

trainer.save_model(f"./reinforcement-learning/aligned/{MODEL}-RLVR_GRPO")

if wandb_run is not None:
    wandb.finish()
