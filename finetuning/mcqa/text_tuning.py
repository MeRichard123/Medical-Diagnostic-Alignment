import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
import json
import pandas as pd 
from datasets import Dataset
from utils import (train_test_split, build_mcqa_prompt, build_vision_prompt, 
                   build_label_token_ids, json_serialiser)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from tqdm import tqdm
import numpy as np
from huggingface_hub import login 

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_ID = "epfl-llm/meditron-7b"
MODEL = MODEL_ID.split("/")[-1]

# Kavyaah/medical-coding-llm
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# epfl-llm/meditron-7b
# google/medgemma-4b-it
# haohao12/qwen2.5-7b-medical
# New MODELS to try:
# microsoft/Phi-3-mini-4k-instruct
# google/gemma-3-4b-it
# Qwen/Qwen2.5-7B-Instruct


data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)


def build_sft_dataset(dataframe: pd.DataFrame) -> Dataset:
    records = []
    for _, row in dataframe.iterrows():
        prompt, correct = build_mcqa_prompt(row)
        records.append({"text": f"{prompt}\nAnswer: {correct}"})
    return Dataset.from_list(records)

print("Training set:", train_df.shape[0], "rows")
print("Test set:", test_df.shape[0], "rows")

train_dataset = build_sft_dataset(train_df)
test_dataset = build_sft_dataset(test_df)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", low_cpu_mem_usage=True,
    offload_buffers=True, offload_folder="./offload",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    dtype=torch.bfloat16,
    )

model.config.tie_word_embeddings = False

try:
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
except Exception as error:
    print(f"Fast tokenizer load failed: {error}")
    print("Falling back to slow tokenizer.")
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

if tokeniser.pad_token is None:
    tokeniser.pad_token = tokeniser.eos_token

LABELS = ['A', 'B', 'C', 'D']

LABEL_TOKEN_IDS = build_label_token_ids(tokeniser, LABELS)


def json_serialiser(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def one_hot_encode(label, label_space=LABELS):
    return np.array([1 if l==label else 0 for l in label_space])

def run_inference(df_row):
    prompt, correct = build_mcqa_prompt(df_row)

    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    predicted_label = ""
    label_probabilities = {}
    
    with torch.no_grad():
        # Only generate logits for the next token
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze()
        
        for label, token_ids in LABEL_TOKEN_IDS.items():
            if token_ids:
                label_probabilities[label] = float(torch.max(probs[token_ids]).item())
            else:
                label_probabilities[label] = 0.0

        predicted_label = max(label_probabilities, key=label_probabilities.get)
    
    return predicted_label, label_probabilities, correct

def compute_metrics(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    total = len(labels)
    accuracy = correct / total if total > 0 else 0.0
    f1 = 2 * correct / (2 * correct + sum(preds) - correct + sum(labels) - correct) if (2 * correct + sum(preds) - correct + sum(labels) - correct) > 0 else 0.0
    recall = correct / sum(labels) if sum(labels) > 0 else 0.0
    return {"accuracy": accuracy, "f1": f1, "recall": recall}


if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
    model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
    init_lora_weights="gaussian",
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    prediction_loss_only=True,
    bf16=True,
    optim="paged_adamw_8bit", 
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model(f"./finetuning/tuned/{MODEL}-lora")