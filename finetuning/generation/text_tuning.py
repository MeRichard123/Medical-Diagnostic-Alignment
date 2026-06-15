import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
import pandas as pd 
from datasets import Dataset
from utils import (build_gen_prompt, train_test_split)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, default_data_collator
from huggingface_hub import login 

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_ID = "haohao12/qwen2.5-7b-medical"
# Next Kavyaah/medical-coding-llm
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
        prompt, correct = build_gen_prompt(row)
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


def tokenise_batch(batch: dict) -> dict:
    tokenised = tokeniser(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    # Some tokenizers/models do not emit token_type_ids; Gemma3 training expects it.
    if "token_type_ids" not in tokenised:
        tokenised["token_type_ids"] = [
            [0] * len(input_ids) for input_ids in tokenised["input_ids"]
        ]

    tokenised["labels"] = [ids.copy() for ids in tokenised["input_ids"]]
    return tokenised


train_dataset = train_dataset.map(
    tokenise_batch,
    batched=True,
    remove_columns=["text"],
)
test_dataset = test_dataset.map(
    tokenise_batch,
    batched=True,
    remove_columns=["text"],
)


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
    output_dir="./finetuning/checkpoints",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator,
)

# if we want to resume from a checkpoint, we can specify set this to true
trainer.train(resume_from_checkpoint=False)

trainer.save_model(f"./finetuning/tuned/{MODEL}-gen-lora")