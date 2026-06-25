from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch
from trl import RewardTrainer, RewardConfig
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

BASE_PATH = Path(__file__).resolve().parent.parent.parent
PREFERENCE_DATA_PATH = BASE_PATH / "reinforcement-learning" / "intermediate" / "preference_data.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

RUN_NAME = "reward-model-4o-Preferences"

# after loading quantization_config and model:
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    device_map="auto",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
)

# prepare for k-bit training and attach LoRA adapters
reward_model = prepare_model_for_kbit_training(reward_model)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # adjust for your model if needed
)
reward_model = get_peft_model(reward_model, peft_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

preference_df = pd.read_csv(PREFERENCE_DATA_PATH)
# Ensure chosen/rejected are strings and drop rows missing either entry
preference_df = preference_df.copy()
if "chosen" not in preference_df.columns or "rejected" not in preference_df.columns:
    raise ValueError("preference_data.csv must contain 'chosen' and 'rejected' columns")
preference_df[["chosen", "rejected"]] = preference_df[["chosen", "rejected"]].fillna("")
preference_df[["chosen", "rejected"]] = preference_df[["chosen", "rejected"]].astype(str)
valid_mask = (preference_df["chosen"].str.strip() != "") & (preference_df["rejected"].str.strip() != "")
dropped = (~valid_mask).sum()
if dropped:
    print(f"Dropping {dropped} rows with empty chosen/rejected fields")
clean_df = preference_df.loc[valid_mask, ["chosen", "rejected"]].copy()
if clean_df.empty:
    raise ValueError("No valid preference pairs found after cleaning chosen/rejected columns")

# Create train/eval split so Trainer has an eval_dataset when eval_strategy != 'no'
eval_frac = 0.1
if len(clean_df) < 2:
    raise ValueError("Not enough data to create train/eval split")
eval_df = clean_df.sample(frac=eval_frac, random_state=42)
train_df = clean_df.drop(eval_df.index).reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

training_args = RewardConfig(
    output_dir=str(BASE_PATH / "reinforcement-learning" / "intermediate" / "reward_model"),
    model_init_kwargs={"dtype": torch.bfloat16},
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb" if _WANDB_AVAILABLE else "none",
    run_name=RUN_NAME,
)

if _WANDB_AVAILABLE:
    # initialize a W&B run (harmless if already initialized)
    try:
        if wandb.run is None:
            wandb.init(project="rl-reward", name=RUN_NAME, config={"model_id": MODEL_ID})
    except Exception:
        pass

reward_trainer = RewardTrainer(
    model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args,
)

reward_trainer.train()
reward_trainer.save_model(str(BASE_PATH / "reinforcement-learning" / "intermediate" / "reward_model_4o-Preferences"))