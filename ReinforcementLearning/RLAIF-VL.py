import os
from pathlib import Path
import types

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForImageTextToText as AutoVLMModel,
)

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
try:    
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from utils import train_test_split

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL = MODEL_ID.split("/")[-1]
MAX_IMAGES_PER_SAMPLE = 1

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID
BASE_REWARD_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REWARD_MODEL_PATH = BASE_PATH / "ReinforcementLearning" / "intermediate" /"reward_model_4o-Preferences"

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def resolve_image_path(filename):
	if pd.isna(filename) or str(filename).strip() in ("", "None", "nan"):
		return None

	name = str(filename).strip()
	candidates = [
		Path("./data", "images", "processed", name),
	]
	for path in candidates:
		if os.path.exists(path):
			return str(path)
	return None


def collect_image_paths(df_row):
	images = []
	for col in ["image_1", "image_2", "image_3"]:
		img = resolve_image_path(df_row.get(col))
		if img:
			images.append(img)
	return images[:MAX_IMAGES_PER_SAMPLE]

def build_rl_prompt(findings: str) -> str:
    return (
        "Task: From findings, output ONLY one diagnosis label (1-4 words).\n"
        "If no active pathology, output: normal.\n"
        "No explanations, no lists, no extra text.\n\n"
        "Examples:\n"
        "Findings: No focal consolidation, pleural effusion, or pneumothorax. Heart size normal.\n"
        "Diagnosis: normal\n\n"
        "Findings: Bilateral interstitial opacities with small pleural effusions.\n"
        "Diagnosis: pulmonary edema\n\n"
        f"Findings: {findings}\n"
        "Diagnosis:"
    )


def load_instruction_dataset(dataframe: pd.DataFrame) -> Dataset:
    records = []
    for _, row in dataframe.iterrows():
        correct = row["copt"]
        prompt = build_rl_prompt(str(row["findings"])) 
        images = collect_image_paths(row)
        if not images:
            continue

        # Keep prompt conversational text-only; TRL injects image placeholders from `images`.
        messages = [{"role": "user", "content": prompt}]

        # Include all samples for training - GRPO will learn to improve
        records.append({
            "uid": row["uid"],
            "prompt": messages,
            "images": images,
            "solution": correct,
        })

    return Dataset.from_list(records)

def build_eval_dataset(dataframe: pd.DataFrame) -> Dataset:
    records = []
    for _, row in dataframe.iterrows():
        correct = row["copt"]
        prompt = build_rl_prompt(str(row["findings"])) 
        images = collect_image_paths(row)
        if not images:
            continue

        # Keep prompt conversational text-only; TRL injects image placeholders from `images`.
        messages = [{"role": "user", "content": prompt}]

        # Include all samples for evaluation
        records.append({
            "uid": row["uid"],
            "prompt": messages,
            "images": images,
            "solution": correct,
        })

    return Dataset.from_list(records)

data = pd.read_csv(BASE_PATH / "data" / "processed_iuxray_mcqa_dataset.csv")
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
qwen_vl_results = pd.read_json("results/Qwen3-VL-8B-Instruct_generation_results.json", orient="records")

qwen_ids_set = set(qwen_vl_results["id"].values)
available_data = data[data["uid"].isin(qwen_ids_set)].reset_index(drop=True) 
print("Available data size:", len(available_data))

train_size = int(len(available_data) * 0.8)
train_df = available_data.iloc[:train_size].reset_index(drop=True)
test_df = available_data.iloc[train_size:].reset_index(drop=True)
print("Train dataset size:", len(train_df))
print("Test dataset size:", len(test_df))


def main() -> None:
    policy_processor = AutoProcessor.from_pretrained(MODEL_ID)
    # if policy_processor.pad_token is None:
    #     policy_processor.pad_token = policy_processor.eos_token
    # if getattr(policy_processor, "pad_token_id", None) is None and policy_processor.eos_token_id is not None:
    #     policy_processor.pad_token_id = policy_processor.eos_token_id

    policy_base = AutoVLMModel.from_pretrained(
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
        if getattr(policy_model.config, "pad_token_id", None) is None and policy_processor.eos_token_id is not None:
            policy_model.config.pad_token_id = policy_processor.eos_token_id
    except Exception:
        pass

    policy_model.gradient_checkpointing_enable()

    if not getattr(policy_model, "_grpo_vl_forward_patch", False):
        _original_forward = policy_model.forward

    def _patched_forward(self, *args, **kwargs):
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        attention_mask = kwargs.get("attention_mask")
        input_ids = kwargs.get("input_ids")

        # If mm_token_type_ids is missing, create a dummy tensor of zeros
        if mm_token_type_ids is None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # dtype must be torch.long (same as input_ids)
            dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
            kwargs["mm_token_type_ids"] = dummy
            print("mm_token_type_ids: CREATED DUMMY (zeros)")

        # Now pad to match attention_mask if needed (existing logic)
        mm_token_type_ids = kwargs.get("mm_token_type_ids")  # refresh
        if mm_token_type_ids is not None and attention_mask is not None:
            target_len = attention_mask.shape[1]
            current_len = mm_token_type_ids.shape[1]
            if current_len < target_len:
                pad = torch.zeros(
                    mm_token_type_ids.shape[0],
                    target_len - current_len,
                    dtype=mm_token_type_ids.dtype,
                    device=mm_token_type_ids.device,
                )
                kwargs["mm_token_type_ids"] = torch.cat([mm_token_type_ids, pad], dim=1)

        # Optional prints (keep or remove)
        print("input_ids:", kwargs["input_ids"].shape)
        if "attention_mask" in kwargs:
            print("attention_mask:", kwargs["attention_mask"].shape)
        if "mm_token_type_ids" in kwargs:
            print("mm_token_type_ids:", kwargs["mm_token_type_ids"].shape)
        else:
            print("mm_token_type_ids: MISSING")
        if "image_grid_thw" in kwargs:
            print("image_grid_thw:", kwargs["image_grid_thw"].shape)

        return _original_forward(*args, **kwargs)

    policy_model.forward = types.MethodType(_patched_forward, policy_model)
    policy_model._grpo_vl_forward_patch = True

    policy_model.config.tie_word_embeddings = False

    reward_tokenizer = AutoTokenizer.from_pretrained(BASE_REWARD_MODEL_ID, use_fast=True)
    reward_tokenizer.chat_template = "User: {{ messages[0]['content'] }}\nAssistant:"
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    if getattr(reward_tokenizer, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

    reward_base = AutoModelForSequenceClassification.from_pretrained(
        BASE_REWARD_MODEL_ID,
        num_labels=1,
        low_cpu_mem_usage=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
        quantization_config=quantization_config,
        device_map="cpu",
    )
    print("Loaded Reward Model Base from:", BASE_REWARD_MODEL_ID)
    reward_model = PeftModel.from_pretrained(
        reward_base,
        REWARD_MODEL_PATH,
        is_trainable=False,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
    )
    print(f"Loaded Reward Model from PEFT path: {REWARD_MODEL_PATH}")
    reward_model.eval()
    reward_model.to("cuda" if torch.cuda.is_available() else "cpu")
    for param in reward_model.parameters():
        param.requires_grad = False
    try:
        if getattr(reward_model.config, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
            reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
    except Exception:
        pass

    grpo_config = GRPOConfig(
        output_dir=str(BASE_PATH / "ReinforcementLearning" / "intermediate" / "grpo_model_4o-Preferences-3"),
        learning_rate=5e-8, 
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        beta=0.2,
        max_completion_length=64,
        num_generations=4,
        temperature=0.5,
        num_train_epochs=3,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy="no",
        report_to="wandb" if _WANDB_AVAILABLE else "none",
    )

    if _WANDB_AVAILABLE:
        try:
            wandb.init(
                project="rl-grpo",
                name="grpo-run-4o-Preferences-3",
                config={
                    "model_id": MODEL_ID,
                    "learning_rate": 5e-8,
                    "per_device_train_batch_size": 2,
                    "per_device_eval_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "num_train_epochs": 3,
                    "max_completion_length": 64,
                    "num_generations": 4,
                    "temperature": 0.5,
                },
                reinit=True,
            )
        except Exception:
            pass

    instruction_dataset = load_instruction_dataset(train_df)

    grpo_trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward_model],
        reward_processing_classes=[reward_tokenizer],
        train_dataset=instruction_dataset,
        processing_class=policy_processor,
        args=grpo_config,
    )
    # TODO: need some code to check if a checkpoint exists and pass resume_from_checkpoint accordingly
    grpo_trainer.train(resume_from_checkpoint=False)
    grpo_trainer.save_model(str(BASE_PATH / "ReinforcementLearning" / "intermediate" / "grpo_model_4o-Preferences-vl"))


if __name__ == "__main__":
    main()
