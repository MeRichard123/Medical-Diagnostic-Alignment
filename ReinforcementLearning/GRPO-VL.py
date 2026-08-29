import os
import types
from pathlib import Path

import pandas as pd
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
import torch
from datasets import Dataset
from .rewards import get_reward_funcs
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from peft import PeftModel
import wandb

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    from transformers import AutoModelForCausalLM as AutoVLMModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

MAX_IMAGES_PER_SAMPLE = 1
MAX_PROMPT_CHARS = 700

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
GRPO_RESULTS_PATH = BASE_PATH / "reinforcement-learning" / "grpo_results_VL_reasoned"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

qwen_results = pd.read_json("results/Qwen3-VL-8B-Instruct_generation_results.json", orient="records")

# Filter data to only include samples with predictions in qwen_results
qwen_ids_set = set(qwen_results["id"].values)
available_data = data[data["uid"].isin(qwen_ids_set)].reset_index(drop=True)
print(f"Samples with predictions: {len(available_data)}")

# Split the available data: 80% train, 20% test
train_size = int(len(available_data) * 0.8)
train_df = available_data[:train_size]
test_df = available_data[train_size:]
print(f"Train split: {len(train_df)}, Test split: {len(test_df)}")

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

def build_rlvr_prompt(findings: str) -> str:
    findings = str(findings)[:MAX_PROMPT_CHARS]
    return (
        "Based ONLY on the following clinical findings, reason through the diagnosis step by step.\n\n"
        "Instructions:\n"
        "1. First, write a short reasoning block inside <think> tags using 1-3 concise sentences\n"
        "2. After </think>, provide ONLY the diagnosis label on the next line in the form 'Diagnosis: <label>'\n"
        "3. Do not add explanation, markdown, lists, or extra text outside the think block and diagnosis line\n\n"
        "Required format:\n"
        "<think>\n"
        "concise reasoning here\n"
        "</think>\n"
        "Diagnosis: <label>\n\n"
        f"Findings:\n{findings}\n\n"
        "Respond now using the required format."
    )


def build_grpo_dataset(dataframe):
    records = []
    for _, row in dataframe.iterrows():
        correct = row["copt"]
        prompt = build_rlvr_prompt(str(row["findings"])) 
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

def build_eval_dataset(dataframe):
    records = []
    for _, row in dataframe.iterrows():
        correct = row["copt"]
        prompt = build_rlvr_prompt(str(row["findings"]))
        images = collect_image_paths(row)
        if not images:
            continue

        # Keep prompt conversational text-only; TRL injects image placeholders from `images`.
        messages = [{"role": "user", "content": prompt}]
        records.append({
            "uid": row["uid"],
            "prompt": messages,
            "images": images,
            "solution": correct,
        })
    return Dataset.from_list(records)

processor = AutoProcessor.from_pretrained(MODEL_ID)

train_dataset = build_grpo_dataset(train_df)
test_dataset = build_eval_dataset(test_df)

print("Train dataset size:", len(train_dataset))

model = AutoVLMModel.from_pretrained(
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

if not getattr(model, "_grpo_vl_forward_patch", False):
    _original_forward = model.forward

    def _patched_forward(self, *args, **kwargs):
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        attention_mask = kwargs.get("attention_mask")

        # GRPO appends completion tokens to attention_mask. Qwen3-VL mm_token_type_ids are prompt-aligned,
        # so if lengths diverge, disable multimodal token types for this pass and clear cached rope deltas.
        if mm_token_type_ids is not None and attention_mask is not None:
            target_len = attention_mask.shape[1]
            current_len = mm_token_type_ids.shape[1]

            if current_len != target_len:
                kwargs.pop("mm_token_type_ids", None)

                try:
                    qwen_vl = self.base_model.model
                    if hasattr(qwen_vl, "model") and hasattr(qwen_vl.model, "rope_deltas"):
                        qwen_vl.model.rope_deltas = None
                except Exception:
                    pass

        return _original_forward(*args, **kwargs)

    model.forward = types.MethodType(_patched_forward, model)
    model._grpo_vl_forward_patch = True

model.config.tie_word_embeddings = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params}")


if processor.tokenizer.pad_token is None:
	processor.tokenizer.pad_token = processor.tokenizer.eos_token


# Import modular reward functions from rewards module

reward_funcs = get_reward_funcs(relaxed=True)

# Passing `generation_config` together with generation-related arguments=({'disable_compile'}) is 
# deprecated and will be removed in future versions. Please pass either a `generation_config`
#  object OR all generation parameters explicitly, but not both.

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
    processing_class=processor,
)

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