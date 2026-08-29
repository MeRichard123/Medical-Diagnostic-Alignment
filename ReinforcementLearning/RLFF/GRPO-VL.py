import os
import types
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import pandas as pd
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
import torch
from datasets import Dataset
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
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_PATH = Path(__file__).parent.parent
REWARD_MODEL_PATH = BASE_PATH / "RLFF" / "Reward_Models" / "reward_model_BT_frugal" / "policy_adapter"

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

MODEL_PATH = BASE_PATH.parent / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
GRPO_RESULTS_PATH = BASE_PATH / "ReinforcementLearning" / "grpo_results" / f"{MODEL}-rlff"

import itertools
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

def patched_get_rope_index(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,   # kept for compatibility, but not used
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Separate video grid thw into multiple grids because timestamps are used to separate videos.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    spatial_merge_size = self.config.vision_config.spatial_merge_size

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        # Use full sequence (ignore attention_mask for position computation)
        input_token_type = mm_token_type_ids[batch_idx]  # shape: (seq_len,)

        # Build groups from the full token type sequence
        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:  # text
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:  # image or video
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = self.get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        seq_len = input_ids.shape[1]
        if llm_positions.shape[1] != seq_len:
            if llm_positions.shape[1] < seq_len:
                # Pad with zeros (text positions) for missing tokens
                pad = torch.zeros(3, seq_len - llm_positions.shape[1],
                                dtype=llm_positions.dtype, device=llm_positions.device)
                llm_positions = torch.cat([llm_positions, pad], dim=1)
            else:
                # Truncate if too long (shouldn't happen, but safe)
                llm_positions = llm_positions[:, :seq_len]
        # Assign to the full sequence (no masking)
        position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        # Delta remains based on the full length (current_input_ids is full)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas

Qwen3VLModel.get_rope_index = patched_get_rope_index

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
    quantization_config= BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

reward_base.config.pad_token_id = reward_tokenizer.pad_token_id

def dummy_prepare_inputs_for_generation(self, input_ids, **kwargs):
    # Return the minimal dict required by the forward pass.
    return {"input_ids": input_ids, **kwargs}

if not hasattr(reward_base, 'prepare_inputs_for_generation'):
    reward_base.prepare_inputs_for_generation = types.MethodType(
        dummy_prepare_inputs_for_generation, reward_base
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
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

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

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
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
    MODEL_ID, device_map="cuda:0", low_cpu_mem_usage=True,
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
        image_grid_thw = kwargs.get("image_grid_thw")
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")

        # If multimodal data is present but mm_token_type_ids is missing, compute it
        if image_grid_thw is not None and mm_token_type_ids is None:
            # Get special token IDs from the processor's tokenizer
            tokenizer = processor.tokenizer
            vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            batch_size, seq_len = input_ids.shape
            # Initialize with zeros (text)
            mm_token_type_ids = torch.zeros(
                (batch_size, seq_len), dtype=torch.int, device=input_ids.device
            )
            for b in range(batch_size):
                ids = input_ids[b]
                # Find positions of vision start/end tokens
                start_indices = (ids == vision_start_id).nonzero(as_tuple=True)[0]
                end_indices = (ids == vision_end_id).nonzero(as_tuple=True)[0]
                # Mark the region between each start–end pair as image (1)
                for start_idx, end_idx in zip(start_indices, end_indices):
                    mm_token_type_ids[b, start_idx:end_idx + 1] = 1
            kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Now handle length mismatch (as before)
        if image_grid_thw is not None:
            if mm_token_type_ids is None:
                raise ValueError(
                    "Still missing mm_token_type_ids after fallback – cannot proceed."
                )
            target_len = input_ids.shape[1]
            current_len = mm_token_type_ids.shape[1]
            if current_len < target_len:
                padding = torch.zeros(
                    mm_token_type_ids.shape[0],
                    target_len - current_len,
                    dtype=mm_token_type_ids.dtype,
                    device=mm_token_type_ids.device,
                )
                kwargs["mm_token_type_ids"] = torch.cat(
                    [mm_token_type_ids, padding], dim=1
                )
            elif current_len > target_len:
                kwargs["mm_token_type_ids"] = mm_token_type_ids[:, :target_len]

        return _original_forward(*args, **kwargs)

    model.forward = types.MethodType(_patched_forward, model)
    model._grpo_vl_forward_patch = True

model.config.tie_word_embeddings = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params}")


if processor.tokenizer.pad_token is None:
	processor.tokenizer.pad_token = processor.tokenizer.eos_token


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
    reward_funcs=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    reward_processing_classes=reward_tokenizer,
)

resume_checkpoint = resolve_resume_checkpoint(GRPO_RESULTS_PATH)
if resume_checkpoint is not None:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint))
else:
    print(f"No checkpoint found under {GRPO_RESULTS_PATH}; starting a fresh run.")
    trainer.train()

trainer.save_model(f"./reinforcement-learning/aligned/{MODEL}-RLFF_GRPO")

if wandb_run is not None:
    wandb.finish()