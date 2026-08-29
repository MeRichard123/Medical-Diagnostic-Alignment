from ..Custom.Config import GRPOConfig, ModelConfigSection, QuantizationConfig, TrainingConfigSection, WandBConfigSection
from ..Custom.FrugalVLGRPOTrainer import FrugalVLGRPOTrainer
from ..Custom.MOGRPOTrainerVL import MORLGRPOTrainerVL
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import torch
import pandas as pd
import os

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL = MODEL_ID.split("/")[-1]
BASE_PATH = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
GRPO_RESULTS_PATH = BASE_PATH / "ReinforcementLearning" / "exp" / f"experiment-morlVL"

SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID
BASE_REWARD_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REWARD_MODEL_PATH = BASE_PATH / "ReinforcementLearning" / "intermediate" / "reward_model_4o-Preferences"
MAX_IMAGES_PER_SAMPLE = 1

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

reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
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
    quantization_config= BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)
print("Loaded Reward Model Base from:", BASE_REWARD_MODEL_ID)
reward_model = PeftModel.from_pretrained(
    reward_base,
    REWARD_MODEL_PATH,
    offload_buffers=True,
    offload_folder=str(BASE_PATH / "offload"),
)
print(f"Loaded Reward Model from PEFT path: {REWARD_MODEL_PATH}")
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False


quantisation_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)


if __name__ == '__main__':
    experimental_cfg = GRPOConfig(
        model = ModelConfigSection(
            model_name = MODEL_ID,
            ref_model_name = MODEL_ID,
            tokenizer_name = MODEL_ID,
            peft_adaptor_path = MODEL_PATH,
            quantization = quantisation_config
        ),
        wandb = WandBConfigSection(
            report_to_wandb=True,
            project="rl-tuning-medical-model-alignment",
            name=f"{MODEL}_grpo_morlVL",
        ),
        training = TrainingConfigSection(
            seed=42,
            log_interval=1,
            save_interval=10,
            output_dir=GRPO_RESULTS_PATH,
            output_dir_grpo=GRPO_RESULTS_PATH,
            device="cuda",
            gradient_checkpointing=True
        ),
        learning_rate=1e-5,
        warmup_steps=10,   
        grad_accum_steps=2,
        mini_batch_size=4,
        rollout_samples=8,
        group_size=4,
    )
    trainer = MORLGRPOTrainerVL(
        args=experimental_cfg, 
        reward_funcs=[reward_model],
        reward_processors=[reward_tokenizer],
    )

    trainer.train()
        
