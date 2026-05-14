import torch
import pandas as pd 
from utils import train_test_split, build_mcqa_prompt
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from transformers import MllamaForConditionalGeneration, Qwen3VLForConditionalGeneration
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from huggingface_hub import login 
import os
import pathlib
from PIL import Image

hf_token = os.getenv("HF_TOKEN")
if hf_token:
	login(token=hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# meta-llama/Llama-3.2-11B-Vision-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# HuggingFaceTB/SmolVLM-Instruct	
# Salesforce/blip2-opt-2.7b


MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
MODEL = MODEL_ID.split("/")[-1]
MAX_IMAGES_PER_SAMPLE = 1
MAX_PROMPT_CHARS = 700

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


def resolve_image_path(filename):
	if pd.isna(filename) or str(filename).strip() in ("", "None", "nan"):
		return None

	name = str(filename).strip()
	candidates = [
		pathlib.Path("./data", "images", "processed", name),
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



def build_sft_dataset(dataframe: pd.DataFrame, processor_obj) -> Dataset:
	records = []
	for _, row in dataframe.iterrows():
		prompt, correct = build_mcqa_prompt(row)
		if len(prompt) > MAX_PROMPT_CHARS:
			prompt = prompt[:MAX_PROMPT_CHARS]

		vision_prompt = (
			"You are a radiology assistant. Use the X-ray image(s) and clinical findings.\n"
			"Return ONLY one letter: A, B, C, or D.\n\n"
			f"{prompt}"
		)

		images = collect_image_paths(row)
		if not images:
			continue

		user_content = [{"type": "text", "text": vision_prompt}]
		for _ in images:
			user_content.append({"type": "image"})

		messages = [
			{"role": "user", "content": user_content},
			{"role": "assistant", "content": [{"type": "text", "text": correct}]},
		]

		records.append(
			{
				"uid": row.get("uid"),
				"messages": messages,
				"chat_text": processor_obj.apply_chat_template(messages, add_generation_prompt=False),
				"images": images,
				"answer": correct,
			}
		)
	return Dataset.from_list(records)


def vlm_collator(examples):
	texts = [item["chat_text"] for item in examples]
	batch_images = [item["images"] for item in examples]

	batch = processor(
		text=texts,
		images=batch_images,
		padding=True,
		truncation=False,
		return_tensors="pt",
	)

	labels = batch["input_ids"].clone()
	pad_id = processor.tokenizer.pad_token_id
	if pad_id is not None:
		labels[labels == pad_id] = -100
	batch["labels"] = labels
	return batch


if processor.tokenizer.pad_token is None:
	processor.tokenizer.pad_token = processor.tokenizer.eos_token


quant_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16,
	bnb_4bit_use_double_quant=True,
)

if "Qwen" in MODEL_ID:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
		dtype=torch.bfloat16,
		device_map={"": 0},
        quantization_config=quant_config,
    )
elif "Llama" in MODEL_ID:
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
		dtype=torch.bfloat16,
		device_map={"": 0},
        quantization_config=quant_config,
    )
elif "smol" in MODEL_ID.lower():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
		dtype=torch.bfloat16,
		device_map={"": 0},
        quantization_config=quant_config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
		dtype=torch.bfloat16,
		device_map={"": 0},
        quantization_config=quant_config,
    )

train_dataset = build_sft_dataset(train_df, processor)
test_dataset = build_sft_dataset(test_df, processor)

if len(train_dataset) == 0:
	raise ValueError("Training dataset is empty after filtering rows with missing images.")

# if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
	r=4,
	lora_alpha=16,
	target_modules=['q_proj', 'v_proj'],
    init_lora_weights="gaussian",
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(
    model,
    lora_config
)

model.config.use_cache = False

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
	per_device_train_batch_size=1,
	gradient_accumulation_steps=2,
	per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    prediction_loss_only=True,
    bf16=True,
    optim="paged_adamw_8bit", 
    logging_steps=10,
    save_strategy="epoch",
	eval_strategy="no",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
	dataloader_pin_memory=True,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
	data_collator=vlm_collator,
)

trainer.train(resume_from_checkpoint=True)

trainer.save_model(f"./finetuning/tuned/{MODEL}-lora")

