import torch
import json
import pandas as pd 
from utils import ( train_test_split, build_mcqa_prompt, build_vision_prompt, 
				   build_label_token_ids, json_serialiser, clean_predictions,
				   resolve_image_path)
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import MllamaForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm
import numpy as np
from huggingface_hub import login 
import os
import pathlib
from peft import PeftModel

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)


print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USING_LORA = False

# meta-llama/Llama-3.2-11B-Vision-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# HuggingFaceTB/SmolVLM-Instruct	
# Salesforce/blip2-opt-2.7b

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
MODEL = MODEL_ID.split("/")[-1]
BASE_PATH = pathlib.Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-lora"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

print("Training set:", train_df.shape[0], "rows")
print("Test set:", test_df.shape[0], "rows")

quant_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16,
	bnb_4bit_use_double_quant=True,
)

if 'Llama' in MODEL_ID:
	base_model = MllamaForConditionalGeneration.from_pretrained(
		MODEL_ID,
		dtype=torch.bfloat16,
		device_map="auto",
		quantization_config=quant_config,
	)
elif 'Qwen' in MODEL_ID:
	base_model = Qwen3VLForConditionalGeneration.from_pretrained(
		MODEL_ID,
		dtype=torch.bfloat16,
		device_map="auto",
		quantization_config=quant_config,
	)
elif "smol" in MODEL_ID.lower():
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
		dtype=torch.bfloat16,
		device_map={"": 0},
        quantization_config=quant_config,
    )
else:
	base_model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		dtype=torch.bfloat16,
		device_map="auto",
		quantization_config=quant_config,
	)

model = PeftModel.from_pretrained(
	base_model, MODEL_PATH, 
	offload_buffers=True, 
	offload_folder="./offload"
)
model = model.to("cuda")
model.eval()

model = base_model.to("cuda")
model.eval()

model.config.tie_word_embeddings = False

if isinstance(model, PeftModel):
	USING_LORA = True
else:	
	USING_LORA = False

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

LABELS = ['A', 'B', 'C', 'D']

LABEL_TOKEN_IDS = build_label_token_ids(processor.tokenizer, LABELS)


def run_inference(df_row):
	prompt, correct = build_mcqa_prompt(df_row)
	# prompt, correct = build_vision_prompt(df_row)
	vision_prompt = (
		"You are a radiology assistant. Use the X-ray image(s) and clinical findings.\n"
		"Return ONLY one letter: A, B, C, or D.\n\n"
		f"{prompt}"
	)
	# vision_prompt = (
	# 	"You are a radiology assistant. Use the X-ray image(s).\n"
	# 	"Return ONLY one letter: A, B, C, or D.\n\n"
	# 	f"{prompt}"
	# )

	images = []

	img1 = resolve_image_path(df_row.get("image_1"))
	img2 = resolve_image_path(df_row.get("image_2"))
	img3 = None
	if df_row.get("image_3") and not pd.isna(df_row.get("image_3")) and str(df_row.get("image_3")).strip() not in ("", "None", "nan"):
		img3 = resolve_image_path(df_row.get("image_3"))

	if img1:
		images.append(img1)
	if img2:
		images.append(img2)
	if img3:
		images.append(img3)

	content = [{"type": "text", "text": vision_prompt}]

	for _ in images:
		content.append({"type": "image"})

	messages = [{"role": "user", "content": content}]

	prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
	processor_images = [images] if images else None
	inputs = processor(text=prompt, images=processor_images, return_tensors="pt")
	# inputs = processor(images=images[0], text=vision_prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
	inputs = inputs.to(DEVICE)


	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits[:, -1, :]
		probs = torch.softmax(logits, dim=-1).squeeze(0)

	label_probabilities = {
		label: float(torch.max(probs[token_ids]).item()) if token_ids else 0.0
		for label, token_ids in LABEL_TOKEN_IDS.items()
	}

	predicted_label = max(label_probabilities, key=label_probabilities.get)
	return predicted_label, label_probabilities, correct


results = []

for i in tqdm(range(len(test_df))):
	obj = {}
	row = test_df.iloc[i]
	inference, probs, correct = run_inference(row)
	pred_label, probs_clean = clean_predictions(inference, probs, LABELS)
	obj["id"] = row["uid"]
	obj['predicted_diagnosis'] = pred_label
	obj['ground_truth'] = correct
	obj['probabilities'] = probs_clean
	results.append(obj)

if USING_LORA:
	modifier = "-lora"
else:
	modifier = ""

counter = 0
if not os.path.exists(f"./results/{MODEL}{modifier}_inference_results.json"):
	with open(f"./results/{MODEL}{modifier}_inference_results.json", "w") as f:
		json.dump(results, f, indent=4, default=json_serialiser)
	print(f"Saved results to ./results/{MODEL}{modifier}_inference_results.json")
else:
	counter += 1
	print(f"File ./results/{MODEL}{modifier}_inference_results.json already exists.")
	with open(f"./results/{MODEL}{modifier}_inference_results_{counter}.json", "w") as f:
		json.dump(results, f, indent=4, default=json_serialiser)
	print(f"Saved results to ./results/{MODEL}{modifier}_inference_results_{counter}.json")


