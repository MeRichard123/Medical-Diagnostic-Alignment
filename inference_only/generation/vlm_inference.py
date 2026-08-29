import torch
import json
import pandas as pd
import re
from utils import (
	train_test_split,
	build_gen_prompt,
	json_serialiser,
	resolve_image_path,
)
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import MllamaForConditionalGeneration, Qwen3VLForConditionalGeneration
from tqdm import tqdm
from huggingface_hub import login
import os
import pathlib
from peft import PeftModel

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)


print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# meta-llama/Llama-3.2-11B-Vision-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# HuggingFaceTB/SmolVLM-Instruct	
# Salesforce/blip2-opt-2.7b

_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_ID = os.getenv("GEN_MODEL_ID", _DEFAULT_MODEL_ID)
MODEL = MODEL_ID.split("/")[-1]
BASE_PATH = pathlib.Path(__file__).parent.parent.parent
# MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
MODEL_PATH = BASE_PATH / "ReinforcementLearning" / "exp" / f"experiment-morlVL" / "final_model"
#MODEL_PATH = BASE_PATH / "ReinforcementLearning" / "exp" / f"experiment-frugalVL" / "final_model"
USE_LORA = os.getenv("USE_VLM_LORA", "1").strip().lower() in {"1", "true", "yes"}

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
_, test_df = train_test_split(data, test_size=0.2, random_state=42)

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
elif 'blip' in MODEL_ID.lower():
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

if USE_LORA and MODEL_PATH.exists():
	model = PeftModel.from_pretrained(
		base_model,
		MODEL_PATH,
		offload_buffers=True,
		offload_folder="./offload",
	)
	print(f"Loaded LoRA fine-tuned model from {MODEL_PATH}")
else:
	if USE_LORA:
		print(f"LoRA requested but not found at {MODEL_PATH}; using base model.")
	model = base_model

model = model.to("cuda")
model.eval()

model.config.tie_word_embeddings = False

USING_LORA = isinstance(model, PeftModel)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

def normalise_diagnosis(text: str) -> str:
	t = text.strip().lower()
	t = re.sub(r"\s*---.*$", "", t)
	t = re.sub(r"\s+", " ", t).strip()
	words = t.split()
	return " ".join(words[:4])


def build_rlvr_prompt(findings: str, impression: str) -> str:
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

def run_inference(df_row):
	_, correct = build_gen_prompt(df_row)
	prompt = build_rlvr_prompt(df_row.get("findings"), df_row.get("impression"))
	vision_prompt = (
		"You are a radiology assistant. Use the X-ray image(s) and clinical findings.\n"
		f"{prompt}"
	)

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

	# Check if processor has chat template; some models don't support it
	try:
		prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
	except (ValueError, AttributeError):
		# Fallback: build prompt manually from the text content
		prompt = vision_prompt

	processor_images = [images] if images else None
	inputs = processor(text=prompt, images=processor_images, return_tensors="pt")
	inputs = inputs.to(DEVICE)
	input_ids = inputs["input_ids"]

	with torch.no_grad():
		generation = model.generate(
			**inputs,
			max_new_tokens=8,
			do_sample=False,
			return_dict_in_generate=True,
			output_scores=True,
			pad_token_id=processor.tokenizer.eos_token_id,
			eos_token_id=processor.tokenizer.eos_token_id,
		)

	generated_ids = generation.sequences[:, input_ids.shape[1]:]
	generated_token_ids = generated_ids[0].tolist()

	output_probabilities = []
	selected_token_ids = []

	for step_idx, step_scores in enumerate(generation.scores):
		if step_idx >= len(generated_token_ids):
			break

		token_id = generated_token_ids[step_idx]
		step_probs = torch.softmax(step_scores[0], dim=-1)
		token_prob = step_probs[token_id].item()
		token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)

		selected_token_ids.append(token_id)
		output_probabilities.append(
			{
				"token": token_text,
				"token_id": int(token_id),
				"probability": float(token_prob),
			}
		)

		partial_output = processor.tokenizer.decode(selected_token_ids, skip_special_tokens=True).strip()
		if len(partial_output.split()) >= 4:
			break

	output_words = processor.tokenizer.decode(selected_token_ids, skip_special_tokens=True).strip().split()
	output = " ".join(output_words[:4])

	return output, output_probabilities, correct


results = []

for i in tqdm(range(len(test_df))):
	obj = {}
	row = test_df.iloc[i]
	inference, probs, correct = run_inference(row)
	obj["id"] = row["uid"]
	obj['predicted_diagnosis'] = inference
	obj['predicted_diagnosis_normalised'] = normalise_diagnosis(inference)
	obj['ground_truth'] = correct
	obj['probabilities'] = probs
	results.append(obj)
if USING_LORA and "grpo_model_4o-Preferences-vl" in MODEL_PATH.parts:
	modifier = "-RLAIF-VL"
elif USING_LORA and "frugalVL" in MODEL_PATH.parts:
	modifier = "-frugalVL"
elif USING_LORA and "morlVL" in MODEL_PATH.parts:
	modifier = "-morlVL"
elif USING_LORA and "reinforcement" in MODEL_PATH.parts:
	modifier = "-RLVR_aligned"
elif USING_LORA:
	modifier = "-finetuned"
else:
	modifier = ""

counter = 0
if not os.path.exists(f"./results/{MODEL}{modifier}_generation_results.json"):
	with open(f"./results/{MODEL}{modifier}_generation_results.json", "w") as f:
		json.dump(results, f, indent=4, default=json_serialiser)
	print(f"Saved results to ./results/{MODEL}{modifier}_generation_results.json")
else:
	print(f"File ./results/{MODEL}{modifier}_generation_results.json already exists.")
	counter += 1
	while os.path.exists(f"./results/{MODEL}{modifier}_generation_results_{counter}.json"):
		counter += 1
		
	with open(f"./results/{MODEL}{modifier}_generation_results_{counter}.json", "w") as f:
		json.dump(results, f, indent=4, default=json_serialiser)
	print(f"Saved results to ./results/{MODEL}{modifier}_generation_results_{counter}.json")


