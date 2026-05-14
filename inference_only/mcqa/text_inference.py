import torch
import json
import pandas as pd 
from utils import ( train_test_split, build_mcqa_prompt, 
				   build_label_token_ids, json_serialiser, clean_predictions)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from huggingface_hub import login 
import os
from pathlib import Path
from peft import PeftModel


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-lora"

print(MODEL_PATH)

# Kavyaah/medical-coding-llm
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# epfl-llm/meditron-7b
# google/medgemma-4b-it
# haohao12/qwen2.5-7b-medical
# New MODELS to try:
# microsoft/Phi-3-mini-4k-instruct
# google/gemma-3-4b-it
# Qwen/Qwen2.5-7B-Instruct

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

_, test_df = train_test_split(data, test_size=0.2, random_state=42)

print("Test set:", test_df.shape[0], "rows")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, low_cpu_mem_usage=True,
    offload_buffers=True, offload_folder="./offload",
    dtype=torch.bfloat16
    )

model = PeftModel.from_pretrained(
    base_model, MODEL_PATH, 
    offload_buffers=True, 
    offload_folder="./offload"
)
model = model.to("cuda")
model.eval()
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


with open(f"./results/{MODEL}-lora_inference_results.json", "w") as f:
    json.dump(results, f, indent=4, default=json_serialiser)
