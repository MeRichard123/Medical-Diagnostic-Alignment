import torch
import json
import pandas as pd
from utils import (
    train_test_split,
    build_gen_prompt,
    json_serialiser,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login
import os, re
from pathlib import Path
from peft import PeftModel

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Allow overriding the base model via environment variable so agents.py
# and other experiment drivers can sweep across models reproducibly.
_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = os.getenv("GEN_MODEL_ID", _DEFAULT_MODEL_ID)
MODEL = MODEL_ID.split("/")[-1]

BASE_PATH = Path(__file__).parent.parent.parent
# MODEL_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
MODEL_PATH = BASE_PATH / "reinforcement-learning" / "grpo_results" / "Qwen2.5-7B-Instruct-gen-lora-grpo-frugal-v1/final_model"

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

# check if using peft model 
if isinstance(model, PeftModel):
    print(f"Loaded PEFT model from {MODEL_PATH}")
else:
    print(f"Loaded base model {MODEL_ID} without PEFT adapters")


try:
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
except Exception as error:
    print(f"Fast tokenizer load failed: {error}")
    print("Falling back to slow tokenizer.")
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

if tokeniser.pad_token is None:
    tokeniser.pad_token = tokeniser.eos_token


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


def extract_diagnosis_label(text: str, max_words: int = 4, reasoning: bool = True) -> str:
    t = str(text)

    if reasoning:
        # Remove explicit reasoning blocks regardless of where they appear.
        t = re.sub(r"<think>.*?</think>", " ", t, flags=re.IGNORECASE | re.DOTALL)
        # Handle malformed or partial tags that can appear in generated text.
        t = re.sub(r"</?think>", " ", t, flags=re.IGNORECASE)

    t = re.split(r"---|###|answer:|explanation:|background:|reasoning:", t, maxsplit=1, flags=re.IGNORECASE)[0]
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if lines:
        selected = ""
        for line in lines:
            candidate = re.sub(
                r"^(diagnosis|diagnosis label|label|final answer)\s*[:\-]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            if candidate:
                selected = candidate
                break
        t = selected if selected else lines[0]

    t = re.sub(r"^(diagnosis|diagnosis label|label|final answer)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w for w in t.split() if re.search(r"[a-z0-9]", w)]
    return " ".join(words[:max_words])


def run_inference(df_row):
    prompt = build_rlvr_prompt(str(df_row["findings"]), str(df_row["impression"]))
    correct = str(df_row["copt"])

    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokeniser.eos_token_id,
            eos_token_id=tokeniser.eos_token_id,
        )

    generated_ids = generation.sequences[:, input_ids.shape[1]:]

    output_probabilities = []
    generated_token_ids = generated_ids[0].tolist()

    for step_idx, step_scores in enumerate(generation.scores):
        if step_idx >= len(generated_token_ids):
            break

        token_id = generated_token_ids[step_idx]
        step_probs = torch.softmax(step_scores[0], dim=-1)
        token_prob = step_probs[token_id].item()
        token_text = tokeniser.decode([token_id], skip_special_tokens=False)

        output_probabilities.append(
            {
                "token": token_text,
                "token_id": int(token_id),
                "probability": float(token_prob),
            }
        )

    raw_output = tokeniser.decode(generated_token_ids, skip_special_tokens=True).strip()
    label_output = extract_diagnosis_label(raw_output, max_words=4)

    return label_output, raw_output, output_probabilities, correct

def normalise_diagnosis(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s*---.*$", "", t)        
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    return " ".join(words[:4])             

results = []

for i in tqdm(range(len(test_df))):
    obj = {}
    row = test_df.iloc[i]
    inference, raw_output, probs, correct = run_inference(row)
    obj["id"] = row["uid"]
    obj['predicted_diagnosis'] = inference
    obj['predicted_diagnosis_raw'] = raw_output
    obj['predicted_diagnosis_normalised'] = normalise_diagnosis(inference)
    obj['ground_truth'] = correct
    obj['probabilities'] = probs
    results.append(obj)   

with open(f"./results/{MODEL}-RLAIF_FrugalGRPO_aligned_generation_results.json", "w") as f:
    json.dump(results, f, indent=4, default=json_serialiser)