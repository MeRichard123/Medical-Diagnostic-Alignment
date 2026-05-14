import os
import json
import re
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import train_test_split

BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
PREFERENCE_DATA_PATH = BASE_PATH / "reinforcement-learning" / "intermediate" / "preference_data.csv"

OFFLOAD_FOLDER = BASE_PATH / "offload"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        offload_folder=str(OFFLOAD_FOLDER),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
)


current_model = PeftModel.from_pretrained(
    base,
    SFT_POLICY_PATH,
    is_trainable=False,
    offload_buffers=True,
    offload_folder=str(BASE_PATH / "offload"),
)



if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token


def build_rlvr_prompt(findings: str) -> str:
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

def generate_response(model, instruction):
    model_device = next(model.parameters()).device
    inputs = policy_tokenizer(instruction, return_tensors="pt").to(model_device)
    input_length = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = policy_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()


def _extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _run_single_judgement(instruction, response_a, response_b, doctor_gt):
    prompt = f"""You are an expert medical judge. Evaluate clinical correctness.

Rubric (1-5):
1=clinically wrong  2=mostly wrong  3=partially correct  4=mostly correct  5=fully correct

Doctor ground truth: {doctor_gt}
Response A: {response_a}
Response B: {response_b}

Score A and B independently against ground truth. Return JSON only:
{{"score_a": <1-5>, "score_b": <1-5>, "winner": "A"|"B"|"Tie", "explanation": "<one sentence>"}}
    """

    response = generate_response(current_model, prompt)
    parsed = _extract_json_object(response)
    if parsed is None:
        return {
            "score_a": 3,
            "score_b": 3,
            "winner": "Tie",
            "explanation": "Judge output was not valid JSON; defaulted to tie.",
            "raw_judge_output": response,
        }

    score_a = int(parsed.get("score_a", 3))
    score_b = int(parsed.get("score_b", 3))
    score_a = min(5, max(1, score_a))
    score_b = min(5, max(1, score_b))

    winner = str(parsed.get("winner", "Tie")).strip()
    if winner not in {"A", "B", "Tie"}:
        winner = "A" if score_a > score_b else "B" if score_b > score_a else "Tie"

    return {
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "explanation": str(parsed.get("explanation", "No explanation provided.")).strip(),
        "raw_judge_output": response,
    }


def get_ai_preference(instruction, response_a, response_b, doctor_gt):
    first_pass = _run_single_judgement(instruction, response_a, response_b, doctor_gt)

    # Simple score smoothing: re-run uncertain or tied comparisons once.
    margin = abs(first_pass["score_a"] - first_pass["score_b"])
    if first_pass["winner"] == "Tie" or margin <= 1:
        second_pass = _run_single_judgement(instruction, response_a, response_b, doctor_gt)
        score_a = round((first_pass["score_a"] + second_pass["score_a"]) / 2)
        score_b = round((first_pass["score_b"] + second_pass["score_b"]) / 2)
        winner = "A" if score_a > score_b else "B" if score_b > score_a else "Tie"
        explanation = (
            f"Pass1: {first_pass['explanation']} Pass2: {second_pass['explanation']}"
        )
        return {
            "winner": winner,
            "score_a": score_a,
            "score_b": score_b,
            "judge_explanation": explanation,
        }

    return {
        "winner": first_pass["winner"],
        "score_a": first_pass["score_a"],
        "score_b": first_pass["score_b"],
        "judge_explanation": first_pass["explanation"],
    }


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Generate preference dataset
preference_data = []
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    doctor_gt = row["impression"]
    instruction = build_rlvr_prompt(str(row["findings"]))

    resp_a = generate_response(current_model, instruction)
    # Generate baseline
    resp_b = generate_response(base, instruction)

    judgement = get_ai_preference(instruction, resp_a, resp_b, doctor_gt)
    winner = judgement.get("winner", "Tie")
    if winner == "Tie":
        winner = "A"

    preference_data.append({
        "instruction": instruction,
        "doctor_gt": doctor_gt,
        "score_a": judgement["score_a"],
        "score_b": judgement["score_b"],
        "judge_explanation": judgement["judge_explanation"],
        "chosen": resp_a if winner == "A" else resp_b,
        "rejected": resp_b if winner == "A" else resp_a,
    })

preference_df = pd.DataFrame(preference_data)
PREFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
preference_df.to_csv(PREFERENCE_DATA_PATH, index=False)
print(f"Saved {len(preference_df)} preference pairs to {PREFERENCE_DATA_PATH}")