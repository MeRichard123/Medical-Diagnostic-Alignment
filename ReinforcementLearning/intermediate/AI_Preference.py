import os
import json
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import pandas as pd
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from utils import train_test_split

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

MAX_NEW_TOKENS = int(os.getenv("PREF_MAX_NEW_TOKENS", "16"))
MIN_MARGIN = int(os.getenv("PREF_MIN_MARGIN", "1"))
MAX_ROWS = int(os.getenv("PREF_MAX_ROWS", "0"))
DO_SAMPLE = os.getenv("PREF_DO_SAMPLE", "1") == "1"
TEMPERATURE_A = float(os.getenv("PREF_TEMPERATURE_A", "0.7"))
TEMPERATURE_B = float(os.getenv("PREF_TEMPERATURE_B", "0.9"))
TOP_P_A = float(os.getenv("PREF_TOP_P_A", "0.9"))
TOP_P_B = float(os.getenv("PREF_TOP_P_B", "0.95"))

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


def _normalize_diagnosis(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "normal"

    text = re.split(r"(?i)diagnosis\s*:", text, maxsplit=1)[-1].strip()
    text = re.split(r"(?i)explanation\s*:", text, maxsplit=1)[0].strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"(?is)```.*?```", " ", text)
    text = re.split(r"[\n\r\.;:,!?]", text, maxsplit=1)[0].strip()
    text = re.sub(r"[^a-zA-Z0-9 _\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    if not text:
        return "normal"

    words = text.split()
    return " ".join(words[:4]) if words else "normal"


def generate_response(model, instruction, temperature: float = 0.7, top_p: float = 0.9):
    model_device = next(model.parameters()).device
    inputs = policy_tokenizer(instruction, return_tensors="pt").to(model_device)
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "eos_token_id": policy_tokenizer.eos_token_id,
        "pad_token_id": policy_tokenizer.pad_token_id,
    }

    if DO_SAMPLE:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": max(0.1, temperature),
                "top_p": max(0.1, min(1.0, top_p)),
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )
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

def generate_openai_response(prompt):
    role_sys = "You are an expert medical judge. Evaluate clinical correctness."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": role_sys}, {"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()

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

    response = generate_openai_response(prompt)
    # response = generate_response(current_model, prompt, temperature=0.0, top_p=0.9)
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


def _build_preference_record(instruction: str, doctor_gt: str, resp_a_raw: str, resp_b_raw: str) -> tuple[Optional[dict], str]:
    resp_a = _normalize_diagnosis(resp_a_raw)
    resp_b = _normalize_diagnosis(resp_b_raw)

    # If normalized labels differ, judge the normalized labels (label-quality comparison).
    # If normalized labels are identical, ask the judge to compare the RAW outputs
    # so it can prefer one for brevity/clarity even when labels agree.
    if resp_a != resp_b:
        judgement = get_ai_preference(instruction, resp_a, resp_b, doctor_gt)
    else:
        # Ask judge to compare raw responses when labels match
        judgement = get_ai_preference(instruction, resp_a_raw, resp_b_raw, doctor_gt)

    winner = judgement.get("winner", "Tie")

    return (
        {
            "instruction": instruction,
            "doctor_gt": doctor_gt,
            "score_a": judgement["score_a"],
            "score_b": judgement["score_b"],
            "judge_explanation": judgement["judge_explanation"],
            "chosen": resp_a if winner == "A" else resp_b,
            "rejected": resp_b if winner == "A" else resp_a,
        }
    )


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Generate preference dataset
preference_data = []
rows_seen = 0 


for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    doctor_gt = row["impression"]
    instruction = build_rlvr_prompt(str(row["findings"]))

    resp_a_raw = generate_response(current_model, instruction, temperature=TEMPERATURE_A, top_p=TOP_P_A)
    # Generate baseline
    resp_b_raw = generate_response(base, instruction, temperature=TEMPERATURE_B, top_p=TOP_P_B)

    record = _build_preference_record(instruction, doctor_gt, resp_a_raw, resp_b_raw)
    if record is None:
        continue
    
    if (rows_seen >= MAX_ROWS) and MAX_ROWS > 0:
        break

    preference_data.append(record)
    rows_seen += 1

preference_df = pd.DataFrame(preference_data)
PREFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
preference_df.to_csv(PREFERENCE_DATA_PATH, index=False)
print(
    f"Saved {len(preference_df)} preference pairs to {PREFERENCE_DATA_PATH}."
    )