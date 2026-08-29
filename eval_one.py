import os
import json
import numpy as np
import pandas as pd
import torch
from Evaluation.Evaluator import QAEvaluator, VisionEvaluator, ReportEvaluator, compute_kl_divergence, compute_kl_and_ppl
from pathlib import Path
from typing import Optional
from Evaluation.reliability import (
    MODEL_ID_MAPPING,
    load_base_model,
    load_adapter,
    compute_self_consistency,
    cleanup_base_model_cache,
)

from ReinforcementLearning.Custom.Config import QuantizationConfig, ModelConfigSection

BASE_PATH = Path(__file__).parent

BASE_DIR = "./results"


def normalise_probabilities(prob_payload):
    if isinstance(prob_payload, list):
        token_probs = []
        for token_info in prob_payload:
            if isinstance(token_info, dict):
                p = float(token_info.get("probability", 0.0))
                token_probs.append(max(p, 1e-10))

        if token_probs:
            log_probs = np.log(token_probs)
            seq_logprob = float(np.sum(log_probs))
            sequence_confidence = float(np.exp(np.mean(log_probs)))
        else:
            seq_logprob = float(np.log(1e-10))
            sequence_confidence = float(1e-10)

        return {
            "type": "token_probs",
            "ground_truth_logprob": seq_logprob,
            "sequence_confidence": sequence_confidence,
            "token_count": len(token_probs),
        }

    return prob_payload

def _results_path(model_name: str, suffix: str) -> str:
    return os.path.join(BASE_DIR, model_name + suffix)

def _extract_report_fields(parse):
    predictions = np.array([data["predicted_diagnosis_normalised"].strip().lower() for data in parse])
    pred_raw = np.array([data["predicted_diagnosis"].strip().lower() for data in parse])
    ground_truth = np.array(
        [
            data["ground_truth"].strip().lower()
            for data in parse
            if not (isinstance(data["ground_truth"], float) and np.isnan(data["ground_truth"]))
        ]
    )
    predicted = [p if p else "N/A" for p in predictions]
    probabilities = [normalise_probabilities(data["probabilities"]) for data in parse]
    prompt_ids = np.array([data["id"] for data in parse])
    return ground_truth, predicted, probabilities, pred_raw, prompt_ids



def _load_results(model_name: str, suffix: str, skip_missing: bool = False):
    path = _results_path(model_name, suffix)
    if not os.path.exists(path):
        if skip_missing:
            print(f"Warning: Results file for {model_name} not found, skipping evaluation.")
            return None
        raise FileNotFoundError(f"Missing required results file: {path}")

    with open(path, "r") as fp:
        return json.load(fp)


quantisation_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

model_paths = [
    'Kavyaah/medical-coding-llm',
    'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    'epfl-llm/meditron-7b',
    'google/medgemma-4b-it',
    'haohao12/qwen2.5-7b-medical',
    'microsoft/Phi-3-mini-4k-instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-7B-Instruct',
]

models = [
    "medical-coding-llm", "Phi-3-mini-4k-instruct",
    "meditron-7b", 
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "Qwen2.5-7b-medical", "Qwen2.5-7B-Instruct",
    "medgemma-4b-it", "gemma-3-4b-it"
]
vlms = [
    "SmolVLM-Instruct",
    "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct",
    "BLIP2-OPT-2.7B", "Llama-3.2-11B-Vision",
    "SmolVLM-Instruct_notext",
    "Qwen3-VL-8B-Instruct_notext",
    ]

vision_models = [
    "vit_b16_head_only", "alexnet",
    "resnet101", "swin_v2_b", "vgg19"
]

finetuned_models = [
    "Qwen2.5-7B-Instruct-lora",
    "meditron-7b-lora",
     
    "Qwen3-VL-8B-Instruct-lora",
    "SmolVLM-Instruct-lora"
]

trained_vision = [
    "alexnet-finetuned", "resnet101-finetuned"
]

finetuned_generation = [
    "Qwen2.5-7B-Instruct-finetuned",
    "gemma-3-4b-it-finetuned",
    "medgemma-4b-it-finetuned",
    "meditron-7b-finetuned",
    "qwen2.5-7b-medical-finetuned"
]

BASE_LORA = BASE_PATH / "finetuning" / "tuned"

split = lambda path: '-'.join(path.split("-")[:-1]) + '-gen-lora'
LORA_Tuned_Paths = {
    k: BASE_LORA / split(k) for k in finetuned_generation
}


# TODO: Add RL models once results are available
reinforcement_learning_models = [
    # "Qwen2.5-7B-Instruct-RLVR_aligned",
    # "Qwen3-VL-8B-Instruct-RLVR_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_tanh-tanh_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_pref3_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_custom_GRPO_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_noscore_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_GroupedGRPO_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_MORL_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_RLFF_aligned",
    # "Qwen3-VL-8B-Instruct-RLAIF-VL",
    "Qwen3-VL-8B-Instruct-frugalVL",
    "Qwen3-VL-8B-Instruct-morlVL",
]

BASE_RL = BASE_PATH / "ReinforcementLearning"



RL_Tuned_Paths = {
    "Qwen2.5-7B-Instruct-RLVR_aligned": BASE_RL / 'aligned' / 'Qwen2.5-7B-Instruct-RLVR_GRPO',
    "Qwen3-VL-8B-Instruct-RLVR_aligned": BASE_RL / 'aligned' / 'Qwen3-VL-8B-Instruct-RLVR_GRPO',
    "Qwen2.5-7B-Instruct-RLAIF_aligned": BASE_RL / 'intermediate' / 'grpo_model_4o-Preferences',
    "Qwen2.5-7B-Instruct-RLAIF_tanh-tanh_aligned": BASE_RL / 'intermediate' / 'grpo_model_BT_tanhminustanh',
    "Qwen2.5-7B-Instruct-RLAIF_pref3_aligned": BASE_RL / 'intermediate' / 'grpo_model_4o-Preferences-3',
    "Qwen2.5-7B-Instruct-RLAIF_custom_GRPO_aligned": BASE_RL / 'grpo_results' / 'custom_grpo',
    "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_noscore_aligned":BASE_RL/'grpo_results'/'Qwen2.5-7B-Instruct-gen-lora-grpo-frugal-accmiscal'/'final_model',
    "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_aligned": BASE_RL / 'grpo_results' / 'Qwen2.5-7B-Instruct-gen-lora-grpo-frugal-v1' / 'final_model',
    "Qwen2.5-7B-Instruct-RLAIF_GroupedGRPO_aligned": BASE_RL / 'exp' / 'experiment-Grouped' / 'final_model',
    "Qwen2.5-7B-Instruct-RLAIF_MORL_aligned": BASE_RL / 'exp' / 'experiment-MORL' / 'final_model',
    "Qwen2.5-7B-Instruct-RLAIF_RLFF_aligned": BASE_RL / "aligned" / f"Qwen2.5-7B-Instruct-RLFF_GRPO",
    "Qwen3-VL-8B-Instruct-RLAIF-VL": BASE_RL / 'aligned' / 'Qwen3-VL-8B-Instruct-RLVR_GRPO',
    "Qwen3-VL-8B-Instruct-morlVL": BASE_RL / 'exp' / 'experiment-morlVL' / 'final_model',
    "Qwen3-VL-8B-Instruct-frugalVL": BASE_RL / 'exp' / 'experiment-frugalVL' / 'final_model',
}


generation_task = [
    "Qwen2.5-7B-Instruct", 
    "medgemma-4b-it", 
    "meditron-7b", 
    "medical-coding-llm",
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit", 
    "Phi-3-mini-4k-instruct", 
    "gemma-3-4b-it",
    "qwen2.5-7b-medical"
]


generation_vlm = [
    "SmolVLM-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Llama-3.2-11B-Vision-Instruct",
    "blip2-opt-2.7b",
]

finetuned_vlm_generation = [
    "SmolVLM-Instruct-finetuned",
    "Qwen3-VL-8B-Instruct-finetuned",
    "Qwen3-VL-4B-Instruct-finetuned",
    "blip2-opt-2.7b-finetuned",
]

VLM_Tuned_Paths = {
    k: BASE_LORA / split(k) for k in finetuned_vlm_generation
}

all_models = reinforcement_learning_models
tuned = LORA_Tuned_Paths | RL_Tuned_Paths | VLM_Tuned_Paths

def get_model_id_and_peft(model: str):
    hugging_face_id = MODEL_ID_MAPPING.get(model)
    peft_path = tuned.get(model)
    return hugging_face_id, peft_path


BASE_PATH = Path(__file__).parent
BASE_LORA = BASE_PATH / "finetuning" / "tuned"
BASE_RL = BASE_PATH / "ReinforcementLearning"
BASE_GENERATION = BASE_PATH / "results" 

MODEL_FOR_EVAL = 'Qwen3-VL-8B-Instruct'  # Change this to the model you want to evaluate


def compute_ece_data(model, tokenizer, prompt, response, max_length=1024):
    """
    Returns:
        confidences: List[float]
        corrects:    List[int]
    """
    # Tokenize full sequence exactly as fed to the model
    inputs = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits[0]
    # Tokenize prompt and response separately
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]
    start = len(prompt_ids) - 1
    available = logits.size(0) - start
    if available <= 0:
        return [], []
    response_ids = response_ids[:available]
    confidences = []
    corrects = []
    for i, true_token in enumerate(response_ids):
        probs = torch.softmax(logits[start + i], dim=-1)
        pred = probs.argmax(dim=-1).item()
        conf = probs[pred].item()
        confidences.append(conf)
        corrects.append(int(pred == true_token))
    return confidences, corrects

def get_all_results_for_model(model):
    results = []
    for model in all_models:
        if MODEL_FOR_EVAL.lower() in model.lower():
            hugging_face_id, peft_path = get_model_id_and_peft(model)
            results.append({
                "model": model,
                "hugging_face_id": hugging_face_id,
                "peft_path": peft_path,
                "generation_results_path": BASE_GENERATION / f"{model}_generation_results.json"
            })
    return results


models = get_all_results_for_model(MODEL_FOR_EVAL)

print(models)

COPY = "VL"

kls = {}
consistencies = {}
reliability_rows = []
perplexities = {}

for model_info in models:
    parse = _load_results(model_info['model'], "_generation_results.json", skip_missing=True)
    if parse is None:
        continue
    
    id_ = model_info['hugging_face_id']
    peft = model_info['peft_path']
    model = model_info['model']

    base_config = ModelConfigSection(
        model_name=id_,
        ref_model_name=id_,
        tokenizer_name=id_,
        quantization=quantisation_config,
    )

    base_model, tokenizer = load_base_model(torch.device("cuda"), base_config)

    gts, preds, _, pred_raw, prompt_ids = _extract_report_fields(parse)

    consistencies[model], policy_model = compute_self_consistency(
        base_model, tokenizer, peft, preds, gts, prompt_ids, model_id=id_
    )

    print(f"PEFT path for {model}: {peft}")

    if peft is None:
        kls[model] = float('nan')
        print(f"No PEFT adapter for {model}; setting KL divergence to NaN")
    else:
        scores = [
            compute_kl_and_ppl(policy_model, base_model, tokenizer, prompt_id, pred)
            for prompt_id, pred in zip(prompt_ids, pred_raw)
        ]
        finite_scores = [score[0] for score in scores if np.isfinite(score[0])]
        finite_perplexities = [score[1] for score in scores if np.isfinite(score[1])]
        kls[model] = float(np.mean(finite_scores)) if finite_scores else float('nan')
        perplexities[model] = float(np.mean(finite_perplexities)) if finite_perplexities else float('nan')
        print(f"KL divergence for {model}: {kls[model]}")
        print(f"Perplexity for {model}: {perplexities[model]}")

    for prompt, response in zip(preds, gts):
        conf, corr = compute_ece_data(policy_model, tokenizer, prompt, response)

        for c, a in zip(conf,corr):
            reliability_rows.append({
                "prompt": prompt,
                "response": response,
                "confidence": c,
                "correct": a,
                "model": model,
            })
    reliability_df = pd.DataFrame(reliability_rows)
    reliability_df.to_csv(
        BASE_PATH / "results" / 'reliability' / f"{model}_reliability_data{COPY}.csv",
        index=False,
    )
    print(f"Saved reliability data for {model} to CSV.")

    del policy_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

consistency_df = pd.DataFrame(list(consistencies.items()), columns=["model", "self_consistency"])
consistency_df.to_csv(
    BASE_PATH / "results" / 'reliability' / f"self_consistencies{COPY}.csv",
    index=False,
)
print(f"Saved self-consistency data to CSV.")

kl_df = pd.DataFrame(list(kls.items()), columns=["model", "kl_divergence"])
kl_df.to_csv(
    BASE_PATH / "results" / 'reliability' / f"kl_divergences{COPY}.csv",
    index=False,
)
print(f"Saved KL divergence data to CSV.")

perplexity_df = pd.DataFrame(list(perplexities.items()), columns=["model", "perplexity"])
perplexity_df.to_csv(
    BASE_PATH / "results" / 'reliability' / f"perplexities{COPY}.csv",
    index=False,
)
print(f"Saved perplexity data to CSV.")