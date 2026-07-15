import os
import json
import numpy as np
import torch
from Evaluation.Evaluator import QAEvaluator, VisionEvaluator, ReportEvaluator
from pathlib import Path
from typing import Optional
from Evaluation.reliability import (
    MODEL_ID_MAPPING,
    load_base_model,
    compute_self_consistency,
    cleanup_base_model_cache,
)
from ReinforcementLearning.Custom.Config import QuantizationConfig, ModelConfigSection



quantisation_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)


BASE_DIR = "./results"
METRIC_PATH = "./results/metrics/"
BASE_PATH = Path(__file__).parent

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
    #"Qwen2.5-7B-Instruct-finetuned",
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
    #"Qwen2.5-7B-Instruct-RLVR_aligned",
    "Qwen3-VL-8B-Instruct-RLVR_aligned",
    #"Qwen2.5-7B-Instruct-RLAIF_aligned",
    #"Qwen2.5-7B-Instruct-RLAIF_tanh-tanh_aligned",
    #"Qwen2.5-7B-Instruct-RLAIF_pref3_aligned",
    #"Qwen2.5-7B-Instruct-RLAIF_custom_GRPO_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_noscore_aligned",
    # "Qwen2.5-7B-Instruct-RLAIF_FrugalGRPO_aligned",
    #"Qwen2.5-7B-Instruct-RLAIF_GroupedGRPO_aligned",
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
}


generation_task = [
    #"Qwen2.5-7B-Instruct", 
    "medgemma-4b-it", 
    "meditron-7b", 
    "medical-coding-llm",
    #"Meta-Llama-3.1-8B-Instruct-bnb-4bit", 
    "Phi-3-mini-4k-instruct", 
    "gemma-3-4b-it",
    "qwen2.5-7b-medical"
]


generation_vlm = [
    "SmolVLM-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    #"Llama-3.2-11B-Vision-Instruct",
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


BASE_PATH = Path(__file__).parent
BASE_LORA = BASE_PATH / "finetuning" / "tuned"
BASE_RL = BASE_PATH / "ReinforcementLearning"
BASE_GENERATION = BASE_PATH / "generation_results"  # Or wherever your generation results are


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


def _load_results(model_name: str, suffix: str, skip_missing: bool = False):
    path = _results_path(model_name, suffix)
    if not os.path.exists(path):
        if skip_missing:
            print(f"Warning: Results file for {model_name} not found, skipping evaluation.")
            return None
        raise FileNotFoundError(f"Missing required results file: {path}")

    with open(path, "r") as fp:
        return json.load(fp)


def _extract_classification_fields(parse):
    predictions = np.array([data["predicted_diagnosis"] for data in parse])
    ground_truth = np.array(
        [
            data["ground_truth"].strip()
            for data in parse
            if not (isinstance(data["ground_truth"], float) and np.isnan(data["ground_truth"]))
        ]
    )
    predicted = [p if p else "N/A" for p in predictions]
    probabilities = [normalise_probabilities(data["probabilities"]) for data in parse]
    return ground_truth, predicted, probabilities


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


def evaluate_classification_group(model_names, evaluator_cls, suffix="_inference_results.json", skip_missing=False):
    stats = []
    for model_name in model_names:
        parse = _load_results(model_name, suffix, skip_missing=skip_missing)
        if parse is None:
            continue

        ground_truth, predicted, probabilities = _extract_classification_fields(parse)
        evaluator = evaluator_cls(model_name)
        data = evaluator.evaluate(ground_truth, predicted, probabilities)
        stats.append({"model": model_name, **data})

    return stats


def evaluate_report_group(model_names, suffix="_generation_results.json", skip_missing=False):
    """
    Evaluate a group of report generation models with automatic path detection.
    """
    stats = []
    
    for model_name in model_names:
        parse = _load_results(model_name, suffix, skip_missing=skip_missing)
        if parse is None:
            continue

        ground_truth, predicted, probabilities, pred_raw, prompt_ids = _extract_report_fields(parse)
        evaluator = ReportEvaluator(model_name)
        
        data = evaluator.evaluate(ground_truth, predicted, probabilities, pred_raw)
        
        stats.append({"model": model_name, **data})

    return stats

def write_metrics_csv(file_name, columns, stats, reliabs=None):
    out_path = os.path.join(METRIC_PATH, file_name)
    with open(out_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for stat in stats:
            row = [stat["model"]]
            for metric_name in columns[1:]:
                value = stat.get(metric_name)
                if reliabs and metric_name == "Reliability" and stat["model"] in reliabs:
                    value = reliabs[stat["model"]]
                if value is not None:
                    row.append(f"{value:.4f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")

stats_text = evaluate_classification_group(models, QAEvaluator)
stats_vlm = evaluate_classification_group(vlms, QAEvaluator)
stats_vision = evaluate_classification_group(vision_models, VisionEvaluator)
trained_vision_stats = evaluate_classification_group(trained_vision, VisionEvaluator)
finetuned_stats = evaluate_classification_group(finetuned_models, QAEvaluator)

report_stats = evaluate_report_group(generation_task)
report_vlm_stats = evaluate_report_group(generation_vlm, skip_missing=True)
finetuned_report_stats = evaluate_report_group(finetuned_generation)
finetuned_vlm_report_stats = evaluate_report_group(finetuned_vlm_generation, skip_missing=True)
rl_report_stats = evaluate_report_group(reinforcement_learning_models, skip_missing=True)


all_models = finetuned_generation + reinforcement_learning_models + generation_task + generation_vlm + finetuned_vlm_generation
tuned = LORA_Tuned_Paths | RL_Tuned_Paths | VLM_Tuned_Paths

def get_model_id_and_peft(model: str):
    hugging_face_id = MODEL_ID_MAPPING.get(model)
    peft_path = tuned.get(model)
    return hugging_face_id, peft_path

consistencies = {}
current_model_id = None
base_model = None
tokenizer = None

for model in sorted(all_models):
    if "phi" in model.lower() or "coding" in model.lower() or 'llama' in model.lower() or 'vl' in model.lower():
        continue

    id_, peft = get_model_id_and_peft(model)
    if not id_:
        continue

    parse = _load_results(model, "_generation_results.json", skip_missing=True)
    if parse is None:
        continue

    if id_ != current_model_id:
        if base_model is not None:
            del base_model
            del tokenizer
            cleanup_base_model_cache()

        base_config = ModelConfigSection(
            model_name=id_,
            ref_model_name=id_,
            tokenizer_name=id_,
            quantization=quantisation_config,
        )
        base_model, tokenizer = load_base_model(torch.device("cuda"), base_config)
        current_model_id = id_

    gts, preds, _, pred_raw, prompt_ids = _extract_report_fields(parse)
    consistencies[model] = compute_self_consistency(
        base_model,
        tokenizer,
        peft,
        preds,
        gts,
        prompt_ids,
        model_id=id_,
    )

if base_model is not None:
    del base_model
    del tokenizer
    cleanup_base_model_cache()


consistencies['Meta-Llama-3.1-8B-Instruct-bnb-4bit'] = 0.2667 
consistencies['"Llama-3.2-11B-Vision-Instruct'] = 0.2917

with open("backup_reliab.json", "w") as f:
    import json
    json.dump(consistencies, f, indent=4,sort_keys=True)

write_metrics_csv(
    "evaluation_stats_text.csv",
    ["model", "EM", "F1", "BA", "MRR", "BS", "KL", "Perp"],
    stats_text, 
)
write_metrics_csv(
    "evaluation_stats_vlm.csv",
    ["model", "EM", "F1", "BA", "MRR", "BS", "KL", "Perp"],
    stats_vlm,
)
write_metrics_csv(
    "evaluation_stats_vision.csv",
    ["model", "EM", "F1", "BA", "MRR", "BS", "KL"],
    stats_vision,
)
write_metrics_csv(
    "evaluation_stats_finetuned.csv",
    ["model", "EM", "F1", "BA", "MRR", "BS", "KL", "Perp"],
    finetuned_stats,
)
write_metrics_csv(
    "evaluation_stats_trained_vision.csv",
    ["model", "EM", "F1", "BA", "MRR", "BS", "KL"],
    trained_vision_stats,
)
write_metrics_csv(
    "evaluation_stats_report.csv",
    ["model", "EM", "TokP", "TokR", "TokF1", "Contain", "CharSim", "CosSim", "KL", "Perp", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "Reliability", "Miscalibration"],
    report_stats, consistencies
)
write_metrics_csv(
    "evaluation_stats_report_vlm.csv",
    ["model", "EM", "TokP", "TokR", "TokF1", "Contain", "CharSim", "CosSim", "KL", "Perp", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "Reliability", "Miscalibration"],
    report_vlm_stats,consistencies
)
write_metrics_csv(
    "evaluation_stats_finetuned_report.csv",
    ["model", "EM", "TokP", "TokR", "TokF1", "Contain", "CharSim", "CosSim", "KL", "Perp", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "Reliability", "Miscalibration"],
    finetuned_report_stats,consistencies
)
write_metrics_csv(
    "evaluation_stats_finetuned_vlm_report.csv",
    ["model", "EM", "TokP", "TokR", "TokF1", "Contain", "CharSim", "CosSim", "KL", "Perp", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "Reliability", "Miscalibration"],
    finetuned_vlm_report_stats,consistencies
)
write_metrics_csv(
    "evaluation_stats_rl_report.csv",
    ["model", "EM", "TokP", "TokR", "TokF1", "Contain", "CharSim", "CosSim", "KL", "Perp", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "Reliability", "Miscalibration"],
    rl_report_stats,consistencies
)
