import os
import json
import random
import re
import time
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpoint

from trl import GRPOConfig, GRPOTrainer
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from utils import train_test_split

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID
REWARD_MODEL_PATH = BASE_PATH / "reinforcement-learning" / "intermediate" / "reward_model_4o-Preferences"

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def build_rl_prompt(findings: str) -> str:
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


endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",  # free, good at instruction following
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    max_new_tokens=256,
    temperature=0.1,
)

class EnvironmentState(TypedDict):
    # Global loop control
    step: int
    max_steps: int
    should_stop: bool

    # Search space + strategy controls
    parameter_space: dict[str, list[Any]]
    fixed_config: dict[str, Any]
    current_config: dict[str, Any]
    pending_configs: list[dict[str, Any]]
    explored_configs: list[dict[str, Any]]
    explored_signatures: list[str]
    exploration_temperature: float

    # Tracking outcomes
    experiment_history: list[dict[str, Any]]
    best_score: float
    best_config: dict[str, Any]
    best_checkpoint_path: str | None
    last_score: float | None
    plateau_count: int
    early_stop_patience: int

    # Runtime + reproducibility
    run_name_prefix: str
    output_root: str
    resume_from_checkpoint: bool
    seed: int
    device: str
    notes: list[str]
    action: str
    selected_action: str | None
    selected_value: Any
    current_output_dir: str | None
    last_runtime_sec: float | None
    trial_sample_size: int
    trial_train_steps: int
    quick_tune_mode: bool


def build_initial_environment_state(max_steps: int | None = None) -> EnvironmentState:
    resolved_max_steps = max(1, int(AGENTIC_RLAIF_FLAGS["max_steps"] if max_steps is None else max_steps))
    resolved_trial_sample_size = max(1, int(AGENTIC_RLAIF_FLAGS["trial_sample_size"]))
    resolved_trial_train_steps = max(1, int(AGENTIC_RLAIF_FLAGS["trial_train_steps"]))
    resolved_early_stop_patience = max(1, int(AGENTIC_RLAIF_FLAGS["early_stop_patience"]))
    resolved_quick_tune_mode = bool(AGENTIC_RLAIF_FLAGS["quick_tune_mode"])
    return {
        "step": 0,
        "max_steps": resolved_max_steps,
        "should_stop": False,
        "parameter_space": {
            "learning_rate": [1e-8, 3e-8, 5e-8, 1e-7],
            "per_device_train_batch_size": [2, 4],
            "gradient_accumulation_steps": [4, 8],
            "beta": [0.1, 0.2, 0.3],
            "temperature": [0.3, 0.5, 0.7],
            "num_generations": [4, 8],
            "max_completion_length": [48, 64, 96],
            "num_train_epochs": [1, 2, 3],
        },
        "fixed_config": {
            "logging_strategy": "steps",
            "save_strategy": "epoch",
            "eval_strategy": "no",
            "report_to": "wandb" if _WANDB_AVAILABLE else "none",
        },
        "current_config": {},
        "pending_configs": [],
        "explored_configs": [],
        "explored_signatures": [],
        "exploration_temperature": 0.2,
        "experiment_history": [],
        "best_score": float("-inf"),
        "best_config": {},
        "best_checkpoint_path": None,
        "last_score": None,
        "plateau_count": 0,
        "early_stop_patience": resolved_early_stop_patience,
        "run_name_prefix": "grpo-agentic",
        "output_root": str(BASE_PATH / "reinforcement-learning" / "intermediate"),
        "resume_from_checkpoint": False,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "notes": [],
        "action": "none",
        "selected_action": None,
        "selected_value": None,
        "current_output_dir": None,
        "last_runtime_sec": None,
        "trial_sample_size": resolved_trial_sample_size,
        "trial_train_steps": resolved_trial_train_steps,
        "quick_tune_mode": resolved_quick_tune_mode,
    }



AVAILABLE_ACTIONS = {
    "change_beta": {
        "description": "Change the beta parameter to increase/decrease exploration",
        "options": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "change_learning_rate": {
        "description": "Change the learning rate to increase/decrease step size in parameter space",
        "options": [1e-8, 3e-8, 5e-8, 1e-7, 3e-7],
    },
    "change_temperature": {
        "description": "Change policy sampling temperature to trade off stability vs exploration",
        "options": [0.2, 0.3, 0.5, 0.7, 0.9],
    },
    "change_num_generations": {
        "description": "Change number of candidate completions per prompt",
        "options": [2, 4, 8, 16],
    },
    "change_per_device_train_batch_size": {
        "description": "Change per-device batch size based on memory and throughput",
        "options": [1, 2, 4],
    },
    "change_gradient_accumulation_steps": {
        "description": "Change accumulation steps to control effective batch size",
        "options": [2, 4, 8, 16],
    },
    "change_max_completion_length": {
        "description": "Change completion token budget",
        "options": [32, 48, 64, 96, 128],
    },
    "change_num_train_epochs": {
        "description": "Change number of epochs per trial",
        "options": [1, 2, 3, 4],
    },
    "change_trial_sample_size": {
        "description": "Change number of training samples used per tuning trial",
        "options": [20, 50, 100, 150],
    },
    "change_trial_train_steps": {
        "description": "Change max optimizer steps per tuning trial",
        "options": [10, 20, 40, 80, 120],
    },
    "change_exploration_temperature": {
        "description": "Change search-policy temperature for picking the next config",
        "options": [0.05, 0.1, 0.2, 0.3, 0.5],
    },
    "stop": {
        "description": "Stop experimentation when performance is satisfactory or budget is exhausted",
        "options": []
    }
}


def get_default_training_config() -> dict[str, Any]:
    return {
        "learning_rate": 5e-8,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "beta": 0.2,
        "max_completion_length": 64,
        "num_generations": 8,
        "temperature": 0.5,
        "num_train_epochs": 3,
    }


def parse_json_object(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config[k] for k in sorted(config.keys())}


def build_experiment_signature(
    trial_config: dict[str, Any],
    trial_sample_size: int,
    trial_train_steps: int,
    quick_tune_mode: bool,
    resume_from_checkpoint: bool,
) -> str:
    payload = {
        "trial_config": normalize_config(trial_config),
        "trial_sample_size": int(trial_sample_size),
        "trial_train_steps": int(trial_train_steps),
        "quick_tune_mode": bool(quick_tune_mode),
        "resume_from_checkpoint": bool(resume_from_checkpoint),
    }
    return json.dumps(payload, sort_keys=True, default=str)


AGENTIC_RLAIF_FLAGS: dict[str, Any] = {
    "max_steps": 12,
    "trial_sample_size": 256,
    "trial_train_steps": 40,
    "early_stop_patience": 5,
    "quick_tune_mode": True,
}


def build_trial_config(state: EnvironmentState) -> dict[str, Any]:
    merged = get_default_training_config()
    merged.update(state["fixed_config"])
    merged.update(state["current_config"])
    return merged


class RLAIFEnvironment:
    def __init__(self) -> None:
        self.instruction_dataset = load_instruction_dataset()

    def _generation_batch_size(self, config: dict[str, Any]) -> int:
        return int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])

    def _valid_num_generations(self, config: dict[str, Any]) -> list[int]:
        generation_batch_size = self._generation_batch_size(config)
        options = [
            int(option)
            for option in AVAILABLE_ACTIONS["change_num_generations"]["options"]
            if int(option) > 0 and generation_batch_size % int(option) == 0
        ]
        if 1 not in options:
            options.append(1)
        return sorted(set(options))

    def _sanitize_trial_config(self, config: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(config)
        valid_num_generations = self._valid_num_generations(sanitized)
        num_generations = int(sanitized["num_generations"])
        if num_generations not in valid_num_generations:
            sanitized["num_generations"] = max(option for option in valid_num_generations if option <= self._generation_batch_size(sanitized))
        return sanitized

    def _model_load_kwargs(self) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "quantization_config": quantization_config,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        return load_kwargs

    def _load_policy(self) -> tuple[Any, Any]:
        policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token
        if getattr(policy_tokenizer, "pad_token_id", None) is None and policy_tokenizer.eos_token_id is not None:
            policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

        load_kwargs = self._model_load_kwargs()
        try:
            policy_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
        except ValueError as exc:
            raise RuntimeError(
                "Policy model could not be placed with 4-bit offload on this machine. "
                "The current Qwen/Qwen2.5-7B-Instruct setup needs enough GPU RAM for the base model or a smaller checkpoint."
            ) from exc
        policy_model = PeftModel.from_pretrained(
            policy_base,
            SFT_POLICY_PATH,
            is_trainable=True,
            offload_buffers=True,
            offload_folder=str(BASE_PATH / "offload"),
        )
        try:
            if getattr(policy_model.config, "pad_token_id", None) is None and policy_tokenizer.eos_token_id is not None:
                policy_model.config.pad_token_id = policy_tokenizer.eos_token_id
        except Exception:
            pass
        return policy_model, policy_tokenizer

    def _load_reward(self) -> tuple[Any, Any]:
        reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        if getattr(reward_tokenizer, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
            reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

        load_kwargs = self._model_load_kwargs()
        try:
            reward_base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_ID,
                num_labels=1,
                **load_kwargs,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Reward model could not be placed with 4-bit offload on this machine. "
                "The current Qwen/Qwen2.5-7B-Instruct setup needs enough GPU RAM for the base model or a smaller checkpoint."
            ) from exc
        reward_model = PeftModel.from_pretrained(
            reward_base,
            REWARD_MODEL_PATH,
            is_trainable=False,
            offload_buffers=True,
            offload_folder=str(BASE_PATH / "offload"),
        )
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False
        try:
            if getattr(reward_model.config, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
                reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
        except Exception:
            pass
        return reward_model, reward_tokenizer

    def _extract_score(self, trainer: GRPOTrainer) -> float:
        history = getattr(getattr(trainer, "state", None), "log_history", []) or []
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            for key in ("reward", "rewards", "objective", "mean_reward"):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            loss_value = row.get("loss")
            if isinstance(loss_value, (int, float)):
                return -float(loss_value)
        return float("-inf")

    def run_experiment(self, state: EnvironmentState, exp_config: dict[str, Any]) -> dict[str, Any]:
        step = state["step"]
        run_name = f"{state['run_name_prefix']}-step-{step:02d}"
        output_dir = Path(state["output_root"]) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        exp_config = self._sanitize_trial_config(exp_config)

        policy_model, policy_tokenizer = self._load_policy()
        reward_model, reward_tokenizer = self._load_reward()

        trial_dataset = self.instruction_dataset
        if state["quick_tune_mode"] and state["trial_sample_size"] > 0:
            sample_size = min(int(state["trial_sample_size"]), len(self.instruction_dataset))
            if sample_size < len(self.instruction_dataset):
                trial_dataset = self.instruction_dataset.shuffle(seed=int(state["seed"]) + int(step)).select(range(sample_size))

        grpo_config = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=float(exp_config["learning_rate"]),
            per_device_train_batch_size=int(exp_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(exp_config["gradient_accumulation_steps"]),
            beta=float(exp_config["beta"]),
            max_completion_length=int(exp_config["max_completion_length"]),
            num_generations=int(exp_config["num_generations"]),
            temperature=float(exp_config["temperature"]),
            num_train_epochs=int(exp_config["num_train_epochs"]),
            logging_strategy=str(exp_config["logging_strategy"]),
            save_strategy=str(exp_config["save_strategy"]),
            eval_strategy=str(exp_config["eval_strategy"]),
            report_to=str(exp_config["report_to"]),
            seed=int(state["seed"]),
            max_steps=int(state["trial_train_steps"]) if state["quick_tune_mode"] else -1,
        )

        if _WANDB_AVAILABLE and str(exp_config["report_to"]) == "wandb":
            try:
                wandb.init(project="rl-grpo", name=run_name, config=exp_config, reinit=True)
            except Exception:
                pass

        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=[reward_model],
            reward_processing_classes=[reward_tokenizer],
            train_dataset=trial_dataset,
            processing_class=policy_tokenizer,
            args=grpo_config,
        )

        start_time = time.time()
        trainer.train(resume_from_checkpoint=bool(state["resume_from_checkpoint"]))
        trainer.save_model(str(output_dir))
        runtime = round(time.time() - start_time, 3)
        score = self._extract_score(trainer)

        return {
            "score": score,
            "runtime": runtime,
            "output_dir": str(output_dir),
            "config": normalize_config(exp_config),
        }


class AgenticRLAIFTuner:
    def __init__(self) -> None:
        self.env = RLAIFEnvironment()
        self.random_counter = 0
        self.total_actions = 0

        graph = StateGraph(EnvironmentState)
        graph.add_node("perceive", self.perceive)
        graph.add_node("select_action", self.select_action)
        graph.add_node("run_experiment", self.run_experiment)
        graph.add_node("update_history", self.update_history)

        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "select_action")
        graph.add_edge("select_action", "run_experiment")
        graph.add_edge("run_experiment", "update_history")
        graph.add_edge("update_history", END)
        self.agent = graph.compile()

    def perceive(self, state: EnvironmentState) -> EnvironmentState:
        done = len(state["experiment_history"])
        remaining = max(0, state["max_steps"] - done)
        print(f"Perceived state at step={state['step']}: trials_done={done}, trials_remaining={remaining}")
        return state

    def _coerce_option(self, action_name: str, raw_value: Any) -> Any:
        options = AVAILABLE_ACTIONS[action_name]["options"]
        if raw_value in options:
            return raw_value
        raw_as_text = str(raw_value)
        for option in options:
            if str(option) == raw_as_text:
                return option
        return None

    def _candidate_signature(self, state: EnvironmentState, action_name: str, action_value: Any) -> str | None:
        if action_name == "stop":
            return None

        candidate_config = dict(state["current_config"])
        candidate_trial_sample_size = int(state["trial_sample_size"])
        candidate_trial_train_steps = int(state["trial_train_steps"])
        candidate_quick_tune_mode = bool(state["quick_tune_mode"])
        candidate_resume = bool(state["resume_from_checkpoint"])

        mapping = {
            "change_beta": "beta",
            "change_learning_rate": "learning_rate",
            "change_temperature": "temperature",
            "change_num_generations": "num_generations",
            "change_per_device_train_batch_size": "per_device_train_batch_size",
            "change_gradient_accumulation_steps": "gradient_accumulation_steps",
            "change_max_completion_length": "max_completion_length",
            "change_num_train_epochs": "num_train_epochs",
        }

        if action_name in mapping:
            candidate_config[mapping[action_name]] = action_value
        elif action_name == "change_trial_sample_size":
            candidate_trial_sample_size = int(action_value)
        elif action_name == "change_trial_train_steps":
            candidate_trial_train_steps = int(action_value)
        elif action_name == "toggle_resume_from_checkpoint":
            candidate_resume = bool(action_value)

        trial_config = get_default_training_config()
        trial_config.update(state["fixed_config"])
        trial_config.update(candidate_config)

        return build_experiment_signature(
            trial_config=trial_config,
            trial_sample_size=candidate_trial_sample_size,
            trial_train_steps=candidate_trial_train_steps,
            quick_tune_mode=candidate_quick_tune_mode,
            resume_from_checkpoint=candidate_resume,
        )

    def _is_novel_action(self, state: EnvironmentState, action_name: str, action_value: Any) -> bool:
        signature = self._candidate_signature(state, action_name, action_value)
        if signature is None:
            return True
        return signature not in set(state["explored_signatures"])

    def _find_novel_action(self, state: EnvironmentState) -> tuple[str, Any] | None:
        randomizer = random.Random(int(state["seed"]) + int(state["step"]))
        actions = [name for name in AVAILABLE_ACTIONS.keys() if name != "stop"]
        randomizer.shuffle(actions)

        for action_name in actions:
            options = list(AVAILABLE_ACTIONS[action_name]["options"])
            randomizer.shuffle(options)
            for option in options:
                if self._is_novel_action(state, action_name, option):
                    return action_name, option
        return None

    def llm_action(self, state: EnvironmentState) -> tuple[str, Any]:
        action_space = {
            key: {"description": value["description"], "options": value["options"]}
            for key, value in AVAILABLE_ACTIONS.items()
        }
        prompt = (
            "You are choosing the next RL tuning action.\n"
            "Return JSON only.\n"
            f"State:\n{json.dumps(state, indent=2, default=str)}\n"
            f"Action space:\n{json.dumps(action_space, indent=2, default=str)}\n"
            "Respond EXACTLY like: {\"action\": \"change_beta\", \"value\": 0.3}\n"
            "If stopping: {\"action\": \"stop\", \"value\": null}"
        )

        raw_text = ""
        try:
            result = endpoint.invoke(prompt)
            raw_text = result if isinstance(result, str) else str(result)
        except Exception:
            pass

        parsed = parse_json_object(raw_text)
        action_name = parsed.get("action")
        action_value = parsed.get("value")

        self.total_actions += 1
        if action_name not in AVAILABLE_ACTIONS:
            self.random_counter += 1
            action_name = random.choice([k for k in AVAILABLE_ACTIONS.keys() if k != "stop"])
            action_value = random.choice(AVAILABLE_ACTIONS[action_name]["options"]) if AVAILABLE_ACTIONS[action_name]["options"] else None
            return action_name, action_value

        if action_name == "stop":
            return "stop", None

        coerced = self._coerce_option(action_name, action_value)
        if coerced is None:
            self.random_counter += 1
            coerced = random.choice(AVAILABLE_ACTIONS[action_name]["options"])
        return action_name, coerced

    def _apply_action(self, state: EnvironmentState, action_name: str, action_value: Any) -> EnvironmentState:
        state["action"] = action_name
        state["selected_action"] = action_name
        state["selected_value"] = action_value

        if action_name == "stop":
            state["should_stop"] = True
            return state

        mapping = {
            "change_beta": "beta",
            "change_learning_rate": "learning_rate",
            "change_temperature": "temperature",
            "change_num_generations": "num_generations",
            "change_per_device_train_batch_size": "per_device_train_batch_size",
            "change_gradient_accumulation_steps": "gradient_accumulation_steps",
            "change_max_completion_length": "max_completion_length",
            "change_num_train_epochs": "num_train_epochs",
        }

        if action_name in mapping:
            state["current_config"][mapping[action_name]] = action_value
        elif action_name == "change_trial_sample_size":
            state["trial_sample_size"] = int(action_value)
        elif action_name == "change_trial_train_steps":
            state["trial_train_steps"] = int(action_value)
        elif action_name == "change_exploration_temperature":
            state["exploration_temperature"] = float(action_value)
        elif action_name == "toggle_resume_from_checkpoint":
            state["resume_from_checkpoint"] = bool(action_value)

        return state

    def select_action(self, state: EnvironmentState) -> EnvironmentState:
        if state["should_stop"] or state["step"] >= state["max_steps"]:
            state["action"] = "stop"
            state["should_stop"] = True
            return state

        action_name, action_value = self.llm_action(state)
        if action_name != "stop" and not self._is_novel_action(state, action_name, action_value):
            forced = self._find_novel_action(state)
            if forced is None:
                state["notes"].append("No novel configurations remain; stopping.")
                state["action"] = "stop"
                state["should_stop"] = True
                return state
            action_name, action_value = forced
            state["notes"].append(f"Duplicate config avoided; forced novel action {action_name}={action_value}.")
        return self._apply_action(state, action_name, action_value)

    def run_experiment(self, state: EnvironmentState) -> EnvironmentState:
        if state["action"] == "stop":
            return state

        trial_config = build_trial_config(state)
        result = self.env.run_experiment(state, trial_config)

        state["last_score"] = float(result["score"])
        state["current_output_dir"] = str(result["output_dir"])
        state["last_runtime_sec"] = float(result["runtime"])
        return state

    def update_history(self, state: EnvironmentState) -> EnvironmentState:
        if state["action"] == "stop":
            return state

        trial_config = normalize_config(build_trial_config(state))
        signature = build_experiment_signature(
            trial_config=trial_config,
            trial_sample_size=state["trial_sample_size"],
            trial_train_steps=state["trial_train_steps"],
            quick_tune_mode=state["quick_tune_mode"],
            resume_from_checkpoint=state["resume_from_checkpoint"],
        )
        record = {
            "step": state["step"],
            "action": state["selected_action"],
            "value": state["selected_value"],
            "config": trial_config,
            "signature": signature,
            "score": state["last_score"],
            "runtime_sec": state["last_runtime_sec"],
            "output_dir": state["current_output_dir"],
        }
        state["experiment_history"].append(record)
        state["explored_configs"].append(trial_config)
        if signature not in state["explored_signatures"]:
            state["explored_signatures"].append(signature)

        current_score = float(state["last_score"]) if state["last_score"] is not None else float("-inf")
        if current_score > state["best_score"]:
            state["best_score"] = current_score
            state["best_config"] = trial_config
            state["best_checkpoint_path"] = state["current_output_dir"]
            state["plateau_count"] = 0
        else:
            state["plateau_count"] += 1

        if state["plateau_count"] >= state["early_stop_patience"]:
            state["notes"].append("Early stop triggered: no improvement within patience window.")
            state["should_stop"] = True

        return state

    def run(self, max_steps: int | None = None) -> EnvironmentState:
        state = build_initial_environment_state(max_steps=max_steps)
        print("STARTING agentic GRPO tuning workflow")

        while not state["should_stop"] and state["step"] < state["max_steps"]:
            state["step"] += 1
            print(f"\nStep {state['step']} =>")
            state = self.agent.invoke(state)

            print(
                {
                    "step": state["step"],
                    "action": state["selected_action"],
                    "value": state["selected_value"],
                    "score": state["last_score"],
                    "best_score": state["best_score"],
                }
            )

            if state["action"] == "stop":
                break

        print("\nFINISHED agentic GRPO tuning workflow")
        print(f"Best score: {state['best_score']}")
        print(f"Best config: {json.dumps(state['best_config'], indent=2, default=str)}")
        if state["best_checkpoint_path"]:
            print(f"Best checkpoint: {state['best_checkpoint_path']}")
        if self.total_actions:
            pct = (self.random_counter / self.total_actions) * 100.0
            print(f"Fallback random choices: {self.random_counter}/{self.total_actions} ({pct:.2f}%)")

        return state


def load_instruction_dataset() -> Dataset:
    data = pd.read_csv(BASE_PATH / "data" / "processed_iuxray_mcqa_dataset.csv")
    train_df, _ = train_test_split(data, test_size=0.2, random_state=42)
    prompts = [build_rl_prompt(str(findings)) for findings in train_df["findings"].tolist()]
    return Dataset.from_dict({"prompt": prompts})




if __name__ == "__main__":
    tuner = AgenticRLAIFTuner()
    tuner.run()
