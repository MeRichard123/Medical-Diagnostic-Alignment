"""
Experiment runner for the IUXray project.

This script encodes the workflow described in AGENTS.md:
- Focus on generation experiments.
- Run generation with text or VLM generators.
- Optionally finetune models with LoRA.
- Sweep over model sets in a reproducible way.
- Run `eval.py` to update metrics once generations are complete.

It assumes you execute it from the project root with your virtual environment
already activated (e.g. `.\.venv\Scripts\Activate.ps1` on Windows).


Example usage:
    # Run VLM generation sweep, then evaluation
    python runner.py --task vlm

    # Run text generation sweep, then evaluation
    python runner.py --task text

    # Run both text and VLM sweeps sequentially, then evaluation
    python runner.py --task both

    # Finetune VLM models, then evaluation
    python runner.py --task finetune-vlm

    # Finetune text models, then evaluation
    python runner.py --task finetune-text

    # Generate with finetuned VLM models, then evaluation
    python runner.py --task generate-finetuned-vlm

    # Generate with finetuned text models, then evaluation
    python runner.py --task generate-finetuned-text
"""
import os
import shutil
import subprocess
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PRIMARY_TEXT_MODELS: List[str] = [
    "Kavyaah/medical-coding-llm",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "epfl-llm/meditron-7b",
    "google/medgemma-4b-it",
    "haohao12/qwen2.5-7b-medical",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]


PRIMARY_VLM_MODELS: List[str] = [
    "HuggingFaceTB/SmolVLM-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Salesforce/blip2-opt-2.7b",
]


def convert_model_to_model_id(model_name: str) -> str:
    return model_name.split("/")[-1]


def _run(cmd: list[str], env: dict | None = None) -> None:
    """Run a subprocess command and stream output, raising on failure."""
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, env=env or os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


@dataclass
class GenerationExperiment:
    model_id: str
    seed: int = 42


def clear_hf_cache(keep_substrings: List[str] | None = None) -> None:
    """Clear Hugging Face hub cache except for entries containing keep_substrings.

    This follows AGENTS.md: clear cache before new model downloads, but try to keep
    roberta large for BERTScore to avoid repeated downloads.
    """
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    if not cache_root.exists():
        return

    keep_substrings = keep_substrings or []

    def should_keep(path: Path) -> bool:
        lower = str(path).lower()
        return any(sub.lower() in lower for sub in keep_substrings)

    for child in cache_root.iterdir():
        if should_keep(child):
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; ignore individual failures.
            continue


def is_model_cached(model_id: str) -> bool:
    """Return True if model appears to already exist in local HF hub cache."""
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    if not cache_root.exists():
        return False

    model_id_lower = model_id.lower()
    # HF cache directory names often encode org/model as models--org--model
    encoded_model_id = model_id.replace("/", "--").lower()
    model_name_only = model_id.split("/")[-1].lower()

    for child in cache_root.iterdir():
        name = child.name.lower()
        if encoded_model_id in name or model_name_only in name or model_id_lower in name:
            return True
    return False


@dataclass
class ExperimentRunner:
    models: List[str] = field(default_factory=list)
    generation_module: str = "inference_only.generation.text_inference"
    python_executable: str = field(default_factory=lambda: sys.executable)

    # check ./results to see which models have already been run
    def get_models_already_run(self, pattern) -> List[str]:
        return [f.stem for f in Path("./results").glob("*.json") if f.stem.endswith(pattern)]

    def run_generation_for_model(self, model_id: str, use_lora: bool = False, clear_cache: bool = True) -> None:
        """Run generation for a single model.

        The model is passed via GEN_MODEL_ID. When use_lora=True, generation
        uses finetuned adapters for models that support it.
        """
        env = os.environ.copy()
        env["GEN_MODEL_ID"] = model_id
        if use_lora and "vlm" in self.generation_module.lower():
            env["USE_VLM_LORA"] = "1"

        mode = "with LoRA" if use_lora else "base"
        print(f"\n=== Preparing generation ({mode}) for: {model_id} ===")
        if clear_cache:
            if is_model_cached(model_id):
                print("Model already cached. Skipping cache clear.")
            else:
                # Only clear cache for fresh downloads, while keeping roberta-large.
                clear_hf_cache(keep_substrings=["roberta-large"])
                print("Model not cached. Hugging Face cache cleared (except kept patterns).")

        print(f"=== Running generation ({mode}) for: {model_id} ===")
        _run(
            [
                self.python_executable,
                "-m",
                self.generation_module,
            ],
            env=env,
        )

    def run_finetune_for_model(self, model_id: str) -> None:
        """Run finetuning for a single model.
        
        For VLM generation, calls finetuning/generation/vlm_tuning.py.
        For text generation, calls finetuning/generation/text_tuning.py (if available).
        """
        env = os.environ.copy()
        env["GEN_MODEL_ID"] = model_id
        print(f"\n=== Preparing finetuning for: {model_id} ===")
        
        # Determine which tuning script to use based on generation_module
        if "vlm" in self.generation_module.lower():
            tuning_script = "finetuning.generation.vlm_tuning"
        else:
            tuning_script = "finetuning.generation.text_tuning"
        
        print(f"=== Running finetuning for: {model_id} using {tuning_script} ===")
        _run(
            [
                self.python_executable,
                "-m",
                tuning_script,
            ],
            env=env,
        )

    def run_finetune_then_generate_for_model(self, model_id: str) -> None:
        """Run one full per-model cycle: clear cache -> finetune -> LoRA generation."""
        print(f"\n=== Starting finetune + generation cycle for: {model_id} ===")
        if is_model_cached(model_id):
            print("Model already cached. Skipping cache clear.")
        else:
            clear_hf_cache(keep_substrings=["roberta-large"])
            print("Model not cached. Hugging Face cache cleared (except kept patterns).")
        self.run_finetune_for_model(model_id)
        # Do not clear cache here so generation reuses the already downloaded base model.
        self.run_generation_for_model(model_id, use_lora=True, clear_cache=False)
        print(f"=== Completed finetune + generation cycle for: {model_id} ===")

    def run_all_generations(self) -> None:
        """Run generation for all configured text models sequentially."""
        models_already_run = self.get_models_already_run("_generation_results")
        for model_name in self.models:
            model_name_id = convert_model_to_model_id(model_name)
            if model_name_id + "_generation_results" in models_already_run:
                continue
            self.run_generation_for_model(model_name, use_lora=False, clear_cache=True)

    def run_all_generations_with_lora(self) -> None:
        """Run generation for all configured models with LoRA adapters enabled."""
        models_already_run = self.get_models_already_run("-finetuned_generation_results")
        for model_name in self.models:
            model_name_id = convert_model_to_model_id(model_name)
            if model_name_id + "-finetuned_generation_results" in models_already_run:
                continue
            self.run_generation_for_model(model_name, use_lora=True, clear_cache=True)

    def run_all_finetunings_evals(self) -> None:
        """Run per-model finetune+LoRA-generation cycles sequentially."""
        models_already_run = self.get_models_already_run("-finetuned_generation_results")
        for model_name in self.models:
            model_name_id = convert_model_to_model_id(model_name)
            if model_name_id + "-finetuned_generation_results" in models_already_run:
                print(f"Skipping {model_name} (already finetuned and evaluated).")
                continue
            self.run_finetune_then_generate_for_model(model_name)

    def run_evaluation(self) -> None:
        """Run eval.py to update metrics CSV files from generated JSON outputs."""
        print("\n=== Running evaluation (eval.py) ===")
        _run([self.python_executable, "eval.py"])

    def run_full_suite(self) -> None:
        """Run all generation experiments then evaluation, end-to-end."""
        print("=" * 60)
        print("IUXray Experimenter: generation + evaluation sweep")
        print("=" * 60)
        print(f"Generation module: {self.generation_module}")
        print("Models to run:")
        for m in self.models:
            print(f"  - {m}")

        self.run_all_generations()
        self.run_evaluation()
        print("\nAll experiments completed successfully.")


def _build_runner(task: str) -> ExperimentRunner:
    task_lower = task.strip().lower()
    if task_lower == "vlm":
        return ExperimentRunner(
            models=PRIMARY_VLM_MODELS.copy(),
            generation_module="inference_only.generation.vlm_inference",
        )
    return ExperimentRunner(
        models=PRIMARY_TEXT_MODELS.copy(),
        generation_module="inference_only.generation.text_inference",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IUXray generation experiment sweeps.")
    parser.add_argument(
        "--task",
        choices=[
            "text",
            "vlm",
            "both",
            "finetune-vlm",
            "finetune-text",
            "generate-finetuned-vlm",
            "generate-finetuned-text",
        ],
        default="vlm",
        help="Select which generation/finetuning sweep to run.",
    )
    args = parser.parse_args()

    if args.task == "both":
        text_agent = _build_runner("text")
        print("Running text generation sweep...")
        text_agent.run_all_generations()
        vlm_agent = _build_runner("vlm")
        print("Running VLM generation sweep...")
        vlm_agent.run_all_generations()
        text_agent.run_evaluation()
        print("\nAll experiments completed successfully.")
    elif args.task == "finetune-vlm":
        vlm_agent = _build_runner("vlm")
        print("Running VLM finetuning sweep...")
        vlm_agent.run_all_finetunings_evals()
        vlm_agent.run_evaluation()
        print("\nVLM finetuning sweep completed successfully.")
    elif args.task == "finetune-text":
        text_agent = _build_runner("text")
        print("Running text finetuning sweep...")
        text_agent.run_all_finetunings_evals()
        text_agent.run_evaluation()
        print("\nText finetuning sweep completed successfully.")
    elif args.task == "generate-finetuned-vlm":
        vlm_agent = _build_runner("vlm")
        print("Running VLM generation with LoRA sweep...")
        vlm_agent.run_all_generations_with_lora()
        vlm_agent.run_evaluation()
        print("\nVLM finetuned generation sweep completed successfully.")
    elif args.task == "generate-finetuned-text":
        text_agent = _build_runner("text")
        print("Running text generation with LoRA sweep...")
        text_agent.run_all_generations_with_lora()
        text_agent.run_evaluation()
        print("\nText finetuned generation sweep completed successfully.")
    else:
        agent = _build_runner(args.task)
        agent.run_full_suite()