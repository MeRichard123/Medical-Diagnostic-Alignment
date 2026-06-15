import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from .BaseRewardModel import (
        finish_wandb, init_wandb, 
        PreferenceRewardModelTRLAdapter,
        CustomBTRewardTrainer
    )
from pathlib import Path
from datasets import Dataset
from trl import RewardConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import InferenceClient

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET =  Path(__file__).parent.parent / 'intermediate' / 'preference_data_normalized.csv'
DEFAULT_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 16
DEFAULT_MAX_LENGTH = 512
DEFAULT_EPOCHS = 1
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42


def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None

def llm_judge(question: str, answer: str, doctor_gt: str) -> float:
    prompt = """You are an expert medical judge. Evaluate clinical correctness.

    Question: {question}

    Rubric (1-5):
    1=clinically wrong  2=mostly wrong  3=partially correct  4=mostly correct  5=fully correct

    Doctor ground truth: {doctor_gt}
    Answer: {answer}

    Return JSON only in the form: {{"score_a": <1-5>, "explanation": "<one sentence>"}}
    """
    output = llm_client.text_generation(
        prompt=prompt.format(question=question, answer=answer, doctor_gt=doctor_gt),
        max_new_tokens=1000,
    )
    # output may be a complex object; attempt to extract text
    response = getattr(output, 'generated_text', None) or str(output)
    score = extract_judge_score(response)
    return score

class FrugalRewardModel(torch.nn.Module):
    def __init__(self, policy_model=None, judge_model=None, tokenizer=None, num_consistency_samples: int = 4, **kwargs):
        super().__init__()
        self.t_acc = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_score = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_reliab = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_miscalib = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.policy_model = policy_model
        self.judge_model = judge_model
        self.tokenizer = tokenizer
        self.num_consistency_samples = num_consistency_samples

    def _placeholder_metric(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        return torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def acc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None):
        """Accuracy: 1.0 if prediction matches doctor's ground truth, 0.0 otherwise.
        
        Args:
            text: The predicted response text (assumes it ends with the diagnosis).
            ground_truth: The correct diagnosis label.
        """
        if ground_truth is None or text is None:
            return self._placeholder_metric(input_ids)
        
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        # Extract diagnosis from text (last line or last 1-4 words)
        if isinstance(text, str):
            text = [text]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        accs = []
        for pred, gt in zip(text, ground_truth):
            pred_diagnosis = pred.strip().split('\n')[-1].strip() if pred else ""
            match = 1.0 if pred_diagnosis.lower() == gt.lower() else 0.0
            accs.append(match)
        
        return torch.tensor(accs, device=input_ids.device, dtype=torch.float32)
    
    def score(self, question, answer, doctor_gt, input_ids: torch.Tensor = None) -> torch.Tensor:
        """LLM-judge score: question, answer, doctor_gt -> tensor in [0,1].

        `question`, `answer`, and `doctor_gt` can be strings or lists of strings.
        If `input_ids` is provided it is used to determine the device and batch size;
        otherwise `self.device` is used.
        """
        if self.judge_model is None or answer is None:
            # Return placeholder matching batch size
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                return self._placeholder_metric(input_ids)
            return torch.tensor([0.0], device=self.device, dtype=torch.float32)

        # Normalize to lists
        if isinstance(question, str):
            question = [question]
        if isinstance(answer, str):
            answer = [answer]
        if isinstance(doctor_gt, str):
            doctor_gt = [doctor_gt]

        scores = []
        for q, a, gt in zip(question, answer, doctor_gt):
            try:
                sc = llm_judge(q, a, gt)
                scores.append((float(sc) / 5.0) if sc is not None else 0.0)
            except Exception as e:
                print(f"Error in score(): {e}")
                scores.append(0.0)

        device = input_ids.device if (input_ids is not None and isinstance(input_ids, torch.Tensor)) else self.device
        return torch.tensor(scores, device=device, dtype=torch.float32)
        
        

    def reliab(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt=None):
        """Reliability: Self-consistency via multiple generations.
        
        Args:
            prompt: The input prompt to generate multiple completions from.
        
        Returns:
            Consistency score in range [0, 1].
        """
        if self.policy_model is None or prompt is None:
            return self._placeholder_metric(input_ids)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        batch_size = len(prompt)
        reliab_scores = []
        
        for p in prompt:
            try:
                # Generate multiple completions
                prompt_input = self.tokenizer(p, return_tensors='pt', truncation=True, max_length=512)
                prompt_input = {k: v.to(self.device) for k, v in prompt_input.items()}
                
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **prompt_input,
                        num_return_sequences=self.num_consistency_samples,
                        max_length=100,
                        temperature=0.7,
                        do_sample=True,
                    )
                
                # Decode completions and extract diagnoses
                diagnoses = []
                for seq in outputs:
                    text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    diagnosis = text.split('\n')[-1].strip() if text else ""
                    diagnoses.append(diagnosis)
                
                # Compute consistency: proportion of samples that match the most common diagnosis
                from collections import Counter
                diagnosis_counts = Counter(diagnoses)
                max_count = max(diagnosis_counts.values()) if diagnosis_counts else 0
                consistency = max_count / self.num_consistency_samples if self.num_consistency_samples > 0 else 0.0
                reliab_scores.append(consistency)
            except Exception as e:
                print(f"Error computing reliability: {e}")
                reliab_scores.append(0.5)
        
        return torch.tensor(reliab_scores, device=input_ids.device, dtype=torch.float32)

    def miscalib(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None):
        """Miscalibration: Expected Calibration Error (ECE) placeholder.
        
        For now, use a simple proxy: if confidence (all 1s from model) doesn't match accuracy.
        
        Args:
            text: The response text.
            ground_truth: The correct label.
        
        Returns:
            Miscalibration score (0 = well-calibrated, higher = worse).
        """
        if ground_truth is None or text is None:
            return self._placeholder_metric(input_ids)
        
        if isinstance(text, str):
            text = [text]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        miscalibs = []
        for pred, gt in zip(text, ground_truth):
            pred_diagnosis = pred.strip().split('\n')[-1].strip() if pred else ""
            is_correct = 1.0 if pred_diagnosis.lower() == gt.lower() else 0.0
            # Assume model confidence is always high (model always outputs something)
            # Miscalibration = |confidence - accuracy|, confidence ≈ 1.0
            miscalib = abs(1.0 - is_correct)
            miscalibs.append(miscalib)
        
        return torch.tensor(miscalibs, device=input_ids.device, dtype=torch.float32)

    def _positive_temp(self, temp: torch.Tensor) -> torch.Tensor:
        return F.softplus(temp) + 1e-6

    def U_plus(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None) -> torch.Tensor:
        delta_plus = (
            self.acc(input_ids, attention_mask, ground_truth=ground_truth, text=text) / self._positive_temp(self.t_acc)
            + self.score(prompt, text, ground_truth, input_ids=input_ids) / self._positive_temp(self.t_score)
        )
        return torch.tanh(delta_plus)
    

    def U_minus(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None) -> torch.Tensor:
        delta_minus = (
            self.reliab(input_ids, attention_mask, prompt=prompt) / self._positive_temp(self.t_reliab)
            + self.miscalib(input_ids, attention_mask, ground_truth=ground_truth, text=text) / self._positive_temp(self.t_miscalib)
        )
        return torch.tanh(delta_minus)


    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None) -> torch.Tensor:
        """Compute the frugal reward as U_plus - U_minus."""
        reward = (
            self.U_plus(input_ids, attention_mask, ground_truth=ground_truth, text=text, prompt=prompt) 
            - self.U_minus(input_ids, attention_mask, ground_truth=ground_truth, text=text, prompt=prompt)
        )
        return reward

    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        ground_truth_chosen=None,
        ground_truth_rejected=None,
        chosen_text=None,
        rejected_text=None,
        prompt=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Bradley-Terry preference loss with frugal reward metrics.

        Returns:
            loss: -log(sigmoid(r_chosen - r_rejected))
            r_chosen: Rewards for chosen responses
            r_rejected: Rewards for rejected responses
        """
        r_chosen = self.get_reward(
            chosen_ids, chosen_mask,
            ground_truth=ground_truth_chosen,
            text=chosen_text,
            prompt=prompt
        )
        r_rejected = self.get_reward(
            rejected_ids, rejected_mask,
            ground_truth=ground_truth_rejected,
            text=rejected_text,
            prompt=prompt
        )

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss, r_chosen, r_rejected

def train_preference_rm_trl(
    model_id: str = DEFAULT_MODEL_ID,
    dataset_path: Path | None = None,
    output_dir: str | None = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    use_wandb: bool = True,
    wandb_run_name: str = 'reward-model-bt-tanh',
    policy_model=None,
    judge_model=None,
):
    """Train FrugalRewardModel with temperature optimization.

    Uses CustomBTRewardTrainer which calls FrugalRewardModel.forward() with ground truth,
    chosen/rejected texts, and prompts to compute real metrics (accuracy, LLM-judge score,
    self-consistency, miscalibration).
    
    Expects CSV at `dataset_path` to contain: chosen, rejected, ground_truth, prompt columns.
    """
    base_path = Path(__file__).parent.parent
    if dataset_path is None:
        dataset_path = base_path / 'intermediate' / 'preference_data_normalized.csv'
    if output_dir is None:
        output_dir = str(base_path / 'RLFF' / 'Reward_Models' / 'reward_model_bt-frugal')

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and clean preference CSV
    import pandas as pd

    pref_df = pd.read_csv(dataset_path)
    required_cols = ['chosen', 'rejected', 'doctor_gt', 'instruction']
    missing_cols = [col for col in required_cols if col not in pref_df.columns]
    if missing_cols:
        raise ValueError(f'preference CSV must contain columns: {required_cols}. Missing: {missing_cols}')
    
    pref_df = pref_df[required_cols].dropna()
    pref_df = pref_df[(pref_df['chosen'].str.strip() != '') & (pref_df['rejected'].str.strip() != '')].reset_index(drop=True)
    if len(pref_df) < 2:
        raise ValueError('Not enough preference pairs after cleaning')

    # Tokenize chosen/rejected into the format TRL expects
    def tokenize_pair(chosen_text, rejected_text):
        chosen_enc = tokenizer(chosen_text, truncation=True, max_length=DEFAULT_MAX_LENGTH, return_tensors=None)
        rejected_enc = tokenizer(rejected_text, truncation=True, max_length=DEFAULT_MAX_LENGTH, return_tensors=None)
        return {
            'chosen_input_ids': chosen_enc['input_ids'],
            'chosen_attention_mask': chosen_enc['attention_mask'],
            'rejected_input_ids': rejected_enc['input_ids'],
            'rejected_attention_mask': rejected_enc['attention_mask'],
        }

    data_list = []
    for _, row in pref_df.iterrows():
        pair = tokenize_pair(row['chosen'], row['rejected'])
        pair['doctor_gt'] = row['doctor_gt']
        pair['instruction'] = row['instruction']
        pair['chosen_text'] = row['chosen']
        pair['rejected_text'] = row['rejected']
        data_list.append(pair)

    # Create dataset
    train_dataset = Dataset.from_dict({
        'chosen_input_ids': [d['chosen_input_ids'] for d in data_list],
        'chosen_attention_mask': [d['chosen_attention_mask'] for d in data_list],
        'rejected_input_ids': [d['rejected_input_ids'] for d in data_list],
        'rejected_attention_mask': [d['rejected_attention_mask'] for d in data_list],
        'doctor_gt': [d['doctor_gt'] for d in data_list],
        'instruction': [d['instruction'] for d in data_list],
        'chosen_text': [d['chosen_text'] for d in data_list],
        'rejected_text': [d['rejected_text'] for d in data_list],
    })

    # Split train/eval
    eval_frac = 0.1
    train_eval_split = train_dataset.train_test_split(test_size=eval_frac, seed=42)
    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']

    # Initialize your custom preference reward model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preference_model = FrugalRewardModel(
        policy_model=policy_model,
        judge_model=judge_model,
        tokenizer=tokenizer,
        num_consistency_samples=4
    ).to(device)
    
    # Wrap it with the adapter (for TRL compatibility)
    adapter_model = PreferenceRewardModelTRLAdapter(preference_model)

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        save_strategy='epoch',
        logging_strategy='steps',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='wandb' if use_wandb else 'none',
        run_name=wandb_run_name,
        learning_rate=DEFAULT_LR,
    )

    if use_wandb:
        try:
            init_wandb(
                default_run_name=wandb_run_name,
                config={
                    'model_id': model_id,
                    'num_train_epochs': num_train_epochs,
                    'per_device_train_batch_size': per_device_train_batch_size,
                    'learning_rate': DEFAULT_LR,
                },
                use_wandb=True,
            )
        except Exception:
            print('Failed to initialize Weights & Biases. Continuing without logging.')
            use_wandb = False

    # Use the adapter itself as the backend model so the preference_model's
    # trainable parameters (e.g. temperature params) are registered and optimized.
    backend_model = adapter_model

    # ensure trainer can access the preference_model attribute
    object.__setattr__(backend_model, 'preference_model', adapter_model.preference_model)

    # Use custom trainer with your BT loss
    trainer = CustomBTRewardTrainer(
        model=backend_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    
    finish_wandb()
    return preference_model

BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID

if __name__ == "__main__":
    policy_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        low_cpu_mem_usage=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
        quantization_config=quantization_config,
        device_map="auto",
    )
    print("Loaded Policy Model Base from:", BASE_MODEL_ID)
    policy_model = PeftModel.from_pretrained(
        policy_base,
        SFT_POLICY_PATH,
        is_trainable=True,
        offload_buffers=True,
        offload_folder=str(BASE_PATH / "offload"),
    )
    print(f"Loaded Policy Model from PEFT path: {SFT_POLICY_PATH}")



    train_preference_rm_trl(
        output_dir=str(Path(__file__).parent.parent / 'RLFF' / 'Reward_Models' / 'reward_model_BT_frugal'),
        wandb_run_name='reward-model-bt-frugal',
        policy_model=policy_model,
        )


