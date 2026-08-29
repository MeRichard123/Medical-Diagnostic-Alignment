import math
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
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
DEFAULT_MAX_LENGTH = 256
DEFAULT_EPOCHS = 1
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42

BASE_PATH = Path(__file__).parent.parent.parent

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL = MODEL_ID.split("/")[-1]
SFT_POLICY_PATH = BASE_PATH / "finetuning" / "tuned" / f"{MODEL}-gen-lora"
BASE_MODEL_ID = MODEL_ID
REWARD_MODEL_PATH = BASE_PATH / "ReinforcementLearning" / "intermediate" / "reward_model_4o-Preferences"

reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
if getattr(reward_tokenizer, "pad_token_id", None) is None and reward_tokenizer.eos_token_id is not None:
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

reward_base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=1,
    low_cpu_mem_usage=True,
    offload_buffers=True,
    offload_folder=str(BASE_PATH / "offload"),
    quantization_config= BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)
print("Loaded Reward Model Base from:", BASE_MODEL_ID)
reward_model = PeftModel.from_pretrained(
    reward_base,
    REWARD_MODEL_PATH,
    is_trainable=False,
    offload_buffers=True,
    offload_folder=str(BASE_PATH / "offload"),
)
print(f"Loaded Reward Model from PEFT path: {REWARD_MODEL_PATH}")
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False

# set reward model padding token 
reward_model.config.pad_token_id = reward_tokenizer.pad_token_id


class FrugalRewardModel(torch.nn.Module):
    def __init__(self, policy_model=None, reward_funcs=None, reward_processors=None, tokenizer=None, num_consistency_samples: int = 4, **kwargs):
        super().__init__()
        self.t_acc = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_score = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_reliab = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.t_miscalib = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 

        
        self.policy_model = policy_model
        self.reward_funcs = reward_funcs or []          # list of callables or models
        self.reward_processors = reward_processors or [] # tokenizers/processors for each model
        self.tokenizer = tokenizer
        self.num_consistency_samples = num_consistency_samples

    def _placeholder_metric(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        return torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def acc(self, input_ids, attention_mask, ground_truth=None, text=None):
        if ground_truth is None or text is None:
            return self._placeholder_metric(input_ids)
        # ensure lists
        if isinstance(text, str): text = [text]
        if isinstance(ground_truth, str): ground_truth = [ground_truth]
        # compute embeddings and cosine similarity
        gt_emb = self.sentence_embedding_model.encode(ground_truth, convert_to_tensor=True)
        pred_emb = self.sentence_embedding_model.encode(text, convert_to_tensor=True)
        sim = F.cosine_similarity(gt_emb, pred_emb, dim=1)  # shape (batch_size,)
        return ((sim + 1) / 2).to(input_ids.device)   # scale to [0,1]
    
    def score(self, text, ground_truth, input_ids=None) -> torch.Tensor | None:
            """Returns tensor of shape (batch_size,) using reward_funcs/reward_processors."""
            if not self.reward_funcs:
                # fallback to placeholder
                return self._placeholder_metric(input_ids) if input_ids is not None else torch.tensor([0.0], device=self.device)

            if isinstance(text, str):
                text = [text]
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            batch_size = len(text)
            device = input_ids.device if input_ids is not None else self.device

        
            total_rewards = None
            for func, processor in zip(self.reward_funcs, self.reward_processors):
                # If it's a callable that takes (text, ground_truth) directly:
                if callable(func) and not isinstance(func, torch.nn.Module):
                    pass
                else:
                    # It's a reward model (torch.nn.Module)
                    if processor is not None:
                        # Tokenize the text (just the response)
                        inputs = processor(text, padding=True, return_tensors='pt', truncation=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    else:
                        inputs = {'input_ids': torch.tensor([0]).to(device)}  # fallback

                    func = func.to(device)
                    with torch.no_grad():
                        outputs = func(**inputs)
                        reward_i = outputs.logits.squeeze(-1) if hasattr(outputs, "logits") else outputs.squeeze(-1)
                        if reward_i.dim() == 0:
                            reward_i = reward_i.unsqueeze(0)
                        if reward_i.shape[0] != batch_size:
                            reward_i = reward_i.repeat(batch_size)

                    if total_rewards is None:
                        total_rewards = reward_i
                    else:
                        total_rewards += reward_i

            if total_rewards is not None:
                return total_rewards
            else:
                # fallback if no model gave a result
                return torch.zeros(batch_size, device=device, dtype=torch.float32)

    def _confidence_from_logprobs(self, mask: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor:
            """Per-sample confidence in [0,1], derived from mean token logprob."""
            mask = mask.float()
            token_counts = mask.sum(dim=1).clamp(min=1)
            mean_logprob = (logprobs * mask).sum(dim=1) / token_counts
            confidence = torch.exp(mean_logprob)
            has_tokens = mask.sum(dim=1) > 0
            confidence = torch.where(has_tokens, confidence, torch.full_like(confidence, 0.5))
            return confidence   

    def calibration_alignment(self, confidence: torch.Tensor, accuracy: torch.Tensor) -> torch.Tensor:
        """
        Per-sample alignment utility (batch_size,): negative perpendicular
        distance from (confidence, accuracy) to the y=x line. 0 = perfectly
        calibrated (confidence exactly matches accuracy); more negative = further
        off the line in either direction (over- or under-confident). Higher
        (closer to 0) is better, matching the 'higher = better' convention of
        other utilities.
        """
        diff = accuracy - confidence
        dist = torch.abs(diff) / math.sqrt(2.0)
        return -dist
        
        

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

    def miscalib(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, logprobs=None):
        """Miscalibration: Expected Calibration Error (ECE) based on logprobs vs accuracy.
        
        Args:
            text: The response text.
            ground_truth: The correct label.
            logprobs: Token-level log probabilities for confidence estimation.
        
        Returns:
            Miscalibration score (0 = well-calibrated, higher = worse).
        """
        if ground_truth is None or text is None:
            return self._placeholder_metric(input_ids)
        
        if isinstance(text, str):
            text = [text]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Compute accuracy
        accuracy = self.acc(input_ids, attention_mask, ground_truth=ground_truth, text=text)
        
        # Compute confidence from logprobs if available
        if logprobs is not None:
            confidence = self._confidence_from_logprobs(attention_mask, logprobs)
        else:
            # Fallback: assume high confidence
            confidence = torch.ones_like(accuracy)
        
        # Miscalibration = |confidence - accuracy|
        miscalib = torch.abs(confidence - accuracy)
        return miscalib

    def _positive_temp(self, temp: torch.Tensor) -> torch.Tensor:
        return F.softplus(temp) + 1e-6

    def get_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract token-level log probabilities from model forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        
        Returns:
            Token log probabilities, shape (batch_size, seq_len)
        """
        if self.policy_model is None:
            return torch.zeros_like(input_ids, dtype=torch.float32)
        
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get logits and compute log probabilities
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Gather log probabilities for the actual tokens
        input_ids_expanded = input_ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
        token_log_probs = log_probs.gather(-1, input_ids_expanded).squeeze(-1)  # (batch_size, seq_len)
        
        # Keep the graph: the preference loss must update the policy adapter.
        return token_log_probs

    def U_plus(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None) -> torch.Tensor:
        """Returns tensor of shape (batch_size, 2) with [score_component, accuracy_component]"""
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        
        # Handle string inputs
        if isinstance(text, str):
            text = [text]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Compute accuracy for each sample in batch (detached from graph)
        accuracy = self.acc(input_ids, attention_mask, ground_truth=ground_truth, text=text)
        accuracy_scaled = accuracy * self.t_acc
        
        # Compute reward model scores (detached from graph)
        reward_model_score = self.score(text, ground_truth, input_ids=input_ids)
        
        if reward_model_score is not None and isinstance(reward_model_score, torch.Tensor):
            # Scale with temperature and tanh to prevent explosion
            reward_scaled = torch.tanh(reward_model_score / 5.0)  # Bound to [-1, 1]
            reward_scaled = reward_scaled * self.t_score
            
            score_col = reward_scaled.unsqueeze(1)  # (batch_size, 1)
            acc_col = accuracy_scaled.unsqueeze(1)  # (batch_size, 1)
            return torch.cat([score_col, acc_col], dim=1)  # (batch_size, 2)
        
        # Fallback: return accuracy for both components
        acc_col = accuracy_scaled.unsqueeze(1)  # (batch_size, 1)
        return torch.cat([acc_col, acc_col], dim=1)  # (batch_size, 2)
    

    def U_minus(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None, logprobs=None) -> torch.Tensor:
        """Returns tensor of shape (batch_size, 2) with [miscalib_component, calib_align_component]"""
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        
        if isinstance(text, str):
            text = [text]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Compute miscalibration (higher = worse)
        miscalib = self.miscalib(
            input_ids,
            attention_mask,
            ground_truth=ground_truth,
            text=text,
            logprobs=logprobs,
        )
        miscalib_scaled = miscalib * self.t_miscalib
        miscalib_col = miscalib_scaled.reshape(-1, 1)

        if logprobs is not None:
            confidence = self._confidence_from_logprobs(attention_mask, logprobs)
            accuracy = self.acc(input_ids, attention_mask, ground_truth=ground_truth, text=text)
            calib_align = self.calibration_alignment(confidence, accuracy)
            calib_align_scaled = calib_align * self.t_reliab
            calib_col = calib_align_scaled.reshape(-1, 1)
            return torch.cat([miscalib_col, calib_col], dim=1)
        
        return torch.cat([miscalib_col, miscalib_col], dim=1)  # fallback, keep width=2


    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ground_truth=None, text=None, prompt=None, logprobs=None) -> torch.Tensor:
        """Compute the frugal reward as U_plus - U_minus."""
        u_plus = self.U_plus(input_ids, attention_mask, ground_truth=ground_truth, text=text, prompt=prompt)
        u_minus = self.U_minus(input_ids, attention_mask, ground_truth=ground_truth, text=text, prompt=prompt, logprobs=logprobs)
        reward = u_plus - u_minus
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
        chosen_logprobs=None,
        rejected_logprobs=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Bradley-Terry preference loss with frugal reward metrics.

        Returns:
            loss: -log(sigmoid(r_chosen - r_rejected))
            r_chosen: Rewards for chosen responses
            r_rejected: Rewards for rejected response
        """
        print(rejected_text)
        print(chosen_text)
        # Compute logprobs if not provided
        if chosen_logprobs is None:
            chosen_logprobs = self.get_logprobs(chosen_ids, chosen_mask)
        if rejected_logprobs is None:
            rejected_logprobs = self.get_logprobs(rejected_ids, rejected_mask)
        
        r_chosen = self.get_reward(
            chosen_ids, chosen_mask,
            ground_truth=ground_truth_chosen,
            text=chosen_text,
            prompt=prompt,
            logprobs=chosen_logprobs
        )
        r_rejected = self.get_reward(
            rejected_ids, rejected_mask,
            ground_truth=ground_truth_rejected,
            text=rejected_text,
            prompt=prompt,
            logprobs=rejected_logprobs
        )

        # Bradley-Terry loss (use mean of both components if 2D)
        r_chosen_scalar = r_chosen.mean(dim=1) if r_chosen.dim() > 1 else r_chosen
        r_rejected_scalar = r_rejected.mean(dim=1) if r_rejected.dim() > 1 else r_rejected

        print("\n--- BT DEBUG ---")
        print("chosen reward:")
        print(r_chosen.detach().cpu())

        print("rejected reward:")
        print(r_rejected.detach().cpu())

        print("chosen scalar:",
            r_chosen_scalar.detach().cpu())

        print("rejected scalar:",
            r_rejected_scalar.detach().cpu())

        print("reward difference:",
            (r_chosen_scalar - r_rejected_scalar).detach().cpu())

        print("temperatures:")
        print("t_acc     =", self._positive_temp(self.t_acc).item())
        print("t_score   =", self._positive_temp(self.t_score).item())
        print("t_reliab  =", self._positive_temp(self.t_reliab).item())
        print("t_miscalib=", self._positive_temp(self.t_miscalib).item())
        loss = -F.logsigmoid(r_chosen_scalar - r_rejected_scalar).mean()

        return loss, r_chosen, r_rejected

def train_preference_rm_trl(
    model_id: str = DEFAULT_MODEL_ID,
    dataset_path: Path | None = None,
    output_dir: str | None = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
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

    def normalize_text(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value if v is not None)
        value = str(value)
        if value == 'nan' or value.strip() == '':
            return ""
        return value

    data_list = []
    for _, row in pref_df.iterrows():
        chosen_text = normalize_text(row['chosen'])
        rejected_text = normalize_text(row['rejected'])
        pair = tokenize_pair(chosen_text, rejected_text)
        pair['doctor_gt'] = normalize_text(row['doctor_gt'])
        pair['instruction'] = normalize_text(row['instruction'])
        pair['chosen'] = chosen_text
        pair['rejected'] = rejected_text
        pair['chosen_text'] = chosen_text
        pair['rejected_text'] = rejected_text

        data_list.append(pair)

    # Create dataset. Keep both raw TRL names (`chosen`/`rejected`) and custom names
    # (`chosen_text`/`rejected_text`) so the trainer can access the text in
    # `compute_loss()` even after TRL tokenization/collation.
    train_dataset = Dataset.from_dict({
        'chosen_input_ids': [d['chosen_input_ids'] for d in data_list],
        'chosen_attention_mask': [d['chosen_attention_mask'] for d in data_list],
        'rejected_input_ids': [d['rejected_input_ids'] for d in data_list],
        'rejected_attention_mask': [d['rejected_attention_mask'] for d in data_list],
        'doctor_gt': [d['doctor_gt'] for d in data_list],
        'instruction': [d['instruction'] for d in data_list],
        'chosen': [d['chosen'] for d in data_list],
        'rejected': [d['rejected'] for d in data_list],
        'chosen_text': [d['chosen_text'] for d in data_list],
        'rejected_text': [d['rejected_text'] for d in data_list],
    })

    # Split train/eval
    eval_frac = 0.1
    train_eval_split = train_dataset.train_test_split(test_size=eval_frac, seed=42)
    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']

    class PreferenceTextCollator:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, features):
            def normalize_text(value):
                if value is None:
                    return ""
                if isinstance(value, (list, tuple)):
                    value = " ".join(str(v) for v in value if v is not None)
                value = str(value)
                if value == 'nan' or value.strip() == '':
                    return ""
                return value

            chosen_texts = []
            rejected_texts = []
            ground_truths = []
            prompts = []

            for f in features:
                # Try to get raw text from various possible keys
                chosen = f.get('chosen_text', f.get('chosen'))
                rejected = f.get('rejected_text', f.get('rejected'))
                gt = f.get('doctor_gt', f.get('ground_truth'))
                prompt = f.get('instruction', f.get('prompt'))

                # If text is missing, decode token IDs if present
                if chosen is None and 'chosen_input_ids' in f:
                    chosen = self.tokenizer.decode(f['chosen_input_ids'], skip_special_tokens=True)
                if rejected is None and 'rejected_input_ids' in f:
                    rejected = self.tokenizer.decode(f['rejected_input_ids'], skip_special_tokens=True)
                # Also support 'chosen_ids' if that's the key
                if chosen is None and 'chosen_ids' in f:
                    chosen = self.tokenizer.decode(f['chosen_ids'], skip_special_tokens=True)
                if rejected is None and 'rejected_ids' in f:
                    rejected = self.tokenizer.decode(f['rejected_ids'], skip_special_tokens=True)

                chosen_texts.append(normalize_text(chosen))
                rejected_texts.append(normalize_text(rejected))
                ground_truths.append(normalize_text(gt))
                prompts.append(normalize_text(prompt))

            # Tokenize chosen and rejected (for the reward model's input)
            chosen_enc = self.tokenizer(
                chosen_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            rejected_enc = self.tokenizer(
                rejected_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            batch = {
                'chosen_input_ids': chosen_enc['input_ids'],
                'chosen_attention_mask': chosen_enc['attention_mask'],
                'rejected_input_ids': rejected_enc['input_ids'],
                'rejected_attention_mask': rejected_enc['attention_mask'],
                'chosen_text': chosen_texts,
                'rejected_text': rejected_texts,
                'doctor_gt': ground_truths,
                'instruction': prompts,
            }
            return batch

    # Initialize your custom preference reward model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preference_model = FrugalRewardModel(
        policy_model=policy_model,
        reward_funcs=[reward_model],
        reward_processors=[reward_tokenizer],
        tokenizer=tokenizer,
        num_consistency_samples=4
    ).to(device)    

    policy_trainable_params = [
        parameter for parameter in preference_model.policy_model.parameters()
        if parameter.requires_grad
    ] if preference_model.policy_model is not None else []
    if not policy_trainable_params:
        raise RuntimeError(
            "The policy model has no trainable parameters. Load the PEFT adapter "
            "with is_trainable=True before starting reward-model training."
        )
    print(
        "Trainable policy parameters:",
        sum(parameter.numel() for parameter in policy_trainable_params),
    )
    
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
        data_collator=PreferenceTextCollator(tokenizer, DEFAULT_MAX_LENGTH),
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)

    # TRL saves the reward-training wrapper. Save the actual policy adapter
    # separately so it can be loaded with PeftModel.from_pretrained().
    policy_adapter_output_dir = Path(output_dir) / "policy_adapter"
    policy_adapter_output_dir.mkdir(parents=True, exist_ok=True)
    preference_model.policy_model.save_pretrained(policy_adapter_output_dir)
    tokenizer.save_pretrained(policy_adapter_output_dir)
    print(f"Saved trained policy adapter to: {policy_adapter_output_dir}")
    
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
    policy_model.gradient_checkpointing_enable()
    print(f"Loaded Policy Model from PEFT path: {SFT_POLICY_PATH}")



    train_preference_rm_trl(
        output_dir=str(Path(__file__).parent.parent / 'RLFF' / 'Reward_Models' / 'reward_model_BT_frugal'),
        wandb_run_name='reward-model-bt-frugal',
        policy_model=policy_model,
        judge_model=reward_model,
        )


