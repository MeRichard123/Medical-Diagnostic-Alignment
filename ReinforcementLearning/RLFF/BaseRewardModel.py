from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, PretrainedConfig
import torch.nn as nn
import wandb, os, torch
from trl import RewardTrainer

class BaseRewardModel(nn.Module):
    """Base class for reward models with peft.
    Adapted from 
    https://github.com/natolambert/rlhf-book/blob/main/code/reward_models/base.py

    Uses a frozen or partially-frozen backbone with a trainable reward head.

    Subclasses should implement:
    - forward(): Define the forward pass and loss computation
    - Optionally override _build_head() for custom reward heads
    """

    def __init__(
        self,
        model_id: str,
        head_dim: int = 1,
        freeze_backbone: bool = False,
        use_4bit: bool = True,
        use_lora: bool = True,
    ):
        super().__init__()

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    
        # BF16 loading - simple for small models
        device_map = {"": 0} if torch.cuda.is_available() else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            dtype="bfloat16",  # Use string to avoid deprecation warning
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            num_labels=1, 
        )
        self.model.config.use_cache = False

        # Quantized models must have trainable adapters for Trainer/TRL fine-tuning.
        if use_4bit and use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            except Exception as exc:
                raise ImportError(
                    "PEFT is required for training a 4-bit quantized reward model. "
                    "Install peft or set use_4bit=False."
                ) from exc

            self.model = prepare_model_for_kbit_training(self.model)
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)

        # Optionally freeze the backbone (for head-only training)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Build head with same dtype as model
        self.head = self._build_head(self.model.config.hidden_size, head_dim)
        self.head = self.head.to(torch.bfloat16)

    def _build_head(self, hidden_size: int, output_dim: int) -> nn.Module:
        """Build the reward head. Override for custom architectures."""
        return nn.Linear(hidden_size, output_dim, bias=output_dim > 1)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the last hidden states from the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def init_wandb(
    default_run_name: str,
    config: dict,
    use_wandb: bool = True,
) -> bool:
    """Initialize wandb with environment variable support.

    Args:
        default_run_name: Default name if WANDB_RUN_NAME not set
        config: Training config to log
        use_wandb: Whether to enable wandb

    Returns:
        True if wandb is enabled, False otherwise
    """
    wandb_project = "rl-tuning-medical-model-alignment"
    wandb_mode = os.getenv("WANDB_MODE", "online")

    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=os.environ.get("WANDB_RUN_NAME", default_run_name),
            config=config,
            mode=wandb_mode,
        )
        print(f"Wandb initialized in {wandb_mode} mode. Project: {wandb_project}, Run Name: {wandb.run.name}")
        return True
    else:
        wandb.init(mode="disabled")
        print("Wandb disabled. Set WANDB_MODE=online and ensure WANDB_PROJECT is configured to enable logging.")
        return False


def log_metrics(metrics: dict, step: int | None = None):
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)


def finish_wandb():
    """Finish wandb run."""
    wandb.finish()


class PreferenceRewardModelTRLAdapter(torch.nn.Module):
    """Adapter to make PreferenceRewardModel compatible with TRL's RewardTrainer.
    
    Wraps the custom PreferenceRewardModel so you can:
    - Keep the custom Bradley-Terry loss fully editable in PreferenceRewardModel
    - Use TRL's RewardTrainer infrastructure (distributed training, gradient accumulation, etc.)
    - Experiment with different loss functions by modifying PreferenceRewardModel.forward()
    """
    def __init__(self, preference_model: BaseRewardModel):
        super().__init__()
        self.preference_model = preference_model
        # Expose config for TRL compatibility
        if hasattr(preference_model, 'model') and getattr(preference_model.model, 'config', None) is not None:
            self.config = preference_model.model.config
        else:
            # TRL RewardTrainer expects model.config.num_labels == 1.
            self.config = PretrainedConfig(num_labels=1)
        self.num_labels = 1
        self.device = preference_model.device

    def __getattr__(self, name):
        """
        Delegate missing attributes to parent models
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, '_modules')
            preference_model = modules.get('preference_model')
            if preference_model is None:
                raise
            base_model = getattr(preference_model, 'model', None)
            if base_model is not None and hasattr(base_model, name):
                return getattr(base_model, name)
            if hasattr(preference_model, name):
                return getattr(preference_model, name)
            raise

    def add_model_tags(self, tags):
        """
        Add tags to the underlying model.
        """
        base_model = getattr(self.preference_model, 'model', None)
        if base_model is not None and hasattr(base_model, 'add_model_tags'):
            base_model.add_model_tags(tags)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing when supported by the wrapped backend model."""
        base_model = getattr(self.preference_model, 'model', None)
        if base_model is not None and hasattr(base_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                return base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            return base_model.gradient_checkpointing_enable()
        return None

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing when supported by the wrapped backend model."""
        base_model = getattr(self.preference_model, 'model', None)
        if base_model is not None and hasattr(base_model, 'gradient_checkpointing_disable'):
            return base_model.gradient_checkpointing_disable()
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """TRL-compatible forward: compute reward score for input sequence.
        
        This is called by RewardTrainer during inference.
        We extract the last token's hidden state and pass through the reward head.
        """
        # Try to extract optional prompt/answer/ground truth from kwargs and pass them
        prompt = kwargs.get('prompt') or kwargs.get('instruction') or kwargs.get('question')
        answer = kwargs.get('answer') or kwargs.get('text') or kwargs.get('chosen_text')
        doctor_gt = kwargs.get('doctor_gt') or kwargs.get('ground_truth')

        reward = self.preference_model.get_reward(
            input_ids,
            attention_mask,
            ground_truth=doctor_gt,
            text=answer,
            prompt=prompt,
        )
        return reward.unsqueeze(-1)  # TRL expects (batch_size, 1)

    def forward_with_pairs(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom Bradley-Terry loss computed on chosen/rejected pairs.
        
        This calls the underlying PreferenceRewardModel's forward
        """
        return self.preference_model.forward(chosen_ids, chosen_mask, rejected_ids, rejected_mask)
    

class CustomBTRewardTrainer(RewardTrainer):
    """Custom RewardTrainer that uses PreferenceRewardModel's Bradley-Terry loss.
    
    Overrides compute_loss to call the custom forward() method of PreferenceRewardModel,
    allowing you to edit the BT loss definition directly in PreferenceRewardModel.forward().
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss using PreferenceRewardModel's custom Bradley-Terry forward."""
        # Extract the underlying PreferenceRewardModel if wrapped
        pref_model = model.preference_model if hasattr(model, 'preference_model') else model
        # Support both legacy column names (`chosen_input_ids`) and the newer
        # processed names (`chosen_ids`) that TRL uses. Provide a helpful
        # error if neither is present.
        if 'chosen_input_ids' in inputs and 'rejected_input_ids' in inputs:
            chosen_ids = inputs['chosen_input_ids']
            chosen_mask = inputs.get('chosen_attention_mask')
            rejected_ids = inputs['rejected_input_ids']
            rejected_mask = inputs.get('rejected_attention_mask')
        elif 'chosen_ids' in inputs and 'rejected_ids' in inputs:
            chosen_ids = inputs['chosen_ids']
            chosen_mask = inputs.get('chosen_attention_mask') or inputs.get('chosen_mask')
            rejected_ids = inputs['rejected_ids']
            rejected_mask = inputs.get('rejected_attention_mask') or inputs.get('rejected_mask')
        elif 'input_ids' in inputs and 'attention_mask' in inputs:
            # Flat batch with alternating chosen/rejected rows: [c0, r0, c1, r1, ...]
            flat_ids = inputs['input_ids']
            flat_mask = inputs['attention_mask']
            if flat_ids.shape[0] % 2 != 0:
                raise ValueError(f"Flat batch has odd size {flat_ids.shape[0]}; expected even number for paired preference data")
            chosen_ids = flat_ids[0::2]
            rejected_ids = flat_ids[1::2]
            chosen_mask = flat_mask[0::2]
            rejected_mask = flat_mask[1::2]
        else:
            raise KeyError(f"Preference trainer expected 'chosen_input_ids'/'rejected_input_ids', 'chosen_ids'/'rejected_ids', or flat 'input_ids'/'attention_mask' in inputs; got keys: {list(inputs.keys())}")

        # Extract optional ground truth and text fields for frugal reward metrics
        # Support both 'ground_truth' and 'doctor_gt' dataset keys
        ground_truth = inputs.get('ground_truth', None) or inputs.get('doctor_gt', None)
        prompt = inputs.get('prompt', None) or inputs.get('instruction', None)
        chosen_text = inputs.get('chosen_text', None) or inputs.get('chosen_texts', None) or inputs.get('chosen', None)
        rejected_text = inputs.get('rejected_text', None) or inputs.get('rejected_texts', None) or inputs.get('rejected', None)

        # Call the custom forward which returns (loss, r_chosen, r_rejected)
        # Try to pass all parameters; if the model doesn't accept them, fall back
        try:
            loss, r_chosen, r_rejected = pref_model.forward(
                chosen_ids,
                chosen_mask,
                rejected_ids,
                rejected_mask,
                ground_truth_chosen=ground_truth,
                ground_truth_rejected=ground_truth,
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                prompt=prompt,
            )
        except TypeError:
            # Fallback for models that don't accept these parameters
            loss, r_chosen, r_rejected = pref_model.forward(
                chosen_ids,
                chosen_mask,
                rejected_ids,
                rejected_mask,
            )
        
        outputs = {
            'loss': loss,
            'r_chosen': r_chosen,
            'r_rejected': r_rejected,
        }
        
        return (loss, outputs) if return_outputs else loss