from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from .BaseRewardModel import (
        BaseRewardModel, 
        finish_wandb, init_wandb, 
        PreferenceRewardModelTRLAdapter,
        CustomBTRewardTrainer
    )
from pathlib import Path
from datasets import Dataset
from trl import RewardConfig
from .custom_fns import logtanh


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

class PreferenceRewardModel(BaseRewardModel):
    """Preference-based Reward Model with PEFT

    Architecture:
    - Base LLM (e.g., Qwen3) loaded in bfloat16
    - Linear head mapping last hidden state to scalar reward

    The model outputs a single scalar reward for each sequence.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        super().__init__(model_id, head_dim=1, **kwargs)

    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute scalar reward for a sequence.

        Returns the reward from the last non-padding token position.
        """
        hidden = self.get_hidden_states(input_ids, attention_mask)

        # Get last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lengths]

        # Ensure dtype matches the head (important when using quantized models with bfloat16 hidden states)
        last_hidden = last_hidden.to(self.head.weight.dtype)
        reward = self.head(last_hidden).squeeze(-1)
        return reward

    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Bradley-Terry preference loss.

        Returns:
            loss: -log(sigmoid(r_chosen - r_rejected))
            r_chosen: Rewards for chosen responses
            r_rejected: Rewards for rejected responses
        """
        r_chosen = self.get_reward(chosen_ids, chosen_mask)
        r_rejected = self.get_reward(rejected_ids, rejected_mask)

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
):
    """Train PreferenceRewardModel using TRL's RewardTrainer with your custom Bradley-Terry loss.

    Uses CustomBTRewardTrainer which calls your PreferenceRewardModel.forward() directly.
    The model's forward() method is fully editable — modify it to experiment with the loss.
    
    Expects CSV at `dataset_path` to contain `chosen` and `rejected` columns.
    """
    base_path = Path(__file__).parent.parent
    if dataset_path is None:
        dataset_path = base_path / 'intermediate' / 'preference_data_normalized.csv'
    if output_dir is None:
        output_dir = str(base_path / 'RLFF' / 'Reward_Models' / 'reward_model_manualBT')

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and clean preference CSV
    import pandas as pd

    pref_df = pd.read_csv(dataset_path)
    if 'chosen' not in pref_df.columns or 'rejected' not in pref_df.columns:
        raise ValueError('preference CSV must contain chosen and rejected columns')
    pref_df = pref_df[['chosen', 'rejected']].dropna()
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
        data_list.append(pair)

    # Create dataset
    train_dataset = Dataset.from_dict({
        'chosen_input_ids': [d['chosen_input_ids'] for d in data_list],
        'chosen_attention_mask': [d['chosen_attention_mask'] for d in data_list],
        'rejected_input_ids': [d['rejected_input_ids'] for d in data_list],
        'rejected_attention_mask': [d['rejected_attention_mask'] for d in data_list],
    })

    # Split train/eval
    eval_frac = 0.1
    train_eval_split = train_dataset.train_test_split(test_size=eval_frac, seed=42)
    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']

    # Initialize your custom preference reward model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preference_model = PreferenceRewardModel(model_id=model_id).to(device)
    
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

    backend_model = None
    if hasattr(adapter_model.preference_model, 'model'):
        backend_model = adapter_model.preference_model.model
    else:
        backend_model = adapter_model

    # set the preference model as an attribute to the backend model so the trainer can access it
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


if __name__ == "__main__":
    train_preference_rm_trl(
        output_dir=str(Path(__file__).parent.parent / 'RLFF' / 'Reward_Models' / 'reward_model_BT_minustanh'),
        wandb_run_name='reward-model-bt-minustanh',
        )


