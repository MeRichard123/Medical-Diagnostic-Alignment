from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class QuantizationConfig:
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    load_in_4bit: bool = True
    load_in_8bit: bool = False

@dataclass
class ModelConfigSection:
    model_name: str = "Qwen/Qwen2-0.5B"
    ref_model_name: Optional[str] = "Qwen/Qwen2-0.5B"
    tokenizer_name: str = "Qwen/Qwen2-0.5B"
    peft_adaptor_path: Optional[str|Path] = None
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    quantization: Optional[QuantizationConfig] = None 

@dataclass
class WandBConfigSection:
    report_to_wandb: bool = False
    project: str = "ppo-grpo-gsm8k"
    name: Optional[str] = None


@dataclass
class DatasetConfigSection:
    dataset_path: str = "./data/processed_iuxray_mcqa_dataset.csv"
    config: Optional[str] = "main"
    split: str = "train"
    max_prompt_length: int = 512
    max_gen_length: int = 56
    num_samples: Optional[int] = None
    num_workers: int = 4

@dataclass
class TrainingConfigSection:
    total_ppo_steps: Optional[int] = None 
    seed: int = 42             
    log_interval: int = 1      
    save_interval: int = 10    
    output_dir: str|Path = "outputs/ppo_gsm8k_model" 
    output_dir_grpo: str|Path = "outputs/grpo_gsm8k_model"
    device: str = "cuda"     
    gradient_checkpointing: bool = True 

@dataclass
class GenerationConfigSection:
    max_new_tokens: int = 56
    min_new_tokens: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


@dataclass
class BaseConfig:
    model: ModelConfigSection = field(default_factory=ModelConfigSection)
    wandb: WandBConfigSection = field(default_factory=WandBConfigSection)
    dataset: DatasetConfigSection = field(default_factory=DatasetConfigSection)
    training: TrainingConfigSection = field(default_factory=TrainingConfigSection)
    generation: GenerationConfigSection = field(default_factory=GenerationConfigSection)


@dataclass
class PPOConfig(BaseConfig):
    learning_rate: float = 2.0e-6 
    epochs: int = 2 
    batch_size: int = 8 
    mini_batch_size: int = 2 
    grad_accum_steps: int = 8  
    kl_coeff: float = 0.05    
    clip_eps: float = 0.2  
    value_clip_eps: float = 0.2 
    vf_coeff: float = 0.1    
    entropy_coeff: float = 0.01 
    gamma: float = 0.99       
    lam: float = 0.95       
    use_8bit_adam: bool = True
    max_grad_norm: float = 1.0 
    rollout_samples: int = 20
    scheduler: str = "cosine_with_min_lr" 
    warmup_steps: int = 5    
    min_lr: float = 1.0e-7   


@dataclass
class GRPOConfig(PPOConfig):
    group_size: int = 4

