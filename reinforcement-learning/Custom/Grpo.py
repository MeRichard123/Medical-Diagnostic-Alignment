from .rlUtils import masked_mean, pad_and_collate_tensors
from .data_utils import load_and_process_dataset, save_model, create_generation_config
from .Config import GRPOConfig
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataclasses import asdict
from peft import PeftModel
from ..RLFF.RLFF import llm_judge, extract_judge_score
try:
    import bitsandbytes.optim as bnb_optim
    bnb_available = True
except ImportError:
    bnb_available = False

from transformers import (
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM, # Use standard LM model
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from peft import PeftModel
import numpy as np
import random
import math
from tqdm.auto import tqdm
import os
from typing import Dict, Any, Tuple, List, Callable, TypeAlias
import time # For timing

RewardFunction: TypeAlias = Callable[[str, str], float] 
RewardModel: TypeAlias = PreTrainedModel | PeftModel
RewardFunc: TypeAlias = RewardFunction | RewardModel

def is_reward_function(func: RewardFunc) -> bool:
    return callable(func) and not isinstance(func, PreTrainedModel) and not isinstance(func, PeftModel)

def is_reward_model(func: RewardFunc) -> bool:
    return isinstance(func, PreTrainedModel) or isinstance(func, PeftModel)

class GRPOTrainer:
    def __init__(self, 
                 reward_funcs: RewardFunc | List[RewardFunc], 
                 reward_processors: List[PreTrainedTokenizerBase], 
                 args: GRPOConfig, 
                ):
        self.reward_funcs = reward_funcs
        self.reward_processors = reward_processors
        self.args = args

        # 1. Initial Setup: Device, Output Directory, Config Saving
        self.device, self.output_dir = self.__training_setup()
        self.model, self.ref_model,self.tokenizer = self.__load_models_and_tokenizer(self.device)

        if not isinstance(self.reward_funcs, list):
            self.reward_funcs: List[RewardFunction | RewardModel] = [self.reward_funcs]
        
        if self.reward_processors is None:
            self.reward_processors = [None] * len(self.reward_funcs)
        elif not isinstance(self.reward_processors, list):
            self.reward_processors = [self.reward_processors]
        if len(self.reward_processors) != len(self.reward_funcs):
            raise ValueError("Length of reward_processors must match length of reward_funcs")
            
        for i, (reward_processing_class, reward_func) in enumerate(zip(self.reward_processors, self.reward_funcs, strict=True)):
            if is_reward_model(reward_func):
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                    reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                    self.reward_processors[i] = reward_processing_class

        if self.args.kl_coeff == 0.0:
            print("Warning: KL coefficient is set to 0.0, GRPO will not apply KL penalties and will behave more like standard PPO.")
            self.ref_model = None  # No reference model needed if KL penalty is not used

        if self.args.training.total_ppo_steps == None:
            self.args.training.total_ppo_steps = self.args.grad_accum_steps


    def __compute_grpo_advantages(
            self,
            rewards: torch.Tensor,
            kl_penalties: torch.Tensor,
            response_mask: torch.Tensor,
            group_size: int,
    ) -> torch.Tensor:
        """
        Compute GRPO advantages by combining rewards and KL penalties, then applying GAE.

        Args:
            rewards: Tensor of shape (batch_size, seq_len) containing token-level rewards.
            kl_penalties: Tensor of shape (batch_size, seq_len) containing token-level KL penalties.
            response_mask: Tensor of shape (batch_size, seq_len) indicating valid response tokens.
            group_size: Number of samples in each group for GRPO.
        Returns:
            advantages: Tensor of shape (batch_size, seq_len) containing the computed advantages.
        """
        with torch.no_grad():
            num_samples = rewards.shape[0]
            num_prompts = num_samples // group_size
            if num_samples % group_size != 0:
                print(f"Warning: Number of samples ({num_samples}) must be divisible by group_size ({group_size}).")
                num_prompts = num_samples // group_size 

            # reshape to (num_prompts, group_size)
            if rewards.dim() == 1:
                rewards = rewards.view(num_prompts, group_size)
        
            # compute KL 
            mean_kl_per_seq = masked_mean(kl_penalties, response_mask, dim=1)  # (num_samples,)
            mean_kl_per_seq = mean_kl_per_seq.view(num_prompts, group_size)  # (num_prompts, group_size)
            adjusted_rewards = rewards - self.args.kl_coeff * mean_kl_per_seq  # (num_prompts, group_size)

            # compute mean and std for each prompt group
            group_mean =  adjusted_rewards.mean(dim=1, keepdim=True)  # (num_prompts, 1)
            # add extra variance using unbiased=False to prevent very small std when group_size is small
            group_std = adjusted_rewards.std(dim=1, keepdim=True, unbiased=False) # (num_prompts, 1) 

            # normalize rewards within each prompt groups
            advantages = (adjusted_rewards - group_mean) / (group_std + 1e-8)  # (num_prompts, group_size)

            # reshape back to (num_prompts * group_size, 1)
            advantages = advantages.view(num_prompts * group_size, 1)  # (num_samples, 1)
            response_len = response_mask.shape[1]
            advantages = advantages.expand(-1, response_len)  # (num_samples, seq_len)

            # Mask out non-response tokens
            advantages = advantages * response_mask.float()  # Mask out non-response tokens
        return advantages


    def __compute_grpo_policy_loss(
            self,
            log_probs_new: torch.Tensor,
            log_probs_old: torch.Tensor,
            advantages: torch.Tensor,
            response_mask: torch.Tensor,
            clip_epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the GRPO policy loss using clipped surrogate objective.

        Args:
            log_probs_new: Tensor of shape (batch_size, seq_len) containing log probabilities from the new policy.
            log_probs_old: Tensor of shape (batch_size, seq_len) containing log probabilities from the old policy.
            log_probs_ref: Tensor of shape (batch_size, seq_len) containing log probabilities from the reference policy.
            advantages: Tensor of shape (batch_size, seq_len) containing computed advantages.
            response_mask: Tensor of shape (batch_size, seq_len) indicating valid response tokens.
            clip_epsilon: Clipping parameter for PPO objective.
        Returns:
            policy_loss: Scalar tensor representing the GRPO policy loss.
            mean_kl: Scalar tensor representing the mean KL divergence across the batch.
        """
        with torch.no_grad():
            mask = response_mask.bool()
            if advantages.shape != log_probs_old.shape:
                raise ValueError(f"Advantages shape {advantages.shape} does not match log_probs_old shape {log_probs_old.shape}")
        
        log_ratio = (log_probs_new - log_probs_old).clamp(-10, 10)  # Prevent extreme ratios
        ratio = torch.exp(log_ratio)  # (batch_size, seq_len)
        surrogate1 = ratio * advantages  # (batch_size, seq_len)
        
        # (batch_size, seq_len)
        surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        # Average over valid response tokens 
        policy_loss = -masked_mean(torch.min(surrogate1, surrogate2), mask)

        with torch.no_grad():
            clip_frac = masked_mean(torch.gt(torch.abs(ratio - 1.0), clip_epsilon).float(), mask)
            approx_kl = masked_mean(log_probs_old - log_probs_new, mask)
        
        return policy_loss, clip_frac, approx_kl


    def __compute_grpo_entropy_loss(
            self,
            logits_new: torch.Tensor,
            response_mask: torch.Tensor,
            log_probs_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the GRPO entropy loss to encourage exploration.

        Args:
            logits_new: Tensor of shape (batch_size, seq_len, vocab_size) containing logits from the new policy.
            response_mask: Tensor of shape (batch_size, seq_len) indicating valid response tokens.
            log_probs_new: Tensor of shape (batch_size, seq_len) containing log probabilities from the new policy.
        Returns:
            entropy_loss: Scalar tensor representing the GRPO entropy loss.
        """
        """Computes entropy loss. Identical logic to PPO."""
        # mask = response_mask.bool()
        # dist = torch.distributions.Categorical(logits=logits_new.float())
        # entropy = dist.entropy()
        # entropy_loss = -masked_mean(entropy, mask) # Maximize entropy
        # Low memory version without creating the full distribution object
        # this estimates the entropy using the log probabilities of the taken actions, which is more memory efficient than computing the full distribution entropy
        entropy_estimate = -log_probs_new.detach()
        entropy_loss = -masked_mean(entropy_estimate, response_mask.bool())
        return entropy_loss

    # ==============================================================================
    # == 2. Actor Model Definition (No Value Head)
    # ==============================================================================

    # We can just use AutoModelForCausalLM directly, or wrap it if needed later.
    # For simplicity, we'll load AutoModelForCausalLM directly in setup.

    # ==============================================================================
    # == 4. Rollout Phase Logic (Modified for Group Generation)
    # ==============================================================================

    def __generate_responses_grouped(
            self,
            prompt_ids: torch.Tensor,
            prompt_mask: torch.Tensor,
            generation_config: GenerationConfig,
            group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        generation_config.do_sample = True
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            # Input needs repeating for generation: (B, L) -> (B*G, L)
            expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
            expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

            generated_output = self.model.generate(
                input_ids=expanded_prompt_ids,
                attention_mask=expanded_prompt_mask,
                generation_config=generation_config,
            )
            # Output shape: (batch_size * group_size, full_len)

            # Extract only generated tokens
            prompt_len = prompt_ids.shape[1]
            response_ids = generated_output[:, prompt_len:] # Shape: (B*G, resp_len)

            # Create response mask
            response_mask = (response_ids != self.tokenizer.pad_token_id).long()

        return response_ids, response_mask # Return grouped results

    def __calculate_rollout_stats(
            self,
            prompt_ids: torch.Tensor,
            prompt_mask: torch.Tensor,
            response_ids: torch.Tensor,
            response_mask: torch.Tensor,
            group_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate log probabilities for the generated responses.

        Args:
            ref_model: The reference model for KL penalty calculation.
            prompt_ids: Tensor of shape (batch_size, prompt_len) containing token IDs of prompts.
            prompt_mask: Tensor of shape (batch_size, prompt_len) indicating valid tokens in prompts.
            response_ids: Tensor of shape (batch_size * group_size, resp_len) containing token IDs of generated responses.
            response_mask: Tensor of shape (batch_size * group_size, resp_len) indicating valid tokens in responses.
            group_size: Number of samples in each group for GRPO.
        Returns:
            A dictionary containing:
                - log_probs_new: Log probabilities from the actor model for the generated responses.
                - log_probs_ref: Log probabilities from the reference model for the generated responses.
        """
        self.model.eval()
        if self.ref_model:
            self.ref_model.eval()

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        resp_len = response_ids.shape[1]

        # Expand prompt to match grouped responses (B, L) -> (B*G, L)
        expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

        # combine prompt and response for model input
        full_ids = torch.cat([expanded_prompt_ids, response_ids], dim=1)  # (B*G, L+R)
        full_mask = torch.cat([expanded_prompt_mask, response_mask], dim=1)
        full_len = full_ids.shape[1]

        with torch.no_grad():
            actor_logits = self.model(input_ids=full_ids, attention_mask=full_mask).logits
            # We only care about log probs for the response part
            start_idx = prompt_len - 1
            end_idx = full_len - 1

            if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
                logprobs = torch.empty(
                    (batch_size * group_size, 0), dtype=torch.float, 
                    device=prompt_ids.device
                )
                ref_logprobs = torch.empty(
                    (batch_size * group_size, 0), dtype=torch.float,
                    device=prompt_ids.device
                )

            else:
                logits_resp = actor_logits[:, start_idx:end_idx, :]
                target_ids = response_ids # (B*G, resp_len)

                # match shapes
                current_resp_len = logits_resp.shape[1]
                if current_resp_len != target_ids.shape[1]:
                    min_len = min(current_resp_len, target_ids.shape[1])
                    logits_resp = logits_resp[:, :min_len, :]
                    target_ids = target_ids[:, :min_len]
                    response_mask_adjusted = response_mask[:, :min_len]
                else:
                    response_mask_adjusted = response_mask

                # Compute log probabilities
                selected_logits = torch.gather(
                    logits_resp, dim=2, index=target_ids.unsqueeze(-1)
                ).squeeze(-1)  # (B*G, resp_len)
                log_denom = torch.logsumexp(logits_resp, dim=-1)  # (B*G, resp_len)

                logprobs = selected_logits - log_denom  # (B*G, resp_len)

                del actor_logits
                del logits_resp
                torch.cuda.empty_cache()

                # print(torch.cuda.memory_summary())

                if self.ref_model:
                    ref_logits = self.ref_model(input_ids=full_ids, attention_mask=full_mask).logits
                else:
                    ref_logits = None

                ref_logits_resp = None

                if start_idx < 0 or end_idx <= start_idx or resp_len == 0:
                    if self.ref_model:
                        ref_logits_resp = ref_logits[:, start_idx:end_idx, :]
                    else:
                        ref_logits_resp = None

                if current_resp_len != target_ids.shape[1]:
                    if ref_logits_resp is not None:
                        ref_logits_resp = ref_logits_resp[:, :min_len, :]


                if ref_logits_resp is not None:
                    ref_selected_logits = torch.gather(
                        ref_logits_resp, dim=2, index=target_ids.unsqueeze(-1)
                    ).squeeze(-1)  # (B*G, resp_len)
                    ref_log_denom = torch.logsumexp(ref_logits_resp, dim=-1)  # (B*G, resp_len)

                    ref_logprobs = ref_selected_logits - ref_log_denom  # (B*G, resp_len)
                else:
                    ref_logprobs = None
                
                # Mask out non-response tokens
                logprobs = logprobs * response_mask_adjusted 
                if ref_logprobs is not None:    
                    ref_logprobs = ref_logprobs * response_mask_adjusted

                del ref_logits
                del ref_logits_resp
                torch.cuda.empty_cache()
        
        return {
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs
        }
    
    def __compute_reward_tensor(self,reward_funcs: List[RewardFunction|RewardModel], full_decoded_texts: List[str], expanded_ground_truths: List[str]) -> torch.Tensor:
        if len(reward_funcs) == 0:
            print("Warning: No reward functions provided, returning zero rewards.")
            return torch.empty((len(full_decoded_texts),), dtype=torch.float32)
        if is_reward_function(reward_funcs[0]):
            total_rewards = torch.tensor([
                np.mean([reward_fnc(full_text, gt) for reward_fnc in self.reward_funcs])
                for full_text, gt in zip(full_decoded_texts, expanded_ground_truths)
            ], dtype=torch.float32, device='cpu') # shape (B*G,)

        elif is_reward_model(reward_funcs[0]):
            total_rewards = None
            for _, (reward_func, reward_processor) in enumerate(zip(self.reward_funcs, self.reward_processors, strict=True)):
                reward_func: RewardModel = reward_func
                reward_processor: PreTrainedTokenizerBase = reward_processor

                if reward_processor is not None:
                    inputs = reward_processor(
                        full_decoded_texts, padding=True, truncation=True, return_tensors="pt"
                    ).to(reward_func.device)
                else:
                    inputs = reward_func.tokenizer(
                        full_decoded_texts, padding=True, truncation=True, return_tensors="pt"
                    ).to(reward_func.device)
                with torch.no_grad():
                    outputs = reward_func(**inputs)
                    # Assuming reward is derived from the last token's logits
                    reward_i = outputs.logits.squeeze(-1)  # shape (B*G,)
                reward_i = reward_i.cpu()
                if total_rewards is None:
                    total_rewards = reward_i
                else:
                    total_rewards += reward_i
        else:
            raise ValueError("reward_funcs must be a list of either RewardFunction or RewardModel")
        reward_func.cpu()
        torch.cuda.empty_cache()
        
        return total_rewards

    def __compute_rollout(
            self,
            prompt_dataloader: DataLoader,
            generation_config: GenerationConfig,
            group_size: int,
            device: torch.device,
    ) -> Dict[str, Any]:
        """
        Perform the rollout phase for a batch of prompts, generating responses and calculating log probabilities.

        Args:
            actor_model: The current policy model.
            ref_model: The reference model for KL penalty calculation.
            prompt_dataloader: DataLoader for iterating over prompts.
            generation_config: Configuration for text generation.
            group_size: Number of samples in each group for GRPO.
            device: The device to perform computations on.
        Returns:
            A dictionary containing:
                - prompt_ids: Tensor of shape (batch_size, prompt_len) containing token IDs of prompts.
                - prompt_mask: Tensor of shape (batch_size, prompt_len) indicating valid tokens in prompts.
                - response_ids: Tensor of shape (batch_size * group_size, resp_len) containing token IDs of generated responses.
                - response_mask: Tensor of shape (batch_size * group_size, resp_len) indicating valid tokens in responses.
                - log_probs_new: Log probabilities from the actor model for the generated responses.
                - log_probs_ref: Log probabilities from the reference model for the generated responses.
        """
        rollout_start_time = time.time()
        buffer_lists = {
            'prompt_input_ids': [],
            'prompt_attention_mask': [],
            'response_input_ids': [],
            'response_attention_mask': [],
            'logprobs': [],
            'ref_logprobs': [],
            'rewards': [],
            'full_texts': [],
            'solutions': []
        }
        timing_data = {'gen_time': 0.0, 'stats_time': 0.0, 'cpu_time': 0.0}

        progress_bar = tqdm(prompt_dataloader, desc="GRPO Rollout", leave=False)

        for batch in progress_bar:
            if batch is None: continue
            prompt_ids = batch['prompt_input_ids'].to(device)
            prompt_mask = batch['prompt_attention_mask'].to(device)
            ground_truths = batch['solutions']

            # Generate G responses per prompt
            gen_start_time = time.time()
            response_ids, response_mask = self.__generate_responses_grouped(
                prompt_ids, prompt_mask, generation_config, group_size
            ) # (B*G, resp_len)
            timing_data['gen_time'] += time.time() - gen_start_time

            # calculate logprobs for generated responses
            stats_start_time = time.time()
            stats = self.__calculate_rollout_stats(
                prompt_ids, prompt_mask, response_ids, response_mask, group_size
            )
            timing_data['stats_time'] += time.time() - stats_start_time

            # Decode texts and compute rewards 
            cpu_work_start_time = time.time()
            # extend to match grouped responses (B, L) -> (B*G, L)
            extended_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
            full_ids = torch.cat([extended_prompt_ids, response_ids], dim=1)
            full_decoded_texts  = self.tokenizer.batch_decode(
                full_ids, skip_special_tokens=True)

            # expand ground truths to match grouped responses [gt1, gt2] -> [gt1]*G + [gt2]*G
            expanded_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]

            rewards = self.__compute_reward_tensor(self.reward_funcs, full_decoded_texts, expanded_ground_truths)

            # Append to buffers
            buffer_lists['prompt_input_ids'].append(prompt_ids.cpu())
            buffer_lists['prompt_attention_mask'].append(prompt_mask.cpu())
            buffer_lists['response_input_ids'].append(response_ids.cpu())
            buffer_lists['response_attention_mask'].append(response_mask.cpu())
            buffer_lists['logprobs'].append(stats['logprobs'].cpu())
            if stats['ref_logprobs'] is not None:
                buffer_lists['ref_logprobs'].append(stats['ref_logprobs'].cpu())
            buffer_lists['rewards'].append(rewards.detach().clone())
            buffer_lists['full_texts'].extend(full_decoded_texts)
            buffer_lists['solutions'].extend(expanded_ground_truths)
            timing_data['cpu_time'] += time.time() - cpu_work_start_time
        
        # collate the buffer lists
        collation_start_time = time.time()
        collated_buffer = {}

        # collate prompts (B, P_len) -> (totalPrompts, Max_P_len)
        collated_buffer['prompt_input_ids'] = pad_and_collate_tensors(
            buffer_lists['prompt_input_ids'], self.tokenizer.pad_token_id)
        collated_buffer['prompt_attention_mask'] = pad_and_collate_tensors(
            buffer_lists['prompt_attention_mask'], 0)
        
        # collate grouped responses/stats (B*G, R_len) -> (totalSamples, Max_R_len)
        collated_buffer['response_input_ids'] = pad_and_collate_tensors(
            buffer_lists['response_input_ids'], self.tokenizer.pad_token_id)
        collated_buffer['response_attention_mask'] = pad_and_collate_tensors(
            buffer_lists['response_attention_mask'], 0)
        collated_buffer['logprobs'] = pad_and_collate_tensors(
            buffer_lists['logprobs'], 0.0)
        if buffer_lists['ref_logprobs']:
            collated_buffer['ref_logprobs'] = pad_and_collate_tensors(
                buffer_lists['ref_logprobs'], 0.0)
        else:
            collated_buffer['ref_logprobs'] = None
        
        # concatenate rewards (B*G,) -> (totalSamples,)
        collated_buffer['rewards'] = torch.cat(
            buffer_lists['rewards'], dim=0) if buffer_lists['rewards'] else torch.empty((0,),
            dtype=torch.float32)
        
        # store lists 
        collated_buffer['full_texts'] = buffer_lists['full_texts']
        collated_buffer['solutions'] = buffer_lists['solutions']
        collation_time = time.time() - collation_start_time

        # Calculate_Average Response Length
        individual_lengths = []
        for mask_batch in buffer_lists['response_attention_mask']:
            if mask_batch.numel() > 0:
                lengths_in_batch = mask_batch.sum(dim=1)
                individual_lengths.extend(lengths_in_batch.cpu().numpy())
        avg_response_length = np.mean(individual_lengths) if individual_lengths else 0
        print(f"Average response length (in tokens): {avg_response_length:.2f}")

        rollout_duration = time.time() - rollout_start_time
        collated_buffer['avg_response_length'] = avg_response_length
        collated_buffer['rollout_duration_sections'] = rollout_duration
        collated_buffer['timing/total_gen_time'] = timing_data['gen_time']
        collated_buffer['timing/total_stats_time'] = timing_data['stats_time']
        collated_buffer['timing/total_cpu_time'] = timing_data['cpu_time']
        collated_buffer['timing/collation_time'] = collation_time
        print(f"Rollout duration: {rollout_duration:.2f}s (Gen: {timing_data['gen_time']:.2f}s, Stats: {timing_data['stats_time']:.2f}s, CPU: {timing_data['cpu_time']:.2f}s, Collation: {collation_time:.2f}s)")

        return collated_buffer

    # ==============================================================================
    # == 5. GRPO update phase 
    # ==============================================================================

    def __run_grpo_update_epoch(
            self,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            collated_buffer: Dict[str, torch.Tensor],
            device: torch.device,
    ) -> Dict[str, float]:
        """
        Run one epoch of GRPO updates using the collected rollout data.

        Args:
            self: The GRPO instance.
            optimizer: The optimizer for updating the actor model.
            lr_scheduler: The learning rate scheduler.
            collated_buffer: A dictionary containing collated rollout data.
            cfg: Configuration dictionary.
            device: The device to perform computations on.
        Returns:
            A dictionary containing average losses and metrics from the update epoch.
        """
        self.model.train()
        aggregate_metrics = {}
        grpo_step_count = 0

        # Load data from collated buffer
        prompt_ids = collated_buffer['prompt_input_ids']
        prompt_mask = collated_buffer['prompt_attention_mask']
        response_ids = collated_buffer['response_input_ids']
        response_mask = collated_buffer['response_attention_mask']
        logprobs_old = collated_buffer['logprobs']
        ref_logprobs = collated_buffer['ref_logprobs']
        rewards = collated_buffer['rewards']
        group_size = self.args.group_size

        # Calculate advantages using GRPO logic
        # need kl 
        with torch.no_grad():
            kl_per_token = logprobs_old - ref_logprobs if ref_logprobs is not None else torch.zeros_like(logprobs_old)
            clip_epsilon = self.args.clip_eps
            grad_accum_steps = self.args.grad_accum_steps
            advantages = self.__compute_grpo_advantages(
                rewards, kl_per_token, response_mask, group_size
            ) # (totalSamples, resp_len)

        # mini-batch update loop
        num_prompts = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        resp_len = response_ids.shape[1]

        # map each sample back to its prompt index
        prompt_indices = np.arange(num_prompts) 
        np.random.shuffle(prompt_indices)

        # iterate over prompts in mini-batch
        # print("Computed Advantages, starting GRPO update epochs...")
        for i in range(0, num_prompts, self.args.mini_batch_size):
            prompt_batch_indicies = prompt_indices[i:i + self.args.mini_batch_size]
            actual_batch_size = len(prompt_batch_indicies)
            if actual_batch_size == 0: continue

            # get prompts from batch
            batch_prompt_ids = prompt_ids[prompt_batch_indicies].to(device)
            batch_prompt_mask = prompt_mask[prompt_batch_indicies].to(device)

            # expand to match grouped responses 
            fwd_prompt_ids = batch_prompt_ids.repeat_interleave(group_size, dim=0).to("cpu")
            fwd_prompt_mask = batch_prompt_mask.repeat_interleave(group_size, dim=0).to("cpu")

            # get corresponding response samples for this prompt batch
            sample_batch_indices = []
            for p_idx in prompt_batch_indicies:
                sample_batch_indices.extend(
                    range(p_idx * group_size, (p_idx + 1) * group_size)
                )
            # print(f"Processing batch with indices: {sample_batch_indices}")
            batch_response_ids = response_ids[sample_batch_indices]
            batch_response_mask = response_mask[sample_batch_indices]
            batch_logprobs_old = logprobs_old[sample_batch_indices]
            batch_advantages = advantages[sample_batch_indices]

            # combine for forward pass 
            batch_full_ids = torch.cat([fwd_prompt_ids, batch_response_ids], dim=1)
            batch_full_mask = torch.cat([fwd_prompt_mask, batch_response_mask], dim=1)

            # print("finished computing tensors on the CPU, starting forward pass on the model...")

            # Minibatch update 
            logits_new = self.model(
                input_ids=batch_full_ids.to(device), 
                attention_mask=batch_full_mask.to(device)).logits
            
            # print("Completed forward pass, attempting to extract response logits and compute logprobs...")
            # Extract response logits and compute new logprobs
            start_idx = prompt_len - 1
            end_idx = prompt_len + resp_len - 1
            if start_idx < 0 or end_idx <= start_idx or end_idx > logits_new.shape[1]:
                print(f"Skipping batch due to invalid indices: start_idx={start_idx}, end_idx={end_idx}, logits_len={logits_new.shape[1]}")
                continue
            
            logits_new_resp = logits_new[:, start_idx:end_idx, :].contiguous()  # (B*G, resp_len, vocab_size)
            del logits_new

            if logits_new_resp.shape[1] != batch_response_ids.shape[1]:
                print(f"Skipping batch due to response length mismatch: logits_resp_len={logits_new_resp.shape[1]}, batch_response_len={batch_response_ids.shape[1]}")
                continue
                
            batch_response_ids = batch_response_ids.to(device)

            selected_logits = torch.gather(logits_new_resp, dim=2, index=batch_response_ids.unsqueeze(-1)).squeeze(-1)
            log_denom = torch.logsumexp(logits_new_resp, dim=-1)
            logprobs_new = selected_logits - log_denom

            # print("Successfully computed new log probabilities, now calculating losses and backpropagating...")

            # Calculate losses
            policy_loss, clip_frac, approx_kl = self.__compute_grpo_policy_loss(
                logprobs_new, batch_logprobs_old.to(device), batch_advantages.to(device), batch_response_mask.to(device), clip_epsilon
            )
            entropy_loss = self.__compute_grpo_entropy_loss(
                logits_new_resp.to(device), batch_response_mask.to(device), logprobs_new.to(device)
            )

            # combine losses
            total_loss = policy_loss + self.args.entropy_coeff * entropy_loss

            # Backpropagation
            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()
            grpo_step_count += 1
            
            # store metrics
            current_metrics = {
                'train/policy_loss': policy_loss.item(),
                'train/entropy_loss': entropy_loss.item(),
                'train/total_loss': total_loss.item(),
                'params/clip_frac': clip_frac.item(),
                'params/approx_kl': approx_kl.item()
            }
            for k, v in current_metrics.items():
                aggregate_metrics.setdefault(k, []).append(v)

            # print("computed forward and backwards now checking if we should step the optimizer...")
            # Gradient step
            if grpo_step_count % grad_accum_steps == 0:
                grads_exist = any(
                    p.grad is not None for p in self.model.parameters() 
                    if p.requires_grad
                )
                if grads_exist:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)
                    aggregate_metrics.setdefault('params/grad_norm', []).append(grad_norm.item())
                    optimizer.step()
                    lr_scheduler.step()
                else:
                    print("Warning: No gradients found during GRPO update step.")
                    raise RuntimeError("No gradients found during GRPO update step.")
                optimizer.zero_grad()
        # end of epoch
        final_metrics = {k: float(np.mean(v)) for k, v in aggregate_metrics.items()}
        return final_metrics

    def __perform_grpo_update(
            self,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            rollout_buffer: Dict[str, Any],
            device: torch.device,
    ) -> Dict[str, float]:
        """
        Perform the full GRPO update process for one epoch.

        Args:
            optimizer: The optimizer for updating the actor model.
            lr_scheduler: The learning rate scheduler.
            rollout_buffer: A dictionary containing collated rollout data.
            cfg: Configuration dictionary.
            device: The device to perform computations on.
        Returns:
            A dictionary containing average losses and metrics from the GRPO update epoch.
        """
        buffer_on_device = rollout_buffer

        if 'response_input_ids' not in buffer_on_device or \
            buffer_on_device['response_input_ids'].numel() == 0:
            print("No response data available for GRPO update. Skipping update.")
            return {}
        
        all_epoch_metrics = {}
        update_epochs = self.args.epochs
        for grpo_epoch in range(update_epochs):
            epoch_metrics = self.__run_grpo_update_epoch(
                optimizer, lr_scheduler, buffer_on_device, device
            )
            print(f"GRPO Update Epoch {grpo_epoch + 1}/{update_epochs} Metrics: {epoch_metrics}")
            all_epoch_metrics = epoch_metrics

        return all_epoch_metrics

    def __training_setup(self) -> Tuple[torch.device, str]:
        random.seed(self.args.training.seed)
        np.random.seed(self.args.training.seed)
        torch.manual_seed(self.args.training.seed)

        if self.args.training.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.training.seed)
            device = torch.device('cuda')
        else:
            if self.args.training.device == 'cuda': 
                print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')

        print(f"Using device: {device}")
        output_dir = self.args.training.output_dir_grpo
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        return device, output_dir

    def __load_models_and_tokenizer(
            self,
            device: torch.device
        ) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:

        print(f"Loading tokenizer: {self.args.model.tokenizer_name}")

        trust_remote_code = self.args.model.trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model.tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            print("Tokenizer does not have a pad token. Setting pad token to eos token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "left"

        print(f"Loading actor model: {self.args.model.model_name}")

        model_kwargs = {"trust_remote_code": trust_remote_code}
        model_dtype_str = self.args.model.dtype
        if model_dtype_str != "auto":
            try:
                model_dtype = getattr(torch, model_dtype_str)
                model_kwargs["dtype"] = model_dtype
                print(f"Setting model dtype to {model_dtype_str}.")
            except AttributeError:
                print(f"Invalid dtype '{model_dtype_str}' specified. Falling back to auto.")

        quantization_cfg = self.args.model.quantization
        if quantization_cfg:
            print(f"Applying quantization with config: {quantization_cfg}")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization_cfg.load_in_4bit,
                load_in_8bit=quantization_cfg.load_in_8bit,
                bnb_4bit_quant_type=quantization_cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, quantization_cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=quantization_cfg.bnb_4bit_use_double_quant,
            )

        attn_implementation = self.args.model.attn_implementation
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            print(f"Using attention implementation: {attn_implementation}.")

        actor_model = AutoModelForCausalLM.from_pretrained(
            self.args.model.model_name, **model_kwargs
        )

        if not model_kwargs.get("quantization_config"):
            actor_model = actor_model.to(device)
        if actor_model.config.pad_token_id is None:
            print("Actor model does not have a pad token. Setting pad token to eos token.")
            actor_model.config.pad_token_id = tokenizer.pad_token_id

        print(f"Actor model loaded with dtype: {actor_model.dtype}")
        # gradient checkpointing can be enabled here if needed, but be cautious with 8-bit models
        if self.args.training.gradient_checkpointing:
            print("Enabling gradient checkpointing for actor model.")
            actor_model.gradient_checkpointing_enable()

        if self.args.model.peft_adaptor_path:
            print(f"Loading PEFT adaptor from {self.args.model.peft_adaptor_path}")
            actor_model = PeftModel.from_pretrained(
                actor_model, 
                self.args.model.peft_adaptor_path, 
                is_trainable=True,
                device_map="auto"
            ).to(device)
            print("PEFT adaptor loaded and applied to actor model.")
        
        # load reference model (for KL penalty)
        reference_model_name = self.args.model.ref_model_name or self.args.model.model_name
        print(f"Loading reference model: {reference_model_name}")
        ref_model_kwargs = model_kwargs.copy()

        ref_model = AutoModelForCausalLM.from_pretrained(
             reference_model_name, **ref_model_kwargs
        )  # Load reference model on CPU to save GPU memory

        if ref_model and ref_model.config.pad_token_id is None:
            print("Reference model does not have a pad token. Setting pad token to eos token.")
            ref_model.config.pad_token_id = tokenizer.pad_token_id

        if self.args.model.peft_adaptor_path and ref_model:
            print(f"Loading PEFT adaptor for reference model from {self.args.model.peft_adaptor_path}")
            ref_model = PeftModel.from_pretrained(
                ref_model, 
                self.args.model.peft_adaptor_path, 
                is_trainable=False,  # Reference model is not updated
                device_map="auto"
            ).to(device)
            print("PEFT adaptor loaded and applied to reference model.")

        # Freeze reference model parameters in case it's not already frozen by PEFT loading 
        # Since peft_adaptor_path: Optional[str]
        if ref_model is not None:
            for param in ref_model.parameters():
                param.requires_grad = False  # Reference model is not updated

            print(f"Reference model loaded with dtype: {ref_model.dtype}")
            ref_model.eval()
        
        return actor_model, ref_model, tokenizer

    def __setup_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, Any]:
        use_8_bit = self.args.use_8bit_adam
        lr = self.args.learning_rate

        if use_8_bit and bnb_available and next(self.model.parameters()).device.type == "cuda":
            wrapped_model = getattr(self.model, "base_model", self.model)
            quantization_config = getattr(wrapped_model, "quantization_config", None)
            is_quantized = bool(
                quantization_config
                and (quantization_config.load_in_8bit or quantization_config.load_in_4bit)
            )
            if is_quantized:
                print("Warning: Using 8-bit AdamW with a quantized model. Consider standard AdamW")
                optimizer = AdamW(self.model.parameters(), lr=lr)
            else:
                print("Using 8-bit AdamW Optimizer (bitsandbytes)")
                optimizer = bnb_optim.AdamW8bit(self.model.parameters(), lr=lr)
        else:
            if use_8_bit: print("Warning: 8-bit AdamW requested but bitsandbytes is not available. Falling back to standard AdamW.")
            else: print("Setting up standard AdamW optimizer.")
            optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Scheduler setup
        num_prompts_per_rollout = self.args.rollout_samples
        num_mini_batches_per_update = math.ceil(
            num_prompts_per_rollout / self.args.mini_batch_size
        )
        num_update_epochs = self.args.epochs
        num_optim_steps_per_step = math.ceil(
            num_mini_batches_per_update / self.args.grad_accum_steps
        ) * num_update_epochs
        num_training_steps = self.args.training.total_ppo_steps * num_optim_steps_per_step

        scheduler_name = self.args.scheduler
        warmup_steps = self.args.warmup_steps
        min_lr = self.args.min_lr

        print(f"Setting up learning rate scheduler {scheduler_name} with warmup steps: {warmup_steps}.")
        print(f"Total optimiser steps calculated: {num_training_steps}")

        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={"min_lr": min_lr} if min_lr is not None else None,
        )
        return optimizer, lr_scheduler

    def train(self):
        if self.args.wandb.report_to_wandb: # Check existence safely
            wandb.init(
                project=self.args.wandb.project,
                config=asdict(self.args),
                name=f"{self.args.wandb.name}GRPO"
            )

        # 3. Setup Optimizer and Scheduler
        optimizer, lr_scheduler = self.__setup_optimizer_and_scheduler()

        # 4. Load and preprocess dataset
        preprocessed_dataset = load_and_process_dataset(self.args, self.tokenizer)

        # 5. Load generation config
        generation_config = create_generation_config(self.args, self.tokenizer)

        def collate_fn(batch):
            input_ids = [item['input_ids'] for item in batch]
            padded_inputs = self.tokenizer.pad({"input_ids": input_ids}, padding='longest', return_tensors="pt", return_attention_mask=True)
            ground_truths = [item['solution'] for item in batch]
            return {"prompt_input_ids": padded_inputs["input_ids"],
                    "prompt_attention_mask": padded_inputs["attention_mask"],
                    "solutions": ground_truths}
        
        print("\n--- Starting GRPO training loop ---")
        group_size = self.args.group_size
        rollout_prompts = self.args.rollout_samples
        print("Using Group Size (G):", group_size)
        print("Using Rollout Samples (R):", rollout_prompts)

        for grpo_step in range(self.args.training.total_ppo_steps):
            print(f"\n=== GRPO Step {grpo_step + 1}/{self.args.training.total_ppo_steps} ===")
            
            # Phase 1: Rollout (Grouped)
            num_prompts_to_select = min(rollout_prompts, len(preprocessed_dataset))
            if num_prompts_to_select < rollout_prompts:
                print(f"Warning: Requested {rollout_prompts} prompts for rollout but only {len(preprocessed_dataset)} available. Using all available prompts.")
            
            prompt_dataloader = DataLoader(
                preprocessed_dataset.shuffle(seed=self.args.training.seed + grpo_step).select(range(num_prompts_to_select)),
                batch_size=self.args.mini_batch_size,
                collate_fn=collate_fn,
                shuffle=False,  # Shuffling is done in dataset selection
            )

            rollout_buffer = self.__compute_rollout(
                prompt_dataloader, generation_config, group_size, self.device
            )

            # Validate Rollout Buffer
            if not rollout_buffer or "rewards" not in rollout_buffer or \
            not isinstance(rollout_buffer["rewards"], torch.Tensor) or \
            rollout_buffer["rewards"].numel() == 0:
                print("Invalid rollout buffer generated. Skipping update."); continue
            
            # Calculate average reward across all generated samples (B*G)
            avg_reward = rollout_buffer["rewards"].mean().item()
            num_generated_samples = rollout_buffer["rewards"].shape[0]
            num_input_prompts = rollout_buffer["prompt_input_ids"].shape[0]
            avg_resp_len = rollout_buffer.get("avg_response_length", 0.0)
            rollout_duration = rollout_buffer.get("rollout_duration_sections", rollout_buffer.get("rollout_duration_seconds", 0.0))
            # Log timing breakdown
            gen_time = rollout_buffer.get("timing/total_gen_time", 0.0)
            stats_time = rollout_buffer.get("timing/total_stats_time", 0.0)
            cpu_time = rollout_buffer.get("timing/total_cpu_time", 0.0)
            collation_time = rollout_buffer.get("timing/collation_time", 0.0)

            print(
                f"Rollout complete ({num_input_prompts} prompts -> {num_generated_samples} samples"
                f"Avg Reward: {avg_reward:.4f}, Avg Resp Len: {avg_resp_len:.2f} tokens, "
                f"Duration: {rollout_duration:.2f}s "
                f"(Gen: {gen_time:.2f}s, Stats: {stats_time:.2f}s, CPU: {cpu_time:.2f}s, Collation: {collation_time:.2f}s)"
            )

            # Phase 2: GRPO Update
            print("Starting GRPO update phase...")
            update_metrics = self.__perform_grpo_update(
                optimizer, lr_scheduler, rollout_buffer, self.device
            )

            # log metrics 
            log_data = {}
            if update_metrics:
                log_data.update(update_metrics)
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in update_metrics.items()])
                print(f"Update metrics (Avg over epoch): {log_str}")
                print(f"   Rollout Reward (Avg over samples): {avg_reward:.4f}")
            else:
                print("Update skipped or failed")
            
            log_data["rollout/reward_mean"] = avg_reward
            log_data["rollout/avg_response_length"] = avg_resp_len
            log_data["rollout/duration_seconds"] = rollout_duration
            log_data["rollout/timing_gen_seconds"] = gen_time
            log_data["rollout/timing_stats_seconds"] = stats_time
            log_data["rollout/timing_cpu_seconds"] = cpu_time
            log_data["rollout/timing_collate_seconds"] = collation_time

            if self.args.wandb.report_to_wandb and update_metrics:
                wandb.log(log_data, step=grpo_step)

            # Phase 3: Checkpointing
            if (grpo_step + 1) % self.args.training.save_interval == 0:
                save_model(
                    self.model, self.tokenizer, os.path.join(self.output_dir, f"step_{grpo_step + 1}")
                )

        # Final logging for this run
        if self.args.wandb.report_to_wandb:
            wandb.finish()
        print("\n--- GRPO training complete ---")
        final_model_path = os.path.join(self.output_dir, "final_model")
        save_model(self.model, self.tokenizer, final_model_path)
        print(f"Final model saved to: {final_model_path}")

