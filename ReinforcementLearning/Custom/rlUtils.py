from contextlib import contextmanager
import time

import torch
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

DEBUG = True

def Debug(fmt, *args):
    if DEBUG:
        if args:
            print('[DEBUG]' + fmt.format(*args))
        else:
            print(f"[DEBUG] {fmt}")  

def masked_mean(tensor: torch.Tensor, 
                mask: Optional[torch.Tensor], 
                dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(tensor, dim=dim)

    mask = mask.to(device=tensor.device)
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor).to(dtype=tensor.dtype)

    masked_sum = (tensor * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=1e-8)
    return masked_sum / denom

def masked_whiten(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor],
        shift_mean: bool = True,
    ) -> torch.Tensor:
    mask = mask.bool()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)

    mean = masked_mean(tensor, mask, dim=None)
    masked_tensor_var = torch.where(
        mask, (tensor - mean) ** 2,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    )
    var = masked_mean(masked_tensor_var, mask, dim=None)
    std = torch.sqrt(var + 1e-8) # Add a small constant to prevent division by zero

    whitened = (tensor - mean) / std if shift_mean else tensor / std
    return torch.where(
        mask, whitened,
        torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    )


def pad_and_collate_tensors(tensor_list: list, pad_val: float) -> torch.Tensor:
    max_len = max(t.shape[1] if t.dim() > 1 else t.shape[0] for t in tensor_list)
    if max_len == 0:
        total_batch_size = sum(t.shape[0] for t in tensor_list)
        original_shape = tensor_list[0].shape if tensor_list else (0,)
        return torch.empty(
            (total_batch_size, 0) + original_shape[2:],
            dtype=tensor_list[0].dtype, device=tensor_list[0].device
        )
    # Pad each tensor to the max length and then concatenate along the sequence length dimension
    padded_tensors = []
    for t in tensor_list:
        current_len = t.shape[1] if t.dim() > 1 else t.shape[0]
        padding_needed = max_len - current_len
        if padding_needed > 0:
            pad_tuple = (0, padding_needed, 0, 0) 
            t = F.pad(t, 
                      tuple(pad_tuple), 
                      mode='constant',
                      value=pad_val)
        padded_tensors.append(t)
    # Concatenate along the sequence length dimension (dim=1)
    return torch.cat(padded_tensors, dim=0)


def _sync_cuda(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def benchmark_section(
        name: str,
        device: Optional[torch.device] = None,
        timings: Optional[Dict[str, float]] = None,
):
    _sync_cuda(device)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda(device)
        elapsed = time.perf_counter() - start_time
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + elapsed
        print(f"[bench] {name}: {elapsed:.4f}s")


def empty_probs(prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logprobs = torch.empty(
        (prompt_ids.shape[0], 0),
        dtype=torch.float,
        device=prompt_ids.device
    )
    ref_logprobs = torch.empty(
        (prompt_ids.shape[0], 0),
        dtype=torch.float,
        device=prompt_ids.device
    )
    values = torch.empty(
        (prompt_ids.shape[0], 0),
        dtype=torch.float,
        device=prompt_ids.device
    )
    return logprobs, ref_logprobs, values
