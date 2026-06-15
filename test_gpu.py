import torch 

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def get_cuda_device_count() -> int:
    """Get the number of CUDA devices available."""
    return torch.cuda.device_count()

def get_cuda_device_name(device_index: int = 0) -> str:
    """Get the name of a specific CUDA device."""
    if is_cuda_available() and device_index < get_cuda_device_count():
        return torch.cuda.get_device_name(device_index)
    else:
        return "No CUDA device available or index out of range."

if __name__ == "__main__":
    if is_cuda_available():
        print("CUDA is available. GPU can be used.")
        print(f"Number of CUDA devices: {get_cuda_device_count()}")
        print(f"CUDA device name: {get_cuda_device_name()}")
    else:
        print("CUDA is not available. GPU cannot be used.")
    