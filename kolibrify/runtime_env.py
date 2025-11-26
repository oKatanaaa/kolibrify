import os
import torch


def prepare_unsloth_environment():
    """
    Configure environment flags before importing unsloth/TRL patches.

    - Disable torch.compile based kernels on low-VRAM GPUs to avoid OOMs.
    """
    if "UNSLOTH_COMPILE_DISABLE" not in os.environ and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_gb = props.total_memory / (1024 ** 3)
        if total_gb < 20:
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
