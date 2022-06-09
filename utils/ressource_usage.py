import os
try:
    import resource
except ModuleNotFoundError:
    # The resource module is available only on unix systems.
    pass
import subprocess
from typing import Tuple


def resource_usage() -> Tuple[int, str]:
    """Returns the ressources used by the process.

    Taken from https://gitlab.com/corentin-pro/torch_utils/-/blob/master/train.py
    Returns peak RAM usage and VRAM usage at the time this function is called.
    Note that this is different from peak VRAM usage (as this usually happens before the training loop).

    Returns:
        tuple: Peak memory usage and peak gpu usage
    """
    try:
        memory_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except NameError:
        pass
    gpu_memory = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True).decode()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_memory = gpu_memory.split('\n')[int(os.environ["CUDA_VISIBLE_DEVICES"])]
    else:
        gpu_memory = ' '.join(gpu_memory.split('\n'))

    return memory_peak, gpu_memory
