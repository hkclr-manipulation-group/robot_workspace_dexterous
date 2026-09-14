"""Opt-in CUDA diagnostics; never retry work on a failed CUDA context."""
from contextlib import contextmanager
import os
import sys


def enable_cuda_debug() -> None:
    # Set before importing torch or creating any CUDA objects.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'


@contextmanager
def cuda_stage(label: str, enabled: bool = False):
    if not enabled:
        yield
        return
    import torch
    print(f'[CUDA debug] START {label}', file=sys.stderr, flush=True)
    try:
        yield
        torch.cuda.synchronize()
    except Exception:
        print(f'[CUDA debug] FAILED {label}. Exit this process before retrying.',
              file=sys.stderr, flush=True)
        raise
    print(f'[CUDA debug] OK {label}', file=sys.stderr, flush=True)


def cuda_environment() -> dict:
    import torch
    import curobo
    result = {'python': sys.version, 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'curobo_path': curobo.__file__,
              'cuda_available': torch.cuda.is_available()}
    if result['cuda_available']:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result.update(device=device, gpu=properties.name,
                      capability=list(torch.cuda.get_device_capability(device)),
                      total_memory_bytes=properties.total_memory)
    return result
