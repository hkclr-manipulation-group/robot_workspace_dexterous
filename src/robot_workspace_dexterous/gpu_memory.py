"""Apply a PyTorch allocator budget before constructing cuRobo GPU objects."""
import math


def configure_gpu_memory(fraction=0.75):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('GPU memory fraction must be in (0, 1]')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('cuRobo workspace computation requires CUDA-enabled PyTorch')
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    # Leave the requested margin even when other processes already use memory.
    limit = min(total*fraction, free-total*(1-fraction))
    if limit <= 0:
        raise RuntimeError('Insufficient free GPU memory to preserve the configured reserve')
    torch.cuda.set_per_process_memory_fraction(limit/total, device)
    print(f'PyTorch GPU allocation limit: {limit/2**30:.1f} GiB; '
          f'target reserve: {total*(1-fraction)/2**30:.1f} GiB. '
          'External CUDA allocations are outside this limit.', flush=True)
    return limit
