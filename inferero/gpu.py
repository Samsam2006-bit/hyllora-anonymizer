"""One model on the GPU at a time, across the whole server: torch's MPS backend segfaults when two threads run
inference at once (the analysis worker plus the draft, or two cameras). Every model call and the reading of its
result tensors holds this lock."""

import threading

GPU_LOCK = threading.RLock()


def half_ok(device: str) -> bool:
    """16-bit floats only where they are fast: Apple's GPU and NVIDIA from Volta (compute capability 7) on. A GTX 10xx
    (Pascal, 6.x) runs them at a small fraction of its 32-bit speed (2026-11-17, the store's GTX 1080)."""
    if device == "mps":
        return True
    if device == "cuda":
        import torch

        return torch.cuda.get_device_capability(0)[0] >= 7
    return False
