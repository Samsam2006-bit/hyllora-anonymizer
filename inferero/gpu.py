"""One model on the GPU at a time, across the whole server: torch's MPS backend segfaults when two threads run
inference at once (the analysis worker plus the draft, or two cameras). Every model call and the reading of its
result tensors holds this lock."""

import threading

GPU_LOCK = threading.RLock()
