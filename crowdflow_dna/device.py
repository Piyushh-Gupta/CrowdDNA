"""Device selection helper with safe CUDA fallback.

PyTorch's ``torch.cuda.is_available()`` only reports whether a CUDA
device is *visible* to the runtime. It does **not** guarantee that the
installed CUDA binaries can actually execute on that device. On GPUs
with a compute capability newer than the one PyTorch was compiled for
(e.g. an RTX 5050 / sm_120 against a build that only supports
sm_50..sm_90), every CUDA kernel launch raises::

    RuntimeError: CUDA error: no kernel image is available for execution on the device

This module exposes a single helper that probes the device at runtime
and falls back to CPU when CUDA is unusable, so training and inference
never crash on a hardware/software mismatch.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def get_safe_device() -> torch.device:
    """Return a ``torch.device`` that is guaranteed to work.

    Returns:
        A CUDA device if CUDA is available **and** a probe tensor
        operation succeeds; otherwise a CPU device. A warning is
        logged whenever the fallback is taken so the operator is
        aware that inference is running on CPU.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA is not available; using CPU.")
        return torch.device("cpu")

    try:
        # Probe: allocate a tiny tensor on the device and run a trivial
        # operation. This is the only reliable way to detect a
        # compute-capability mismatch, which ``is_available()`` misses.
        probe = torch.zeros(1, device="cuda")
        _ = (probe + 1).to("cpu")
        logger.info("CUDA probe succeeded; using GPU %s.", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    except RuntimeError as exc:
        logger.warning(
            "CUDA device is present but unusable (%s). Falling back to CPU.",
            exc,
        )
        return torch.device("cpu")