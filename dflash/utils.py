"""Utility functions for dflash.

Provides helpers for device detection, logging, token counting,
and other shared functionality used across the package.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "dflash") -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Device / backend detection
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Detect the best available backend: 'cuda', 'mps', 'mlx', or 'cpu'."""
    # MLX is Apple-silicon only; prefer it over MPS when available.
    try:
        import mlx.core as mx  # noqa: F401
        return "mlx"
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def is_mlx_available() -> bool:
    """Return True if the mlx package is importable."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def is_torch_available() -> bool:
    """Return True if PyTorch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager that logs elapsed wall-clock time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        tag = f"[{label}] " if label else ""
        logger.debug("%sElapsed: %.4f s", tag, elapsed)


# ---------------------------------------------------------------------------
# Token / text helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens in *text* using *tokenizer*.

    Works with both HuggingFace tokenizers and any callable that accepts a
    string and returns an object with an ``input_ids`` attribute or a list.
    """
    encoded = tokenizer(text)
    if hasattr(encoded, "input_ids"):
        return len(encoded.input_ids)
    if isinstance(encoded, (list, tuple)):
        return len(encoded)
    raise TypeError(f"Unexpected tokenizer output type: {type(encoded)}")


def truncate_to_max_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    side: str = "left",
) -> str:
    """Truncate *text* so that it fits within *max_tokens*.

    Args:
        text: Input string.
        tokenizer: A HuggingFace-compatible tokenizer.
        max_tokens: Maximum number of tokens to keep.
        side: ``'left'`` keeps the *end* of the text (useful for context
              windows); ``'right'`` keeps the beginning.

    Returns:
        Decoded string that fits within the token budget.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    if side == "left":
        ids = ids[-max_tokens:]
    else:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def format_size(num_bytes: int) -> str:
    """Human-readable byte size, e.g. ``'1.23 GiB'``."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.2f} PiB"
