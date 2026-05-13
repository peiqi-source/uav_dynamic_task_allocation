"""utils 数据模块中的随机种子实现。"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible experiments.

    This function sets seeds for:
        - Python random
        - NumPy
        - PyTorch, if installed
        - TensorFlow, if installed
        - Python hash seed

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic behavior for deep learning frameworks.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed)}")

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # 让 Python 的哈希行为更加稳定,比如字典、集合在某些情况下的顺序
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 设置 Python 自带随机库
    random.seed(seed)
    # 确定 NumPy 随机数
    np.random.seed(seed)

    _set_torch_seed(seed, deterministic=deterministic)
    _set_tensorflow_seed(seed)


def _set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set PyTorch random seed if PyTorch is installed.

    The function does nothing if PyTorch is not available.
    """
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _set_tensorflow_seed(seed: int) -> None:
    """
    Set TensorFlow random seed if TensorFlow is installed.

    The function does nothing if TensorFlow is not available.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.random.set_seed(seed)


def get_random_state(seed: int | None = None) -> np.random.Generator:
    """
    Create a NumPy random generator.

    Args:
        seed: Optional random seed.

    Returns:
        NumPy random generator.
    """
    return np.random.default_rng(seed)


def seed_worker(worker_id: int) -> None:
    """
    Set seed for data loading workers.

    This is mainly useful for PyTorch DataLoader in future training code.

    Args:
        worker_id: Worker ID.
    """
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)