"""utils 数据模块中的计算设备实现。"""
from __future__ import annotations


def is_torch_available() -> bool:
    """
    Check whether PyTorch is installed.

    Returns:
        True if PyTorch is available, otherwise False.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def is_cuda_available() -> bool:
    """
    Check whether CUDA is available through PyTorch.

    Returns:
        True if CUDA is available, otherwise False.
    """
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available()


def get_device(preferred_device: str = "auto") -> str:
    """
    Select running device.

    Args:
        preferred_device:
            - "auto": use CUDA if available, otherwise CPU
            - "cpu": force CPU
            - "cuda": force CUDA, raise error if unavailable

    Returns:
        Selected device name: "cpu" or "cuda".

    Raises:
        ValueError: If preferred_device is invalid or CUDA is requested but unavailable.
    """
    preferred_device = preferred_device.lower()

    if preferred_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Invalid device option: {preferred_device}. "
            "Available options are: auto, cpu, cuda."
        )

    if preferred_device == "cpu":
        return "cpu"

    if preferred_device == "cuda":
        if not is_torch_available():
            raise ValueError(
                "CUDA was requested, but PyTorch is not installed."
            )

        if not is_cuda_available():
            raise ValueError(
                "CUDA was requested, but CUDA is not available."
            )

        return "cuda"

    # auto mode
    if is_cuda_available():
        return "cuda"

    return "cpu"


def get_device_info() -> dict[str, str | bool | int | None]:
    """
    Get basic device information.

    Returns:
        Dictionary containing runtime device information.
    """
    info: dict[str, str | bool | int | None] = {
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
    }

    try:
        import torch
    except ImportError:
        return info

    info["torch_available"] = True
    info["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)

    return info