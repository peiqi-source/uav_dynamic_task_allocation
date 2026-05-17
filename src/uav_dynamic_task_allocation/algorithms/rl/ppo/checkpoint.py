"""Checkpoint helpers for PPO target regrouping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PPOCheckpointError(Exception):
    """Raised when a PPO checkpoint cannot be saved or loaded."""


def _checkpoint_paths(checkpoint_dir: str | Path) -> tuple[Path, Path]:
    path = Path(checkpoint_dir)
    return path / "ppo_clusterer.pt", path / "ppo_clusterer_checkpoint.json"


def save_ppo_checkpoint(
    checkpoint_dir: str | Path,
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model_state: Any | None = None,
) -> Path:
    """
    Save PPO model weights and metadata.

    The preferred signature is save_ppo_checkpoint(checkpoint_dir, model, metadata).
    The legacy model_state keyword is still accepted for backward compatibility.
    """
    checkpoint_path, metadata_path = _checkpoint_paths(checkpoint_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None and isinstance(model, dict) and model_state is not None:
        metadata_to_save = dict(model)
        model = None
    elif metadata is None and isinstance(model, dict) and model_state is None:
        metadata_to_save = dict(model)
        model = None
    else:
        metadata_to_save = dict(metadata or {})

    state_dict = model_state
    if state_dict is None and model is not None:
        state_dict = model.state_dict() if hasattr(model, "state_dict") else model

    if state_dict is not None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise PPOCheckpointError(
                "PyTorch is required to save ppo_clusterer.pt."
            ) from exc
        torch.save({"model_state_dict": state_dict}, checkpoint_path)
        metadata_to_save["model_path"] = str(checkpoint_path)

    metadata_path.write_text(
        json.dumps(metadata_to_save, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return metadata_path


def load_ppo_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load PPO checkpoint metadata."""
    _, metadata_path = _checkpoint_paths(checkpoint_dir)
    if not metadata_path.exists():
        raise PPOCheckpointError(f"PPO metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_ppo_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load PPO model_state_dict and metadata."""
    checkpoint_path, metadata_path = _checkpoint_paths(checkpoint_dir)
    if not checkpoint_path.exists():
        raise PPOCheckpointError(f"PPO model checkpoint not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise PPOCheckpointError(f"PPO metadata file not found: {metadata_path}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PPOCheckpointError("PyTorch is required to load PPO checkpoint.") from exc

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise PPOCheckpointError(
            f"Unsupported PPO checkpoint payload type: {type(payload)}"
        )

    return {
        "model_state_dict": state_dict,
        "metadata": load_ppo_metadata(checkpoint_dir),
    }
