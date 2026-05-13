from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_ppo_checkpoint(
    checkpoint_dir: str | Path,
    metadata: dict[str, Any],
    model_state: Any | None = None,
) -> Path:
    """Save PPO metadata and optional torch state dict."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "ppo_clusterer_checkpoint.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if model_state is not None:
        try:
            import torch

            torch.save(model_state, path / "ppo_clusterer.pt")
        except Exception:
            pass
    return metadata_path


def load_ppo_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load PPO checkpoint metadata if present."""
    path = Path(checkpoint_dir) / "ppo_clusterer_checkpoint.json"
    if not path.exists():
        return {"checkpoint_loaded": False}
    return json.loads(path.read_text(encoding="utf-8"))
