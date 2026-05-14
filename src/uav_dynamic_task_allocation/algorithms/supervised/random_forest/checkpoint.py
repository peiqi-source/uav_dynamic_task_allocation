"""Random Forest 模型 checkpoint 读写。"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier


def save_random_forest_model(
    model: RandomForestClassifier,
    path: str | Path,
) -> Path:
    """
    保存 Random Forest 模型。

    参数：
        model: 已训练好的 RandomForestClassifier。
        path: 模型输出路径，通常为 checkpoints/random_forest/destroy_target_rf.joblib。

    返回：
        实际写入的模型路径。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_random_forest_model(path: str | Path) -> RandomForestClassifier:
    """
    加载 Random Forest 模型。

    参数：
        path: joblib 模型文件路径。

    返回：
        从磁盘反序列化得到的 RandomForestClassifier。
    """
    model_path = Path(path)
    return joblib.load(model_path)

