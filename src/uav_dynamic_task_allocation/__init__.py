"""项目顶层兼容包。

当前源码目录采用 `src/core`、`src/planning`、`src/envs` 等顶层子包组织，
但历史脚本、测试和模块内部导入仍使用 `uav_dynamic_task_allocation.xxx`。
这里把包搜索路径指向 `src/`，让旧导入路径继续解析到现有模块。
"""
from __future__ import annotations

from pathlib import Path

# __path__: 告诉 Python 在 src/ 目录下查找 core、planning、envs 等子包。
__path__ = [str(Path(__file__).resolve().parent.parent)]

# __version__: 当前工程包版本号，供日志、调试和外部工具读取。
__version__ = "0.1.0"
