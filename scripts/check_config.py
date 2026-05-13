"""check配置脚本，封装可直接运行的实验、检查或可视化流程。"""
from pathlib import Path

from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
    resolve_path,
)


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)

    print("Config loaded successfully.")
    print(f"Project root: {project_root}")
    print(f"Config path: {config_path}")

    print("\nBasic information:")
    print(f"  Project name: {get_config_value(config, 'project.name')}")
    print(f"  Version: {get_config_value(config, 'project.version')}")
    print(f"  Experiment name: {get_config_value(config, 'experiment.name')}")
    print(f"  Seed: {get_config_value(config, 'experiment.seed')}")
    print(f"  Device: {get_config_value(config, 'experiment.device')}")

    print("\nData paths:")
    uav_file = resolve_path(get_config_value(config, "data.uav_file"), project_root)
    target_file = resolve_path(get_config_value(config, "data.target_file"), project_root)
    battlefield_file = resolve_path(
        get_config_value(config, "data.battlefield_file"),
        project_root,
    )

    print(f"  UAV file: {uav_file}")
    print(f"  Target file: {target_file}")
    print(f"  Battlefield file: {battlefield_file}")

    print("\nEnvironment:")
    print(f"  Environment name: {get_config_value(config, 'env.name')}")
    print(f"  Map width: {get_config_value(config, 'env.map_width')}")
    print(f"  Map height: {get_config_value(config, 'env.map_height')}")
    print(f"  Max steps: {get_config_value(config, 'env.max_steps')}")
    print(
        "  Dynamic events enabled: "
        f"{get_config_value(config, 'env.enable_dynamic_events')}"
    )

    print("\nAlgorithm:")
    print(f"  Algorithm name: {get_config_value(config, 'algorithm.name')}")
    print(f"  Learning rate: {get_config_value(config, 'algorithm.learning_rate')}")
    print(f"  Batch size: {get_config_value(config, 'algorithm.batch_size')}")

    print("\nTraining:")
    print(f"  Episodes: {get_config_value(config, 'training.episodes')}")
    print(f"  Eval interval: {get_config_value(config, 'training.eval_interval')}")
    print(f"  Save interval: {get_config_value(config, 'training.save_interval')}")

    print("\nConfig check finished.")


if __name__ == "__main__":
    main()