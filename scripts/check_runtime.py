from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.device import get_device, get_device_info
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Runtime check started.")

    seed = get_config_value(config, "experiment.seed", default=42)
    preferred_device = get_config_value(config, "experiment.device", default="auto")

    set_seed(seed)
    device = get_device(preferred_device)
    device_info = get_device_info()

    logger.info(f"Seed set to: {seed}")
    logger.info(f"Preferred device: {preferred_device}")
    logger.info(f"Selected device: {device}")

    logger.info("Device information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("Runtime check finished successfully.")


if __name__ == "__main__":
    main()