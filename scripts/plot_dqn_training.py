"""绘图DQN 算法训练脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from uav_dynamic_task_allocation.utils.config import get_project_root


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认读取 outputs/metrics/dqn_training_metrics.csv，
    并将训练曲线保存到 outputs/figures/dqn/。

    示例：
        python scripts/plot_dqn_training.py

        python scripts/plot_dqn_training.py ^
            --metrics-path outputs/metrics/dqn_training_metrics.csv ^
            --output-dir outputs/figures/dqn ^
            --smooth-window 10
    """
    parser = argparse.ArgumentParser(
        description="Plot DQN training curves from metrics CSV."
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        default="outputs/metrics/dqn_training_metrics.csv",
        help="Path to DQN training metrics CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures/dqn",
        help="Directory to save generated figures.",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help=(
            "Rolling average window size. "
            "If set to 1, smoothing is disabled."
        ),
    )

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    """
    解析路径。

    如果输入是相对路径，则默认相对于项目根目录。
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return get_project_root() / path_obj


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    读取训练指标 CSV。

    这里会把关键数值列统一转换成 numeric。
    因为 CSV 中某些字段可能为空字符串，比如训练前期没有 update 时 mean_loss 为空，
    如果不转换，后面画图容易出现类型问题。
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Please run scripts/train_dqn.py first."
        )

    df = pd.read_csv(metrics_path, encoding="utf-8-sig")

    required_columns = ["episode", "total_reward"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in metrics CSV: {missing_columns}"
        )

    numeric_columns = [
        "episode",
        "total_reward",
        "mean_loss",
        "random_action_ratio",
        "epsilon_after_episode",
        "best_reward_so_far",
        "num_steps",
        "num_updates",
        "replay_buffer_size",
        "global_env_steps",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("episode").reset_index(drop=True)

    return df


def add_rolling_mean(
    df: pd.DataFrame,
    column: str,
    smooth_window: int,
) -> pd.Series:
    """
    计算滚动平均曲线。

    强化学习曲线通常波动比较大，尤其是 episode reward。
    直接看原始曲线可能比较乱，所以增加 rolling mean 辅助观察趋势。
    """
    if smooth_window <= 1:
        return df[column]

    valid_window = min(smooth_window, max(len(df), 1))

    return df[column].rolling(
        window=valid_window,
        min_periods=1,
    ).mean()


def save_line_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    smooth_window: int = 10,
    include_smooth: bool = True,
) -> None:
    """
    保存单条指标曲线。

    如果 include_smooth=True，会额外画一条滚动平均曲线。
    这里不在代码中指定颜色，让 matplotlib 使用默认配色。
    """
    if y_column not in df.columns:
        print(f"[Skip] Column not found: {y_column}")
        return

    plot_df = df[[x_column, y_column]].dropna()

    if plot_df.empty:
        print(f"[Skip] Column has no valid values: {y_column}")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df[x_column],
        plot_df[y_column],
        label=y_column,
        linewidth=1.5,
    )

    if include_smooth and smooth_window > 1 and len(plot_df) > 1:
        smooth_values = add_rolling_mean(
            plot_df,
            y_column,
            smooth_window=smooth_window,
        )
        plt.plot(
            plot_df[x_column],
            smooth_values,
            label=f"{y_column}_rolling_mean_{smooth_window}",
            linewidth=2.0,
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Saved] {output_path}")


def save_reward_plot(
    df: pd.DataFrame,
    output_dir: Path,
    smooth_window: int,
) -> None:
    """
    保存 reward 曲线。

    total_reward 是最重要的训练指标；
    best_reward_so_far 用于观察当前最优表现是否持续刷新。
    """
    output_path = output_dir / "reward_curve.png"

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["episode"],
        df["total_reward"],
        label="total_reward",
        linewidth=1.5,
    )

    if smooth_window > 1 and len(df) > 1:
        reward_smooth = add_rolling_mean(
            df,
            "total_reward",
            smooth_window=smooth_window,
        )
        plt.plot(
            df["episode"],
            reward_smooth,
            label=f"total_reward_rolling_mean_{smooth_window}",
            linewidth=2.0,
        )

    if "best_reward_so_far" in df.columns:
        best_reward_df = df[["episode", "best_reward_so_far"]].dropna()
        if not best_reward_df.empty:
            plt.plot(
                best_reward_df["episode"],
                best_reward_df["best_reward_so_far"],
                label="best_reward_so_far",
                linewidth=2.0,
            )

    plt.title("DQN Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Saved] {output_path}")


def save_steps_updates_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存每个 episode 的步数和更新次数曲线。

    这个图用于检查训练是否真的在进行：
    - num_steps 太低，说明 episode 过早结束；
    - num_updates 长期为 0，说明 replay buffer 或 warmup 设置有问题。
    """
    required_columns = ["num_steps", "num_updates"]
    if not all(column in df.columns for column in required_columns):
        print("[Skip] num_steps or num_updates not found.")
        return

    output_path = output_dir / "steps_updates_curve.png"

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["episode"],
        df["num_steps"],
        label="num_steps",
        linewidth=1.5,
    )

    plt.plot(
        df["episode"],
        df["num_updates"],
        label="num_updates",
        linewidth=1.5,
    )

    plt.title("DQN Steps and Updates per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Saved] {output_path}")


def print_training_summary(df: pd.DataFrame) -> None:
    """
    在控制台输出训练结果摘要。

    这一步不是画图必需的，但很方便快速判断训练是否正常。
    """
    print("\n========== DQN Training Summary ==========")
    print(f"Total episodes: {len(df)}")

    if "total_reward" in df.columns:
        print(f"Best reward: {df['total_reward'].max()}")
        print(f"Last reward: {df['total_reward'].iloc[-1]}")
        print(f"Mean reward: {df['total_reward'].mean()}")

    if "mean_loss" in df.columns:
        valid_loss = df["mean_loss"].dropna()
        if not valid_loss.empty:
            print(f"Last mean loss: {valid_loss.iloc[-1]}")
            print(f"Min mean loss: {valid_loss.min()}")
            print(f"Max mean loss: {valid_loss.max()}")
        else:
            print("Mean loss: no valid values")

    if "epsilon_after_episode" in df.columns:
        valid_epsilon = df["epsilon_after_episode"].dropna()
        if not valid_epsilon.empty:
            print(f"Initial epsilon: {valid_epsilon.iloc[0]}")
            print(f"Final epsilon: {valid_epsilon.iloc[-1]}")

    if "random_action_ratio" in df.columns:
        valid_ratio = df["random_action_ratio"].dropna()
        if not valid_ratio.empty:
            print(f"Final random action ratio: {valid_ratio.iloc[-1]}")

    if "num_updates" in df.columns:
        print(f"Total updates: {df['num_updates'].sum()}")

    print("==========================================\n")


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    metrics_path = resolve_path(args.metrics_path)
    output_dir = resolve_path(args.output_dir)

    df = load_metrics(metrics_path)

    print(f"[Loaded] {metrics_path}")
    print(f"[Output dir] {output_dir}")

    print_training_summary(df)

    save_reward_plot(
        df=df,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )

    save_line_plot(
        df=df,
        x_column="episode",
        y_column="mean_loss",
        output_path=output_dir / "loss_curve.png",
        title="DQN Mean Loss",
        ylabel="Mean Loss",
        smooth_window=args.smooth_window,
        include_smooth=True,
    )

    save_line_plot(
        df=df,
        x_column="episode",
        y_column="epsilon_after_episode",
        output_path=output_dir / "epsilon_curve.png",
        title="DQN Epsilon Decay",
        ylabel="Epsilon",
        smooth_window=1,
        include_smooth=False,
    )

    save_line_plot(
        df=df,
        x_column="episode",
        y_column="random_action_ratio",
        output_path=output_dir / "random_action_ratio_curve.png",
        title="DQN Random Action Ratio",
        ylabel="Random Action Ratio",
        smooth_window=args.smooth_window,
        include_smooth=True,
    )

    save_steps_updates_plot(
        df=df,
        output_dir=output_dir,
    )

    print("[Done] DQN training figures generated successfully.")


if __name__ == "__main__":
    main()