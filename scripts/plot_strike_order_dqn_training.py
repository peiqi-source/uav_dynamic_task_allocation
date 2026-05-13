"""绘图打击顺序DQN 算法训练脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from uav_dynamic_task_allocation.utils.config import get_project_root


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认读取 StrikeOrder DQN 的训练指标文件：
        outputs/metrics/strike_order_dqn_metrics.csv

    默认输出到：
        outputs/figures/strike_order_dqn/
    """
    parser = argparse.ArgumentParser(
        description="Plot StrikeOrder DQN training curves."
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        default="outputs/metrics/strike_order_dqn_metrics.csv",
        help="Path to StrikeOrder DQN metrics CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures/strike_order_dqn",
        help="Directory to save figures.",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Rolling average window size. Use 1 to disable smoothing.",
    )

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    """将相对路径解析到项目根目录下。"""
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return get_project_root() / path_obj


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    读取 StrikeOrder DQN metrics CSV。

    这里会把常用列转换成数值类型。
    如果 total_path_distance 因为旧版 metrics 没有被保存成单独列，
    会尝试从 extra 字段中解析。
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Please run scripts/train_strike_order_dqn.py first."
        )

    df = pd.read_csv(metrics_path, encoding="utf-8-sig")

    if "episode" not in df.columns:
        raise ValueError("Missing required column: episode")

    if "total_reward" not in df.columns:
        raise ValueError("Missing required column: total_reward")

    df = recover_columns_from_extra(df)

    numeric_columns = [
        "episode",
        "cluster_id",
        "total_reward",
        "num_steps",
        "num_updates",
        "mean_loss",
        "random_action_ratio",
        "epsilon_after_episode",
        "best_reward_so_far",
        "global_env_steps",
        "replay_buffer_size",
        "total_path_distance",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("episode").reset_index(drop=True)

    return df


def recover_columns_from_extra(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 extra 字段中恢复 StrikeOrder 专用字段。

    这是为了兼容之前已经生成的 metrics.csv。
    如果 CSV 里本来就有 total_path_distance 等列，则不会覆盖有效值。
    """
    if "extra" not in df.columns:
        return df

    candidate_columns = [
        "cluster_id",
        "ordered_target_ids",
        "total_path_distance",
    ]

    for column in candidate_columns:
        if column not in df.columns:
            df[column] = None

    for row_index, extra_value in df["extra"].items():
        if pd.isna(extra_value) or not str(extra_value).strip():
            continue

        try:
            extra_data: dict[str, Any] = json.loads(str(extra_value))
        except json.JSONDecodeError:
            continue

        for column in candidate_columns:
            current_value = df.at[row_index, column]

            if pd.isna(current_value) or current_value in ("", None):
                if column in extra_data:
                    df.at[row_index, column] = extra_data[column]

    return df


def rolling_mean(
    values: pd.Series,
    smooth_window: int,
) -> pd.Series:
    """计算滚动平均。"""
    if smooth_window <= 1:
        return values

    return values.rolling(
        window=min(smooth_window, max(len(values), 1)),
        min_periods=1,
    ).mean()


def save_line_plot(
    df: pd.DataFrame,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    smooth_window: int = 10,
    include_smooth: bool = True,
) -> None:
    """
    保存单指标曲线。
    """
    if y_column not in df.columns:
        print(f"[Skip] Missing column: {y_column}")
        return

    plot_df = df[["episode", y_column]].dropna()

    if plot_df.empty:
        print(f"[Skip] No valid values for column: {y_column}")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df["episode"],
        plot_df[y_column],
        label=y_column,
        linewidth=1.5,
    )

    if include_smooth and smooth_window > 1 and len(plot_df) > 1:
        smooth_values = rolling_mean(plot_df[y_column], smooth_window)
        plt.plot(
            plot_df["episode"],
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

    对 StrikeOrder DQN 来说，reward 应该和：
    - 距离收益
    - 目标重要性收益
    - 完成目标群 bonus
    有关。
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
        plt.plot(
            df["episode"],
            rolling_mean(df["total_reward"], smooth_window),
            label=f"total_reward_rolling_mean_{smooth_window}",
            linewidth=2.0,
        )

    if "best_reward_so_far" in df.columns:
        best_df = df[["episode", "best_reward_so_far"]].dropna()
        if not best_df.empty:
            plt.plot(
                best_df["episode"],
                best_df["best_reward_so_far"],
                label="best_reward_so_far",
                linewidth=2.0,
            )

    plt.title("StrikeOrder DQN Episode Reward")
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
    保存每个 episode 的步数和更新次数。
    """
    if "num_steps" not in df.columns or "num_updates" not in df.columns:
        print("[Skip] Missing num_steps or num_updates.")
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

    plt.title("StrikeOrder DQN Steps and Updates")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Saved] {output_path}")


def save_cluster_reward_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存不同 cluster 的 reward 分布图。

    这个图用于检查模型是否只在某些 cluster 上表现好，
    或者不同目标群难度差异是否很大。
    """
    if "cluster_id" not in df.columns:
        print("[Skip] Missing cluster_id.")
        return

    plot_df = df[["cluster_id", "total_reward"]].dropna()

    if plot_df.empty:
        print("[Skip] No valid cluster reward data.")
        return

    output_path = output_dir / "cluster_reward_boxplot.png"

    cluster_ids = sorted(plot_df["cluster_id"].unique())
    reward_groups = [
        plot_df[plot_df["cluster_id"] == cluster_id]["total_reward"].values
        for cluster_id in cluster_ids
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        reward_groups,
        tick_labels=[str(int(cid)) for cid in cluster_ids],
    )
    plt.title("StrikeOrder DQN Reward Distribution by Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Saved] {output_path}")


def print_training_summary(df: pd.DataFrame) -> None:
    """输出训练摘要。"""
    print("\n========== StrikeOrder DQN Training Summary ==========")
    print(f"Total episodes: {len(df)}")

    if "cluster_id" in df.columns:
        valid_clusters = df["cluster_id"].dropna().unique()
        print(f"Clusters used: {sorted(valid_clusters.tolist())}")

    if "total_reward" in df.columns:
        print(f"Best reward: {df['total_reward'].max()}")
        print(f"Last reward: {df['total_reward'].iloc[-1]}")
        print(f"Mean reward: {df['total_reward'].mean()}")

    if "total_path_distance" in df.columns:
        valid_distance = df["total_path_distance"].dropna()
        if not valid_distance.empty:
            print(f"Best path distance: {valid_distance.min()}")
            print(f"Last path distance: {valid_distance.iloc[-1]}")
            print(f"Mean path distance: {valid_distance.mean()}")

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

    print("======================================================\n")


def main() -> None:
    """解析命令行参数并生成 StrikeOrder DQN 训练曲线图。"""
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
        y_column="mean_loss",
        output_path=output_dir / "loss_curve.png",
        title="StrikeOrder DQN Mean Loss",
        ylabel="Mean Loss",
        smooth_window=args.smooth_window,
        include_smooth=True,
    )

    save_line_plot(
        df=df,
        y_column="epsilon_after_episode",
        output_path=output_dir / "epsilon_curve.png",
        title="StrikeOrder DQN Epsilon Decay",
        ylabel="Epsilon",
        smooth_window=1,
        include_smooth=False,
    )

    save_line_plot(
        df=df,
        y_column="random_action_ratio",
        output_path=output_dir / "random_action_ratio_curve.png",
        title="StrikeOrder DQN Random Action Ratio",
        ylabel="Random Action Ratio",
        smooth_window=args.smooth_window,
        include_smooth=True,
    )

    save_line_plot(
        df=df,
        y_column="total_path_distance",
        output_path=output_dir / "path_distance_curve.png",
        title="StrikeOrder DQN Total Path Distance",
        ylabel="Total Path Distance",
        smooth_window=args.smooth_window,
        include_smooth=True,
    )

    save_steps_updates_plot(
        df=df,
        output_dir=output_dir,
    )

    save_cluster_reward_boxplot(
        df=df,
        output_dir=output_dir,
    )

    print("[Done] StrikeOrder DQN training figures generated successfully.")


if __name__ == "__main__":
    main()
