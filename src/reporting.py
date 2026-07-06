"""
Reporting utilities for final project outputs.

Creates clean benchmark tables and saves charts for README/Streamlit/demo use.
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def calculate_improvement(
    baseline_wait: float,
    model_wait: float,
) -> float:
    if baseline_wait <= 0:
        return 0.0

    return (baseline_wait - model_wait) / baseline_wait * 100


def build_final_results_table(
    results: Dict[str, Dict[str, object]],
    baseline_controller: str = "fixed_equal",
) -> pd.DataFrame:
    """
    Convert controller results into a clean summary table.
    """
    baseline_wait = (
        results[baseline_controller]["metrics"].avg_wait_time_seconds
    )

    rows = []

    for controller_name, result in results.items():
        metrics = result["metrics"]

        rows.append(
            {
                "controller": controller_name,
                "avg_wait_seconds": metrics.avg_wait_time_seconds,
                "throughput": metrics.throughput,
                "final_queue": metrics.final_queue_length,
                "max_queue": metrics.max_queue_length,
                "avg_queue": metrics.avg_queue_length,
                "pedestrian_avg_wait_seconds": (
                    metrics.pedestrian_avg_wait_seconds
                ),
                "pedestrian_target_met": metrics.pedestrian_target_met,
                "emergency_delay_seconds": (
                    metrics.emergency_delay_seconds
                ),
                "improvement_vs_fixed_equal_pct": calculate_improvement(
                    baseline_wait=baseline_wait,
                    model_wait=metrics.avg_wait_time_seconds,
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    return (
        summary_df
        .sort_values("avg_wait_seconds", ascending=True)
        .reset_index(drop=True)
    )


def save_final_results_charts(
    summary_df: pd.DataFrame,
    output_dir: str = "outputs/charts",
) -> None:
    """
    Save simple charts for README and presentation use.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Average wait chart
    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["controller"],
        summary_df["avg_wait_seconds"],
    )
    plt.title("Average Vehicle Wait Time by Controller")
    plt.xlabel("Controller")
    plt.ylabel("Average Wait Time (seconds)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_wait_by_controller.png", dpi=150)
    plt.close()

    # Improvement chart
    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["controller"],
        summary_df["improvement_vs_fixed_equal_pct"],
    )
    plt.title("Improvement vs Fixed Equal Timing")
    plt.xlabel("Controller")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_vs_fixed_equal.png", dpi=150)
    plt.close()

    # Final queue chart
    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["controller"],
        summary_df["final_queue"],
    )
    plt.title("Final Queue Length by Controller")
    plt.xlabel("Controller")
    plt.ylabel("Final Vehicle Queue")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_queue_by_controller.png", dpi=150)
    plt.close()