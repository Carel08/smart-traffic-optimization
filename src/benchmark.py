"""
Benchmarking utilities for traffic signal controllers.

This module compares:
- fixed equal timing
- fixed calibrated timing
- adaptive pressure controller

It also supports simple grid search tuning for the adaptive controller.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config import (
    ADAPTIVE_QUEUE_WEIGHT_GRID,
    ADAPTIVE_PREDICTION_WEIGHT_GRID,
    ADAPTIVE_PEDESTRIAN_WEIGHT_GRID,
    ADAPTIVE_PRESSURE_EXPONENT_GRID,
    ADAPTIVE_MIN_GREEN_GRID,
    SCENARIO_BENCHMARK_SCENARIOS,
    SCENARIO_BENCHMARK_GA_GENERATIONS,
    SCENARIO_BENCHMARK_GA_POPULATION_SIZE,
)
from src.simulator import (
    run_fixed_equal_simulation,
    run_fixed_calibrated_simulation,
    run_adaptive_simulation,
    run_enhanced_adaptive_simulation,
)


from src.data_generator import generate_traffic_demand
from src.ga_optimizer import optimize_ga_timing_plan

def metrics_to_row(
    controller_name: str,
    metrics,
    fixed_equal_avg_wait: float | None = None,
) -> Dict[str, float]:
    """
    Convert Simulation Metrics into a benchmark table row.
    """
    if fixed_equal_avg_wait is None or fixed_equal_avg_wait <= 0:
        improvement_vs_fixed_equal = 0.0
    else:
        improvement_vs_fixed_equal = (
            (fixed_equal_avg_wait - metrics.avg_wait_time_seconds)
            / fixed_equal_avg_wait
            * 100
        )

    return {
        "controller": controller_name,
        "avg_wait_seconds": metrics.avg_wait_time_seconds,
        "throughput": metrics.throughput,
        "final_queue": metrics.final_queue_length,
        "max_queue": metrics.max_queue_length,
        "avg_queue": metrics.avg_queue_length,
        "pedestrian_avg_wait_seconds": metrics.pedestrian_avg_wait_seconds,
        "pedestrian_final_queue": metrics.pedestrian_final_queue,
        "improvement_vs_fixed_equal_pct": improvement_vs_fixed_equal,
    }


def run_controller_benchmark(
    num_intersections: int,
    demand_df: pd.DataFrame,
    adaptive_params: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """
    Run all main controllers on the same demand data.

    This ensures a fair comparison because all controllers face the same
    arrivals, weather, pedestrian demand, and accident scenario.
    """
    if adaptive_params is None:
        adaptive_params = {
            "queue_weight": 1.0,
            "prediction_weight": 1.0,
            "pedestrian_weight": 0.15,
        }

    results = {}
    rows = []

    fixed_equal_result = run_fixed_equal_simulation(
        num_intersections=num_intersections,
        demand_df=demand_df,
    )
    results["fixed_equal"] = fixed_equal_result

    fixed_equal_avg_wait = (
        fixed_equal_result["metrics"].avg_wait_time_seconds
    )

    rows.append(
        metrics_to_row(
            controller_name="fixed_equal",
            metrics=fixed_equal_result["metrics"],
            fixed_equal_avg_wait=fixed_equal_avg_wait,
        )
    )

    fixed_calibrated_result = run_fixed_calibrated_simulation(
        num_intersections=num_intersections,
        demand_df=demand_df,
    )
    results["fixed_calibrated"] = fixed_calibrated_result

    rows.append(
        metrics_to_row(
            controller_name="fixed_calibrated",
            metrics=fixed_calibrated_result["metrics"],
            fixed_equal_avg_wait=fixed_equal_avg_wait,
        )
    )

    adaptive_result = run_adaptive_simulation(
        num_intersections=num_intersections,
        demand_df=demand_df,
        queue_weight=adaptive_params["queue_weight"],
        prediction_weight=adaptive_params["prediction_weight"],
        pedestrian_weight=adaptive_params["pedestrian_weight"],
    )
    results["adaptive"] = adaptive_result

    rows.append(
        metrics_to_row(
            controller_name="adaptive",
            metrics=adaptive_result["metrics"],
            fixed_equal_avg_wait=fixed_equal_avg_wait,
        )
    )

    enhanced_adaptive_result = run_enhanced_adaptive_simulation(
        num_intersections=num_intersections,
        demand_df=demand_df,
    )
    results["enhanced_adaptive"] = enhanced_adaptive_result

    rows.append(
        metrics_to_row(
            controller_name="enhanced_adaptive",
            metrics=enhanced_adaptive_result["metrics"],
            fixed_equal_avg_wait=fixed_equal_avg_wait,
        )
    )

    comparison_df = pd.DataFrame(rows)

    return {
        "comparison": comparison_df,
        "results": results,
    }


def tune_adaptive_controller(
    num_intersections: int,
    demand_df: pd.DataFrame,
    queue_weight_grid: List[float] | None = None,
    prediction_weight_grid: List[float] | None = None,
    pedestrian_weight_grid: List[float] | None = None,
) -> Dict[str, object]:
    """
    Grid-search adaptive controller weights.

    Objective:
        minimize average vehicle wait time

    We keep this simple and explainable.
    """
    if queue_weight_grid is None:
        queue_weight_grid = ADAPTIVE_QUEUE_WEIGHT_GRID

    if prediction_weight_grid is None:
        prediction_weight_grid = ADAPTIVE_PREDICTION_WEIGHT_GRID

    if pedestrian_weight_grid is None:
        pedestrian_weight_grid = ADAPTIVE_PEDESTRIAN_WEIGHT_GRID

    tuning_rows = []
    best_result = None
    best_params = None
    best_avg_wait = float("inf")

    for queue_weight in queue_weight_grid:
        for prediction_weight in prediction_weight_grid:
            for pedestrian_weight in pedestrian_weight_grid:
                result = run_adaptive_simulation(
                    num_intersections=num_intersections,
                    demand_df=demand_df,
                    queue_weight=queue_weight,
                    prediction_weight=prediction_weight,
                    pedestrian_weight=pedestrian_weight,
                )

                metrics = result["metrics"]

                row = {
                    "queue_weight": queue_weight,
                    "prediction_weight": prediction_weight,
                    "pedestrian_weight": pedestrian_weight,
                    "avg_wait_seconds": metrics.avg_wait_time_seconds,
                    "throughput": metrics.throughput,
                    "final_queue": metrics.final_queue_length,
                    "max_queue": metrics.max_queue_length,
                    "pedestrian_avg_wait_seconds": (
                        metrics.pedestrian_avg_wait_seconds
                    ),
                    "pedestrian_final_queue": metrics.pedestrian_final_queue,
                }

                tuning_rows.append(row)

                if metrics.avg_wait_time_seconds < best_avg_wait:
                    best_avg_wait = metrics.avg_wait_time_seconds
                    best_params = {
                        "queue_weight": queue_weight,
                        "prediction_weight": prediction_weight,
                        "pedestrian_weight": pedestrian_weight,
                    }
                    best_result = result

    tuning_df = (
        pd.DataFrame(tuning_rows)
        .sort_values("avg_wait_seconds", ascending=True)
        .reset_index(drop=True)
    )

    return {
        "best_params": best_params,
        "best_result": best_result,
        "tuning_results": tuning_df,
    }

def tune_enhanced_adaptive_controller(
    num_intersections: int,
    demand_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Grid-search enhanced adaptive controller.

    Objective:
        minimize average vehicle wait time
    """
    tuning_rows = []
    best_result = None
    best_params = None
    best_avg_wait = float("inf")

    for queue_weight in ADAPTIVE_QUEUE_WEIGHT_GRID:
        for prediction_weight in ADAPTIVE_PREDICTION_WEIGHT_GRID:
            for pedestrian_weight in ADAPTIVE_PEDESTRIAN_WEIGHT_GRID:
                for pressure_exponent in ADAPTIVE_PRESSURE_EXPONENT_GRID:
                    for adaptive_min_green in ADAPTIVE_MIN_GREEN_GRID:
                        result = run_enhanced_adaptive_simulation(
                            num_intersections=num_intersections,
                            demand_df=demand_df,
                            queue_weight=queue_weight,
                            prediction_weight=prediction_weight,
                            pedestrian_weight=pedestrian_weight,
                            pressure_exponent=pressure_exponent,
                            adaptive_min_green=adaptive_min_green,
                        )

                        metrics = result["metrics"]

                        row = {
                            "queue_weight": queue_weight,
                            "prediction_weight": prediction_weight,
                            "pedestrian_weight": pedestrian_weight,
                            "pressure_exponent": pressure_exponent,
                            "adaptive_min_green": adaptive_min_green,
                            "avg_wait_seconds": metrics.avg_wait_time_seconds,
                            "throughput": metrics.throughput,
                            "final_queue": metrics.final_queue_length,
                            "max_queue": metrics.max_queue_length,
                            "pedestrian_avg_wait_seconds": (
                                metrics.pedestrian_avg_wait_seconds
                            ),
                            "pedestrian_final_queue": metrics.pedestrian_final_queue,
                        }

                        tuning_rows.append(row)

                        if metrics.avg_wait_time_seconds < best_avg_wait:
                            best_avg_wait = metrics.avg_wait_time_seconds
                            best_params = {
                                "queue_weight": queue_weight,
                                "prediction_weight": prediction_weight,
                                "pedestrian_weight": pedestrian_weight,
                                "pressure_exponent": pressure_exponent,
                                "adaptive_min_green": adaptive_min_green,
                            }
                            best_result = result

    tuning_df = (
        pd.DataFrame(tuning_rows)
        .sort_values("avg_wait_seconds", ascending=True)
        .reset_index(drop=True)
    )

    return {
        "best_params": best_params,
        "best_result": best_result,
        "tuning_results": tuning_df,
    }

def run_ga_scenario_benchmark(
    num_intersections: int,
    duration: int,
    scenarios: list[str] | None = None,
    ga_generations: int = SCENARIO_BENCHMARK_GA_GENERATIONS,
    ga_population_size: int = SCENARIO_BENCHMARK_GA_POPULATION_SIZE,
) -> Dict[str, object]:
    """
    Run fixed_equal vs GA across multiple scenarios.

    This is used to test whether the optimized controller achieves
    20%+ improvement under different traffic conditions.
    """
    if scenarios is None:
        scenarios = SCENARIO_BENCHMARK_SCENARIOS

    rows = []
    scenario_results = {}

    for scenario in scenarios:
        print(f"\nRunning scenario benchmark: {scenario}")

        demand_df = generate_traffic_demand(
            num_intersections=num_intersections,
            simulation_minutes=duration,
            scenario=scenario,
        )

        fixed_result = run_fixed_equal_simulation(
            num_intersections=num_intersections,
            demand_df=demand_df,
        )

        fixed_metrics = fixed_result["metrics"]

        ga_result = optimize_ga_timing_plan(
            num_intersections=num_intersections,
            demand_df=demand_df,
            population_size=ga_population_size,
            generations=ga_generations,
        )

        ga_metrics = ga_result["best_result"]["metrics"]

        improvement_pct = (
            (
                fixed_metrics.avg_wait_time_seconds
                - ga_metrics.avg_wait_time_seconds
            )
            / fixed_metrics.avg_wait_time_seconds
            * 100
            if fixed_metrics.avg_wait_time_seconds > 0
            else 0.0
        )

        pass_20_pct = improvement_pct >= 20.0

        row = {
            "scenario": scenario,
            "fixed_equal_avg_wait_seconds": (
                fixed_metrics.avg_wait_time_seconds
            ),
            "ga_avg_wait_seconds": ga_metrics.avg_wait_time_seconds,
            "improvement_pct": improvement_pct,
            "passes_20_pct_target": pass_20_pct,
            "fixed_equal_throughput": fixed_metrics.throughput,
            "ga_throughput": ga_metrics.throughput,
            "fixed_equal_final_queue": fixed_metrics.final_queue_length,
            "ga_final_queue": ga_metrics.final_queue_length,
            "ga_best_fitness": ga_result["best_fitness"],
        }

        rows.append(row)

        scenario_results[scenario] = {
            "fixed_equal": fixed_result,
            "ga": ga_result,
        }

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("improvement_pct", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "comparison": comparison_df,
        "scenario_results": scenario_results,
    }