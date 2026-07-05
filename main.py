"""
CLI entry point for the Smart Traffic Light Optimization project.

This file will later run the full simulation benchmark.
For Phase 1, it only confirms that the project is wired correctly.
"""

import argparse

from src.data_generator import generate_traffic_demand
from src.predictor import (
    train_and_evaluate_predictor,
    add_predictions_to_demand,
)

from src.config import DEFAULT_TRAINING_DAYS
from src.simulator import (
    run_fixed_equal_simulation,
    run_fixed_calibrated_simulation,
    run_adaptive_simulation,
)
from src.config import (
    DEFAULT_NUM_INTERSECTIONS,
    MAX_NUM_INTERSECTIONS,
    DEFAULT_SIMULATION_MINUTES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart City Traffic Light Optimization"
    )

    parser.add_argument(
        "--num-intersections",
        type=int,
        default=DEFAULT_NUM_INTERSECTIONS,
        help="Number of intersections to simulate.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "rain", "accident", "pedestrian", "emergency", "combined"],
        help="Traffic scenario to simulate.",
    )

    parser.add_argument(
        "--controller",
        type=str,
        default="fixed_equal",
        choices=[
            "fixed_equal",
            "fixed_calibrated",
            "adaptive",
            "ga",
            "rl_experimental",
        ],
        help="Traffic light controller to use.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_SIMULATION_MINUTES,
        help="Simulation duration in minutes.",
    )

    parser.add_argument(
        "--preview-data",
        action="store_true",
        help="Preview generated synthetic traffic demand.",
    )

    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train and evaluate the ML traffic demand predictor.",
    )

    parser.add_argument(
        "--training-days",
        type=int,
        default=DEFAULT_TRAINING_DAYS,
        help="Number of synthetic historical days to generate for ML training.",
    )


    return parser.parse_args()


def validate_args(args):
    if args.num_intersections < 1:
        raise ValueError("num_intersections must be at least 1.")

    if args.num_intersections > MAX_NUM_INTERSECTIONS:
        raise ValueError(
            f"num_intersections cannot exceed {MAX_NUM_INTERSECTIONS}."
        )


def main():
    args = parse_args()
    validate_args(args)

    print("Smart City Traffic Light Optimization")
    print("=" * 50)
    print(f"Intersections : {args.num_intersections}")
    print(f"Scenario      : {args.scenario}")
    print(f"Controller    : {args.controller}")
    print(f"Duration      : {args.duration} minutes")
    print("=" * 50)
    print("Project configuration loaded successfully.")

    if args.preview_data:
        demand_df = generate_traffic_demand(
            num_intersections=args.num_intersections,
            simulation_minutes=args.duration,
            scenario=args.scenario,
        )

        print("\nGenerated demand preview:")
        print(demand_df.head(12))

        print("\nDemand summary:")
        summary = (
            demand_df
            .groupby(["intersection_id", "direction"])["vehicle_demand"]
            .agg(["mean", "max", "sum"])
            .reset_index()
        )
        print(summary)

        output_path = "outputs/generated_demand_preview.csv"
        demand_df.to_csv(output_path, index=False)
        print(f"\nSaved generated demand to {output_path}")

    ml_result = None

    if args.train_model or args.controller == "adaptive":
        print("\nTraining ML demand predictor...")

        training_minutes = args.training_days * 24 * 60

        training_df = generate_traffic_demand(
            num_intersections=args.num_intersections,
            simulation_minutes=training_minutes,
            scenario=args.scenario,
        )

        ml_result = train_and_evaluate_predictor(
            demand_df=training_df,
            test_size=0.2,
        )

        baseline_eval = ml_result["baseline_evaluation"]
        model_eval = ml_result["model_evaluation"]
        feature_importance = ml_result["feature_importance"]

        print("\nML Prediction Results")
        print("-" * 50)
        print(
            f"{baseline_eval.model_name:25s} "
            f"MAE={baseline_eval.mae:.3f} "
            f"RMSE={baseline_eval.rmse:.3f} "
            f"R2={baseline_eval.r2:.3f}"
        )
        print(
            f"{model_eval.model_name:25s} "
            f"MAE={model_eval.mae:.3f} "
            f"RMSE={model_eval.rmse:.3f} "
            f"R2={model_eval.r2:.3f}"
        )

        print("\nTop 10 Feature Importances")
        print("-" * 50)
        print(feature_importance.head(10))

        feature_importance_path = "outputs/feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)

        print(f"\nSaved feature importance to {feature_importance_path}")

    print("\nGenerating demand for simulation...")
    demand_df = generate_traffic_demand(
        num_intersections=args.num_intersections,
        simulation_minutes=args.duration,
        scenario=args.scenario,
    )
    if args.controller == "adaptive":
        print("Adding ML predictions to simulation demand...")
        demand_df = add_predictions_to_demand(
            demand_df=demand_df,
            predictor=ml_result["predictor"],
        )


    if args.controller == "fixed_equal":
        print("Running fixed equal timing simulation...")
        result = run_fixed_equal_simulation(
            num_intersections=args.num_intersections,
            demand_df=demand_df,
        )
    elif args.controller == "fixed_calibrated":
        print("Running fixed calibrated timing simulation...")
        result = run_fixed_calibrated_simulation(
            num_intersections=args.num_intersections,
            demand_df=demand_df,
        )
    elif args.controller == "adaptive":
        print("Running adaptive pressure controller simulation...")
        result = run_adaptive_simulation(
            num_intersections=args.num_intersections,
            demand_df=demand_df,
        )
    else:
        raise NotImplementedError(
            f"Controller '{args.controller}' will be implemented in a later phase."
        )

    metrics = result["metrics"]
    history = result["history"]

    print(f"\nSimulation Results: {args.controller}")
    print("-" * 50)
    print(f"Total vehicle arrivals       : {metrics.total_arrivals}")
    print(f"Vehicle throughput           : {metrics.throughput}")
    print(f"Final vehicle queue          : {metrics.final_queue_length}")
    print(f"Average wait time, seconds   : {metrics.avg_wait_time_seconds:.2f}")
    print(f"Total wait time, seconds     : {metrics.total_wait_time_seconds:.2f}")
    print(f"Max vehicle queue            : {metrics.max_queue_length}")
    print(f"Average vehicle queue        : {metrics.avg_queue_length:.2f}")

    print("\nPedestrian Metrics")
    print("-" * 50)
    print(f"Pedestrian arrivals          : {metrics.pedestrian_total_arrivals}")
    print(f"Pedestrian throughput        : {metrics.pedestrian_throughput}")
    print(f"Final pedestrian queue       : {metrics.pedestrian_final_queue}")
    print(f"Max pedestrian queue         : {metrics.pedestrian_max_queue}")
    print(f"Average pedestrian queue     : {metrics.pedestrian_avg_queue:.2f}")
    print(f"Average pedestrian wait, sec : {metrics.pedestrian_avg_wait_seconds:.2f}")
    print(f"Pedestrian wait, seconds     : {metrics.pedestrian_total_wait_seconds:.2f}")

    history_output_path = f"outputs/{args.controller}_history.csv"
    history.to_csv(history_output_path, index=False)

    print(f"\nSaved simulation history to {history_output_path}")
if __name__ == "__main__":
    main()