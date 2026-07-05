"""
CLI entry point for the Smart Traffic Light Optimization project.

This file will later run the full simulation benchmark.
For Phase 1, it only confirms that the project is wired correctly.
"""

import argparse

from src.data_generator import generate_traffic_demand
from src.simulator import run_fixed_equal_simulation

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
    print("Phase 1 setup successful. Simulation engine will be added next.")

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

    print("\nGenerating demand for simulation...")
    demand_df = generate_traffic_demand(
        num_intersections=args.num_intersections,
        simulation_minutes=args.duration,
        scenario=args.scenario,
    )

    print("Running fixed equal timing simulation...")
    result = run_fixed_equal_simulation(
        num_intersections=args.num_intersections,
        demand_df=demand_df,
    )

    metrics = result["metrics"]
    history = result["history"]

    print("\nSimulation Results: Fixed Equal Timing")
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

    history_output_path = "outputs/fixed_equal_history.csv"
    history.to_csv(history_output_path, index=False)

    print(f"\nSaved simulation history to {history_output_path}")

if __name__ == "__main__":
    main()