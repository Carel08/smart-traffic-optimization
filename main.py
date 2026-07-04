"""
CLI entry point for the Smart Traffic Light Optimization project.

This file will later run the full simulation benchmark.
For Phase 1, it only confirms that the project is wired correctly.
"""

import argparse

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


if __name__ == "__main__":
    main()