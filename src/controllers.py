"""
Traffic signal controllers.

Controllers decide how much green time each direction receives.

Phase 4 includes:
- FixedEqualController
- FixedCalibratedController

Later phases will add:
- AdaptiveOptimizerController
- GAOptimizerController
- RLExperimentalController
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.config import (
    DIRECTIONS,
    CYCLE_TIME,
    YELLOW_TIME,
    ALL_RED_TIME,
    MIN_GREEN,
    MAX_GREEN,
)


GreenTimes = Dict[int, Dict[str, float]]


@dataclass
class ControllerDecision:
    """
    Stores green-time decisions for all intersections.

    Example:
    {
        0: {"NS": 25.0, "EW": 25.0},
        1: {"NS": 30.0, "EW": 20.0}
    }
    """

    green_times: GreenTimes
    controller_name: str


class BaseTrafficController:
    """
    Base class for all traffic-light controllers.
    """

    name = "base_controller"

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        """
        Return green time allocation per intersection and direction.
        """
        raise NotImplementedError("Controller must implement get_green_times().")

    @staticmethod
    def available_green_time() -> float:
        """
        Green time left after safety transitions.

        With:
        - cycle = 60
        - yellow = 4 per transition
        - all-red = 1 per transition

        available green = 60 - 2*4 - 2*1 = 50 seconds
        """
        return CYCLE_TIME - 2 * YELLOW_TIME - 2 * ALL_RED_TIME

    @staticmethod
    def _safe_two_direction_split(ns_green: float) -> Dict[str, float]:
        """
        Convert proposed NS green time into a safe NS/EW split.

        Ensures:
        - NS >= MIN_GREEN
        - EW >= MIN_GREEN
        - NS + EW = available green time
        """
        available = BaseTrafficController.available_green_time()

        # Maximum feasible green for one direction while still giving
        # the other direction at least MIN_GREEN.
        feasible_max = min(MAX_GREEN, available - MIN_GREEN)

        ns_green = max(MIN_GREEN, min(ns_green, feasible_max))
        ew_green = available - ns_green

        return {
            "NS": float(ns_green),
            "EW": float(ew_green),
        }


class FixedEqualController(BaseTrafficController):
    """
    Fixed baseline with equal green time for NS and EW.
    """

    name = "fixed_equal"

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        available = self.available_green_time()
        green_per_direction = available / len(DIRECTIONS)

        green_times = {
            intersection_id: {
                direction: float(green_per_direction)
                for direction in DIRECTIONS
            }
            for intersection_id in range(num_intersections)
        }

        return ControllerDecision(
            green_times=green_times,
            controller_name=self.name,
        )


class FixedCalibratedController(BaseTrafficController):
    """
    Fixed baseline calibrated using average historical demand.

    Instead of giving NS and EW equal time, this controller assigns more
    green time to the direction with higher average demand.

    This is a fairer baseline than fixed equal timing.
    """

    name = "fixed_calibrated"

    def __init__(self, calibration_demand: pd.DataFrame):
        self.calibration_demand = calibration_demand.copy()
        self.green_time_lookup = self._build_green_time_lookup()

    def _build_green_time_lookup(self) -> GreenTimes:
        """
        Calculate one fixed green-time split per intersection.
        """
        available = self.available_green_time()

        grouped = (
            self.calibration_demand
            .groupby(["intersection_id", "direction"])["vehicle_demand"]
            .mean()
            .reset_index()
        )

        lookup: GreenTimes = {}

        intersection_ids = sorted(
            self.calibration_demand["intersection_id"].unique()
        )

        for intersection_id in intersection_ids:
            intersection_data = grouped[
                grouped["intersection_id"] == intersection_id
            ]

            demand_by_direction = {
                direction: 0.0
                for direction in DIRECTIONS
            }

            for _, row in intersection_data.iterrows():
                direction = row["direction"]
                demand_by_direction[direction] = float(row["vehicle_demand"])

            total_demand = sum(demand_by_direction.values())

            if total_demand <= 0:
                ns_green = available / 2
            else:
                ns_share = demand_by_direction["NS"] / total_demand
                ns_green = available * ns_share

            lookup[int(intersection_id)] = self._safe_two_direction_split(
                ns_green=ns_green
            )

        return lookup

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        green_times = {}

        for intersection_id in range(num_intersections):
            if intersection_id in self.green_time_lookup:
                green_times[intersection_id] = self.green_time_lookup[
                    intersection_id
                ]
            else:
                # Fallback for safety.
                green_times[intersection_id] = self._safe_two_direction_split(
                    ns_green=self.available_green_time() / 2
                )

        return ControllerDecision(
            green_times=green_times,
            controller_name=self.name,
        )