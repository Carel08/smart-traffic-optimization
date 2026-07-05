"""
Traffic signal controllers.

Controllers decide how much green time each direction receives.

includes:
- FixedEqualController
- FixedCalibratedController
- AdaptiveOptimizerController

Later phases will add:

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
    ADAPTIVE_QUEUE_WEIGHT,
    ADAPTIVE_PREDICTION_WEIGHT,
    ADAPTIVE_PEDESTRIAN_WEIGHT,
    ADAPTIVE_EPSILON,
    ADAPTIVE_MIN_GREEN,
    ADAPTIVE_MAX_GREEN,
    ADAPTIVE_PRESSURE_EXPONENT,
    ADAPTIVE_CAPACITY_AWARE,
)


GreenTimes = Dict[int, Dict[str, float]]


@dataclass
class ControllerDecision:
    """
    Stores controller decisions for one simulation step.

    allow_actuated_reallocation:
        If True, the simulator may reallocate unused green time within
        the cycle. This should be enabled only for smart/actuated controllers,
        not fixed timing baselines.
    """

    green_times: GreenTimes
    controller_name: str
    allow_actuated_reallocation: bool = False


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

    @staticmethod
    def _safe_two_direction_split_custom(
            ns_green: float,
            min_green: float,
            max_green: float,
    ) -> Dict[str, float]:
        """
        Convert proposed NS green time into a safe NS/EW split
        using custom min/max green constraints.

        """
        available = BaseTrafficController.available_green_time()

        feasible_max = min(max_green, available - min_green)

        ns_green = max(min_green, min(ns_green, feasible_max))
        ew_green = available - ns_green

        # Safety fallback in case custom inputs are inconsistent.
        ew_green = max(min_green, ew_green)

        # Rebalance if needed.
        total = ns_green + ew_green
        if total > available:
            scale = available / total
            ns_green *= scale
            ew_green *= scale

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
            allow_actuated_reallocation=False,
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
            allow_actuated_reallocation=False,
        )

class AdaptivePressureController(BaseTrafficController):
    """
    Adaptive controller using queue pressure and predicted demand.

    The controller allocates green time per intersection based on:

        pressure = queue_weight * current_queue
                 + prediction_weight * predicted_demand

    This is a practical max-pressure style controller.
    """

    name = "adaptive"

    def __init__(
        self,
        queue_weight: float = ADAPTIVE_QUEUE_WEIGHT,
        prediction_weight: float = ADAPTIVE_PREDICTION_WEIGHT,
        pedestrian_weight: float = ADAPTIVE_PEDESTRIAN_WEIGHT,
    ):
        self.queue_weight = queue_weight
        self.prediction_weight = prediction_weight
        self.pedestrian_weight = pedestrian_weight

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        """
        Return adaptive green times for each intersection.

        Uses:
        - current simulator queues
        - predicted vehicle demand if available
        - current demand as fallback
        - pedestrian pressure as a small balancing term
        """
        if simulator_state is None:
            simulator_state = {}

        queues = simulator_state.get("queues", {})
        pedestrian_queues = simulator_state.get("pedestrian_queues", {})

        green_times = {}

        for intersection_id in range(num_intersections):
            intersection_rows = step_demand[
                step_demand["intersection_id"] == intersection_id
            ]

            demand_by_direction = self._predicted_demand_by_direction(
                intersection_rows
            )

            ns_queue = queues.get(intersection_id, {}).get("NS", 0)
            ew_queue = queues.get(intersection_id, {}).get("EW", 0)

            pedestrian_queue = pedestrian_queues.get(intersection_id, 0)

            ns_pressure = (
                self.queue_weight * ns_queue
                + self.prediction_weight * demand_by_direction["NS"]
            )

            ew_pressure = (
                self.queue_weight * ew_queue
                + self.prediction_weight * demand_by_direction["EW"]
            )

            # Pedestrian pressure slightly reduces aggressive vehicle allocation.
            # For Phase 6 this is intentionally light-touch; proper pedestrian
            # optimization comes later.
            pedestrian_penalty = self.pedestrian_weight * pedestrian_queue

            ns_pressure = max(0.0, ns_pressure - pedestrian_penalty / 2)
            ew_pressure = max(0.0, ew_pressure - pedestrian_penalty / 2)

            total_pressure = ns_pressure + ew_pressure + ADAPTIVE_EPSILON

            available = self.available_green_time()

            ns_share = ns_pressure / total_pressure
            ns_green = available * ns_share

            green_times[intersection_id] = self._safe_two_direction_split(
                ns_green=ns_green
            )

        return ControllerDecision(
            green_times=green_times,
            controller_name=self.name,
            allow_actuated_reallocation=True,
        )

    @staticmethod
    def _predicted_demand_by_direction(
        intersection_rows: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Get predicted demand by direction.

        If ML predictions exist, use predicted_vehicle_demand.
        Otherwise, fall back to current generated vehicle_demand.
        """
        demand_by_direction = {
            direction: 0.0
            for direction in DIRECTIONS
        }

        prediction_column = (
            "predicted_vehicle_demand"
            if "predicted_vehicle_demand" in intersection_rows.columns
            else "vehicle_demand"
        )

        for _, row in intersection_rows.iterrows():
            direction = row["direction"]
            demand_by_direction[direction] = float(row[prediction_column])

        return demand_by_direction

class EnhancedAdaptiveController(BaseTrafficController):
    """
    Enhanced adaptive controller.

    Improvements over basic AdaptivePressureController:
    - pressure exponent for more decisive green allocation
    - adaptive min/max green limits
    - optional capacity-aware pressure adjustment

    """

    name = "enhanced_adaptive"

    def __init__(
        self,
        queue_weight: float = ADAPTIVE_QUEUE_WEIGHT,
        prediction_weight: float = ADAPTIVE_PREDICTION_WEIGHT,
        pedestrian_weight: float = ADAPTIVE_PEDESTRIAN_WEIGHT,
        pressure_exponent: float = ADAPTIVE_PRESSURE_EXPONENT,
        adaptive_min_green: float = ADAPTIVE_MIN_GREEN,
        adaptive_max_green: float = ADAPTIVE_MAX_GREEN,
        capacity_aware: bool = ADAPTIVE_CAPACITY_AWARE,
    ):
        self.queue_weight = queue_weight
        self.prediction_weight = prediction_weight
        self.pedestrian_weight = pedestrian_weight
        self.pressure_exponent = pressure_exponent
        self.adaptive_min_green = adaptive_min_green
        self.adaptive_max_green = adaptive_max_green
        self.capacity_aware = capacity_aware

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        if simulator_state is None:
            simulator_state = {}

        queues = simulator_state.get("queues", {})
        pedestrian_queues = simulator_state.get("pedestrian_queues", {})

        green_times = {}

        for intersection_id in range(num_intersections):
            intersection_rows = step_demand[
                step_demand["intersection_id"] == intersection_id
            ]

            demand_by_direction = self._predicted_demand_by_direction(
                intersection_rows
            )

            capacity_by_direction = self._capacity_by_direction(
                intersection_rows
            )

            ns_queue = queues.get(intersection_id, {}).get("NS", 0)
            ew_queue = queues.get(intersection_id, {}).get("EW", 0)

            pedestrian_queue = pedestrian_queues.get(intersection_id, 0)

            ns_pressure = (
                self.queue_weight * ns_queue
                + self.prediction_weight * demand_by_direction["NS"]
            )

            ew_pressure = (
                self.queue_weight * ew_queue
                + self.prediction_weight * demand_by_direction["EW"]
            )

            # Light pedestrian penalty to avoid totally ignoring crossings.
            pedestrian_penalty = self.pedestrian_weight * pedestrian_queue

            ns_pressure = max(0.0, ns_pressure - pedestrian_penalty / 2)
            ew_pressure = max(0.0, ew_pressure - pedestrian_penalty / 2)

            if self.capacity_aware:
                # Capacity-aware pressure:
                ns_capacity = max(capacity_by_direction["NS"], ADAPTIVE_EPSILON)
                ew_capacity = max(capacity_by_direction["EW"], ADAPTIVE_EPSILON)

                ns_pressure = ns_pressure / ns_capacity
                ew_pressure = ew_pressure / ew_capacity

            ns_pressure = max(ns_pressure, ADAPTIVE_EPSILON)
            ew_pressure = max(ew_pressure, ADAPTIVE_EPSILON)

            ns_adjusted = ns_pressure ** self.pressure_exponent
            ew_adjusted = ew_pressure ** self.pressure_exponent

            total_adjusted_pressure = ns_adjusted + ew_adjusted

            available = self.available_green_time()

            ns_share = ns_adjusted / total_adjusted_pressure
            ns_green = available * ns_share

            green_times[intersection_id] = self._safe_two_direction_split_custom(
                ns_green=ns_green,
                min_green=self.adaptive_min_green,
                max_green=self.adaptive_max_green,
            )

        return ControllerDecision(
            green_times=green_times,
            controller_name=self.name,
            allow_actuated_reallocation=True,
        )

    @staticmethod
    def _predicted_demand_by_direction(
        intersection_rows: pd.DataFrame,
    ) -> Dict[str, float]:
        demand_by_direction = {
            direction: 0.0
            for direction in DIRECTIONS
        }

        prediction_column = (
            "predicted_vehicle_demand"
            if "predicted_vehicle_demand" in intersection_rows.columns
            else "vehicle_demand"
        )

        for _, row in intersection_rows.iterrows():
            direction = row["direction"]
            demand_by_direction[direction] = float(row[prediction_column])

        return demand_by_direction

    @staticmethod
    def _capacity_by_direction(
        intersection_rows: pd.DataFrame,
    ) -> Dict[str, float]:
        capacity_by_direction = {
            direction: 1.0
            for direction in DIRECTIONS
        }

        for _, row in intersection_rows.iterrows():
            direction = row["direction"]

            weather_capacity = float(row["weather_capacity_multiplier"])
            accident_active = int(row["accident_active"])
            accident_capacity = 0.5 if accident_active == 1 else 1.0

            capacity_by_direction[direction] = (
                weather_capacity * accident_capacity
            )

        return capacity_by_direction

class GATimingPlanController(BaseTrafficController):
    """
    Controller using a GA-optimized timing plan.

    The timing plan provides a green-time split for each:
    - time band
    - intersection

    This is an offline black-box optimized policy.
    """

    name = "ga_timing_plan"

    def __init__(self, timing_plan: Dict[str, GreenTimes]):
        self.timing_plan = timing_plan

    def get_green_times(
        self,
        time_step: int,
        num_intersections: int,
        step_demand: pd.DataFrame,
        simulator_state: Dict | None = None,
    ) -> ControllerDecision:
        band_name = self._time_band_name(time_step)

        if band_name not in self.timing_plan:
            raise ValueError(f"No GA timing plan found for band: {band_name}")

        green_times = self.timing_plan[band_name]

        return ControllerDecision(
            green_times=green_times,
            controller_name=self.name,
            allow_actuated_reallocation=True,
        )

    @staticmethod
    def _time_band_name(time_step: int) -> str:
        from src.config import GA_TIME_BANDS

        for band_name, start_step, end_step in GA_TIME_BANDS:
            if start_step <= time_step <= end_step:
                return band_name

        return GA_TIME_BANDS[-1][0]