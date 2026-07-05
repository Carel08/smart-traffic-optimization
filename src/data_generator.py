"""
Synthetic traffic demand generation.

This module creates realistic traffic demand for the simulator:
- time-of-day peak patterns
- weather effects
- intersection-level variation
- direction-level variation
- pedestrian demand
- accidents/incidents

The output is intentionally tabular so it can be used by:
- the simulator
- the ML predictor
- the Streamlit demo
- benchmark reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import DIRECTIONS, DEFAULT_RANDOM_SEED


@dataclass
class WeatherState:
    """
    Represents weather assumptions for a time step.

    rain_level:
        0 = clear
        1 = light rain
        2 = heavy rain
    """
    weather: str
    rain_level: int
    demand_multiplier: float
    capacity_multiplier: float
    accident_multiplier: float


class TrafficDemandGenerator:
    """
    Generates synthetic traffic demand for each:
    - time step
    - intersection
    - direction

    We use a Poisson distribution because vehicle arrivals are count data.
    The arrival rate changes with time of day, weather, events, intersection,
    and direction.
    """

    def __init__(
        self,
        num_intersections: int,
        simulation_minutes: int,
        start_hour: int = 7,
        seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.num_intersections = num_intersections
        self.simulation_minutes = simulation_minutes
        self.start_hour = start_hour
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, scenario: str = "normal") -> pd.DataFrame:
        """
        Generate traffic demand.

        Parameters
        ----------
        scenario:
            One of:
            - normal
            - rain
            - accident
            - pedestrian
            - emergency
            - combined

        Returns
        -------
        pd.DataFrame
            One row per time_step, intersection, and direction.
        """
        rows = []

        for time_step in range(self.simulation_minutes):
            hour = self._hour_of_day(time_step)
            weather_state = self._weather_for_step(time_step, scenario)
            accident_info = self._accident_for_step(time_step, scenario)

            for intersection_id in range(self.num_intersections):
                intersection_multiplier = self._intersection_multiplier(
                    intersection_id
                )

                for direction in DIRECTIONS:
                    direction_multiplier = self._direction_multiplier(
                        direction=direction,
                        hour=hour,
                    )

                    peak_multiplier = self._peak_time_multiplier(hour)

                    event_multiplier = self._event_multiplier(
                        time_step=time_step,
                        scenario=scenario,
                    )

                    base_rate = 6.0

                    arrival_rate = (
                        base_rate
                        * peak_multiplier
                        * weather_state.demand_multiplier
                        * event_multiplier
                        * intersection_multiplier
                        * direction_multiplier
                    )

                    vehicle_demand = self.rng.poisson(arrival_rate)

                    pedestrian_demand = self._pedestrian_demand(
                        hour=hour,
                        scenario=scenario,
                    )

                    accident_active = int(
                        accident_info["active"]
                        and intersection_id == accident_info["intersection_id"]
                        and direction == accident_info["direction"]
                    )

                    rows.append(
                        {
                            "time_step": time_step,
                            "hour": hour,
                            "intersection_id": intersection_id,
                            "direction": direction,
                            "weather": weather_state.weather,
                            "rain_level": weather_state.rain_level,
                            "weather_demand_multiplier": weather_state.demand_multiplier,
                            "weather_capacity_multiplier": weather_state.capacity_multiplier,
                            "accident_multiplier": weather_state.accident_multiplier,
                            "event_multiplier": event_multiplier,
                            "peak_multiplier": peak_multiplier,
                            "intersection_multiplier": intersection_multiplier,
                            "direction_multiplier": direction_multiplier,
                            "vehicle_demand": int(vehicle_demand),
                            "pedestrian_demand": int(pedestrian_demand),
                            "accident_active": accident_active,
                        }
                    )

        df = pd.DataFrame(rows)
        return df

    def _hour_of_day(self, time_step: int) -> int:
        """
        Convert simulation minute into hour of day.

        Example:
        start_hour = 7
        time_step = 90 minutes
        hour = 8
        """
        return int((self.start_hour + time_step // 60) % 24)

    def _peak_time_multiplier(self, hour: int) -> float:
        """
        Time-of-day traffic demand pattern.

        This creates rush-hour behavior.
        """
        if 0 <= hour <= 5:
            return 0.4

        if 6 <= hour <= 9:
            return 2.0

        if 10 <= hour <= 15:
            return 1.0

        if 16 <= hour <= 18:
            return 2.2

        if 19 <= hour <= 21:
            return 1.2

        return 0.7

    def _direction_multiplier(self, direction: str, hour: int) -> float:
        """
        Directional demand pattern.

        Morning and evening flows are directionally biased.
        For this simplified model:
        - Morning: NS is heavier
        - Evening: EW is heavier
        """
        if 6 <= hour <= 9:
            return 1.25 if direction == "NS" else 0.90

        if 16 <= hour <= 18:
            return 1.25 if direction == "EW" else 0.90

        return 1.0

    def _intersection_multiplier(self, intersection_id: int) -> float:
        """
        Some intersections are naturally busier than others.

        This gives the network spatial variation.
        """
        base = 1.0

        # Make central intersections slightly busier.
        center = (self.num_intersections - 1) / 2
        distance_from_center = abs(intersection_id - center)

        # Closer to center => higher multiplier.
        multiplier = base + 0.25 * (1 - distance_from_center / max(center, 1))

        # Add small deterministic random variation.
        local_noise = self.rng.normal(0, 0.05)

        return max(0.7, multiplier + local_noise)

    def _weather_for_step(self, time_step: int, scenario: str) -> WeatherState:
        """
        Generate weather assumptions.

        Weather affects:
        - vehicle demand
        - road capacity
        - accident risk
        """
        if scenario in ["rain", "combined"]:
            # Force rain during the middle part of the demo.
            if 20 <= time_step <= 80:
                return WeatherState(
                    weather="heavy_rain",
                    rain_level=2,
                    demand_multiplier=1.25,
                    capacity_multiplier=0.75,
                    accident_multiplier=2.0,
                )

            return WeatherState(
                weather="light_rain",
                rain_level=1,
                demand_multiplier=1.10,
                capacity_multiplier=0.90,
                accident_multiplier=1.3,
            )

        # Normal scenario still has small random weather variation.
        random_weather = self.rng.choice(
            ["clear", "light_rain"],
            p=[0.85, 0.15],
        )

        if random_weather == "light_rain":
            return WeatherState(
                weather="light_rain",
                rain_level=1,
                demand_multiplier=1.10,
                capacity_multiplier=0.90,
                accident_multiplier=1.3,
            )

        return WeatherState(
            weather="clear",
            rain_level=0,
            demand_multiplier=1.00,
            capacity_multiplier=1.00,
            accident_multiplier=1.0,
        )

    def _event_multiplier(self, time_step: int, scenario: str) -> float:
        """
        Simulate an event surge, such as a stadium event or school release.

        This gives the model something non-smooth to react to.
        """
        if scenario in ["combined"]:
            if 45 <= time_step <= 75:
                return 1.5

        return 1.0

    def _pedestrian_demand(self, hour: int, scenario: str) -> int:
        """
        Generate pedestrian arrivals.

        Pedestrian arrivals also increase during peak times.
        """
        base_rate = 1.0

        if scenario in ["pedestrian", "combined"]:
            base_rate = 3.0

        if 7 <= hour <= 9:
            rate = base_rate * 2.0
        elif 12 <= hour <= 14:
            rate = base_rate * 1.6
        elif 16 <= hour <= 18:
            rate = base_rate * 2.2
        else:
            rate = base_rate

        return self.rng.poisson(rate)

    def _accident_for_step(self, time_step: int, scenario: str) -> Dict[str, Optional[object]]:
        """
        Deterministic accident window for demos.

        Later we can add stochastic accident generation, but deterministic
        scenarios are better for live demos because they are reproducible.
        """
        if scenario in ["accident", "combined"]:
            if 50 <= time_step <= 90:
                return {
                    "active": True,
                    "intersection_id": min(2, self.num_intersections - 1),
                    "direction": "EW",
                }

        return {
            "active": False,
            "intersection_id": None,
            "direction": None,
        }


def generate_traffic_demand(
    num_intersections: int,
    simulation_minutes: int,
    scenario: str = "normal",
    start_hour: int = 7,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Convenience function used by main.py, Streamlit, and future tests.
    """
    generator = TrafficDemandGenerator(
        num_intersections=num_intersections,
        simulation_minutes=simulation_minutes,
        start_hour=start_hour,
        seed=seed,
    )
    return generator.generate(scenario=scenario)