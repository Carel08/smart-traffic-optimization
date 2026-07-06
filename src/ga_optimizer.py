"""
Genetic Algorithm optimizer for traffic signal timing plans.

The GA treats the simulator as a black box:
    timing plan -> run simulation -> evaluate metrics -> improve plan

This is useful because the simulation objective is noisy, nonlinear,
and not easily differentiable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DIRECTIONS,
    GA_RANDOM_STATE,
    GA_POPULATION_SIZE,
    GA_GENERATIONS,
    GA_ELITE_COUNT,
    GA_MUTATION_RATE,
    GA_MUTATION_SCALE,
    GA_TOURNAMENT_SIZE,
    GA_MIN_GREEN,
    GA_MAX_GREEN,
    GA_TIME_BANDS,
    GA_FINAL_QUEUE_WEIGHT,
    GA_MAX_QUEUE_WEIGHT,
    GA_PEDESTRIAN_WAIT_WEIGHT,
)
from src.controllers import (
    BaseTrafficController,
    GATimingPlanController,
)
from src.simulator import run_simulation_with_controller


def chromosome_size(num_intersections: int) -> int:
    """
    One NS green-time decision per time band and intersection.
    """
    return len(GA_TIME_BANDS) * num_intersections


def initialize_population(
    population_size: int,
    num_intersections: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random initial population.

    Each gene is an NS green time between GA_MIN_GREEN and GA_MAX_GREEN.
    """
    size = chromosome_size(num_intersections)

    return rng.uniform(
        low=GA_MIN_GREEN,
        high=GA_MAX_GREEN,
        size=(population_size, size),
    )


def decode_chromosome(
    chromosome: np.ndarray,
    num_intersections: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Convert chromosome into a timing plan.

    Output shape:
    {
        "pre_event": {
            0: {"NS": 30, "EW": 20},
            1: {"NS": 25, "EW": 25},
            ...
        },
        ...
    }
    """
    timing_plan = {}

    gene_index = 0

    for band_name, _, _ in GA_TIME_BANDS:
        band_plan = {}

        for intersection_id in range(num_intersections):
            ns_green = float(chromosome[gene_index])

            split = BaseTrafficController._safe_two_direction_split_custom(
                ns_green=ns_green,
                min_green=GA_MIN_GREEN,
                max_green=GA_MAX_GREEN,
            )

            band_plan[intersection_id] = split
            gene_index += 1

        timing_plan[band_name] = band_plan

    return timing_plan


def evaluate_timing_plan(
    chromosome: np.ndarray,
    num_intersections: int,
    demand_df: pd.DataFrame,
    scenario: str = "normal",
    enable_emergency_priority: bool = True,
) -> Tuple[float, Dict[str, object]]:
    """
    Evaluate one chromosome by running the simulator.

    Lower fitness is better.
    """
    timing_plan = decode_chromosome(
        chromosome=chromosome,
        num_intersections=num_intersections,
    )

    controller = GATimingPlanController(
        timing_plan=timing_plan,
    )

    result = run_simulation_with_controller(
        num_intersections=num_intersections,
        demand_df=demand_df,
        controller=controller,
        scenario=scenario,
        enable_emergency_priority=enable_emergency_priority,
    )

    metrics = result["metrics"]

    fitness = (
        metrics.avg_wait_time_seconds
        + GA_FINAL_QUEUE_WEIGHT * metrics.final_queue_length
        + GA_MAX_QUEUE_WEIGHT * metrics.max_queue_length
        + GA_PEDESTRIAN_WAIT_WEIGHT * metrics.pedestrian_avg_wait_seconds
    )

    return float(fitness), result


def tournament_selection(
    population: np.ndarray,
    fitness_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select one parent using tournament selection.
    """
    candidate_indices = rng.choice(
        len(population),
        size=GA_TOURNAMENT_SIZE,
        replace=False,
    )

    best_index = candidate_indices[
        np.argmin(fitness_values[candidate_indices])
    ]

    return population[best_index].copy()


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blend crossover between two parents.
    """
    alpha = rng.uniform(0.0, 1.0, size=parent_a.shape)

    child = alpha * parent_a + (1 - alpha) * parent_b

    return child


def mutate(
    chromosome: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gaussian mutation with clipping to valid green-time bounds.
    """
    mutated = chromosome.copy()

    mutation_mask = rng.random(size=mutated.shape) < GA_MUTATION_RATE

    noise = rng.normal(
        loc=0.0,
        scale=GA_MUTATION_SCALE,
        size=mutated.shape,
    )

    mutated[mutation_mask] += noise[mutation_mask]

    mutated = np.clip(mutated, GA_MIN_GREEN, GA_MAX_GREEN)

    return mutated

def encode_timing_plan(
    timing_plan: Dict[str, Dict[int, Dict[str, float]]],
    num_intersections: int,
) -> np.ndarray:
    """
    Convert timing plan into chromosome.
    """
    genes = []

    for band_name, _, _ in GA_TIME_BANDS:
        for intersection_id in range(num_intersections):
            genes.append(timing_plan[band_name][intersection_id]["NS"])

    return np.array(genes, dtype=float)


def build_equal_seed_chromosome(num_intersections: int) -> np.ndarray:
    """
    Seed chromosome using equal NS/EW green split.
    """
    available = BaseTrafficController.available_green_time()
    ns_green = available / 2

    size = chromosome_size(num_intersections)

    return np.full(size, ns_green, dtype=float)


def build_directional_seed_chromosome(
    num_intersections: int,
    ns_green: float,
) -> np.ndarray:
    """
    Seed chromosome with a directional bias.
    """
    size = chromosome_size(num_intersections)

    ns_green = float(np.clip(ns_green, GA_MIN_GREEN, GA_MAX_GREEN))

    return np.full(size, ns_green, dtype=float)


def build_calibrated_seed_chromosome(
    num_intersections: int,
    demand_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a demand-calibrated seed chromosome by time band.

    For each band and intersection:
        NS green = available_green * NS demand share
    """
    genes = []
    available = BaseTrafficController.available_green_time()

    for _, start_step, end_step in GA_TIME_BANDS:
        band_df = demand_df[
            (demand_df["time_step"] >= start_step)
            & (demand_df["time_step"] <= end_step)
        ]

        for intersection_id in range(num_intersections):
            intersection_df = band_df[
                band_df["intersection_id"] == intersection_id
            ]

            demand_by_direction = (
                intersection_df
                .groupby("direction")["vehicle_demand"]
                .mean()
                .to_dict()
            )

            ns_demand = demand_by_direction.get("NS", 0.0)
            ew_demand = demand_by_direction.get("EW", 0.0)
            total_demand = ns_demand + ew_demand

            if total_demand <= 0:
                ns_green = available / 2
            else:
                ns_green = available * ns_demand / total_demand

            ns_green = float(np.clip(ns_green, GA_MIN_GREEN, GA_MAX_GREEN))
            genes.append(ns_green)

    return np.array(genes, dtype=float)


def seed_population(
    population: np.ndarray,
    num_intersections: int,
    demand_df: pd.DataFrame,
) -> np.ndarray:
    """
    Replace a few random chromosomes with sensible seeded policies.
    """
    seeded_population = population.copy()

    seeds = [
        build_equal_seed_chromosome(num_intersections),
        build_calibrated_seed_chromosome(num_intersections, demand_df),
        build_directional_seed_chromosome(num_intersections, ns_green=35.0),
        build_directional_seed_chromosome(num_intersections, ns_green=15.0),
    ]

    for i, seed in enumerate(seeds):
        if i < len(seeded_population):
            seeded_population[i] = seed

    return seeded_population

def optimize_ga_timing_plan(
    num_intersections: int,
    demand_df: pd.DataFrame,
    population_size: int = GA_POPULATION_SIZE,
    generations: int = GA_GENERATIONS,
    random_state: int = GA_RANDOM_STATE,
    scenario: str = "normal",
    enable_emergency_priority: bool = True,
) -> Dict[str, object]:
    """
    Run GA optimization.

    Returns:
        best timing plan, best result, and generation history.
    """
    rng = np.random.default_rng(random_state)

    population = initialize_population(
        population_size=population_size,
        num_intersections=num_intersections,
        rng=rng,
    )

    population = seed_population(
        population=population,
        num_intersections=num_intersections,
        demand_df=demand_df,
    )

    best_chromosome = None
    best_fitness = float("inf")
    best_result = None

    generation_rows: List[Dict[str, float]] = []

    for generation in range(generations):
        fitness_values = []
        results = []

        for chromosome in population:
            fitness, result = evaluate_timing_plan(
                chromosome=chromosome,
                num_intersections=num_intersections,
                demand_df=demand_df,
                scenario=scenario,
                enable_emergency_priority=enable_emergency_priority,
            )

            fitness_values.append(fitness)
            results.append(result)

        fitness_values = np.array(fitness_values)

        generation_best_index = int(np.argmin(fitness_values))
        generation_best_fitness = float(fitness_values[generation_best_index])
        generation_mean_fitness = float(np.mean(fitness_values))

        generation_best_result = results[generation_best_index]
        generation_best_metrics = generation_best_result["metrics"]

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_chromosome = population[generation_best_index].copy()
            best_result = generation_best_result

        generation_rows.append(
            {
                "generation": generation,
                "best_fitness": generation_best_fitness,
                "mean_fitness": generation_mean_fitness,
                "best_avg_wait_seconds": (
                    generation_best_metrics.avg_wait_time_seconds
                ),
                "best_throughput": generation_best_metrics.throughput,
                "best_final_queue": generation_best_metrics.final_queue_length,
                "best_max_queue": generation_best_metrics.max_queue_length,
            }
        )

        # Elitism: carry best chromosomes forward
        elite_indices = np.argsort(fitness_values)[:GA_ELITE_COUNT]
        elites = population[elite_indices].copy()

        next_population = [elite for elite in elites]

        while len(next_population) < population_size:
            parent_a = tournament_selection(
                population=population,
                fitness_values=fitness_values,
                rng=rng,
            )

            parent_b = tournament_selection(
                population=population,
                fitness_values=fitness_values,
                rng=rng,
            )

            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng)

            next_population.append(child)

        population = np.array(next_population)

    best_timing_plan = decode_chromosome(
        chromosome=best_chromosome,
        num_intersections=num_intersections,
    )

    generation_history = pd.DataFrame(generation_rows)

    return {
        "best_fitness": best_fitness,
        "best_chromosome": best_chromosome,
        "best_timing_plan": best_timing_plan,
        "best_result": best_result,
        "generation_history": generation_history,
    }