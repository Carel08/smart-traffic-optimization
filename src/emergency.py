"""
Emergency routing and signal priority.

Emergency routing uses Dijkstra shortest path over a simple intersection graph.
This is deliberately deterministic and explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import math
import networkx as nx


@dataclass
class EmergencyRoute:
    origin: int
    destination: int
    route: List[int]
    route_cost: float


def grid_columns(num_intersections: int) -> int:
    """
    Use a compact grid layout.

    Example:
        10 intersections -> 4 columns approximately.
    """
    return math.ceil(math.sqrt(num_intersections))


def build_intersection_graph(num_intersections: int) -> nx.Graph:
    """
    Build a simple grid graph for intersections.

    Nodes:
        intersection ids

    Edges:
        adjacent intersections in a grid.
    """
    graph = nx.Graph()
    cols = grid_columns(num_intersections)

    for intersection_id in range(num_intersections):
        graph.add_node(intersection_id)

    for intersection_id in range(num_intersections):
        right = intersection_id + 1
        down = intersection_id + cols

        same_row_right = (
            right < num_intersections
            and right // cols == intersection_id // cols
        )

        if same_row_right:
            graph.add_edge(intersection_id, right)

        if down < num_intersections:
            graph.add_edge(intersection_id, down)

    return graph


def infer_movement_direction(
    source: int,
    target: int,
    num_intersections: int,
) -> str:
    """
    Convert movement between graph nodes into simplified signal direction.

    Horizontal movement -> EW
    Vertical movement   -> NS
    """
    cols = grid_columns(num_intersections)

    if abs(source - target) == 1:
        return "EW"

    if abs(source - target) == cols:
        return "NS"

    # Fallback for unusual graph movement.
    return "NS"


def compute_edge_weight(
    source: int,
    target: int,
    simulator_state: Dict,
    base_travel_time: float,
    queue_weight: float,
    accident_penalty: float,
) -> float:
    """
    Emergency route edge cost.

    Cost combines:
    - base travel time
    - queue/congestion delay
    - accident penalty
    """
    queues = simulator_state.get("queues", {})

    source_queue = queues.get(source, {})
    target_queue = queues.get(target, {})

    queue_delay = queue_weight * (
        sum(source_queue.values()) + sum(target_queue.values())
    )

    accident_active = simulator_state.get("accident_active", False)
    accident_intersection = simulator_state.get(
        "accident_intersection",
        None,
    )

    accident_delay = (
        accident_penalty
        if accident_active and target == accident_intersection
        else 0.0
    )

    return base_travel_time + queue_delay + accident_delay


def find_emergency_route(
    num_intersections: int,
    origin: int,
    destination: int,
    simulator_state: Dict,
    base_travel_time: float,
    queue_weight: float,
    accident_penalty: float,
) -> EmergencyRoute:
    """
    Find emergency route using Dijkstra shortest path.
    """
    graph = build_intersection_graph(num_intersections)

    for source, target in graph.edges():
        graph[source][target]["weight"] = compute_edge_weight(
            source=source,
            target=target,
            simulator_state=simulator_state,
            base_travel_time=base_travel_time,
            queue_weight=queue_weight,
            accident_penalty=accident_penalty,
        )

    route = nx.shortest_path(
        graph,
        source=origin,
        target=destination,
        weight="weight",
    )

    route_cost = nx.shortest_path_length(
        graph,
        source=origin,
        target=destination,
        weight="weight",
    )

    return EmergencyRoute(
        origin=origin,
        destination=destination,
        route=route,
        route_cost=float(route_cost),
    )