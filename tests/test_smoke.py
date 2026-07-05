"""
Basic smoke tests for the project setup.
"""


def test_import_config():
    from src.config import DEFAULT_NUM_INTERSECTIONS, MAX_NUM_INTERSECTIONS

    assert DEFAULT_NUM_INTERSECTIONS > 0
    assert MAX_NUM_INTERSECTIONS >= DEFAULT_NUM_INTERSECTIONS

def test_generate_traffic_demand():
    from src.data_generator import generate_traffic_demand

    df = generate_traffic_demand(
        num_intersections=5,
        simulation_minutes=10,
        scenario="normal",
    )

    assert not df.empty
    assert "vehicle_demand" in df.columns
    assert "pedestrian_demand" in df.columns
    assert df["intersection_id"].nunique() == 5

def test_fixed_equal_simulation_runs():
    from src.data_generator import generate_traffic_demand
    from src.simulator import run_fixed_equal_simulation

    demand_df = generate_traffic_demand(
        num_intersections=4,
        simulation_minutes=10,
        scenario="normal",
    )

    result = run_fixed_equal_simulation(
        num_intersections=4,
        demand_df=demand_df,
    )

    metrics = result["metrics"]
    history = result["history"]

    assert metrics.total_arrivals >= 0
    assert metrics.throughput >= 0
    assert metrics.avg_wait_time_seconds >= 0
    assert not history.empty