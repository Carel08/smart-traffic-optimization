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