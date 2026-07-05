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

def test_fixed_equal_simulation_has_pedestrian_metrics():
    from src.data_generator import generate_traffic_demand
    from src.simulator import run_fixed_equal_simulation

    demand_df = generate_traffic_demand(
        num_intersections=4,
        simulation_minutes=20,
        scenario="pedestrian",
    )

    result = run_fixed_equal_simulation(
        num_intersections=4,
        demand_df=demand_df,
    )

    metrics = result["metrics"]
    history = result["history"]

    assert metrics.pedestrian_total_arrivals >= 0
    assert metrics.pedestrian_throughput >= 0
    assert metrics.pedestrian_avg_wait_seconds >= 0
    assert "pedestrian_phase_active" in history.columns
    assert "avg_pedestrian_queue_per_intersection" in history.columns

def test_fixed_calibrated_simulation_runs():
    from src.data_generator import generate_traffic_demand
    from src.simulator import run_fixed_calibrated_simulation

    demand_df = generate_traffic_demand(
        num_intersections=4,
        simulation_minutes=20,
        scenario="normal",
    )

    result = run_fixed_calibrated_simulation(
        num_intersections=4,
        demand_df=demand_df,
    )

    metrics = result["metrics"]
    history = result["history"]

    assert metrics.total_arrivals >= 0
    assert metrics.throughput >= 0
    assert metrics.avg_wait_time_seconds >= 0
    assert "avg_ns_green" in history.columns
    assert "avg_ew_green" in history.columns

def test_train_and_evaluate_predictor_runs():
    from src.data_generator import generate_traffic_demand
    from src.predictor import train_and_evaluate_predictor

    demand_df = generate_traffic_demand(
        num_intersections=4,
        simulation_minutes=180,
        scenario="normal",
    )

    result = train_and_evaluate_predictor(
        demand_df=demand_df,
        test_size=0.2,
    )

    assert "baseline_evaluation" in result
    assert "model_evaluation" in result
    assert "feature_importance" in result

    model_eval = result["model_evaluation"]
    assert model_eval.mae >= 0
    assert model_eval.rmse >= 0

def test_adaptive_simulation_runs():
    from src.data_generator import generate_traffic_demand
    from src.predictor import train_and_evaluate_predictor, add_predictions_to_demand
    from src.simulator import run_adaptive_simulation

    demand_df = generate_traffic_demand(
        num_intersections=4,
        simulation_minutes=120,
        scenario="normal",
    )

    ml_result = train_and_evaluate_predictor(
        demand_df=demand_df,
        test_size=0.2,
    )

    demand_df = add_predictions_to_demand(
        demand_df=demand_df,
        predictor=ml_result["predictor"],
    )

    result = run_adaptive_simulation(
        num_intersections=4,
        demand_df=demand_df,
    )

    metrics = result["metrics"]
    history = result["history"]

    assert metrics.total_arrivals >= 0
    assert metrics.throughput >= 0
    assert metrics.avg_wait_time_seconds >= 0
    assert "avg_ns_green" in history.columns
    assert "avg_ew_green" in history.columns