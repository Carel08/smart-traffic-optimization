"""
Streamlit demo for Smart City Traffic Light Optimization.

"""

import streamlit as st
import pandas as pd

from src.data_generator import generate_traffic_demand
from src.predictor import (
    train_and_evaluate_predictor,
    add_predictions_to_demand,
)
from src.benchmark import (
    run_controller_benchmark,
    tune_adaptive_controller,
)
from src.simulator import (
    run_fixed_equal_simulation,
    run_fixed_calibrated_simulation,
    run_adaptive_simulation,
)
from src.config import (
    DEFAULT_NUM_INTERSECTIONS,
    MAX_NUM_INTERSECTIONS,
    DEFAULT_SIMULATION_MINUTES,
)


def main():
    st.set_page_config(
        page_title="Smart Traffic Light Optimization",
        page_icon="🚦",
        layout="wide",
    )

    st.title("🚦 Smart City Traffic Light Optimization")
    st.write(
        """
        This demo will compare fixed traffic lights against an adaptive
        ML + optimization-based controller.
        """
    )

    st.sidebar.header("Simulation Settings")

    num_intersections = st.sidebar.slider(
        "Number of intersections",
        min_value=1,
        max_value=MAX_NUM_INTERSECTIONS,
        value=DEFAULT_NUM_INTERSECTIONS,
    )

    scenario = st.sidebar.selectbox(
        "Scenario",
        [
            "normal",
            "rain",
            "accident",
            "pedestrian",
            "emergency",
            "combined",
        ],
    )
    enable_emergency_priority = st.sidebar.checkbox(
        "Enable emergency signal priority",
        value=True,
    )
    controller = st.sidebar.selectbox(
        "Controller",
        [
            "fixed_equal",
            "fixed_calibrated",
            "adaptive",
            "ga",
            "rl_experimental",
        ],
    )

    duration = st.sidebar.slider(
        "Simulation duration, minutes",
        min_value=30,
        max_value=240,
        value=DEFAULT_SIMULATION_MINUTES,
        step=30,
    )

    training_days = st.sidebar.slider(
        "ML training days",
        min_value=1,
        max_value=7,
        value=3,
    )

    st.subheader("Current Configuration")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Intersections", num_intersections)
    col2.metric("Scenario", scenario)
    col3.metric("Controller", controller)
    col4.metric("Duration", f"{duration} min")

    st.info(" Metrics have been added")


    if st.button("Generate traffic demand preview"):
        demand_df = generate_traffic_demand(
            num_intersections=num_intersections,
            simulation_minutes=duration,
            scenario=scenario,
        )

        st.subheader("Generated Demand Sample")
        st.dataframe(demand_df.head(60), use_container_width=True)

        st.subheader("Demand Summary by Intersection and Direction")
        summary = (
            demand_df
            .groupby(["intersection_id", "direction"])["vehicle_demand"]
            .agg(["mean", "max", "sum"])
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("Vehicle Demand Over Time")
        chart_data = (
            demand_df
            .groupby("time_step")["vehicle_demand"]
            .sum()
            .reset_index()
        )
        st.line_chart(chart_data, x="time_step", y="vehicle_demand")

    if st.button("Run fixed timing simulation"):
        demand_df = generate_traffic_demand(
            num_intersections=num_intersections,
            simulation_minutes=duration,
            scenario=scenario,
        )

        if controller == "fixed_equal":
            result = run_fixed_equal_simulation(
                num_intersections=num_intersections,
                demand_df=demand_df,
                scenario=scenario,
                enable_emergency_priority=enable_emergency_priority,
            )

        elif controller == "fixed_calibrated":
            result = run_fixed_calibrated_simulation(
                num_intersections=num_intersections,
                demand_df=demand_df,
                scenario=scenario,
                enable_emergency_priority=enable_emergency_priority,
            )

        elif controller == "adaptive":
            with st.spinner("Training ML predictor for adaptive controller..."):
                training_minutes = training_days * 24 * 60

                training_df = generate_traffic_demand(
                    num_intersections=num_intersections,
                    simulation_minutes=training_minutes,
                    scenario=scenario,
                )

                ml_result = train_and_evaluate_predictor(
                    demand_df=training_df,
                    test_size=0.2,
                )

                demand_df = add_predictions_to_demand(
                    demand_df=demand_df,
                    predictor=ml_result["predictor"],
                )

            result = run_adaptive_simulation(
                num_intersections=num_intersections,
                demand_df=demand_df,
                scenario=scenario,
                enable_emergency_priority=enable_emergency_priority,
            )

        else:
            st.warning(
                f"Controller '{controller}' will be implemented in a later phase."
            )
            st.stop()

        metrics = result["metrics"]
        history = result["history"]

        st.subheader("Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Avg Vehicle Wait", f"{metrics.avg_wait_time_seconds:.1f}s")
        col2.metric("Vehicle Throughput", metrics.throughput)
        col3.metric("Max Vehicle Queue", metrics.max_queue_length)
        col4.metric("Final Vehicle Queue", metrics.final_queue_length)

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Avg Ped Wait", f"{metrics.pedestrian_avg_wait_seconds:.1f}s")
        col6.metric("Ped Throughput", metrics.pedestrian_throughput)
        col7.metric("Max Ped Queue", metrics.pedestrian_max_queue)
        col8.metric("Final Ped Queue", metrics.pedestrian_final_queue)

        st.subheader("Emergency Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Emergency Delay",
            f"{metrics.emergency_delay_seconds:.1f}s",
        )

        col2.metric(
            "Preemptions",
            metrics.emergency_preemptions,
        )

        col3.metric(
            "Route Length",
            metrics.emergency_route_length,
        )

        st.subheader("Vehicle Queue Over Time")
        st.line_chart(
            history,
            x="time_step",
            y=["total_queue", "ns_queue", "ew_queue"],
        )

        st.subheader("Pedestrian Queue Over Time")
        st.line_chart(
            history,
            x="time_step",
            y=["pedestrian_queue", "avg_pedestrian_queue_per_intersection"],
        )
        st.subheader("Average Green Time Allocation")
        st.line_chart(
            history,
            x="time_step",
            y=["avg_ns_green", "avg_ew_green"],
        )

        st.subheader("Pedestrian Phase Activation")
        st.line_chart(
            history,
            x="time_step",
            y=["pedestrian_phase_active", "pedestrian_active_intersections"],
        )

        st.subheader("Simulation History")
        st.dataframe(history.head(100), use_container_width=True)
    st.divider()
    st.subheader("ML Traffic Demand Predictor")

    if st.button("Train and evaluate ML predictor"):
        training_minutes = training_days * 24 * 60

        with st.spinner("Generating training data and fitting model..."):
            training_df = generate_traffic_demand(
                num_intersections=num_intersections,
                simulation_minutes=training_minutes,
                scenario=scenario,
            )

            ml_result = train_and_evaluate_predictor(
                demand_df=training_df,
                test_size=0.2,
            )

        baseline_eval = ml_result["baseline_evaluation"]
        model_eval = ml_result["model_evaluation"]
        feature_importance = ml_result["feature_importance"]

        st.subheader("Prediction Performance")

        performance_df = pd.DataFrame(
            [
                {
                    "model": baseline_eval.model_name,
                    "MAE": baseline_eval.mae,
                    "RMSE": baseline_eval.rmse,
                    "R2": baseline_eval.r2,
                },
                {
                    "model": model_eval.model_name,
                    "MAE": model_eval.mae,
                    "RMSE": model_eval.rmse,
                    "R2": model_eval.r2,
                },
            ]
        )

        st.dataframe(performance_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE Improvement",
            f"{baseline_eval.mae - model_eval.mae:.2f}",
        )

        rmse_improvement_pct = (
            (baseline_eval.rmse - model_eval.rmse) / baseline_eval.rmse * 100
            if baseline_eval.rmse > 0
            else 0.0
        )

        col2.metric(
            "RMSE Improvement",
            f"{rmse_improvement_pct:.1f}%",
        )

        col3.metric(
            "Model R²",
            f"{model_eval.r2:.3f}",
        )

        st.subheader("Top Feature Importances")
        st.dataframe(feature_importance.head(15), use_container_width=True)

        st.bar_chart(
            feature_importance.head(10),
            x="feature",
            y="importance",
        )

    st.divider()
    st.subheader("Controller Benchmark")

    if st.button("Run controller benchmark"):
        training_minutes = training_days * 24 * 60

        with st.spinner("Training predictor and running benchmark..."):
            training_df = generate_traffic_demand(
                num_intersections=num_intersections,
                simulation_minutes=training_minutes,
                scenario=scenario,
            )

            ml_result = train_and_evaluate_predictor(
                demand_df=training_df,
                test_size=0.2,
            )

            benchmark_demand_df = generate_traffic_demand(
                num_intersections=num_intersections,
                simulation_minutes=duration,
                scenario=scenario,
            )

            benchmark_demand_df = add_predictions_to_demand(
                demand_df=benchmark_demand_df,
                predictor=ml_result["predictor"],
            )

            benchmark_result = run_controller_benchmark(
                num_intersections=num_intersections,
                demand_df=benchmark_demand_df,
            )

        comparison_df = benchmark_result["comparison"]

        st.dataframe(comparison_df, use_container_width=True)

        st.bar_chart(
            comparison_df,
            x="controller",
            y="avg_wait_seconds",
        )

        st.bar_chart(
            comparison_df,
            x="controller",
            y="improvement_vs_fixed_equal_pct",
        )

    st.divider()
    st.subheader("Adaptive Controller Tuning")

    if st.button("Tune adaptive controller"):
        training_minutes = training_days * 24 * 60

        with st.spinner("Training predictor and tuning adaptive controller..."):
            training_df = generate_traffic_demand(
                num_intersections=num_intersections,
                simulation_minutes=training_minutes,
                scenario=scenario,
            )

            ml_result = train_and_evaluate_predictor(
                demand_df=training_df,
                test_size=0.2,
            )

            tuning_demand_df = generate_traffic_demand(
                num_intersections=num_intersections,
                simulation_minutes=duration,
                scenario=scenario,
            )

            tuning_demand_df = add_predictions_to_demand(
                demand_df=tuning_demand_df,
                predictor=ml_result["predictor"],
            )

            tuning_result = tune_adaptive_controller(
                num_intersections=num_intersections,
                demand_df=tuning_demand_df,
            )

        st.write("Best parameters:")
        st.json(tuning_result["best_params"])

        tuning_df = tuning_result["tuning_results"]

        st.dataframe(tuning_df.head(20), use_container_width=True)

        st.bar_chart(
            tuning_df.head(10),
            x="queue_weight",
            y="avg_wait_seconds",
        )
    event_log = result.get("event_log")

    if event_log is not None and not event_log.empty:
        st.subheader("Event Log")
        st.dataframe(event_log, use_container_width=True)

    if "emergency_preemption_active" in history.columns:
        st.subheader("Emergency Priority Activation")

        st.line_chart(
            history,
            x="time_step",
            y=[
                "emergency_active",
                "emergency_preemption_active",
            ],
        )
    st.subheader("Pedestrian Fairness")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Pedestrian Avg Wait",
        f"{metrics.pedestrian_avg_wait_seconds:.1f}s",
    )

    col2.metric(
        "Target Wait",
        f"{metrics.pedestrian_target_wait_seconds:.0f}s",
    )

    col3.metric(
        "Target Met",
        "Yes" if metrics.pedestrian_target_met else "No",
    )

    col4.metric(
        "Fairness Gap",
        f"{metrics.pedestrian_fairness_gap_seconds:.1f}s",
    )
    if "pedestrian_max_estimated_wait_seconds" in history.columns:
        st.line_chart(
            history,
            x="time_step",
            y=[
                "pedestrian_max_estimated_wait_seconds",
                "pedestrian_target_wait_seconds",
            ],
        )
if __name__ == "__main__":
    main()