"""
Streamlit demo for Smart City Traffic Light Optimization.

For Phase 1, this only creates the basic UI shell.
Later, it will run the simulation and visualize results.
"""

import streamlit as st

from src.data_generator import generate_traffic_demand
from src.simulator import run_fixed_equal_simulation
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
        ["normal", "rain", "accident", "pedestrian", "emergency", "combined"],
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

    st.subheader("Current Configuration")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Intersections", num_intersections)
    col2.metric("Scenario", scenario)
    col3.metric("Controller", controller)
    col4.metric("Duration", f"{duration} min")

    st.info("Phase 1 setup successful. Simulation engine will be added next.")

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

        result = run_fixed_equal_simulation(
            num_intersections=num_intersections,
            demand_df=demand_df,
        )

        metrics = result["metrics"]
        history = result["history"]

        st.subheader("Fixed Equal Timing Results")

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

        st.subheader("Pedestrian Phase Activation")
        st.line_chart(
            history,
            x="time_step",
            y="pedestrian_phase_active",
        )

        st.subheader("Simulation History")
        st.dataframe(history.head(100), use_container_width=True)

if __name__ == "__main__":
    main()