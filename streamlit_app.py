"""
Streamlit demo for Smart City Traffic Light Optimization.

For Phase 1, this only creates the basic UI shell.
Later, it will run the simulation and visualize results.
"""

import streamlit as st

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


if __name__ == "__main__":
    main()