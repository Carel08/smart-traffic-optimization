"""
The dashboard shows:
- demand preview
- single-controller simulation
- ML predictor diagnostics
- benchmark / final report
- pedestrian fairness, emergency preemption, and event log views
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import math
import time

import plotly.graph_objects as go
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - fallback for minimal installs
    px = None

from src.data_generator import generate_traffic_demand
from src.predictor import train_and_evaluate_predictor, add_predictions_to_demand
from src.simulator import (
    run_fixed_equal_simulation,
    run_fixed_calibrated_simulation,
    run_adaptive_simulation,
)

# Optional controllers added in later phases. The dashboard degrades gracefully
# if a local branch does not yet have one of these helpers.
try:
    from src.simulator import run_scipy_mpc_simulation
except Exception:  # pragma: no cover
    run_scipy_mpc_simulation = None

try:
    from src.simulator import run_enhanced_adaptive_simulation
except Exception:  # pragma: no cover
    run_enhanced_adaptive_simulation = None

try:
    from src.ga_optimizer import optimize_ga_timing_plan
except Exception:  # pragma: no cover
    optimize_ga_timing_plan = None

try:
    from src.benchmark import tune_adaptive_controller
except Exception:  # pragma: no cover
    tune_adaptive_controller = None

from src.config import (
    DEFAULT_NUM_INTERSECTIONS,
    MAX_NUM_INTERSECTIONS,
    DEFAULT_SIMULATION_MINUTES,
    SCENARIOS,
)



CONTROLLERS = [
    "fixed_equal",
    "fixed_calibrated",
    "adaptive",
    "scipy_mpc",
    "ga",
]

if run_enhanced_adaptive_simulation is not None:
    CONTROLLERS.insert(4, "enhanced_adaptive")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def metric_value(metrics: Any, attr: str, default: Any = 0) -> Any:
    return getattr(metrics, attr, default)


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def calculate_improvement(baseline_wait: float, model_wait: float) -> float:
    if baseline_wait <= 0:
        return 0.0
    return (baseline_wait - model_wait) / baseline_wait * 100


def generate_demand(num_intersections: int,duration: int, scenario: str,start_hour: int,) -> pd.DataFrame:
    return generate_traffic_demand(
        num_intersections=num_intersections,
        simulation_minutes=duration,
        scenario=scenario,
        start_hour=start_hour,
    )


@st.cache_data(show_spinner=False)
def train_predictor( num_intersections: int,training_days: int,scenario: str,start_hour: int,) -> Dict[str, Any]:
    training_minutes = training_days * 24 * 60
    training_df = generate_traffic_demand(
        num_intersections=num_intersections,
        simulation_minutes=training_minutes,
        scenario=scenario,
        start_hour=start_hour,
    )
    return train_and_evaluate_predictor(demand_df=training_df, test_size=0.2)


def add_ml_predictions_if_needed(
    demand_df: pd.DataFrame,
    controller: str,
    num_intersections: int,
    training_days: int,
    scenario: str,
    start_hour: int,
) -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    needs_predictions = controller in {"adaptive", "enhanced_adaptive", "scipy_mpc"}

    if not needs_predictions:
        return demand_df, None

    ml_result = train_predictor(num_intersections, training_days, scenario, start_hour)

    demand_with_predictions = add_predictions_to_demand(
        demand_df=demand_df,
        predictor=ml_result["predictor"],
    )

    return demand_with_predictions, ml_result


def run_controller(
    controller: str,
    num_intersections: int,
    demand_df: pd.DataFrame,
    scenario: str,
    enable_emergency_priority: bool,
    training_days: int,
    ga_generations: int,
    ga_population_size: int,
    start_hour: int,) -> Dict[str, Any]:
    """Run one selected controller and return simulator result."""
    demand_for_run, ml_result = add_ml_predictions_if_needed(
        demand_df=demand_df,
        controller=controller,
        num_intersections=num_intersections,
        training_days=training_days,
        scenario=scenario,
        start_hour=start_hour,)

    if controller == "fixed_equal":
        result = run_fixed_equal_simulation(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
        )

    elif controller == "fixed_calibrated":
        result = run_fixed_calibrated_simulation(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
        )

    elif controller == "adaptive":
        result = run_adaptive_simulation(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
        )

    elif controller == "scipy_mpc":
        if run_scipy_mpc_simulation is None:
            raise NotImplementedError("run_scipy_mpc_simulation is not available in this branch.")
        result = run_scipy_mpc_simulation(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
        )

    elif controller == "enhanced_adaptive":
        if run_enhanced_adaptive_simulation is None:
            raise NotImplementedError("run_enhanced_adaptive_simulation is not available in this branch.")
        result = run_enhanced_adaptive_simulation(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
        )

    elif controller == "ga":
        if optimize_ga_timing_plan is None:
            raise NotImplementedError("optimize_ga_timing_plan is not available in this branch.")
        ga_result = optimize_ga_timing_plan(
            num_intersections=num_intersections,
            demand_df=demand_for_run,
            population_size=ga_population_size,
            generations=ga_generations,
        )
        result = ga_result["best_result"]
        result["ga_generation_history"] = ga_result.get("generation_history")
        result["ga_best_fitness"] = ga_result.get("best_fitness")

    else:
        raise NotImplementedError(f"Controller '{controller}' is not supported yet.")

    result["ml_result"] = ml_result
    result["controller_name"] = controller
    return result


def build_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    baseline_wait = results["fixed_equal"]["metrics"].avg_wait_time_seconds
    rows = []

    for controller_name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "controller": controller_name,
                "avg_wait_seconds": metrics.avg_wait_time_seconds,
                "throughput": metrics.throughput,
                "final_queue": metrics.final_queue_length,
                "max_queue": metrics.max_queue_length,
                "avg_queue": metrics.avg_queue_length,
                "pedestrian_avg_wait_seconds": metrics.pedestrian_avg_wait_seconds,
                "pedestrian_target_met": metric_value(metrics, "pedestrian_target_met", 0),
                "emergency_delay_seconds": metric_value(metrics, "emergency_delay_seconds", 0.0),
                "improvement_vs_fixed_equal_pct": calculate_improvement(
                    baseline_wait=baseline_wait,
                    model_wait=metrics.avg_wait_time_seconds,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("avg_wait_seconds").reset_index(drop=True)


def run_final_report(
    num_intersections: int,
    duration: int,
    scenario: str,
    training_days: int,
    enable_emergency_priority: bool,
    ga_generations: int,
    ga_population_size: int,
    start_hour: int,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    demand_df = generate_demand(num_intersections, duration, scenario, start_hour)

    controllers = ["fixed_equal", "fixed_calibrated", "adaptive"]
    if run_scipy_mpc_simulation is not None:
        controllers.append("scipy_mpc")
    if optimize_ga_timing_plan is not None:
        controllers.append("ga")

    results = {}
    for controller in controllers:
        results[controller] = run_controller(
            controller=controller,
            num_intersections=num_intersections,
            demand_df=demand_df.copy(),
            scenario=scenario,
            enable_emergency_priority=enable_emergency_priority,
            training_days=training_days,
            ga_generations=ga_generations,
            ga_population_size=ga_population_size,
            start_hour=start_hour,
        )

    return build_summary_table(results), results


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------


def show_dataframe_download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """
    Render a robust bar chart.

    This defensively converts the y-axis to numeric before plotting.
    It prevents charts such as final_queue from rendering blank when the
    dataframe column is stored as object/string after display, download,
    or session-state operations.
    """
    if df is None or df.empty:
        st.info(f"No data available for {title}.")
        return

    if x not in df.columns or y not in df.columns:
        st.warning(f"Cannot plot {title}: missing '{x}' or '{y}' column.")
        return

    chart_df = df[[x, y]].copy()
    chart_df[y] = pd.to_numeric(chart_df[y], errors="coerce")
    chart_df = chart_df.dropna(subset=[x, y])

    if chart_df.empty:
        st.info(f"No numeric values available for {title}.")
        return

    if px is not None:
        fig = px.bar(
            chart_df,
            x=x,
            y=y,
            title=title,
        )
        fig.update_traces(
            text=chart_df[y],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(
            xaxis_title=x.replace("_", " ").title(),
            yaxis_title=y.replace("_", " ").title(),
            showlegend=False,
            height=420,
            margin=dict(t=70, b=80),
        )
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart_df, x=x, y=y)


def line_chart(df: pd.DataFrame, x: str, y: list[str], title: str) -> None:
    existing_cols = [col for col in y if col in df.columns]
    if not existing_cols:
        return

    st.subheader(title)
    if px is not None:
        long_df = df[[x] + existing_cols].melt(id_vars=x, var_name="metric", value_name="value")
        fig = px.line(long_df, x=x, y="value", color="metric", title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df, x=x, y=existing_cols)


def display_metric_cards(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]

    st.subheader("Executive Summary")
    cols = st.columns(5)
    cols[0].metric("Avg Vehicle Wait", f"{metrics.avg_wait_time_seconds:.1f}s")
    cols[1].metric("Throughput", f"{metrics.throughput:,}")
    cols[2].metric("Final Queue", f"{metrics.final_queue_length:,}")
    cols[3].metric("Ped Avg Wait", f"{metrics.pedestrian_avg_wait_seconds:.1f}s")
    cols[4].metric("Emergency Delay", f"{metric_value(metrics, 'emergency_delay_seconds', 0.0):.1f}s")

    cols = st.columns(5)
    cols[0].metric("Max Vehicle Queue", f"{metrics.max_queue_length:,}")
    cols[1].metric("Avg Vehicle Queue", f"{metrics.avg_queue_length:.1f}")
    cols[2].metric("Ped Target Met", "Yes" if metric_value(metrics, "pedestrian_target_met", 0) else "No")
    cols[3].metric("Ped Final Queue", f"{metrics.pedestrian_final_queue:,}")
    cols[4].metric("Preemptions", f"{metric_value(metrics, 'emergency_preemptions', 0):,}")


def display_single_result(result: Dict[str, Any], num_intersections: int) -> None:
    metrics = result["metrics"]
    history = result.get("history", pd.DataFrame())
    event_log = result.get("event_log", pd.DataFrame())

    display_metric_cards(result)

    tabs = st.tabs([
        "Vehicle Flow",
        "Pedestrians",
        "Emergency & Events",
        "Controller Timing",
        "Network Replay",
        "Raw Data",
    ])

    with tabs[0]:
        line_chart(history, "time_step", ["total_queue", "ns_queue", "ew_queue"], "Vehicle Queue Over Time")
        line_chart(history, "time_step", ["throughput"], "Cumulative Throughput")

    with tabs[1]:
        cols = st.columns(4)
        cols[0].metric("Pedestrian Avg Wait", f"{metrics.pedestrian_avg_wait_seconds:.1f}s")
        cols[1].metric("Target Wait", f"{metric_value(metrics, 'pedestrian_target_wait_seconds', 0):.0f}s")
        cols[2].metric("Target Met", "Yes" if metric_value(metrics, "pedestrian_target_met", 0) else "No")
        cols[3].metric("Fairness Gap", f"{metric_value(metrics, 'pedestrian_fairness_gap_seconds', 0.0):.1f}s")

        line_chart(
            history,
            "time_step",
            ["pedestrian_queue", "avg_pedestrian_queue_per_intersection"],
            "Pedestrian Queue Over Time",
        )
        line_chart(
            history,
            "time_step",
            ["pedestrian_max_estimated_wait_seconds", "pedestrian_target_wait_seconds"],
            "Pedestrian Estimated Max Wait vs Target",
        )
        line_chart(
            history,
            "time_step",
            ["pedestrian_phase_active", "pedestrian_active_intersections"],
            "Pedestrian Phase Activation",
        )

    with tabs[2]:
        cols = st.columns(4)
        cols[0].metric("Emergency Dispatched", metric_value(metrics, "emergency_dispatched", 0))
        cols[1].metric("Emergency Completed", metric_value(metrics, "emergency_completed", 0))
        cols[2].metric("Emergency Delay", f"{metric_value(metrics, 'emergency_delay_seconds', 0.0):.1f}s")
        cols[3].metric("Completion Step", metric_value(metrics, "emergency_completion_step", -1))

        line_chart(
            history,
            "time_step",
            ["emergency_active", "emergency_preemption_active"],
            "Emergency Priority Activation",
        )

        st.subheader("Event Log")
        if event_log is not None and not event_log.empty:
            st.dataframe(event_log, use_container_width=True)
            show_dataframe_download(event_log, "Download event log", "event_log.csv")
        else:
            st.info("No events were logged for this run.")

    with tabs[3]:
        line_chart(history, "time_step", ["avg_ns_green", "avg_ew_green"], "Average Green Time Allocation")
        if "allow_actuated_reallocation" in history.columns:
            line_chart(history, "time_step", ["allow_actuated_reallocation"], "Actuated Reallocation Enabled")

        ga_history = result.get("ga_generation_history")
        if ga_history is not None and isinstance(ga_history, pd.DataFrame) and not ga_history.empty:
            st.subheader("GA Optimization History")
            st.dataframe(ga_history, use_container_width=True)
            line_chart(
                ga_history,
                "generation",
                ["best_avg_wait_seconds", "best_fitness", "mean_fitness"],
                "GA Convergence",
            )

    with tabs[4]:
        render_network_replay(
            result=result,
            num_intersections=num_intersections,
        )

    with tabs[5]:
        st.subheader("Simulation History")
        st.dataframe(history.head(500), use_container_width=True)
        show_dataframe_download(history, "Download simulation history", "simulation_history.csv")


def display_demand_preview(demand_df: pd.DataFrame) -> None:
    st.subheader("Demand Preview")

    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(demand_df):,}")
    cols[1].metric("Total Vehicle Demand", f"{demand_df['vehicle_demand'].sum():,}")
    cols[2].metric("Total Ped Demand", f"{demand_df['pedestrian_demand'].max() if 'pedestrian_demand' in demand_df.columns else 0:,}")
    cols[3].metric("Max Rain Level", f"{demand_df['rain_level'].max() if 'rain_level' in demand_df.columns else 0}")

    st.dataframe(demand_df.head(100), use_container_width=True)

    demand_over_time = demand_df.groupby("time_step", as_index=False)["vehicle_demand"].sum()
    if px is not None:
        fig = px.line(demand_over_time, x="time_step", y="vehicle_demand", title="Vehicle Demand Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(demand_over_time, x="time_step", y="vehicle_demand")

    summary = (
        demand_df.groupby(["intersection_id", "direction"], as_index=False)["vehicle_demand"]
        .agg(["mean", "max", "sum"])
        .reset_index()
    )
    st.subheader("Demand by Intersection and Direction")
    st.dataframe(summary, use_container_width=True)

def build_network_replay_figure(
    intersection_history: pd.DataFrame,
    selected_time_step: int,
    num_intersections: int,
    ) -> go.Figure:
    """
    Build a queue-based network replay figure.

    This visualizes the state of each intersection at a selected time step.
    Marker size represents queue pressure.
    Labels show queue size, green direction, pedestrian phase, and emergency state.
    """
    frame = intersection_history[
        intersection_history["time_step"] == selected_time_step
    ].copy()

    if frame.empty:
        return go.Figure()

    columns = min(5, num_intersections)
    rows = math.ceil(num_intersections / columns)

    frame["grid_x"] = frame["intersection_id"] % columns
    frame["grid_y"] = rows - 1 - (frame["intersection_id"] // columns)

    max_queue = max(float(intersection_history["total_queue"].max()),1.0,)

    frame["marker_size"] = 22 + (
        frame["total_queue"] / max_queue * 55
    )

    marker_colors = []

    for _, row in frame.iterrows():
        if int(row.get("emergency_preemption_active", 0)) == 1:
            marker_colors.append("#d62728")
        elif int(row.get("pedestrian_phase_active", 0)) == 1:
            marker_colors.append("#9467bd")
        elif row["green_direction"] == "NS":
            marker_colors.append("#2ca02c")
        else:
            marker_colors.append("#1f77b4")

    hover_text = []

    for _, row in frame.iterrows():
        hover_text.append(
            "<b>Intersection "
            + str(int(row["intersection_id"]))
            + "</b><br>"
            + f"Total queue: {int(row['total_queue'])}<br>"
            + f"NS queue: {int(row['ns_queue'])}<br>"
            + f"EW queue: {int(row['ew_queue'])}<br>"
            + f"Ped queue: {int(row['pedestrian_queue'])}<br>"
            + f"NS green: {row['ns_green']:.1f}s<br>"
            + f"EW green: {row['ew_green']:.1f}s<br>"
            + f"Green priority: {row['green_direction']}<br>"
            + f"Pedestrian phase: {int(row['pedestrian_phase_active'])}<br>"
            + f"Emergency preemption: {int(row['emergency_preemption_active'])}"
        )

    text_labels = []

    for _, row in frame.iterrows():
        label = (
            f"I{int(row['intersection_id'])}<br>"
            f"Q:{int(row['total_queue'])}<br>"
            f"{row['green_direction']}"
        )

        if int(row.get("pedestrian_phase_active", 0)) == 1:
            label += "<br>🚶"

        if int(row.get("emergency_preemption_active", 0)) == 1:
            label += "<br>🚑"

        text_labels.append(label)

    fig = go.Figure()

    # Draw simple road grid lines.
    for y in range(rows):
        fig.add_trace(
            go.Scatter(
                x=[0, columns - 1],
                y=[y, y],
                mode="lines",
                line=dict(width=2, color="lightgray"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for x in range(columns):
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, rows - 1],
                mode="lines",
                line=dict(width=2, color="lightgray"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=frame["grid_x"],
            y=frame["grid_y"],
            mode="markers+text",
            marker=dict(
                size=frame["marker_size"],
                color=marker_colors,
                line=dict(width=2, color="black"),
                opacity=0.85,
            ),
            text=text_labels,
            textposition="middle center",
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Network Replay — Minute {selected_time_step}",
        height=520,
        xaxis=dict(
            visible=False,
            range=[-0.7, columns - 0.3],
        ),
        yaxis=dict(
            visible=False,
            range=[-0.7, rows - 0.3],
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig

def render_network_replay(result: dict,num_intersections: int,) -> None:
    """
    Render an interactive queue-based network replay.
    """
    intersection_history = result.get("intersection_history")
    event_log = result.get("event_log")

    if intersection_history is None or intersection_history.empty:
        st.warning(
            "Network replay data is not available. "
            "Make sure the simulator returns 'intersection_history'."
        )
        return

    st.subheader("🗺️ Network Simulation Replay")

    st.caption(
        """
        This is a queue-based replay. Marker size represents congestion.
        Green means NS priority, blue means EW priority, purple indicates a
        pedestrian phase, and red indicates emergency preemption.
        """
    )

    min_time = int(intersection_history["time_step"].min())
    max_time = int(intersection_history["time_step"].max())

    col1, col2, col3 = st.columns([2, 1, 1])

    selected_time_step = col1.slider(
        "Replay minute",
        min_value=min_time,
        max_value=max_time,
        value=min_time,
        step=1,
    )

    playback_speed = col2.selectbox(
        "Playback speed",
        ["Slow", "Medium", "Fast"],
        index=1,
    )

    speed_map = {
        "Slow": 0.45,
        "Medium": 0.18,
        "Fast": 0.08,
    }

    show_events = col3.checkbox(
        "Show nearby events",
        value=True,
    )

    replay_placeholder = st.empty()

    fig = build_network_replay_figure(
        intersection_history=intersection_history,
        selected_time_step=selected_time_step,
        num_intersections=num_intersections,
    )

    replay_placeholder.plotly_chart(
        fig,
        use_container_width=True,
    )

    if st.button("▶ Play replay"):
        progress = st.progress(0)

        time_steps = list(range(min_time, max_time + 1))

        for index, time_step_value in enumerate(time_steps):
            fig = build_network_replay_figure(
                intersection_history=intersection_history,
                selected_time_step=time_step_value,
                num_intersections=num_intersections,
            )

            replay_placeholder.plotly_chart(
                fig,
                use_container_width=True,
            )

            progress.progress(
                int((index + 1) / len(time_steps) * 100)
            )

            time.sleep(speed_map[playback_speed])

    if show_events and event_log is not None and not event_log.empty:
        st.markdown("#### Events near selected minute")

        nearby_events = event_log[
            event_log["time_step"].between(
                selected_time_step - 3,
                selected_time_step + 3,
            )
        ]

        if nearby_events.empty:
            st.info("No major logged events near this minute.")
        else:
            st.dataframe(nearby_events,use_container_width=True,)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Smart Traffic Light Optimization",
        page_icon="🚦",
        layout="wide",
    )

    st.title("🚦 Smart City Traffic Light Optimization")
    st.caption(
        "Queue-based simulation, ML demand prediction, real-time optimization, "
        "GA timing plans, pedestrian fairness, and emergency preemption."
    )

    with st.sidebar:
        st.header("Simulation Settings")

        num_intersections = st.slider(
            "Number of intersections",
            min_value=1,
            max_value=MAX_NUM_INTERSECTIONS,
            value=DEFAULT_NUM_INTERSECTIONS,
        )

        scenario = st.selectbox("Scenario", SCENARIOS, index=SCENARIOS.index("combined"))

        controller = st.selectbox("Controller", CONTROLLERS, index=CONTROLLERS.index("ga") if "ga" in CONTROLLERS else 0)

        duration = st.slider(
            "Simulation duration, minutes",
            min_value=30,
            max_value=240,
            value=DEFAULT_SIMULATION_MINUTES,
            step=30,
        )
        start_hour = st.selectbox(
            "Simulation start time",
            options=list(range(24)),
            index=7,
            format_func=lambda h: f"{h:02d}:00",
            help=(
                "Controls the starting hour of the simulated day. "
                "Useful for comparing morning peak, lunch traffic, evening peak, and night traffic."
            ),
        )

        training_days = st.slider(
            "Synthetic ML training history, days",
            min_value=1,
            max_value=7,
            value=1,
            help=(
                "Controls how many synthetic historical days are generated for ML "
                "model training and hyperparameter tuning. Higher values may improve "
                "model stability but increase runtime."
            ),
        )

        st.divider()
        st.subheader("Emergency")
        enable_emergency_priority = st.checkbox("Enable emergency signal priority", value=True)

        st.divider()
        st.subheader("GA Settings")
        ga_generations = st.slider("GA generations", min_value=5, max_value=40, value=20, step=5)
        ga_population_size = st.slider("GA population size", min_value=8, max_value=64, value=32, step=8)

    config_cols = st.columns(6)
    config_cols[0].metric("Intersections", num_intersections)
    config_cols[1].metric("Scenario", scenario)
    config_cols[2].metric("Controller", controller)
    config_cols[3].metric("Duration", f"{duration} min")
    config_cols[4].metric("Start Time", f"{start_hour:02d}:00")
    config_cols[5].metric("Training Days", training_days)

    st.info(
        "Recommended demo: Run `combined` with `ga`, then open Final Report. "
        "Use `emergency` scenario to show signal preemption and event logs."
    )

    main_tabs = st.tabs([
        "Run Simulation",
        "Demand Preview",
        "ML Predictor",
        "Final Report",
        "Adaptive Tuning",
    ])

    with main_tabs[0]:
        st.subheader("Run Selected Controller")
        run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

        if run_clicked:
            with st.spinner(f"Running {controller} controller on {scenario} scenario..."):
                demand_df = generate_demand(num_intersections, duration, scenario, start_hour)
                result = run_controller(
                    controller=controller,
                    num_intersections=num_intersections,
                    demand_df=demand_df,
                    scenario=scenario,
                    enable_emergency_priority=enable_emergency_priority,
                    training_days=training_days,
                    ga_generations=ga_generations,
                    ga_population_size=ga_population_size,
                    start_hour=start_hour,
                )
                st.session_state["last_result"] = result

        if "last_result" in st.session_state:
            display_single_result(st.session_state["last_result"], num_intersections)
        else:
            st.info("Run a simulation to display results.")

    with main_tabs[1]:
        if st.button("Generate demand preview", use_container_width=True):
            with st.spinner("Generating demand preview..."):
                demand_df = generate_demand(num_intersections, duration, scenario, start_hour)
                st.session_state["last_demand_preview"] = demand_df

        if "last_demand_preview" in st.session_state:
            display_demand_preview(st.session_state["last_demand_preview"])
        else:
            st.info("Generate a demand preview to inspect synthetic arrivals.")

    with main_tabs[2]:
        st.subheader("ML Demand Predictor Diagnostics")
        st.caption(
            "Compares the historical-average baseline, tuned Random Forest, "
            "and tuned HistGradientBoosting models using a time-based split."
        )

        if st.button("Train and evaluate ML predictor", use_container_width=True):
            with st.spinner("Training and tuning ML demand predictors..."):
                ml_result = train_predictor(num_intersections, training_days, scenario, start_hour)
                st.session_state["last_ml_result"] = ml_result

        if "last_ml_result" in st.session_state:
            ml_result = st.session_state["last_ml_result"]

            model_comparison = ml_result.get("model_comparison")
            feature_importance = ml_result["feature_importance"]
            selected_model_name = ml_result.get(
                "selected_model_name",
                "Unknown",
            )
            selected_model_params = ml_result.get(
                "selected_model_params",
                {},
            )
            tuning_results = ml_result.get("tuning_results")

            if model_comparison is not None:
                performance_df = model_comparison.copy()
            else:
                # Backward-compatible fallback for the older predictor.py
                # implementation.
                baseline_eval = ml_result["baseline_evaluation"]
                model_eval = ml_result["model_evaluation"]

                performance_df = pd.DataFrame(
                    [
                        {
                            "model": baseline_eval.model_name,
                            "MAE": baseline_eval.mae,
                            "RMSE": baseline_eval.rmse,
                            "R2": baseline_eval.r2,
                            "selected": False,
                        },
                        {
                            "model": model_eval.model_name,
                            "MAE": model_eval.mae,
                            "RMSE": model_eval.rmse,
                            "R2": model_eval.r2,
                            "selected": True,
                        },
                    ]
                )

            st.subheader("Model Comparison")
            st.dataframe(performance_df, use_container_width=True)

            best_row = performance_df.sort_values("RMSE").iloc[0]

            cols = st.columns(3)
            cols[0].metric("Selected Model", selected_model_name)
            cols[1].metric("Best RMSE", f"{best_row['RMSE']:.3f}")
            cols[2].metric("Best R²", f"{best_row['R2']:.3f}")

            st.subheader("Selected Model Hyperparameters")
            st.json(selected_model_params)

            if tuning_results is not None and not tuning_results.empty:
                st.subheader("Hyperparameter Tuning Results")
                st.dataframe(tuning_results, use_container_width=True)
                show_dataframe_download(
                    tuning_results,
                    "Download ML tuning results",
                    "ml_tuning_results.csv",
                )

            st.subheader("Feature Importance")
            st.dataframe(feature_importance.head(20), use_container_width=True)
            bar_chart(
                feature_importance.head(12),
                "feature",
                "importance",
                "Top Feature Importances",
            )
        else:
            st.info("Train the predictor to see model comparison and feature importance.")


    with main_tabs[3]:
        st.subheader("Final Benchmark Report")
        st.write(
            "Runs fixed equal, fixed calibrated, adaptive, Scipy MPC, and GA on the same demand scenario. "
        )

        if st.button("Run final report", type="primary", use_container_width=True):
            with st.spinner("Running final report. GA may take a little longer..."):
                summary_df, final_results = run_final_report(
                    num_intersections=num_intersections,
                    duration=duration,
                    scenario=scenario,
                    training_days=training_days,
                    enable_emergency_priority=enable_emergency_priority,
                    ga_generations=ga_generations,
                    ga_population_size=ga_population_size,
                    start_hour=start_hour,
                )
                st.session_state["final_summary_df"] = summary_df
                st.session_state["final_results"] = final_results

        if "final_summary_df" in st.session_state:
            summary_df = st.session_state["final_summary_df"]
            st.dataframe(summary_df, use_container_width=True)
            show_dataframe_download(summary_df, "Download final results CSV", "final_results_summary.csv")

            cols = st.columns(3)
            best = summary_df.iloc[0]
            fixed = summary_df[summary_df["controller"] == "fixed_equal"].iloc[0]
            cols[0].metric("Best Controller", best["controller"])
            cols[1].metric("Best Avg Wait", f"{best['avg_wait_seconds']:.1f}s")
            cols[2].metric("Best Improvement", format_pct(best["improvement_vs_fixed_equal_pct"]))

            chart_cols = st.columns(2)
            with chart_cols[0]:
                bar_chart(summary_df, "controller", "avg_wait_seconds", "Average Wait by Controller")
            with chart_cols[1]:
                bar_chart(summary_df, "controller", "improvement_vs_fixed_equal_pct", "Improvement vs Fixed Equal")
            bar_chart(summary_df, "controller", "final_queue", "Final Queue by Controller")
        else:
            st.info("Run the final report to generate presentation-ready results.")

    with main_tabs[4]:
        st.subheader("Adaptive Controller Tuning")
        if tune_adaptive_controller is None:
            st.warning("tune_adaptive_controller is not available in this branch.")
        elif st.button("Tune adaptive controller", use_container_width=True):
            with st.spinner("Training predictor and tuning adaptive controller..."):
                ml_result = train_predictor(num_intersections, training_days, scenario, start_hour)
                tuning_demand_df = generate_demand(num_intersections, duration, scenario, start_hour)
                tuning_demand_df = add_predictions_to_demand(
                    demand_df=tuning_demand_df,
                    predictor=ml_result["predictor"],
                )
                tuning_result = tune_adaptive_controller(
                    num_intersections=num_intersections,
                    demand_df=tuning_demand_df,
                )
                st.session_state["last_tuning_result"] = tuning_result

        if "last_tuning_result" in st.session_state:
            tuning_result = st.session_state["last_tuning_result"]
            st.write("Best parameters:")
            st.json(tuning_result["best_params"])
            tuning_df = tuning_result["tuning_results"]
            st.dataframe(tuning_df.head(30), use_container_width=True)
            show_dataframe_download(tuning_df, "Download tuning results", "adaptive_tuning_results.csv")
            bar_chart(tuning_df.head(15), "queue_weight", "avg_wait_seconds", "Top Adaptive Tuning Results")


if __name__ == "__main__":
    main()
