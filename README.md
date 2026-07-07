# Smart City Traffic Light Optimization

A queue-based smart traffic-light optimization project that combines **ML demand forecasting**, **constrained signal optimization**, **genetic algorithm timing plans**, **pedestrian fairness**, **emergency vehicle preemption**, and an interactive **Streamlit dashboard**.

The project was designed to demonstrate three capabilities:

1. **ML Modeling** — predicting near-term vehicle demand by time, intersection, direction, weather, event conditions, and recent demand.
2. **Mathematical Optimization** — allocating signal green time under cycle, min/max green, pedestrian, and emergency constraints.
3. **Software Engineering** — clean modular code, CLI execution, Streamlit demo, event logging, replay visualization, and automated tests.

---

## 1. Problem Statement

Urban traffic networks must balance competing objectives:

- Reduce vehicle wait time.
- Avoid excessive queue build-up.
- Serve pedestrian crossings fairly.
- Prioritize emergency vehicles.
- Adapt to changing traffic conditions such as rain, accidents, and event surges.

This project models the problem as a **queue-based traffic signal control system**.

The system simulates demand across multiple intersections and compares several controllers:

| Controller | Purpose |
|---|---|
| `fixed_equal` | Simple baseline: equal green split between NS and EW |
| `fixed_calibrated` | Stronger baseline: fixed timing based on average demand |
| `adaptive` | Real-time pressure controller using queues and ML predictions |
| `enhanced_adaptive` | Experimental adaptive controller with stronger pressure logic |
| `scipy_mpc` | Real-time constrained optimization using Scipy |
| `ga` | Offline genetic algorithm that searches timing plans by scenario phase |

---

## 2. High-Level Approach

The solution follows this pipeline:

```text
Synthetic scenario demand
        ↓
Feature engineering
        ↓
ML demand forecasting
        ↓
Traffic simulation
        ↓
Signal controller decision
        ↓
Vehicle, pedestrian, and emergency service
        ↓
Benchmark metrics and dashboard visualization
```

The project separates responsibilities across modules:

```text
src/config.py          Central assumptions and parameters
src/data_generator.py  Synthetic traffic, pedestrian, rain, event, and accident demand
src/predictor.py       ML demand prediction and model comparison
src/controllers.py     Fixed, adaptive, enhanced adaptive, and MPC controllers
src/simulator.py       Queue-based traffic simulation engine
src/ga_optimizer.py    Genetic algorithm timing-plan optimizer
src/emergency.py       Emergency routing and preemption support
src/event_log.py       Scenario, accident, weather, pedestrian, and emergency event logging
src/benchmark.py       Controller comparison and tuning utilities
src/reporting.py       Final tables and chart outputs
main.py                CLI entry point
streamlit_app.py       Interactive dashboard and network replay
tests/test_smoke.py    Smoke tests for major project workflows
```

---

## 3. Environment Setup

The project supports Windows and Bash-based setup.

### Option A: Windows Anaconda Prompt

Run this from the project root:

```bat
setup_env.bat trafficopt
```

Then activate the environment:

```bat
conda activate trafficopt
```

### Option B: Windows PowerShell

Run this from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1 trafficopt
```

Then activate the environment:

```powershell
conda activate trafficopt
```

### Option C: Bash, Linux, macOS, Git Bash, or WSL

Run this from the project root:

```bash
bash setup_env.sh trafficopt
```

Then activate the environment:

```bash
conda activate trafficopt
```

### Manual setup

```bash
conda create -n trafficopt python=3.11 -y
conda activate trafficopt
pip install -r requirements.txt
```

Core packages:

```text
numpy
pandas
scikit-learn
scipy
networkx
matplotlib
plotly
streamlit
pytest
```

---

## 4. How to Run the Project

### 4.1 Run automated tests

```bash
pytest
```

Expected latest result:

```text
14 passed
```

---

### 4.2 Preview generated demand

```bash
python main.py --num-intersections 4 --scenario normal --duration 60 --preview-data
```

Available scenarios:

```text
normal
rain
accident
pedestrian
emergency
combined
combined_emergency
```

---

### 4.3 Train and evaluate the ML demand predictor

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --train-model --training-days 1
```

This trains and compares:

- Historical-average baseline
- Tuned `RandomForestRegressor`
- Tuned `HistGradientBoostingRegressor`

Outputs:

```text
outputs/ml_tuning_results.csv
outputs/ml_model_comparison.csv
outputs/feature_importance.csv
```

---

### 4.4 Run a single controller

Fixed equal baseline:

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --controller fixed_equal
```

Adaptive controller with ML predictions:

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --controller adaptive --training-days 1
```

Scipy MPC controller:

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --controller scipy_mpc --training-days 1
```

GA optimizer:

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --controller ga --ga-generations 20 --ga-population-size 32
```

---

### 4.5 Run the final benchmark report

Use an ML-enabled controller so the CLI trains the predictor before the final report:

```bash
python main.py --num-intersections 10 --scenario combined --duration 120 --controller adaptive --training-days 1 --final-report --ga-generations 20 --ga-population-size 32
```

Outputs:

```text
outputs/final_results_summary.csv
outputs/charts/avg_wait_by_controller.png
outputs/charts/improvement_vs_fixed_equal.png
outputs/charts/final_queue_by_controller.png
```

---

### 4.6 Run multi-scenario GA benchmark

```bash
python main.py --num-intersections 10 --duration 120 --scenario-benchmark --ga-generations 20 --ga-population-size 32
```

Output:

```text
outputs/scenario_ga_benchmark.csv
```

---

## 5. Run the Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard includes:

- Scenario selection
- Controller selection
- Simulation duration
- Simulation start-time selector
- ML training history slider
- Emergency priority toggle
- GA settings
- Demand preview
- ML model comparison
- Hyperparameter tuning results
- Final benchmark report
- Vehicle, pedestrian, emergency, and controller charts
- Event log
- Queue-based network replay

Recommended demo:

```text
Scenario: combined
Controller: ga
Duration: 120 minutes
Start time: 07:00 or 16:00
GA generations: 20
GA population size: 32
```

For emergency priority:

```text
Scenario: emergency or combined_emergency
Emergency priority: enabled
```

---

## 6. ML Modeling

### 6.1 Model selection and justification

The ML layer predicts near-term vehicle demand for each intersection and direction.

Three models are compared:

| Model | Role | Reason |
|---|---|---|
| HistoricalAverage | Baseline | Simple benchmark using average demand by hour, intersection, and direction |
| RandomForestRegressor | Candidate ML model | Strong tabular model, nonlinear, robust, interpretable feature importance |
| HistGradientBoostingRegressor | Candidate ML model | Strong gradient boosting model for structured tabular data, efficient and accurate |

The selected model is chosen using validation RMSE and creates:

```text
predicted_vehicle_demand
```

This prediction is used by:

```text
adaptive controller
enhanced adaptive controller
scipy_mpc controller
```

---

### 6.2 Feature engineering

The predictor uses:

| Feature group | Examples |
|---|---|
| Time | `time_step`, `hour`, `hour_sin`, `hour_cos` |
| Network position | `intersection_id` |
| Direction | `direction_is_ns`, `direction_multiplier` |
| Weather | `rain_level`, `weather_demand_multiplier`, `weather_capacity_multiplier` |
| Events | `event_multiplier`, `accident_active` |
| Demand pattern | `peak_multiplier`, `intersection_multiplier` |
| Pedestrian pressure | `pedestrian_demand` |
| Lag features | `lag_1_vehicle_demand` |
| Rolling features | `rolling_mean_3_vehicle_demand`, `rolling_mean_5_vehicle_demand` |

Cyclical hour encoding helps the model understand that 23:00 and 00:00 are close in time.

Lag and rolling features allow the model to use recent demand patterns without using future information.

---

### 6.3 Validation methodology

The project uses a **time-based split**, not a random row split.

```text
Earlier time periods → training
Later time periods   → testing
```

Hyperparameter tuning also uses an internal time-based validation split inside the training window. This avoids look-ahead bias and better reflects a forecasting workflow.

---

### 6.4 ML performance example

Latest observed ML comparison on a combined scenario:

| Model | MAE | RMSE | R² | Selected |
|---|---:|---:|---:|---|
| HistGradientBoostingRegressor | 1.96 | 2.69 | 0.79 | Yes |
| RandomForestRegressor | 2.67 | 3.45 | 0.66 | No |
| HistoricalAverage | 5.91 | 6.72 | -0.31 | No |

Selected model:

```text
HistGradientBoostingRegressor
```

Selected hyperparameters:

```python
{
    "max_iter": 180,
    "learning_rate": 0.03,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 0.01
}
```

---

## 7. Mathematical Optimization

### 7.1 Problem formulation

At each intersection, the controller allocates green time between two signal directions:

```text
NS = north/south movement
EW = east/west movement
```

The core decision is:

```text
How much green time should NS and EW receive during the signal cycle?
```

Simplified optimization objective:

```text
Minimize:
average vehicle wait
+ final queue penalty
+ max queue penalty
+ pedestrian wait penalty
```

For the MPC-style controller:

```text
Minimize:
expected post-service queue
+ squared queue penalty
+ directional imbalance penalty
+ predicted demand pressure
```

Subject to:

```text
NS green + EW green = available green time
minimum green <= green time <= maximum green
pedestrian fairness rules
emergency preemption rules
```

---

### 7.2 Algorithm choices

| Algorithm | Type | Why it was used |
|---|---|---|
| Fixed Equal | Baseline | Simple reference point |
| Fixed Calibrated | Stronger baseline | Separates true optimization value from simple demand calibration |
| Adaptive Pressure | Real-time heuristic | Fast, explainable, responsive |
| Scipy MPC | Real-time constrained optimization | Solves a green-split optimization problem per intersection |
| Genetic Algorithm | Offline black-box optimization | Searches timing plans across scenario phases without requiring gradients |

This gives both:

- **Operational practicality** through adaptive and MPC controllers.
- **Scenario-level optimization** through the GA controller.

---

### 7.3 Constraint handling

| Constraint | How it is handled |
|---|---|
| Cycle time | Green time is split within available green time |
| Minimum green | Prevents starving a direction |
| Maximum green | Prevents one direction taking the full cycle |
| Yellow and all-red time | Reserved before available green split |
| Pedestrian service | Pedestrian phases triggered by wait and queue thresholds |
| Pedestrian fairness | Target average pedestrian wait is monitored |
| Emergency priority | Emergency route receives signal preemption when enabled |
| Accident capacity reduction | Accident scenario reduces discharge capacity |
| Weather capacity reduction | Rain reduces road service capacity |

---

### 7.4 Solution quality

Representative combined-scenario benchmark:

| Controller | Avg vehicle wait, sec | Throughput | Final queue | Pedestrian avg wait, sec | Improvement vs fixed equal |
|---|---:|---:|---:|---:|---:|
| Fixed Equal | 484.73 | 41,428 | 4,195 | 174.40 | 0.00% |
| Fixed Calibrated | 416.36 | 42,966 | 2,657 | 174.40 | 14.10% |
| Adaptive | 405.61 | 43,080 | 2,543 | 174.40 | 16.32% |
| Scipy MPC | 399.82 | 43,150 | 2,473 | 174.40 | 17.52% |
| GA | 378.03 | 43,390 | 2,233 | 174.40 | 22.01% |

Main result:

```text
The GA-optimized controller reduced average vehicle wait time by about 22% versus fixed equal timing while still meeting the pedestrian fairness target.
```

---

## 8. Software Engineering

### 8.1 Code quality and structure

The project is modular:

- Config is centralized.
- Demand generation is separate from simulation.
- ML prediction is separate from control.
- Controllers are separate from the simulator.
- GA optimization is separate from real-time control.
- Reporting and dashboard logic are separate from modelling logic.

---

### 8.2 Working demo and visualization

The Streamlit dashboard supports:

- Demand preview
- Model comparison
- Feature importance
- Hyperparameter tuning display
- Final report charts
- Pedestrian fairness metrics
- Emergency metrics
- Event logs
- GA convergence
- Queue-based network replay

The replay visualizes:

```text
queue pressure
green direction
pedestrian phases
emergency preemption
nearby logged events
```

---

### 8.3 Scalability considerations

| Area | Current approach | Future scaling path |
|---|---|---|
| Intersections | Compact grid layout and dictionary-based queues | More efficient graph/network representation |
| ML training | Scikit-learn models on synthetic history | Batch training on real sensor data |
| GA optimization | Population and generation controls | Parallel evaluation of candidates |
| Simulation | Queue-based per-minute simulation | Event-based or microscopic simulator integration |
| Dashboard | Streamlit interactive demo | Deployed app with cached scenarios |

---

### 8.4 Testing and error handling

The smoke tests cover:

- Config import
- Demand generation
- Fixed equal simulation
- Pedestrian metrics
- Fixed calibrated controller
- ML predictor
- Adaptive simulation
- Controller benchmark
- Emergency scenario
- Scipy MPC
- Accident event logging
- Rain event logging
- Combined emergency dispatch
- Network replay history

Latest result:

```text
14 passed
```

---

## 9. Key Design Decisions and Tradeoffs

### 9.1 Queue-based simulation instead of microscopic simulation

The simulator models queues and service rates rather than individual cars.

**Why:** It is easier to explain, faster to run, and suitable for comparing controller strategies.

**Tradeoff:** It does not model lane-level behavior, driver behavior, acceleration, turning movements, or spillback in the same detail as a microscopic simulator.

---

### 9.2 Synthetic demand instead of real sensor data

Demand is generated using scenario assumptions for peaks, rain, accidents, pedestrians, and events.

**Why:** It makes the project reproducible without needing proprietary traffic data.

**Tradeoff:** Results demonstrate methodology, not calibrated city-level operational performance.

---

### 9.3 Time-based validation instead of random split

The ML model is trained on earlier periods and tested on later periods.

**Why:** This better reflects forecasting and avoids look-ahead bias.

**Tradeoff:** There may be fewer training examples than with random shuffling.

---

### 9.4 Explainable ML models instead of deep learning

The project uses Random Forest and HistGradientBoosting.

**Why:** They are strong for tabular data, fast to train, easy to tune, and easier to explain.

**Tradeoff:** A sequence model could potentially capture richer temporal dynamics, but would add complexity and reduce explainability.

---

### 9.5 MPC and GA instead of reinforcement learning

The project uses deterministic optimization and GA rather than reinforcement learning.

**Why:** MPC and GA are easier to validate, explain, and benchmark in a short case-study setting.

**Tradeoff:** RL may learn richer multi-agent policies, but requires careful reward design, many training episodes, and stability testing.

---

### 9.6 Pedestrian fairness included as a constraint

The project does not only optimize vehicle throughput.

**Why:** Real-world signal control must balance vehicles with pedestrian safety and fairness.

**Tradeoff:** Serving pedestrians can reduce vehicle throughput, so the design balances fairness against congestion.

---

### 9.7 Emergency preemption included as operational priority

Emergency vehicles can receive signal priority.

**Why:** Traffic systems should minimize emergency delay even when it temporarily disrupts normal signal plans.

**Tradeoff:** Preemption may temporarily increase delays for conflicting traffic.

---

## 10. Known Limitations

1. **Synthetic data** — The demand generator is scenario-based and not calibrated to real city sensor data.
2. **Queue-based simulation** — The simulator does not model individual vehicle movement, lane changing, turning movements, or driver behavior.
3. **No physical road geometry** — The network is a simplified intersection grid.
4. **Simplified pedestrian model** — Pedestrian demand and crossing behavior are represented through queues and service phases.
5. **Simplified emergency routing** — Emergency route choice is graph-based and queue-aware, but not based on real road maps.
6. **GA runtime** — Larger populations and more generations improve search quality but increase runtime.
7. **Limited scalability testing** — The project is designed for a small-to-medium simulated grid rather than a full city network.
8. **No real deployment integration** — The system does not connect to live traffic signals, sensors, or city control infrastructure.
9. **RL not implemented** — Reinforcement learning is a future-work option, not part of the current solution.
10. **CLI and dashboard are not identical in every feature** — The Streamlit dashboard includes a start-time selector. The CLI can be extended with a matching `--start-hour` argument for full parity.

---

## 11. Future Work

Possible extensions:

- Calibrate the simulator with real traffic sensor data.
- Add turning movements and lane-level capacity.
- Integrate with SUMO or another microscopic traffic simulator.
- Add Bayesian optimization for controller parameters.
- Parallelize GA candidate evaluation.
- Add multi-agent reinforcement learning as a research extension.
- Deploy the Streamlit dashboard.
- Add scenario comparison exports to PowerPoint or PDF.
- Add CI/CD testing through GitHub Actions.
- Add Docker support.

---

## 12. Recommended Demo Script

1. Open the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

2. Use:

```text
Scenario: combined
Controller: ga
Duration: 120 minutes
Start time: 07:00
GA generations: 20
GA population size: 32
```

3. Run the simulation and show:

- Executive summary metrics
- Vehicle queue over time
- Pedestrian fairness
- Controller timing
- Network replay
- Event log

4. Open **ML Predictor** and show:

- Model comparison
- Selected model
- Hyperparameters
- Feature importance

5. Open **Final Report** and show:

- Average wait by controller
- Improvement versus fixed equal
- Final queue by controller

6. Explain the final result:

```text
The GA controller produced the strongest result, reducing average vehicle wait by about 22% relative to fixed equal timing while still meeting the pedestrian fairness target.
```

---

## 13. Repository Notes

Generated output files are written to:

```text
outputs/
outputs/charts/
```

Common files produced during runs:

```text
outputs/feature_importance.csv
outputs/ml_model_comparison.csv
outputs/ml_tuning_results.csv
outputs/controller_benchmark.csv
outputs/final_results_summary.csv
outputs/ga_generation_history.csv
outputs/*_history.csv
outputs/*_event_log.csv
```

These outputs can be used in the README, Streamlit dashboard, or presentation.
