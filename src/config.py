"""
Central configuration for the Smart Traffic Light Optimization project.
This file holds constant values we will reuse 
"""

DEFAULT_NUM_INTERSECTIONS = 4
MAX_NUM_INTERSECTIONS = 10

DIRECTIONS = ["NS", "EW"]

# Traffic signal timing assumptions, in seconds
CYCLE_TIME = 60
YELLOW_TIME = 4
MIN_GREEN = 10
MAX_GREEN = 50

# Simulation assumptions
DEFAULT_SIMULATION_MINUTES = 120
DEFAULT_RANDOM_SEED = 42

# Vehicle flow assumptions
BASE_SERVICE_RATE = 1.0  # vehicles released per second during green

# Pedestrian assumptions
PEDESTRIAN_CROSSING_TIME = 10
PEDESTRIAN_WAIT_THRESHOLD = 10
PEDESTRIAN_PHASE_TRIGGER = 20
PEDESTRIAN_SERVICE_RATE = 1.5

# Minimum number of simulation steps between pedestrian crossings
# at the same intersection.
PEDESTRIAN_MIN_GAP_STEPS = 3

# Emergency vehicle assumptions
EMERGENCY_PRIORITY_LOOKAHEAD = 2

# Safety buffer where all directions are red between phase changes
ALL_RED_TIME = 1

# Simulation assumptions
SECONDS_PER_STEP = 60

# ML training assumptions
DEFAULT_TRAINING_DAYS = 3
DEFAULT_TEST_SIZE = 0.2

# Random Forest assumptions
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5
RF_RANDOM_STATE = DEFAULT_RANDOM_SEED

# Feature engineering assumptions
ROLLING_WINDOW_SHORT = 3
ROLLING_WINDOW_LONG = 5

# Adaptive controller assumptions
ADAPTIVE_QUEUE_WEIGHT = 1.0
ADAPTIVE_PREDICTION_WEIGHT = 1.0
ADAPTIVE_PEDESTRIAN_WEIGHT = 0.15
ADAPTIVE_EPSILON = 1e-6

# Adaptive tuning assumptions

# Enhanced adaptive controller assumptions
ADAPTIVE_MIN_GREEN = 5
ADAPTIVE_MAX_GREEN = 45
ADAPTIVE_PRESSURE_EXPONENT = 2.0
ADAPTIVE_CAPACITY_AWARE = True
ADAPTIVE_QUEUE_WEIGHT_GRID = [1.0, 2.0, 5.0]
ADAPTIVE_PREDICTION_WEIGHT_GRID = [0.0, 1.0, 2.0]
ADAPTIVE_PEDESTRIAN_WEIGHT_GRID = [0.0, 0.1]
ADAPTIVE_PRESSURE_EXPONENT_GRID = [1.0, 2.0, 3.0]
ADAPTIVE_MIN_GREEN_GRID = [5, 8]
# Actuated signal assumptions
ENABLE_ACTUATED_REALLOCATION = True
MAX_REALLOCATED_GREEN_SECONDS = 15

# Genetic Algorithm assumptions
GA_RANDOM_STATE = DEFAULT_RANDOM_SEED
GA_POPULATION_SIZE = 24
GA_GENERATIONS = 12
GA_ELITE_COUNT = 4
GA_MUTATION_RATE = 0.20
GA_MUTATION_SCALE = 5.0
GA_TOURNAMENT_SIZE = 3

GA_MIN_GREEN = 5.0
GA_MAX_GREEN = 45.0

# Time bands used by GA timing plan controller
GA_TIME_BANDS = [
    ("early_rush", 0, 19),
    ("rain_build_up", 20, 44),
    ("event_accident_peak", 45, 75),
    ("accident_clearance", 76, 90),
    ("recovery", 91, 10_000),
]

# GA objective weights
GA_FINAL_QUEUE_WEIGHT = 0.02
GA_MAX_QUEUE_WEIGHT = 0.01
GA_PEDESTRIAN_WAIT_WEIGHT = 0.001

# Scenario benchmark assumptions
SCENARIO_BENCHMARK_SCENARIOS = [
    "normal",
    "rain",
    "accident",
    "pedestrian",
    "combined",
]

SCENARIO_BENCHMARK_GA_GENERATIONS = 20
SCENARIO_BENCHMARK_GA_POPULATION_SIZE = 32