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
PEDESTRIAN_CROSSING_TIME = 12
PEDESTRIAN_WAIT_THRESHOLD = 10

# Emergency vehicle assumptions
EMERGENCY_PRIORITY_LOOKAHEAD = 2