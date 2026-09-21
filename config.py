"""
Configuration file for the Mars Colony model.

This file contains the experimental settings shared by all teams.

During the hackathon, students should NOT modify these values.
The purpose is to ensure that every team solves the same problem.
"""

# ============================================================
# ENVIRONMENT
# ============================================================

GRID_WIDTH = 20
GRID_HEIGHT = 20


# ============================================================
# AGENT POPULATION
# ============================================================

NUMBER_EXPLORERS = 2
NUMBER_COLLECTORS = 3


# ============================================================
# RESOURCE SETTINGS
# ============================================================

NUMBER_RESOURCES = 12

# Amount stored in each resource deposit.
RESOURCE_AMOUNT = 20

# Resources initially available at the Mars base.
INITIAL_BASE_RESOURCES = 100

# The colony consumes this amount every simulation step.
CONSUMPTION_RATE = 1


# ============================================================
# ROBOT ENERGY
# ============================================================

ROBOT_INITIAL_ENERGY = 100

# Initial threshold used by Explorers.
EXPLORER_RETURN_THRESHOLD = 20

# Initial threshold used by Collectors.
COLLECTOR_RETURN_THRESHOLD = 25


# ============================================================
# COLLECTOR SETTINGS
# ============================================================

COLLECTOR_CAPACITY = 10


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MAX_STEPS = 500

# Fixed seed for the introductory experiment.
DEFAULT_SEED = 42


# Seeds used later during the hackathon evaluation.
BENCHMARK_SEEDS = [
    10,
    20,
    30,
    40,
    50,
]