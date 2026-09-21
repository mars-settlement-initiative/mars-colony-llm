import mesa
from llm_decision import DecisionEngine
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import (
    ResourceAgent,
    RobotAgent,
    ExplorerAgent,
    CollectorAgent,
)

from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    NUMBER_EXPLORERS,
    NUMBER_COLLECTORS,
    NUMBER_RESOURCES,
    RESOURCE_AMOUNT,
    INITIAL_BASE_RESOURCES,
    CONSUMPTION_RATE,
    ROBOT_INITIAL_ENERGY,
)


class MarsColonyModel(mesa.Model):
    """
    Main Mars Colony simulation model.

    The model creates the environment and agents and advances
    the simulation through time.

    Operational decisions, however, remain inside the agents.
    """

    def __init__(
        self,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        number_explorers=NUMBER_EXPLORERS,
        number_collectors=NUMBER_COLLECTORS,
        number_resources=NUMBER_RESOURCES,
        initial_base_resources=INITIAL_BASE_RESOURCES,
        consumption_rate=CONSUMPTION_RATE,
        seed=None,
        decision_engine=None,
    ):

        super().__init__(seed=seed)

        # Only the collector decision mechanism differs from Folder 2.
        self.decision_engine = decision_engine if decision_engine is not None else DecisionEngine()

        self.width = width
        self.height = height

        # ====================================================
        # ENVIRONMENT
        # ====================================================

        self.grid = MultiGrid(
            width,
            height,
            torus=False
        )

        # Mars base is placed approximately in the center.
        self.base_position = (
            width // 2,
            height // 2
        )

        # ====================================================
        # COLONY STATE
        # ====================================================

        self.base_resources = initial_base_resources
        self.consumption_rate = consumption_rate

        # Shared set containing resource locations discovered
        # by Explorer agents.
        self.known_resources = set()

        # Statistics.
        self.total_resources_collected = 0
        self.steps_survived = 0

        # ====================================================
        # CREATE RESOURCE DEPOSITS
        # ====================================================

        for _ in range(number_resources):

            resource = ResourceAgent(
                self,
                amount=RESOURCE_AMOUNT
            )

            position = self.random_empty_position()

            self.grid.place_agent(
                resource,
                position
            )

        # ====================================================
        # CREATE EXPLORERS
        # ====================================================

        for _ in range(number_explorers):

            explorer = ExplorerAgent(
                self,
                energy=ROBOT_INITIAL_ENERGY
            )

            self.grid.place_agent(
                explorer,
                self.base_position
            )

        # ====================================================
        # CREATE COLLECTORS
        # ====================================================

        for _ in range(number_collectors):

            collector = CollectorAgent(
                self,
                energy=ROBOT_INITIAL_ENERGY
            )

            self.grid.place_agent(
                collector,
                self.base_position
            )

        # ====================================================
        # DATA COLLECTION
        # ====================================================

        self.datacollector = DataCollector(

            model_reporters={

                "Base Resources":
                    lambda model:
                    model.base_resources,

                "Resources Collected":
                    lambda model:
                    model.total_resources_collected,

                "Known Resources":
                    lambda model:
                    len(model.known_resources),

                "Active Robots":
                    lambda model:
                    sum(
                        1
                        for agent in model.agents
                        if isinstance(agent, RobotAgent)
                        and agent.active
                    ),

                "Distance Travelled":
                    lambda model:
                    sum(
                        agent.distance_travelled
                        for agent in model.agents
                        if isinstance(agent, RobotAgent)
                    ),
            }
        )

        self.datacollector.collect(self)

    def random_empty_position(self):
        """
        Select a random grid position that does not already
        contain another resource deposit.
        """

        while True:

            position = (
                self.random.randrange(self.width),
                self.random.randrange(self.height)
            )

            # Do not place resources on the Mars base.
            if position == self.base_position:
                continue

            contents = self.grid.get_cell_list_contents(
                [position]
            )

            contains_resource = any(
                isinstance(agent, ResourceAgent)
                for agent in contents
            )

            if not contains_resource:
                return position

    def step(self):
        """
        Execute one simulation time step.
        """

        # ====================================================
        # 1. ACTIVATE ROBOTS
        # ====================================================

        robots = self.agents.select(
            lambda agent:
                isinstance(agent, RobotAgent)
        )

        robots.shuffle_do("step")

        # ====================================================
        # 2. COLONY CONSUMES RESOURCES
        # ====================================================

        self.base_resources -= self.consumption_rate

        self.steps_survived += 1

        # ====================================================
        # 3. COLLECT DATA
        # ====================================================

        self.datacollector.collect(self)

        # ====================================================
        # 4. CHECK COLONY SURVIVAL
        # ====================================================

        if self.base_resources <= 0:

            self.base_resources = 0
            self.running = False
