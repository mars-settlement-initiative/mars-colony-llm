import mesa
from llm_decision import collect_action

from config import (
    EXPLORER_RETURN_THRESHOLD,
    COLLECTOR_RETURN_THRESHOLD,
    COLLECTOR_CAPACITY,
)

def manhattan_distance(position_a, position_b):
    """Calculate Manhattan distance between two grid positions."""
    x1, y1 = position_a
    x2, y2 = position_b

    return abs(x1 - x2) + abs(y1 - y2)


class ResourceAgent(mesa.Agent):
    """
    A resource deposit on Mars.

    It does not move and does not make decisions.
    """

    def __init__(self, model, amount=20):
        super().__init__(model)

        self.amount = amount
        self.discovered = False

    def step(self):
        pass


class RobotAgent(mesa.Agent):
    """
    Parent class for all robot agents.

    ExplorerAgent and CollectorAgent inherit common functionality
    from this class.
    """

    def __init__(self, model, energy=100):
        super().__init__(model)

        self.max_energy = energy
        self.energy = energy
        self.active = True
        self.distance_travelled = 0

    def move_one_step_towards(self, target):
        """
        Move one grid cell toward a target.

        Robots move horizontally first and then vertically.
        """

        if not self.active:
            return

        current_x, current_y = self.pos
        target_x, target_y = target

        new_x = current_x
        new_y = current_y

        if current_x < target_x:
            new_x += 1

        elif current_x > target_x:
            new_x -= 1

        elif current_y < target_y:
            new_y += 1

        elif current_y > target_y:
            new_y -= 1

        new_position = (new_x, new_y)

        self.model.grid.move_agent(
            self,
            new_position
        )

        self.energy -= 1
        self.distance_travelled += 1

        if self.energy <= 0:
            self.energy = 0
            self.active = False

    def random_move(self):
        """
        Move randomly to one neighboring cell.

        This is deliberately a simple exploration strategy.
        """

        if not self.active:
            return

        x, y = self.pos

        possible_positions = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        valid_positions = []

        for position in possible_positions:
            px, py = position

            if (
                0 <= px < self.model.width
                and 0 <= py < self.model.height
            ):
                valid_positions.append(position)

        if valid_positions:
            new_position = self.random.choice(
                valid_positions
            )

            self.model.grid.move_agent(
                self,
                new_position
            )

            self.energy -= 1
            self.distance_travelled += 1

            if self.energy <= 0:
                self.energy = 0
                self.active = False

    def recharge_if_at_base(self):
        """
        Fully recharge the robot when it reaches the Mars base.
        """

        if self.pos == self.model.base_position:
            self.energy = self.max_energy
            self.active = True


class ExplorerAgent(RobotAgent):
    """
    Explorer robots search for resources.

    Baseline rule:

        IF energy < 20:
            return to base
        ELSE:
            move randomly
    """

    def discover_resource(self):
        """
        Check whether a resource exists in the current cell.
        """

        agents_here = self.model.grid.get_cell_list_contents(
            [self.pos]
        )

        for agent in agents_here:

            if isinstance(agent, ResourceAgent):

                if not agent.discovered:

                    agent.discovered = True

                    self.model.known_resources.add(
                        agent.pos
                    )

                    print(
                        f"Explorer {self.unique_id} "
                        f"discovered resource at {agent.pos}"
                    )

    def step(self):
        """
        One local decision cycle of the Explorer.
        """

        if not self.active:
            return

        self.discover_resource()

        # ==============================================
        # STUDENT MODIFICATION AREA
        # ==============================================

        if self.energy < EXPLORER_RETURN_THRESHOLD:

            self.move_one_step_towards(
                self.model.base_position
            )

        else:

            self.random_move()

        self.recharge_if_at_base()


class CollectorAgent(RobotAgent):
    """
    Collector robots travel to known resources,
    collect them, and return them to base.

    The initial strategy is intentionally simple.
    """

    def __init__(self, model, energy=100):
        super().__init__(model, energy)

        self.cargo = 0
        self.capacity = COLLECTOR_CAPACITY
        self.target_resource = None

        self.last_action = ""
        self.last_reason = ""
        self.last_source = ""

    def choose_nearest_resource(self):
        """
        Choose the nearest known resource.

        Collectors do not coordinate their choices, so several
        collectors may choose the same target.
        """

        if not self.model.known_resources:
            return None

        return min(
            self.model.known_resources,
            key=lambda position:
                manhattan_distance(
                    self.pos,
                    position
                )
        )

    def collect_resource(self):
        """
        Collect resource if the robot has reached its target.
        """

        if self.target_resource is None:
            return

        if self.pos != self.target_resource:
            return

        agents_here = self.model.grid.get_cell_list_contents(
            [self.pos]
        )

        for agent in agents_here:

            if isinstance(agent, ResourceAgent):

                amount_to_collect = min(
                    self.capacity,
                    agent.amount
                )

                self.cargo += amount_to_collect
                agent.amount -= amount_to_collect

                if agent.amount <= 0:

                    self.model.known_resources.discard(
                        agent.pos
                    )

                    self.model.grid.remove_agent(agent)
                    agent.remove()

                self.target_resource = None

                return

        # Another collector may already have taken the resource.
        self.model.known_resources.discard(
            self.pos
        )

        self.target_resource = None

    def unload_cargo(self):
        """
        Deliver collected resources to the Mars base.
        """

        if (
            self.pos == self.model.base_position
            and self.cargo > 0
        ):

            self.model.base_resources += self.cargo
            self.model.total_resources_collected += self.cargo

            self.cargo = 0

    def baseline_step(self):
        """Original Folder 2 policy, retained for reading and comparison."""

        if not self.active:
            return

        self.collect_resource()
        self.unload_cargo()
        self.recharge_if_at_base()

        # ==============================================
        # STUDENT MODIFICATION AREA
        # ==============================================

        # Low energy -> return to base.
        if self.energy < COLLECTOR_RETURN_THRESHOLD:

            self.move_one_step_towards(
                self.model.base_position
            )

            return

        # Carrying something -> return to base.
        if self.cargo > 0:

            self.move_one_step_towards(
                self.model.base_position
            )

            return

        # No target yet -> choose nearest known resource.
        if self.target_resource is None:

            self.target_resource = (
                self.choose_nearest_resource()
            )

        # Move toward selected resource.
        if self.target_resource is not None:

            self.move_one_step_towards(
                self.target_resource
            )

    def create_observation(self):
        """Only own state, discovered locations and shared collector targets.

        Deposit quantities and undiscovered positions are deliberately absent.
        A stale current target is remembered locally, just as in Folder 2.
        """
        targets = set(self.model.known_resources)
        if self.target_resource is not None:
            targets.add(self.target_resource)
        if self.cargo:
            allowed = ["RETURN_BASE"]
        else:
            allowed = ["WAIT"]
            if self.pos != self.model.base_position:
                allowed.append("RETURN_BASE")
            allowed.extend(collect_action(pos) for pos in sorted(targets))
        return {
            "step": self.model.steps_survived + 1,
            "agent_id": self.unique_id,
            "position": list(self.pos), "base_position": list(self.model.base_position),
            "energy": self.energy, "max_energy": self.max_energy,
            "cargo": self.cargo, "capacity": self.capacity,
            "current_target": list(self.target_resource) if self.target_resource is not None else None,
            "known_resources": [list(pos) for pos in sorted(self.model.known_resources)],
            "other_collector_intentions": [
                {"agent_id": agent.unique_id, "target": list(agent.target_resource)}
                for agent in self.model.agents
                if isinstance(agent, CollectorAgent) and agent is not self
                and agent.active and agent.target_resource is not None
            ],
            "allowed_actions": allowed,
            # These two fields are for the offline/fallback policy only.
            "return_threshold": COLLECTOR_RETURN_THRESHOLD,
            "baseline_target": self.target_resource if self.target_resource is not None else self.choose_nearest_resource(),
        }

    def step(self):
        """Same physical operations as Folder 2; replace the decision block."""
        if not self.active:
            return
        self.collect_resource()
        self.unload_cargo()
        self.recharge_if_at_base()

        # ======================================================
        # CHALLENGE 2: OBSERVE -> LLM DECISION -> EXECUTE ONE ACTION
        # Students edit prompt.txt; llm_decision.py validates output.
        # ======================================================
        decision = self.model.decision_engine.decide(self.create_observation())
        self.last_action = decision["action"]
        self.last_reason = decision["reason"]
        self.last_source = self.model.decision_engine.records[-1]["source"]
        if self.last_action == "RETURN_BASE":
            # Preserve remembered target, matching the original return rule.
            self.move_one_step_towards(self.model.base_position)
        elif self.last_action.startswith("COLLECT_"):
            _, x, y = self.last_action.split("_")
            self.target_resource = (int(x), int(y))
            self.move_one_step_towards(self.target_resource)
        # WAIT retains memory and consumes no movement energy.
