"""Interactive Solara dashboard for the configured Mars Colony model.

Run from this directory with::

    solara run app.py
"""

import importlib
import os

import mesa
import pandas as pd
import solara
from matplotlib.lines import Line2D

from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from agents import CollectorAgent, ExplorerAgent, ResourceAgent, RobotAgent
from config import (
    CONSUMPTION_RATE,
    DEFAULT_SEED,
    GRID_HEIGHT,
    GRID_WIDTH,
    INITIAL_BASE_RESOURCES,
    NUMBER_COLLECTORS,
    NUMBER_EXPLORERS,
    NUMBER_RESOURCES,
)
from model import MarsColonyModel
from metrics import MissionMetrics
from config import MAX_STEPS

RUN_SEED = int(os.getenv("MARS_RUN_SEED", str(DEFAULT_SEED)))
RUN_STEPS = int(os.getenv("MARS_RUN_STEPS", str(MAX_STEPS)))


def dashboard_layout(number_components):
    """Give the prompt/decision log a full-width row for readable text."""
    primary_cards = [
        {"i": 0, "w": 4, "h": 14, "x": 0, "y": 0, "moved": False},
        {"i": 1, "w": 4, "h": 14, "x": 4, "y": 0, "moved": False},
        {"i": 2, "w": 4, "h": 14, "x": 8, "y": 0, "moved": False},
        {"i": 3, "w": 12, "h": 32, "x": 0, "y": 14, "moved": False},
    ]
    plot_row = [
        {
            "i": index,
            "w": 4,
            "h": 10,
            "x": 4 * (index - 4),
            "y": 46,
            "moved": False,
        }
        for index in range(4, number_components)
    ]
    return (primary_cards + plot_row)[:number_components]


# Mesa 3.5 does not yet expose the initial draggable-card layout as a
# SolaraViz argument, so configure its layout hook before creating the page.
solara_viz_module = importlib.import_module("mesa.visualization.solara_viz")
solara_viz_module.make_initial_grid_layout = dashboard_layout


class BaseMarkerAgent(mesa.Agent):
    """A stationary agent used only to keep the base visible in every frame."""


class VisualMarsColonyModel(MarsColonyModel):
    """Mars Colony model with a permanent visualization marker for the base."""

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
    ):
        # Keep an explicit signature: Mesa uses it to decide which parameters
        # are safe to pass back to the model when the Reset button is pressed.
        super().__init__(
            width=width,
            height=height,
            number_explorers=number_explorers,
            number_collectors=number_collectors,
            number_resources=number_resources,
            initial_base_resources=initial_base_resources,
            consumption_rate=consumption_rate,
            seed=seed,
        )
        self.base_marker = BaseMarkerAgent(self)
        self.grid.place_agent(self.base_marker, self.base_position)
        self.mission_metrics = MissionMetrics()

    def step(self):
        if not self.running:
            return
        before = self.mission_metrics.before_step(self)
        super().step()
        self.mission_metrics.after_step(self, before)
        if self.steps_survived >= RUN_STEPS:
            self.running = False


def agent_portrayal(agent):
    """Return the visual style for each kind of colony agent."""
    if isinstance(agent, BaseMarkerAgent):
        return AgentPortrayalStyle(
            color="none",
            marker="*",
            size=480,
            zorder=0.5,
            edgecolors="#f4d03f",
            linewidths=2.2,
        )

    if isinstance(agent, ResourceAgent):
        return AgentPortrayalStyle(
            color="#2ecc71" if agent.discovered else "#7f8c8d",
            marker="D",
            size=max(35, agent.amount * 4),
            zorder=1,
            alpha=0.9,
            edgecolors="#17202a",
        )

    common = {
        "size": 105,
        "zorder": 3,
        "alpha": 1.0 if agent.active else 0.35,
        "edgecolors": "white",
        "linewidths": 1.2,
    }
    if isinstance(agent, ExplorerAgent):
        return AgentPortrayalStyle(color="#3498db", marker="^", **common)
    if isinstance(agent, CollectorAgent):
        collector_color = "#e74c3c" if agent.cargo > 0 else "#f39c12"
        return AgentPortrayalStyle(color=collector_color, marker="s", **common)

    return AgentPortrayalStyle(color="#ecf0f1", **common)


def style_grid(ax):
    """Apply Mars styling to the grid."""
    ax.set_facecolor("#3b1f1a")
    ax.set_title("Mars surface — star marks the colony base")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(color="white", alpha=0.10, linewidth=0.5)

    legend = [
        Line2D([], [], marker="^", linestyle="", color="#3498db", label="Explorer"),
        Line2D([], [], marker="s", linestyle="", color="#f39c12", label="Empty collector"),
        Line2D([], [], marker="s", linestyle="", color="#e74c3c", label="Loaded collector"),
        Line2D([], [], marker="D", linestyle="", color="#7f8c8d", label="Undiscovered resource"),
        Line2D([], [], marker="D", linestyle="", color="#2ecc71", label="Discovered resource"),
        Line2D([], [], marker="*", linestyle="", markerfacecolor="none", markeredgecolor="#f4d03f", markersize=13, label="Base"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        frameon=False,
    )


def mission_status(model):
    """Show the current values that matter most during a run."""
    robots = [agent for agent in model.agents if isinstance(agent, RobotAgent)]
    active = sum(robot.active for robot in robots)
    energy = sum(robot.energy for robot in robots)
    cargo = sum(
        agent.cargo for agent in robots if isinstance(agent, CollectorAgent)
    )
    status = "Operational" if model.running else ("Step limit reached" if model.base_resources > 0 else "Colony resources depleted")
    engine = model.decision_engine
    mode = "Offline demo (coded rules)" if engine.mode == "mock" else f"{engine.mode.title()}: {engine.model_name}"

    with solara.Card("Mission status"):
        solara.Markdown(
            f"""
**Status:** {status}  
**Step:** {model.steps_survived}  
**Base resources:** {model.base_resources}  
**Delivered:** {model.total_resources_collected}  
**Known deposits:** {len(model.known_resources)}  
**Active robots:** {active}/{len(robots)}  
**Robot energy / cargo:** {energy} / {cargo}

**Decision mode:** {mode}  
**API calls:** {engine.api_calls}/{engine.max_calls}  
**LLM decisions:** {engine.llm_decisions}  
**Continued LLM actions:** {engine.continued_decisions}  
**Fallback decisions:** {engine.fallbacks}  
**Idle collector ticks:** {model.mission_metrics.idle_collector_steps}  
**Duplicate target ticks:** {model.mission_metrics.duplicate_target_steps}
"""
        )
        if engine.records:
            last = engine.records[-1]
            solara.Text(f"Collector {last['agent_id']}: {last['action']} ({last['source']})")
            solara.Text(last["reason"])
        if engine.mode == "gemini":
            solara.Text(f"Requests spaced by {engine.min_request_seconds:g}s. Use a Free Tier project key.")


def agent_table(model):
    """Display the current state of every agent in the model."""
    rows = []
    for agent in model.agents:
        if isinstance(agent, BaseMarkerAgent):
            agent_type = "Base"
        elif isinstance(agent, ExplorerAgent):
            agent_type = "Explorer"
        elif isinstance(agent, CollectorAgent):
            agent_type = "Collector"
        elif isinstance(agent, ResourceAgent):
            agent_type = "Resource"
        else:
            agent_type = type(agent).__name__

        rows.append(
            {
                "ID": agent.unique_id,
                "Type": agent_type,
                "Position": str(agent.pos),
                "Active": getattr(agent, "active", ""),
                "Energy": getattr(agent, "energy", ""),
                "Cargo": getattr(agent, "cargo", ""),
                "Capacity": getattr(agent, "capacity", ""),
                "Target": str(getattr(agent, "target_resource", ""))
                if getattr(agent, "target_resource", None) is not None
                else "",
                "Resource amount": getattr(agent, "amount", ""),
                "Discovered": getattr(agent, "discovered", ""),
                "Distance": getattr(agent, "distance_travelled", ""),
                "Decision": getattr(agent, "last_action", ""),
                "Source": getattr(agent, "last_source", ""),
            }
        )

    data = pd.DataFrame(rows).sort_values(["Type", "ID"], ignore_index=True)
    with solara.Card("Agent data"):
        solara.DataFrame(data, items_per_page=10, scrollable=True)


def decision_log(model):
    """Show recent per-agent prompts and decisions recorded by the engine."""
    engine = model.decision_engine
    rows = [
        {
            "Step": record["step"],
            "Agent": record["agent_id"],
            "API called": record["api_called"],
            "Source": record["source"],
            "Action": record["action"],
            "Reason": record["reason"],
            "API reply": record["api_reply"],
            "Error": record["error"] or "",
            "Seconds": record["seconds"],
            "Observation prompt": record["prompt"],
        }
        for record in reversed(engine.records[-200:])
    ]
    with solara.Card("Agent prompts and decisions"):
        solara.Markdown("**System prompt used for API decisions**")
        solara.Markdown(f"```text\n{engine.prompt}\n```")
        if engine.csv_path is not None:
            solara.Text(f"CSV log: {engine.csv_path}")
        if rows:
            solara.DataFrame(pd.DataFrame(rows), items_per_page=10, scrollable=True)
        else:
            solara.Text("No collector decisions have been made yet.")


# Fixed experiment settings; reset reruns the same world and resets API budget.
model_params = {
    "width": GRID_WIDTH, "height": GRID_HEIGHT,
    "number_explorers": NUMBER_EXPLORERS,
    "number_collectors": NUMBER_COLLECTORS,
    "number_resources": NUMBER_RESOURCES,
    "initial_base_resources": INITIAL_BASE_RESOURCES,
    "consumption_rate": CONSUMPTION_RATE,
    "seed": RUN_SEED,
}

model = VisualMarsColonyModel(seed=RUN_SEED)
renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.draw_structure()
renderer.draw_agents()
renderer.post_process = style_grid

page = SolaraViz(
    model,
    renderer,
    components=[
        (agent_table, 0),
        (mission_status, 0),
        (decision_log, 0),
        make_plot_component(
            {"Base Resources": "#c0392b", "Resources Collected": "#27ae60"},
            page=0,
        ),
        make_plot_component(
            {"Active Robots": "#2980b9", "Known Resources": "#8e44ad"},
            page=0,
        ),
        make_plot_component("Distance Travelled", page=0),
    ],
    model_params=model_params,
    name="Mars Colony — Mission 2: AI Reasoning",
    play_interval=250,
)

page  # noqa: B018
