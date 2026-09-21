"""Identical observational measurements for all three tutorial folders."""

from collections import Counter


class MissionMetrics:
    """Observe ticks without changing the agents or consuming random draws."""

    def __init__(self):
        self.idle_collector_steps = 0
        self.duplicate_target_steps = 0

    def before_step(self, model):
        return {a.unique_id: (a.active, a.distance_travelled)
                for a in model.agents if hasattr(a, "cargo")}

    def after_step(self, model, before):
        collectors = [a for a in model.agents if hasattr(a, "cargo")]
        for agent in collectors:
            was_active, distance = before[agent.unique_id]
            self.idle_collector_steps += int(was_active and agent.distance_travelled == distance)
        targets = Counter(a.target_resource for a in collectors
                          if a.active and a.cargo == 0
                          and a.target_resource in model.known_resources)
        self.duplicate_target_steps += sum(max(0, count - 1) for count in targets.values())

    def report(self, model, seed, max_steps):
        robots = [a for a in model.agents if hasattr(a, "distance_travelled")]
        return {
            "seed": seed, "max_steps": max_steps,
            "steps_survived": model.steps_survived,
            "resources_collected": model.total_resources_collected,
            "base_resources": model.base_resources,
            "distance_travelled": sum(a.distance_travelled for a in robots),
            "failed_robots": sum(not a.active for a in robots),
            "idle_collector_steps": self.idle_collector_steps,
            "duplicate_target_steps": self.duplicate_target_steps,
        }


def simulate(model, max_steps, seed):
    metrics = MissionMetrics()
    for _ in range(max_steps):
        if not model.running:
            break
        before = metrics.before_step(model)
        model.step()
        metrics.after_step(model, before)
    return metrics.report(model, seed, max_steps)

