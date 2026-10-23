"""
Co-presence tracking for the virtual town.

The town has a single location: every agent is assigned to "Workplace"
each round, with no movement between locations, as the sandbox
environment of Park et al. (2023) is not used.
"""

from pathlib import Path
import json

LOCATIONS = ["Workplace"]
ROUNDS = ["Morning", "Afternoon", "Evening"]


class Town:
    def __init__(self, agents: list):
        """
        Initialises the town with a list of Agent objects.
        """
        self.agents = agents
        self.current_round = None
        self.current_day = 0

        # location name -> agents currently there
        self.locations = {loc: [] for loc in LOCATIONS}

    def assign_locations(self, round_name: str, curr_time, tracker=None):
        """
        Assigns every agent to "Workplace" for the given round.
        """
        self.current_round = round_name

        for loc in self.locations:
            self.locations[loc] = []

        for agent in self.agents:
            agent.location = "Workplace"
            self.locations["Workplace"].append(agent)

    def get_agents_at(self, location: str) -> list:
        """
        Returns list of agents currently at a given location.
        """
        return self.locations.get(location, [])

    def get_co_present_agents(self, agent) -> list:
        """
        Returns list of agents at the same location as the given agent,
        excluding the agent itself.
        """
        return [
            a for a in self.locations.get(agent.location, [])
            if a.name != agent.name
        ]

    def get_location_summary(self) -> str:
        """
        Returns a readable summary of agent locations for logging.
        """
        summary = f"Day {self.current_day}, {self.current_round} Round\n"
        summary += "-" * 40 + "\n"
        for loc, agents in self.locations.items():
            names = ", ".join([a.name for a in agents]) if agents else "empty"
            summary += f"{loc}: {names}\n"
        return summary

    def save_location_log(self, log_dir: str):
        """
        Saves the current location assignment to a JSON log file.
        """
        path = Path(log_dir) / f"day{self.current_day}_{self.current_round.lower()}_locations.json"
        data = {
            "day": self.current_day,
            "round": self.current_round,
            "locations": {
                loc: [a.name for a in agents]
                for loc, agents in self.locations.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
