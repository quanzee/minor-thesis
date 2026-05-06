"""
File: town.py
Description: Manages the three locations in the virtual town and 
agent location assignment per simulation round.
"""

from pathlib import Path
import json


LOCATIONS = ["University", "Workplace", "Communal Space"]
ROUNDS = ["Morning", "Afternoon", "Evening"]


class Town:
    def __init__(self, agents: list):
        """
        Initialises the town with a list of Agent objects.
        Assigns each agent to their primary location based on job type.
        
        INPUT:
            agents: list of Agent objects
        """
        self.agents = agents
        self.current_round = None
        self.current_day = 0

        # maps location name to list of agents currently there
        self.locations = {loc: [] for loc in LOCATIONS}

    def assign_locations(self, round_name: str, curr_time, tracker=None):
        """
        Assigns agents to locations for a given round.
        
        - Morning and Afternoon: agents use plan_location to decide,
          defaulting to their primary institution unless memories
          suggest otherwise
        - Evening: all agents go to Communal Space
        
        INPUT:
            round_name: "Morning", "Afternoon", or "Evening"
            curr_time: current simulation datetime
        """
        self.current_round = round_name

        # reset all locations
        for loc in self.locations:
            self.locations[loc] = []

        if round_name == "Evening":
            # everyone goes to communal space
            for agent in self.agents:
                agent.location = "Communal Space"
                self.locations["Communal Space"].append(agent)
        else:
            # morning and afternoon — agent decides location
            for agent in self.agents:
                location = agent.plan_location(curr_time, tracker=tracker)
                agent.location = location
                self.locations[location].append(agent)

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
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

