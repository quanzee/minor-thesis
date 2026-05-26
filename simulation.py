"""
File: simulation.py
Description: Main simulation loop for cultural drift study.
Runs a 15-day virtual town simulation with 3 rounds per day,
administering VSM13 at T0, T5, T10, and T15.
Adapted from Park et al. (2023) Generative Agents architecture.
"""

import json
import yaml
import random
from pathlib import Path

from simplified_persona import Agent
from town import Town, ROUNDS
from vsm13 import run_vsm13, compute_dimensions
from kendall_tau import compute_kendall_tau, HUMAN_SCORES
from config import MODEL_NAME, client
from cognitive_modules.simplified_retrieve import get_embedding
from token_tracker import TokenTracker
print("Imports loaded successfully")

# ── Configuration ─────────────────────────────────────────────────────────────

TOTAL_DAYS = 3
VSM13_DAYS = {3}
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("simulation_results")

INSTITUTIONAL_CONTEXT = {
    ("University", "Morning"):      "Students and staff are engaged in academic work and discussion.",
    ("University", "Afternoon"):    "Students and staff are engaged in academic work and discussion.",
    ("University", "Evening"):      "A few students are studying late or attending evening sessions.",
    ("Workplace", "Morning"):       "Colleagues are working together on shared tasks and responsibilities.",
    ("Workplace", "Afternoon"):     "Colleagues are working together on shared tasks and responsibilities.",
    ("Workplace", "Evening"):       "A few staff members are finishing up their work for the day.",
    ("Communal Space", "Morning"):  "A few people are starting their day with coffee and casual conversation.",
    ("Communal Space", "Afternoon"): "People are taking a break and socialising.",
    ("Communal Space", "Evening"):  "People are socialising and relaxing after the day's work.",
}

# ── Logging helpers ────────────────────────────────────────────────────────────

def append_log(filepath: Path, entry: dict):
    """Appends a single JSON entry to a .jsonl log file."""
    with open(filepath, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def setup_logs():
    """Creates log and results directories."""
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


# ── Observation generation ─────────────────────────────────────────────────────

def build_observation(agent, round_name: str) -> str:
    """
    Builds a contextual observation string for an agent
    based on their current location and round.
    """
    context = INSTITUTIONAL_CONTEXT.get(
        (agent.location, round_name),
        "People are going about their day." #FALLBACK
    )
    return (
        f"{agent.name} is at the {agent.location} "
        f"during the {round_name} round. {context}"
    )

# ── VSM13 measurement ──────────────────────────────────────────────────────────

def run_measurement(agents: list, day: int, sim_time=None, tracker=None, n_runs=5):
    print(f"\n{'='*60}")
    print(f"VSM13 Measurement — Day {day} ({n_runs} runs per agent)")
    print(f"{'='*60}")

    cultures = ["Australia", "Japan", "Malaysia"]
    culture_responses = {c: {} for c in cultures}
    culture_explanations = {c: {} for c in cultures}
    culture_all_runs = {c: {} for c in cultures}
    culture_run_dimensions = {c: {} for c in cultures}  # NEW

    for agent in agents:
        print(f"  Administering VSM13 to {agent.name} ({n_runs} runs)...")
        all_runs = []
        last_explanations = {}
        per_run_dims = {}  # NEW

        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}...")
            responses, explanations = run_vsm13(
                agent, curr_time=sim_time, tracker=tracker
            )
            all_runs.append(responses)
            last_explanations = explanations
            per_run_dims[f"run_{run + 1}"] = compute_dimensions(responses)  # NEW

        averaged = {}
        for q in range(1, 25):
            averaged[q] = sum(r[q] for r in all_runs) / n_runs

        culture_responses[agent.country][agent.name] = averaged
        culture_explanations[agent.country][agent.name] = last_explanations
        culture_all_runs[agent.country][agent.name] = {
            f"run_{i+1}": {str(k): v for k, v in run.items()}
            for i, run in enumerate(all_runs)
        }
        culture_run_dimensions[agent.country][agent.name] = per_run_dims  # NEW

    pooled = {}
    individual_dimensions = {c: {} for c in cultures}
    pooled_run_dimensions = {c: {} for c in cultures}  # NEW

    for culture in cultures:
        agent_responses = culture_responses[culture]
        if not agent_responses:
            continue
        n = len(agent_responses)

        for name, responses in agent_responses.items():
            individual_dimensions[culture][name] = compute_dimensions(responses)

        pooled_means = {}
        for q in range(1, 25):
            pooled_means[q] = sum(
                agent_responses[name][q] for name in agent_responses
            ) / n
        pooled[culture] = compute_dimensions(pooled_means)

        # NEW: pooled dimension scores per run across all agents in this culture
        for run_idx in range(n_runs):
            run_key = f"run_{run_idx + 1}"
            run_pooled_means = {}
            for q in range(1, 25):
                run_pooled_means[q] = sum(
                    culture_all_runs[culture][name][run_key][str(q)]
                    for name in culture_all_runs[culture]
                ) / n
            pooled_run_dimensions[culture][run_key] = compute_dimensions(run_pooled_means)

    tau_results = compute_kendall_tau(HUMAN_SCORES, pooled)

    output = {
        "day": day,
        "n_runs": n_runs,
        "individual_responses_averaged": {
            culture: {
                name: {str(k): v for k, v in resp.items()}
                for name, resp in responses.items()
            }
            for culture, responses in culture_responses.items()
        },
        "individual_all_runs": culture_all_runs,
        "individual_explanations_last_run": {
            culture: {
                name: {str(k): v for k, v in expl.items()}
                for name, expl in explanations.items()
            }
            for culture, explanations in culture_explanations.items()
        },
        "individual_dimension_scores": individual_dimensions,
        "individual_run_dimensions": culture_run_dimensions,   # NEW
        "pooled_dimension_scores": pooled,
        "pooled_run_dimensions": pooled_run_dimensions,        # NEW
        "kendall_tau": tau_results
    }

    path = RESULTS_DIR / f"vsm13_day{day}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Average Kendall tau (Day {day}): {tau_results.get('average_tau', 'N/A'):.4f}")
    print(f"  Results saved to {path}")

# ── Scenario Helper Functions ──────────────────────────────────────────────────────────
def load_scenarios(path="scenarios_v3.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["scenarios"]

def inject_scenarios(town, scenarios, day, round_name, sim_time, tracker=None):
    matching = [
        s for s in scenarios
        if s["day"] == day and s["round"] == round_name
    ]

    if not matching:
        return

    for scenario in matching:
        location = scenario["location"]
        agents_at_location = [
            a for a in town.agents
            if a.location == location
        ]
        if not agents_at_location:
            print(f"      WARNING: No agents at {location} for Day {day} {round_name} scenario")
            continue

        for agent in agents_at_location:
            agent.active_scenario = {
                "text": scenario["text"],
                "day": day,
                "round": round_name,
                "location": location,
                "sim_time": sim_time
            }
            print(f"      Scenario activated for {agent.name} ({location})")
            
# ── Round execution ────────────────────────────────────────────────────────────

def run_round(town: Town, round_name: str, day: int,
              sim_time: str, scenarios: list, tracker=None):
    print(f"\n  [{round_name} Round]")

    # clear scenarios from previous day at the start of each Morning round
    if round_name == "Morning":
        for agent in town.agents:
            if agent.active_scenario and agent.active_scenario["day"] < day:
                agent.active_scenario = None

    # assign locations for this round
    print(f"  Assigning locations...")
    town.assign_locations(round_name, sim_time, tracker=tracker)
    print(f"  Processing {len(town.agents)} agents...")
    town.current_round = round_name
    print(town.get_location_summary())

    # log location assignments
    append_log(
        LOGS_DIR / "locations.jsonl",
        {
            "day": day,
            "round": round_name,
            "locations": {
                loc: [a.name for a in agents]
                for loc, agents in town.locations.items()
            }
        }
    )

    # track conversations that have already happened this round
    already_conversed = set()

    # inject scenarios for this round
    inject_scenarios(town, scenarios, day, round_name, sim_time, tracker)

    # randomise agent processing order each round
    agents_this_round = list(town.agents)
    random.shuffle(agents_this_round)

    # run each agent sequentially, following Park et al.
    for agent in agents_this_round:
        print(f"    Processing {agent.name}...")
        # step 1: perceive
        observation = build_observation(agent, round_name)
        agent.perceive(observation, sim_time, tracker=tracker)
        print(f"      Perceived environment")

        # step 2: plan — who to talk to
        co_present = town.get_co_present_agents(agent)

        # filter out agents already conversed with this round
        available = [
            a for a in co_present
            if tuple(sorted([agent.name, a.name])) not in already_conversed
        ]

        target = agent.plan(available, sim_time, tracker=tracker) if available else None
        print(f"      Plan: {'talking to ' + target.name if target else 'no conversation'}")

        # step 3: converse if target found
        if target:
            pair = tuple(sorted([agent.name, target.name]))
            if pair not in already_conversed:
                already_conversed.add(pair)

                print(f"    {agent.name} ({agent.country}) → "
                    f"{target.name} ({target.country})")

                conversation, reasoning_log = agent.converse(target, sim_time, tracker=tracker)

                #update recent partners
                agent.recent_partners.append(target.name)
                target.recent_partners.append(agent.name)

                # keep only last 3
                agent.recent_partners = agent.recent_partners[-3:]
                target.recent_partners = target.recent_partners[-3:]

                # log interaction
                append_log(
                    LOGS_DIR / "interactions.jsonl",
                    {
                        "day": day,
                        "round": round_name,
                        "agent_a": agent.name,
                        "culture_a": agent.country,
                        "agent_b": target.name,
                        "culture_b": target.country,
                        "location": agent.location,
                        "cross_cultural": agent.country != target.country,
                        "reasoning": reasoning_log,
                        "dialogue": conversation
                    }
                )

        # step 4: reflect if threshold met
        reflected, focal_points, all_insights = agent.reflect(sim_time, tracker=tracker)

        # log reflection
        if reflected:
            recent_thoughts = [
                n.description
                for n in agent.memory.seq_thought[-3:]
            ]
            append_log(
                LOGS_DIR / "reflections.jsonl",
                {
                    "day": day,
                    "round": round_name,
                    "agent": agent.name,
                    "culture": agent.country,
                    "focal_points": focal_points,
                    "recent_thoughts": recent_thoughts,
                    "insights": all_insights
                }
            )

# ── Main simulation loop ───────────────────────────────────────────────────────

def run_simulation(agents: list):
    """
    Main simulation loop. Runs for TOTAL_DAYS days with 3 rounds per day.
    Administers VSM13 at days specified in VSM13_DAYS.
    """
    scenarios = load_scenarios()
    setup_logs()
    town = Town(agents)
    tracker = TokenTracker()

    TOKEN_LOG = LOGS_DIR / "token_usage.jsonl"

    tracker.start_run(TOKEN_LOG, run_label="Test Dimension 1 Simulation")

    # # T0 measurement before simulation starts
    # run_measurement(agents, day=0, sim_time="Day 00", tracker=tracker)
    # tracker.report("Day 0 (T0 Measurement)")
    # tracker.append_log("Day 00 T0 Measurement", TOKEN_LOG)
    # tracker.reset()

    # save initial memory state
    snapshot_dir = RESULTS_DIR / "memory_snapshots" / "day00"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        agent.save(str(snapshot_dir))

    for day in range(1, TOTAL_DAYS + 1):
        tracker.reset()
        print(f"\n{'='*60}")
        print(f"Day {day}")
        print(f"{'='*60}")

        town.current_day = day

        for round_name in ROUNDS:
            if round_name == "Morning":
                sim_time = f"Day {day:02d}, Morning"   # produces "Day 01, Morning"
            elif round_name == "Afternoon":
                sim_time = f"Day {day:02d}, Afternoon"
            else:
                sim_time = f"Day {day:02d}, Evening"

            run_round(town, round_name, day, sim_time, scenarios, tracker)

        tracker.report(f"Day {day}")
        tracker.append_log(f"Day {day:02d} Rounds", TOKEN_LOG)

        # debug: dump memory streams after day 1
        memory_debug = []
        for agent in agents:
            agent_memory = {
                "agent": agent.name,
                "country": agent.country,
                "nodes": []
            }
            for node in agent.memory.seq_event + agent.memory.seq_thought + agent.memory.seq_chat:
                agent_memory["nodes"].append({
                    "created": node.created,
                    "node_type": node.node_type,
                    "description": node.description,
                    "poignancy": node.poignancy
                })
            memory_debug.append(agent_memory)

        with open(LOGS_DIR / "memory_debug.json", "w") as f:
            json.dump(memory_debug, f, indent=2, ensure_ascii=False)

        # VSM13 measurement at specified days
        if day in VSM13_DAYS:
            run_measurement(agents, day=day, sim_time=sim_time, tracker=tracker)
            tracker.report(f"Day {day} VSM13 Measurement")
            tracker.append_log(f"Day {day:02d} VSM13 Measurement", TOKEN_LOG)

            # save memory state at measurement points
            snapshot_dir = RESULTS_DIR / "memory_snapshots" / f"day{day:02d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for agent in agents:
                agent.save(str(snapshot_dir))

    print(f"\n{'='*60}")
    print("Simulation complete.")
    print(f"Logs saved to: {LOGS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")

    tracker.end_run(TOKEN_LOG)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Starting simulation...")

    def embedding_fn(text: str):
        return get_embedding(text, client)

    # load all agents from YAML files
    agent_dirs = {
        "australia": Path("agents_australia"),
        "japan": Path("agents_japan"),
        "malaysia": Path("agents_malaysia"),
    }

    agents = []
    for culture, directory in agent_dirs.items():
        yaml_files = sorted(directory.glob("*.yaml"))
        for yaml_path in yaml_files:
            agent = Agent.from_yaml(yaml_path, embedding_fn)
            agents.append(agent)
            print(f"Loaded: {agent.name} ({agent.country})")

    print(f"\nTotal agents loaded: {len(agents)}")
    run_simulation(agents)


if __name__ == "__main__":
    main()